using Lux, NNlib, Random, Tools, BFloat16s, YAML, Statistics, LuxCore
# include("../embodied/lux/rms.jl");
# include("../embodied/lux/nets.jl");
# include("../embodied/lux/BlockLinear.jl");
# include("../embodied/lux/ReArrange.jl");
# include("../embodied/lux/UpSample.jl");
config = YAML.load_file("dreamerv3/configs.yaml");

struct Decoder{SD, SS, DC} <: Lux.AbstractLuxContainerLayer{(:spatialize_deter, :spatialize_stoch, :deconvolve)}
    act::Function
    mults::Vector
    depth::Int
    kernel::Int
    obs::Space
    depths::Vector{Int}
    shape::Vector{Int}
    bspace::Int
    deter_dim::Int
    stoch_dim::Int
    classes_dim::Int
    units::Int
    # Network components (Lux layers)
    spatialize_deter::SD
    spatialize_stoch::SS
    deconvolve::DC
end


function Decoder(;
    obs::Space,
    deter_dim::Int,
    units::Int,
    stoch_dim::Int,
    classes_dim::Int,
    act::Function=gelu,
    mults::Vector=[2, 3, 4, 4],
    depth::Int=64,
    kernel::Int=5,
    bspace::Integer = 8)

    depths = [depth * m for m in mults];
    # Workout the final spatial dimensions after all CNN layers:
    # 1. Calculate the total downsampling factor
    # Example: if self.depths = [64, 128, 256, 512] and self.outer=False
    # factor = 2^4 = 16 (image will be downsampled by 16)
    factor = 2 ^ length(depths);
    # 2. Calculate the minimum spatial dimensions after all CNN layers
    minres = [Integer(x // factor) for x in obs.size[1:2]];
    # If input image is 96x96 and factor=16
    # minres = [96//16, 96//16] = [6, 6]
    # 3. Assert the final resolution is reasonable
    @assert 3 <= minres[1] <= 16 "minres[1] must be between 3 and 16 $minres"
    @assert 3 <= minres[2] <= 16 "minres[2] must be between 3 and 16 $minres"
    # 4. Final shape includes the number of channels from last layer
    shape = [minres..., depths[end]]; # final spatial dimensions after all CNN layers

    u, g = prod(shape), bspace;
    w, h, c = shape;

    # construct the network ---------------------------------------------------

    # 1. Spatialize the latent vector (width, height, channels) to feed into the CNN ---
    # it bridges the gap between the flat state representation (from the RSSM) and
    # the spatial structure needed for image generation through the decoder's
    # convolutional layers.
    # This is going to be applied to the deter part of the state
    spatialize_deter = Chain(
        # BlockLinear layer transforms the latent vector into spatial features:
        # Input:  (deter_dim, batch*seq)        # e.g. (8, 80) for batch=8, seq=10
        # Output: (u, batch*seq)                # e.g. (288, 80) where u = w*h*c
        # where u = prod(shape) = w*h*c         # e.g. 6*6*8 = 288
        # and g = bspace (number of blocks)     # e.g. 8 blocks
        # Blocks create sparse connections by processing input in independent groups
        BlockLinear(deter_dim, u, g; init_weight=cast_glorot_uniform, init_bias=cast_zeros), # sp0 in python
        # ReArrange layer reshapes the output of BlockLinear to spatial dimensions:
        # Input:  (u, batch*seq)                # e.g. (288, 80)
        # Output: (w, h, c, batch*seq)         # e.g. (6, 6, 8, 80)
        ReArrange((w, h, c, :))
    );

    # 2. Spatialize the stochastic variables (stoch_dim, classes_dim, seq_length, batch_size) ---
    # to feed into the CNN
    # x1 dimension: (stoch_dim x classes_dim, seq_length x batch_size)
    # This is going to be applied to the stoch part of the state
    spatialize_stoch = Chain(
        # Dense layer transforms the stochastic variables into spatial features:
        # Input: (stoch_dim * classes_dim, batch*seq) # e.g. (8, 80)
        # Output: (2*units, batch*seq)                     # e.g. (16, 80)
        Dense(stoch_dim * classes_dim => 2units, act; init_weight=cast_glorot_uniform, init_bias=cast_zeros), # sp1 in python
        # Normalize along feature dim (dim 1), includes activation
        # Input/Output: (2*units, batch*seq)               # e.g. (16, 80)
        RMSNorm((2units,), 1, act; dims=(1,), init_scale=cast_ones), # sp1norm in python
        # Dense layer transforms intermediate features into spatial features matching deter:
        # Input: (2*units, batch*seq)                      # e.g. (16, 80)
        # Output: (u, batch*seq)                           # e.g. (288, 80)
        Dense(2units => u; init_weight=cast_glorot_uniform, init_bias=cast_zeros), # sp2 in python (without act, handled by ReArrange below)
        # ReArrange layer reshapes the output to spatial dimensions:
        # Input:  (u, batch*seq)                           # e.g. (288, 80)
        # Output: (w, h, c, batch*seq)                    # e.g. (6, 6, 8, 80)
        ReArrange((w, h, c, :))
    );

    # 3. Deconvolve the spatialized deter and stoch parts ---
    deconv_layers_list = []
    println("DEBUG: Inside Decoder constructor, about to add first RMSNorm")
    # FINAL ATTEMPT: Add feature_dim=3, keep dims=(3,) for channel norm
    push!(deconv_layers_list, RMSNorm((c,), 3, act; dims=(3,), init_scale=cast_ones)); # spnorm

    current_channels = c
    for depth_out in reverse(depths[1:end-1])
        push!(deconv_layers_list, Lux.Upsample(:nearest; scale=(2, 2, 1, 1)))
        push!(deconv_layers_list, Conv((kernel, kernel), current_channels => depth_out; pad=SamePad(), init_weight=cast_glorot_uniform, init_bias=cast_zeros))
        # FINAL ATTEMPT: Add feature_dim=3, keep dims=(3,) for channel norm
        push!(deconv_layers_list, RMSNorm((depth_out,), 3, act; dims=(3,), init_scale=cast_ones))
        current_channels = depth_out
    end

    # Map to image channels
    imgdep = obs.size[3]
    # Final Upsample - Use Lux.Upsample
    push!(deconv_layers_list, Lux.Upsample(:nearest; scale=(2, 2, 1, 1)))
    # Final Convolution
    push!(deconv_layers_list, Conv((kernel, kernel), current_channels => imgdep; pad=SamePad(), init_weight=cast_glorot_uniform, init_bias=cast_zeros))

    deconv_layers = Chain(deconv_layers_list...)

    # return the decoder struct ------------------------------------------------------
    return Decoder(act, mults, depth, kernel, obs, depths, shape, bspace, deter_dim, stoch_dim, classes_dim, units, spatialize_deter, spatialize_stoch, deconv_layers)
end


"""
Decoder for RSSM
"""
function (dec::Decoder)(feat, ps, st)

    # Get runtime dimensions T, B from input 'feat'
    # Assuming feat["deter"] has shape (deter_dim, T, B)
    _, T, B = size(feat.deter)

    # Prepare inputs: Flatten sequence and batch dimensions
    deter_flat = reshape(feat.deter, (dec.deter_dim, :));
    stoch_flat = reshape(feat.stoch, (dec.stoch_dim * dec.classes_dim, :));

    # 1. Spatialize the deter and stoch parts
    # Pass the corresponding subset of parameters and states
    out_deter, _ = dec.spatialize_deter(deter_flat, ps.spatialize_deter, st.spatialize_deter)
    out_stoch, _ = dec.spatialize_stoch(stoch_flat, ps.spatialize_stoch, st.spatialize_stoch)

    # 2. Combine spatialized features
    input_decoder = out_deter + out_stoch;

    # 3. Deconvolve the combined input
    out_deconv, st_deconv_new = dec.deconvolve(input_decoder, ps.deconvolve, st.deconvolve)

    # 4. Apply final activation (sigmoid)
    out_sigmoid = sigmoid.(out_deconv)

    # 5. Reshape the output to include T and B dimensions
    # Target shape: (Width, Height, Channels, Time, Batch)
    output = reshape(out_sigmoid, (dec.obs.size..., T, B))

    return output, st
end;






# # Usage example with debug parameters:
# obs = Dict(
#     :image => Tools.Space(UInt8, (96, 96, 1)),
# );

# rng = Random.default_rng();
# enc = Encoder(; obs=obs[:image], kernel=5, depth=2);
# ps, st = Lux.setup(rng, enc.net);

# # Forward pass
# seq_length = 10;
# batch_size = 8;
# obs = (image = rand(UInt8, 96, 96, 1, seq_length, batch_size),);
# output, new_st = forward(enc, st, ps, obs);
# size(output)
