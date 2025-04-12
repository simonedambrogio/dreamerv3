using Lux, NNlib, Random, Tools, BFloat16s, YAML, Statistics, LuxCore
include("../embodied/lux/rms.jl");
include("../embodied/lux/nets.jl");
include("../embodied/lux/BlockLinear.jl");
include("../embodied/lux/ReArrange.jl");
include("../embodied/lux/UpSample.jl");
config = YAML.load_file("dreamerv3/configs.yaml");

struct Decoder{SD, SS, DC} <: Lux.AbstractLuxContainerLayer{(:spatialize_deter, :spatialize_stoch, :deconvolve)}
    act::Function
    mults::Tuple
    depth::Int
    kernel::Int
    obs::Space
    depths::Tuple
    shape::Vector{Int}
    bspace::Int
    deter_dim::Int
    stoch_vars::Int
    classes_per_vars::Int
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
    stoch_vars::Int,
    classes_per_vars::Int,
    act::Function=gelu,
    mults::Tuple=(2, 3, 4, 4),
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

    # 2. Spatialize the stochastic variables (stoch_vars, classes_per_vars, seq_length, batch_size) ---
    # to feed into the CNN
    # x1 dimension: (stoch_vars x classes_per_vars, seq_length x batch_size)
    # This is going to be applied to the stoch part of the state
    spatialize_stoch = Chain(
        # Dense layer transforms the stochastic variables into spatial features:
        # Input: (stoch_vars * classes_per_vars, batch*seq) # e.g. (8, 80)
        # Output: (2*units, batch*seq)                     # e.g. (16, 80)
        Dense(stoch_vars * classes_per_vars => 2units, act; init_weight=cast_glorot_uniform, init_bias=cast_zeros), # sp1 in python
        # Normalize along feature dim (dim 1), includes activation
        # Input/Output: (2*units, batch*seq)               # e.g. (16, 80)
        RMSNorm((2units,), act; dims=(1,), init_scale=cast_ones), # sp1norm in python
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
    # Input:  (w, h, c, batch*seq) after element-wise addition # e.g. (6, 6, 8, 80)
    # Output: (w, h, c, batch*seq)                             # e.g. (6, 6, 8, 80)
    # Normalizes across channel dimension (dim=3), includes activation
    push!(deconv_layers_list, RMSNorm((c,), act; dims=(3,), init_scale=cast_ones)); # spnorm in python

    # Define the main convolutional upsampling layers
    current_channels = c # Channels after combine step (e.g. 8)
    # Iterate from second-to-last depth down to the first depth
    # Python: for i, depth in reversed(list(enumerate(self.depths[:-1])))
    # Example depths = [4, 6, 8, 8]. Loop iterates over [8, 6, 4] (indices 2, 1, 0)
    # Julia depths = [4, 6, 8, 8]. reverse(depths[1:end-1]) gives [8, 6, 4]
    for depth_out in reverse(depths[1:end-1])
        # Upsample spatial dimensions (W, H) by 2x using nearest neighbor
        # Input: (Wi, Hi, current_channels, batch*seq)   # e.g. iter 1: (6, 6, 8, 80)
        # Output: (2*Wi, 2*Hi, current_channels, batch*seq) # e.g. iter 1: (12, 12, 8, 80)
        push!(deconv_layers_list, UpSample(factor=2, dims=(1, 2))) # Equivalent to Python's repeat(2,-2).repeat(2,-3)

        # Apply 2D Convolution
        # Input: (2*Wi, 2*Hi, current_channels, batch*seq) # e.g. iter 1: (12, 12, 8, 80)
        # Output: (2*Wi, 2*Hi, depth_out, batch*seq)       # e.g. iter 1: (12, 12, 8, 80)
        push!(deconv_layers_list, Conv((kernel, kernel), current_channels => depth_out; pad=SamePad(), init_weight=cast_glorot_uniform, init_bias=cast_zeros))

        # Apply RMS Normalization and activation
        # Input/Output: (2*Wi, 2*Hi, depth_out, batch*seq) # e.g. iter 1: (12, 12, 8, 80)
        # Normalizes across channel dimension (dim=3), includes activation
        push!(deconv_layers_list, RMSNorm((depth_out,), act; dims=(3,), init_scale=cast_ones))

        current_channels = depth_out # Update channels for next iteration input (e.g. iter 1: 8)
    end

    # Map to image channels
    imgdep = obs.size[3] # Target image channels (e.g., 1)
    # Final Upsample
    # Input: (W_last, H_last, depth_in_final, batch*seq)     # e.g. (48, 48, 4, 80)
    # Output: (2*W_last, 2*H_last, depth_in_final, batch*seq) # e.g. (96, 96, 4, 80)
    push!(deconv_layers_list, UpSample(factor=2, dims=(1, 2))) # Final repeat(2,-2).repeat(2,-3)
    # Final Convolution
    # Input: (96, 96, depth_in_final, batch*seq)             # e.g. (96, 96, 4, 80)
    # Output: (96, 96, imgdep, batch*seq)                   # e.g. (96, 96, 1, 80)
    push!(deconv_layers_list, Conv((kernel, kernel), current_channels => imgdep; pad=SamePad(), init_weight=cast_glorot_uniform, init_bias=cast_zeros))

    # Sigmoid activation is applied *after* this layer in the call function
    # push!(deconv_layers_list, sigmoid)

    # Final ReArrange layer
    # Input: (96, 96, imgdep, batch*seq)                   # e.g. (96, 96, 1, 80)
    # Output: (96, 96, batch*seq, imgdep)                   # e.g. (96, 96, 80, 1)
    # push!(deconv_layers_list, ReArrange((obs.size..., T, B)))

    deconv_layers = Chain(deconv_layers_list...)

    # return the decoder struct ------------------------------------------------------
    return Decoder(act, mults, depth, kernel, obs, Tuple(depths), shape, bspace, deter_dim, stoch_vars, classes_per_vars, units, spatialize_deter, spatialize_stoch, deconv_layers)
end


"""
Decoder for RSSM
"""
function (dec::Decoder)(feat, ps, st)

    # Get runtime dimensions T, B from input 'feat'
    # Assuming feat["deter"] has shape (deter_dim, T, B)
    _, T, B = size(feat["deter"])

    # Prepare inputs: Flatten sequence and batch dimensions
    deter_flat = reshape(feat["deter"], (dec.deter_dim, :));
    stoch_flat = reshape(feat["stoch"], (dec.stoch_vars * dec.classes_per_vars, :));

    # 1. Spatialize the deter and stoch parts
    # Pass the corresponding subset of parameters and states
    out_deter, st_deter_new = dec.spatialize_deter(deter_flat, ps.spatialize_deter, st.spatialize_deter)
    out_stoch, st_stoch_new = dec.spatialize_stoch(stoch_flat, ps.spatialize_stoch, st.spatialize_stoch)

    # 2. Combine spatialized features
    input_decoder = out_deter + out_stoch;

    # 3. Deconvolve the combined input
    out_deconv, st_deconv_new = dec.deconvolve(input_decoder, ps.deconvolve, st.deconvolve)

    # 4. Apply final activation (sigmoid)
    out_sigmoid = sigmoid.(out_deconv)

    # 5. Reshape the output to include T and B dimensions
    # Target shape: (Width, Height, Channels, Time, Batch)
    output = reshape(out_sigmoid, (dec.obs.size..., T, B))

    # 6. Combine updated states into a new nested NamedTuple
    st_new = (
        spatialize_deter=st_deter_new,
        spatialize_stoch=st_stoch_new,
        deconvolve=st_deconv_new
    )

    return output, st_new
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
