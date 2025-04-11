using Lux, NNlib, Random, Tools, BFloat16s, YAML, Statistics
# include("../embodied/lux/RMSNorm.jl");
include("../embodied/lux/rms.jl");
include("../embodied/lux/nets.jl");
include("../embodied/lux/BlockLinear.jl");
include("../embodied/lux/ReArrange.jl");
config = YAML.load_file("dreamerv3/configs.yaml");

struct Decoder
    act::Function
    mults::Tuple
    depth::Int
    kernel::Int
    net::Chain
    obs::Space
    depths::Tuple
    shape::Vector{Integer}
    bspace
    deter_dim::Int
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
        BlockLinear(deter_dim, u, g),
        # ReArrange layer reshapes the output of BlockLinear to spatial dimensions:
        # Input:  (u, batch*seq)                # e.g. (288, 80)
        # Output: (w, h, c, batch*seq)         # e.g. (6, 6, 8, 80)
        ReArrange((w, h, c, :))
    );

    # 2. Spatialize the stochastic variables (stoch_vars, classes_per_vars, seq_length, batch_size)
    # to feed into the CNN
    # x1 dimension: (stoch_vars x classes_per_vars, seq_length x batch_size)
    # This is going to be applied to the stoch part of the state
    spatialize_stoch = Chain(
        # Dense layer transforms the stochastic variables into spatial features:
        Dense(stoch_vars * classes_per_vars, 2units, act), # Output: (2*units, batch*seq) # sp1
        # Normalize along feature dim (dim 1), includes activation
        RMSNorm((2 * units,), act; dims=(1,), init_scale=cast_ones), # sp1norm
        # Dense layer transforms the stochastic variables into spatial features:
        Dense(2units, prod(shape), act), ReArrange((w, h, c, :)) # sp2
    );
    
   
    # Test BlockLinear --------------------------------------------------------
    bl = BlockLinear(deter_dim, u, g)
    ps, st = Lux.setup(rng, bl);
    x = rand32(deter_dim, seq_length * batch_size);
    y, st = bl(x, ps, st);
    size(y)

    # Test ReArrange -----------------------------------------------------------
    r = ReArrange((w, h, c, :))
    ps, st = Lux.setup(rng, r);
    y, st = r(y, ps, st);
    size(y)
    

    # Test Linear -------------------------------------------------------------
    l = Dense(stoch_vars * classes_per_vars, 2units, act)
    ps, st = Lux.setup(rng, l);
    x = rand32(stoch_vars * classes_per_vars, seq_length * batch_size);
    y, st = l(x, ps, st);
    size(y)

    # Test RMSNorm -------------------------------------------------------------
    rn = RMSNorm((2units,), act; dims = (1,), init_scale=cast_ones)
    ps, st = Lux.setup(rng, rn);
    y, st = rn(y, ps, st);
    size(y)

    # Test Dense + Reshape -----------------------------------------------------
    # Implement a Dense layer that takes as input the stoch part of the state
    # of size 16, 80 (16 = stoch_vars * classes_per_vars, 80 = T * B) and outputs a vector of size 
    # shape (where shape is defined above, and for instance is (6, 6, 8)). So the input
    # goes from 16 to 6, 6, 8 so the final output dimention is 6, 6, 8, 80 (a 4 dimentional array)
    x = rand32(16, 80);
    l = Chain(Dense(16 => u, act), ReArrange((w, h, c, :)))
    ps, st = Lux.setup(rng, l);
    y, st = l(x, ps, st);
    size(y)

    
    # layers = []
    
    # l = Chain(
    #     BlockLinear(deter_dim, u, g),
    #     ReArrange((w, h, c, :))
    # )
    # ps, st = Lux.setup(rng, l)
    # x = rand32(deter_dim, seq_length * batch_size)
    # y, st = l(x, ps, st)
    # size(y)

    # channels = obs.size[3]
    # for (d_in, d_out) in zip(vcat(channels,depths[1:end-1]), depths)
    #     push!(layers, Conv((kernel, kernel), d_in => d_out, pad=SamePad()))  # Add padding
    #     push!(layers, MaxPool((2, 2), stride=(2, 2)))
    #     push!(layers, RMSNorm(d_out, act))
    # end
    # nn = Chain(layers...)

    # return the encoder ------------------------------------------------------
    return Decoder(act, mults, depth, kernel, nn, obs, Tuple(depths), shape, bspace, deter_dim)
end

B = batch_size = config["debug"]["batch_size"];
T = seq_length = config["debug"]["batch_length"];
deter_dim = config["debug"]["agent"][".*\\.deter"];
obs = Tools.Space(UInt8, (96, 96, 1));
depth = config["debug"]["agent"][".*\\.depth"];
units = config["debug"]["agent"][".*\\.units"];
stoch_vars = config["debug"]["agent"][".*\\.stoch"];
classes_per_vars = config["debug"]["agent"][".*\\.classes"];

act=gelu
mults=(2, 3, 4, 4)
kernel=5
bspace=8

rng = Random.default_rng();
dec = Decoder(; obs, deter_dim, depth, units, stoch_vars, classes_per_vars);


feat = Dict(
    "deter" => cast(rand(rng, Float32, deter_dim, seq_length, batch_size)),
    "stoch" => cast(rand(rng, Float32, stoch_vars, classes_per_vars, seq_length, batch_size)),
);
reset = rand(rng, Bool, seq_length, batch_size);


"""
Decoder for RSSM
"""
function (dec::Decoder)(state, ps, feat, reset)
    
    bshape = size(reset); # sequence length, batch size
    u, g = prod(dec.shape), dec.bspace;
    deter, stoch = feat["deter"], feat["stoch"];
    # x0 (deter): (8, 10, 8)        # deter_dim, length, batch
    # x1 (stoch): (2, 4, 10, 8)     # stoch_vars, classes, length, batch
    stoch = reshape(stoch, (:, size(stoch)[end-1:end]...)); # (2, 4, 10, 8) -> (8, 10, 8) 
    stoch = reshape(stoch, (size(stoch, 1), :)); # (8, 10, 8) -> (8, 80)
    deter = reshape(deter, (size(deter, 1), :)); # (8, 10, 8) -> (8, 80)
    x = vcat(stoch, deter); # (16, 80)

    
    # 4. Calculate the final spatial dimensions after all CNN layers

    # inp = [cast(feat[k]) for k in ("stoch", "deter")]; 
    # inp = [reshape(x, (:, prod(bshape))) for x in inp]; # flatten the sequence and batch dimensions
    # inp = vcat(inp...); # n features x (seq_length * batch_size)

    # flatten the sequence and batch dimensions
    W, H, C, T, B = size(imgs);
    imgs = reshape(imgs, (W, H, C, T*B));
    @assert typeof(imgs) == Array{UInt8, 4} "Image must be an array of UInt8"
    imgs = Float32.(imgs) ./ 255f0 .- 0.5f0;
    
    output, new_state = enc.net(imgs, ps, state);

    # Reshape the output to be a 3D array of size (embedding_dim, T, B)
    W, H, C, A = size(output);
    WHC = W*H*C
    output = reshape(output, (WHC, A));
    output = reshape(output, (WHC, T, B))

    return output, new_state
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

