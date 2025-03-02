using Lux, NNlib, Random, Tools, BFloat16s, YAML, Statistics
include("../embodied/lux/RMSNorm.jl");
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
    spatialize_stoch = Chain(
        Dense(stoch_vars * classes_per_vars, 2units, act),
        # RMSNorm(2units)
    );
    
    
    ps, st = Lux.setup(rng, spatialize_stoch);
    x = rand32(stoch_vars * classes_per_vars, seq_length * batch_size);
    y, st = spatialize_stoch(x, ps, st);
    size(y)

    # 1. Check RMS (root mean square) is approximately 1 along feature dimension
    rms = sqrt.(mean(abs2.(y), dims=1))
    println("RMS values should be close to 1: ", mean(rms))
    println("RMS std deviation: ", std(rms))

    layers = []
    
    # l = Chain(
    #     BlockLinear(deter_dim, u, g),
    #     ReArrange((w, h, c, :))
    # )
    # ps, st = Lux.setup(rng, l)
    # x = rand32(deter_dim, seq_length * batch_size)
    # y, st = l(x, ps, st)
    # size(y)

    channels = obs.size[3]
    for (d_in, d_out) in zip(vcat(channels,depths[1:end-1]), depths)
        push!(layers, Conv((kernel, kernel), d_in => d_out, pad=SamePad()))  # Add padding
        push!(layers, MaxPool((2, 2), stride=(2, 2)))
        push!(layers, RMSNorm(d_out, act))
    end
    nn = Chain(layers...)

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
    x0, x1 = feat["deter"], feat["stoch"];
    # x0 (deter): (8, 10, 8)        # deter_dim, batch, length
    # x1 (stoch): (8, 10, 2, 4)     # stoch_vars, classes, batch, length
    x1 = reshape(x1, (:, size(x1)[end-1:end]...));
    x0 = reshape(x0, (size(x0, 1), :));
    x1 = reshape(x1, (size(x1, 1), :));
    
    
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

