using Lux, NNlib, Random, Tools
include("../embodied/lux/RMSNorm.jl")


struct Encoder
    act::Function
    mults::Tuple
    depth::Int
    kernel::Int
    net::Chain
end


function Encoder(;
    obs::Space,
    act::Function=gelu,
    mults::Tuple=(2, 3, 4, 4),
    depth::Int=64,
    kernel::Int=5)


    # construct the network ---------------------------------------------------
    depths = [depth * m for m in mults]
    channels = obs.size[3]
    layers = []
    for (d_in, d_out) in zip(vcat(channels,depths[1:end-1]), depths)
        push!(layers, Conv((kernel, kernel), d_in => d_out, pad=SamePad()))  # Add padding
        push!(layers, MaxPool((2, 2), stride=(2, 2)))
        push!(layers, RMSNorm(d_out, act))
    end
    nn = Chain(layers...)

    # return the encoder ------------------------------------------------------
    return Encoder(act, mults, depth, kernel, nn)
end



enc = Encoder(; obs);

enc.net(rand(UInt8, 96, 96, 1, 1)) 

rng = Random.default_rng();
ps, state = Lux.setup(rng, enc.net);


"""
Encoder for RSSM


    Conv((3, 3), 1 => 64) ------------------------------------------------------
    kernel = 3, depth = 1 (input channel), d = 64 (output channels)
    Input Image (1 channel):
    [
        1  2  3  4
        5  6  7  8
        9  10 11 12
        13 14 15 16
    ]

    3x3 Kernel (simplified, showing just one of the 64 output channels):
    [
        a b c
        d e f
        g h i
    ]

    Output computation for one position:
    result = a*1 + b*2 + c*3 +
            d*5 + e*6 + f*7 +
            g*9 + h*10 + i*11

    The process repeats with 64 different kernels to produce 64 output channels.
    --------------------------------------------------------------------------

    MaxPool((2, 2), stride=(2, 2)) ----------------------------------------------
    Window moves 2 positions each time:
    [1 2]  [3 4]
    [5 6]  [7 8]
    ↓      ↓
    [9 10]  [11 12]
    [13 14] [15 16]

    Output (max of each window):
    6  8
    14 16
    --------------------------------------------------------------------------

    RMSNorm(2, 2, 3, 2) ------------------------------------------------------
    Input Shape: (2, 2, 3, 2)  # (Width, Height, Channels, Batch)

    Batch 1:
    Channel 1:    Channel 2:    Channel 3:
    [1  2]       [5  6]       [9   10]
    [3  4]       [7  8]       [11  12]

    Batch 2:
    Channel 1:    Channel 2:    Channel 3:
    [13 14]      [17 18]      [21 22]
    [15 16]      [19 20]      [23 24]

    Step 1: Calculate mean square (dims=Colon() "all dimentions"):
    ms = (1² + 2² + 3² + ... + 24²)/24 
       = (1 + 4 + 9 + ... + 576)/24
       = 4900/24 
       ≈ 204.17

    Step 2: Add epsilon and take sqrt for RMS:
    rms = √(ms + ε)

    Step 3: Normalize by dividing input by RMS:
    For position (1,1) in Batch 1:
    y = [
        Batch 1:
        Channel 1:           Channel 2:           Channel 3:
        [1/rms  2/rms]  [5/rms  6/rms]  [9/rms  10/rms]
        [3/rms  4/rms]  [7/rms  8/rms]  [11/rms 12/rms]

        Batch 2:
        Channel 1:            Channel 2:            Channel 3:
        [13/rms 14/rms]  [17/rms 18/rms]  [21/rms 22/rms]
        [15/rms 16/rms]  [19/rms 20/rms]  [23/rms 24/rms]
    ]

    Step 4: Apply learnable scale:
    final = y * scale
    --------------------------------------------------------------------------
"""
function forward(enc::Encoder, state, ps, obs::Dict)
    
    # to do  
    # - test if input is image or vector (implemented only image)
    # - implement different types of convolution (e.g. outer, strided)

    imgs = obs[:image]; # image is a 4D array of UInt8 (W, H, C, sequence_length, batch_size)
    # flatten the sequence and batch dimensions
    W, H, C, T, B = size(imgs)
    imgs = reshape(imgs, (W, H, C, T*B));
    @assert typeof(imgs) == Array{UInt8, 4} "Image must be an array of UInt8"
    imgs = Float32.(imgs) ./ 255f0 .- 0.5f0;
    
    output, new_state = enc.net(imgs, ps, state);
    return output, new_state
end;


# Usage example:
obs_space = Dict(
    :image => Tools.Space(UInt8, (96, 96, 1)),
);

rng = Random.default_rng()
enc = encoder(rng; obs=obs_space)
ps, st = Lux.setup(rng, enc);

# Forward pass
obs = (image = rand(Float32, 64, 64, 3),);
output, new_st = enc(obs, ps, st)


