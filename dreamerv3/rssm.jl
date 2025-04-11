using Lux, NNlib, Random, Tools
include("../embodied/lux/RMSNorm.jl")
include("../embodied/lux/BlockLinear.jl")

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
function encoder(rng=Random.default_rng(); 
    obs_space::Dict,
    units::Int=1024,
    depth::Int=64,
    mults::Tuple=(2, 3, 4, 4),
    layers::Int=3,
    kernel::Int=5,
    symlog::Bool=true,
    outer::Bool=false,
    strided::Bool=false)
    
    # Determine input type based on dimentionality
    veckeys = Symbol[k for (k,v) in obs_space if length(v.size) ≤ 2];
    imgkeys = Symbol[k for (k,v) in obs_space if length(v.size) == 3];
    
    # Vector processing branch
    vec_branch = if !isempty(veckeys)
        Chain([
            Chain(
                Dense(units => units),
                BatchNorm(units, gelu)
            ) for _ in 1:layers
        ]...)
    else
        identity
    end
    
    # Image processing branch
    img_branch = if !isempty(imgkeys)
        depths = [depth * m for m in mults];
        Chain([
            Chain(
                if outer && i == 1
                    Conv((kernel, kernel), 3 => d)
                elseif strided
                    Conv((kernel, kernel), (i == 1 ? 3 : depths[i-1]) => d, stride=2)
                else
                    Chain(
                        Conv((kernel, kernel), (i == 1 ? 3 : depths[i-1]) => d),
                        x -> begin
                            B, H, W, C = size(x)
                            x = reshape(x, B, H ÷ 2, 2, W ÷ 2, 2, C)
                            x = maximum(x, dims=(3, 5))
                            dropdims(x, dims=(3, 5))
                        end
                    )
                end,
                BatchNorm(d, gelu)
            )
            for (i, d) in enumerate(depths)
        ]...)
    else
        identity
    end
    
    return Chain(
        NamedTuple{(:vec, :img)}((vec_branch, img_branch)),
        x -> begin
            outs = []
            if !isempty(veckeys)
                vecs = Dict(k => x[k] for k in veckeys)
                vec_input = reduce(vcat, values(vecs))
                vec_input = symlog ? symlog.(vec_input) : vec_input
                push!(outs, vec_branch(vec_input))
            end
            
            if !isempty(imgkeys)
                imgs = [x[k] for k in sort(imgkeys)]
                img_input = cat(imgs..., dims=3)
                img_input = Float32.(img_input) ./ 255f0 .- 0.5f0
                img_input = permutedims(img_input, (1, 2, 3, 4))  # Now shape is (W, H, C, N)
                img_out = img_branch(img_input)
                img_out = reshape(img_out, (W, H ÷ 2, 2, W ÷ 2, 2, C, B))
                pooled = maximum(img_out, dims=(3, 5))
                final = dropdims(pooled, dims=(3, 5))
                push!(outs, final)
            end
            
            return length(outs) > 1 ? vcat(outs...) : outs[1]
        end
    )
end;

# # Usage example:
# obs = Space(UInt8, (96, 96, 1));

# units = 1024; depth = 64; mults = (2, 3, 4, 4); layers = 3; kernel = 5; symlog = true; outer = false; strided = false
# rng = Random.default_rng();
# enc = encoder(rng; obs_space=obs_space);
# ps, st = Lux.setup(rng, enc);
# # Forward pass
# output, new_st = enc(x, ps, st)


# batch = 10;
# units = 1024; depth = 64; mults = (2, 3, 4, 4); layers = 3; kernel = 5; symlog = true; outer = false; strided = false

# # Debug
# units = 8; depth = 2; mults = (2, 3, 4, 4); layers = 1; kernel = 5; symlog = true; outer = false; strided = false
# batch_size = 2;
# batch_length = 4;
# obs = (image = rand(UInt8, 64, 64, 1, batch_length, batch_size), );

# size(x.image)
# imgs = [x.image];
# img_input = cat(imgs..., dims=3);
# size(img_input)
# img_input = Float32.(img_input) ./ 255f0 .- 0.5f0;
# # img_input = permutedims(img_input, (1, 2, 3, 4));  # Now shape is (W, H, C, N)

# depths = [depth * m for m in mults]
# (i, d) = first(enumerate(depths))

# d_in, d_out = first(zip(
#     vcat(size(img_input, 3), depths[1:end-1]), 
#     depths
# ))
# W,H,C,N = size(img_input)

# layers = []
# for (d_in, d_out) in zip(vcat(C,depths[1:end-1]), depths)
#     push!(layers, Conv((kernel, kernel), d_in => d_out, pad=SamePad()))  # Add padding
#     push!(layers, MaxPool((2, 2), stride=(2, 2)))
#     push!(layers, RMSNorm(d_out, swish))
# end
# en = Chain(layers...)


# rng = Random.default_rng();
# ps, st = Lux.setup(rng, en);
# output, new_st = en(img_input, ps, st);
# size(output)

# norm = RMSNorm((depths[i], ))
# ps, st = Lux.setup(rng, norm);
# # output, new_st = norm(output, ps, st);








# x′ = match_eltype(norm, ps, st, output);

# # Calculate RMS statistics over specified dimensions
# ms = mean(abs2.(x′), dims=norm.dims)
# rms = sqrt.(ms .+ convert(unwrapped_eltype(x′), norm.epsilon))

# y = x′ ./ rms;
# scale = reshape(safe_getproperty(ps, Val(:scale)), (1, 1, :, 1))

# y = y .* scale;
# size(y)

# # 1. Check RMS (Root Mean Square) of normalized output
# # Should be close to 1 along normalized dimensions
# function check_rms(x, dims)
#     ms = mean(abs2.(x), dims=dims)
#     rms = sqrt.(ms)
#     println("RMS values mean: ", mean(rms))
#     println("RMS values std: ", Statistics.std(rms))
#     # Should be close to 1 if normalized correctly
# end

# # 2. Check scale of values before and after
# println("\nBefore normalization:")
# println("Mean: ", mean(output))
# println("Std: ", std(output))

# # Apply normalization
# x′ = match_eltype(norm, ps, st, output);
# ms = mean(abs2.(x′), dims=norm.dims);
# rms = sqrt.(ms .+ convert(unwrapped_eltype(x′), norm.epsilon));
# y = x′ ./ rms;
# scale = reshape(safe_getproperty(ps, Val(:scale)), (1, 1, :, 1))
# y = y .* scale;

# println("\nAfter normalization:")
# println("Mean: ", mean(output))
# println("Std: ", Statistics.std(output))

# # 3. Check RMS of normalized output
# println("\nRMS check of normalized output:")
# check_rms(y, norm.dims)

# # 4. Verify that relative relationships are preserved
# println("\nCorrelation between input and output:")
# println(cor(vec(output), vec(y)))

# # Chain([
# #     Chain(
# #         Chain(
# #             Conv((kernel, kernel), (i == 1 ? 3 : depths[i-1]) => d)(img_input),
# #             x -> begin

# #                 B, H, W, C = size(x)
# #                 x = reshape(x, B, H ÷ 2, 2, W ÷ 2, 2, C)
# #                 x = maximum(x, dims=(3, 5))
# #                 dropdims(x, dims=(3, 5))
# #             end
# #         ),
# #         BatchNorm(d, gelu)
# #     )
# #     for (i, d) in enumerate(depths)
# # ]...)
