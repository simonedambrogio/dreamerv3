using Lux, NNlib, Random, Tools
include("RMSNorm.jl")
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
        depths = [depth * m for m in mults]
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
                img_out = img_branch(img_input)
                img_out = reshape(img_out, :, size(img_out, 4))
                push!(outs, img_out)
            end
            
            return length(outs) > 1 ? vcat(outs...) : outs[1]
        end
    )
end;

# Usage example:
obs_space = Dict(
    :image => Tools.Space(UInt8, (96, 96, 1)),
);

rng = Random.default_rng()
enc = encoder(rng; obs_space=obs_space)
ps, st = Lux.setup(rng, enc);

# Forward pass
x = (image = rand(Float32, 64, 64, 3), vector = rand(Float32, 10))
output, new_st = enc(x, ps, st)


