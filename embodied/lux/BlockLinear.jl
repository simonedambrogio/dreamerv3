using Lux, Random
import LuxCore  # Import LuxCore directly
using ConcreteStructs: @concrete

@doc """
    BlockLinear(units::Int, blocks::Int; bias::Bool=true, outscale::Float32=1.0f0)

A block-wise linear transformation layer.

# Arguments
- `units::Int`: Number of output units
- `blocks::Int`: Number of blocks to split input and output into
- `bias::Bool=true`: Whether to include a bias term
- `outscale::Float32=1.0f0`: Scaling factor for weight initialization

# Example
rng = MersenneTwister(1234)
u = 288
g = 8
x = rand32(8, 80)
l = BlockLinear(size(x, 1), u, g)

ps, st = Lux.setup(rng, l)
x = rand32(8, 80)
y, st = l(x, ps, st)
size(y)
"""
@concrete struct BlockLinear <: AbstractLuxLayer
    in_features::Int    # Add input size
    units::Int
    blocks::Int
    bias::Bool
    outscale::Float32
    init_weight
    init_bias
end

function BlockLinear(in_features::Int, units::Int, blocks::Int; 
                    bias::Bool=true, 
                    outscale::Float32=1.0f0,
                    init_weight=glorot_uniform,
                    init_bias=zeros32)
    @assert blocks <= units && units % blocks == 0 "blocks must divide units evenly"
    @assert in_features % blocks == 0 "input features must be divisible by blocks"
    return BlockLinear(in_features, units, blocks, bias, outscale, init_weight, init_bias)
end

function Lux.initialparameters(rng::AbstractRNG, l::BlockLinear)
    # Initialize with correct block sizes
    block_in_size = l.in_features ÷ l.blocks
    block_out_size = l.units ÷ l.blocks
    
    # Weight shape: (blocks, in_per_block, out_per_block)
    weight_shape = (l.blocks, block_in_size, block_out_size)
    weight = l.init_weight(rng, weight_shape...) .* l.outscale
    
    if l.bias
        bias = l.init_bias(rng, l.units)
        return (weight=weight, bias=bias)
    else
        return (weight=weight,)
    end
end

function (l::BlockLinear)(x::AbstractArray, ps, st::NamedTuple)
    # Get input dimensions
    x_dims = size(x)
    insize = x_dims[1]  # First dimension is features in Lux
    @assert insize == l.in_features "Input features must match layer's in_features"
        
    # Reshape input for block multiplication
    # From: (features, batch...)
    # To: (features_per_block, blocks, batch...)
    x_blocked = reshape(x, (insize ÷ l.blocks, l.blocks, x_dims[2:end]...))
        
    # For batched_mul, we need:
    # - Matrix dimensions first (2 dims)
    # - Batch dimension last
    x_batched = permutedims(x_blocked, (1, 3, 2))  # (in_per_block, batch, blocks)
    
    # Weight should be (out_per_block, in_per_block, blocks)
    weight = permutedims(ps.weight, (3, 2, 1))
        
    y = batched_mul(weight, x_batched)
    
    # Reshape output to (features, batch...)
    y = permutedims(y, (1, 3, 2))  # (out_features, blocks, batch)
    y = reshape(y, (l.units, x_dims[2:end]...))
    
    # Add bias if present
    if l.bias
        y = y .+ reshape(ps.bias, :, ones(Int, length(x_dims[2:end]))...)
    end
    
    return y, st
end

# Pretty printing
function Base.show(io::IO, l::BlockLinear)
    print(io, "BlockLinear($(l.units), $(l.blocks)")
    !l.bias && print(io, ", bias=false")
    l.outscale != 1.0f0 && print(io, ", outscale=$(l.outscale)")
    print(io, ")")
end 
