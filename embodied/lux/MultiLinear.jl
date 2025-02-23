using Lux

@concrete struct MultiLinear{use_bias} <: AbstractExplicitLayer
    activation
    in_dims::Union{Int, Tuple}
    out_dims::Union{Int, Tuple}
    init_weight
    init_bias
end

# Constructor with Pair syntax (like Dense)
function MultiLinear(mapping::Pair{<:Union{Int,Tuple}, <:Union{Int,Tuple}}, 
                    activation=identity;
                    init_weight=glorot_uniform,
                    init_bias=zeros32,
                    use_bias::Bool=true,
                    allow_fast_activation::Bool=true)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    return MultiLinear{use_bias}(activation, first(mapping), last(mapping), 
                                init_weight, init_bias)
end

# Constructor with separate arguments
function MultiLinear(in_dims::Union{Int,Tuple}, out_dims::Union{Int,Tuple}, 
                    activation=identity; kwargs...)
    return MultiLinear(in_dims => out_dims, activation; kwargs...)
end

function Lux.initialparameters(rng::AbstractRNG, l::MultiLinear{use_bias}) where {use_bias}
    # Convert dimensions to total sizes
    in_size = l.in_dims isa Int ? l.in_dims : prod(l.in_dims)
    out_size = l.out_dims isa Int ? l.out_dims : prod(l.out_dims)
    
    # Initialize parameters
    weight = l.init_weight(rng, out_size, in_size)
    
    if use_bias
        bias = l.init_bias(rng, out_size)
        return (weight=weight, bias=bias)
    else
        return (weight=weight,)
    end
end

Lux.initialstates(::AbstractRNG, ::MultiLinear) = NamedTuple()

function (l::MultiLinear{use_bias})(x::AbstractArray, ps, st::NamedTuple) where {use_bias}
    # Get input dimensions
    x_dims = size(x)
    insize = x_dims[1]  # First dimension is features in Lux
    
    # Reshape input to 2D for matrix multiplication
    x_flat = reshape(x, (insize, :))
    
    # Apply linear transformation
    y = ps.weight * x_flat
    
    if use_bias
        y = y .+ reshape(ps.bias, :, 1)
    end
    
    # Apply activation
    y = l.activation.(y)
    
    # If output is multi-dimensional, reshape accordingly
    if l.out_dims isa Tuple
        y = reshape(y, (l.out_dims..., x_dims[2:end]...))
    end
    
    return y, st
end

# Pretty printing
function Base.show(io::IO, l::MultiLinear{use_bias}) where {use_bias}
    print(io, "MultiLinear($(l.in_dims) => $(l.out_dims)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    use_bias || print(io, ", bias=false")
    return print(io, ")")
end

rng = MersenneTwister(1234)
l = MultiLinear((288, 8))
ps, st = Lux.setup(rng, l)
x = rand32(288, 8)
y, st = l(x, ps, st)
size(y)

# Like Dense layer
layer1 = MultiLinear(64 => 128)  # Single integers
layer2 = MultiLinear(128 => 256, relu)  # With activation

# With multi-dimensional output (like JAX)
layer3 = MultiLinear(64 => (6, 6, 8))  # 3D output
layer4 = MultiLinear((2, 32) => 64)    # 2D input to 1D output
