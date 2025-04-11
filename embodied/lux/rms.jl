using Lux
using Lux: BoolType, has_affine, match_eltype, safe_getproperty, unwrapped_eltype, initialparameters, AbstractLuxLayer
using ConcreteStructs: @concrete
using Markdown: @doc_str
using Static: StaticBool, True, static
using Random: AbstractRNG
using NNlib
using Statistics: mean, std, var
using Random

# RMSNorm is a normalization technique that scales inputs by their root mean square (RMS)
# It's a simpler alternative to LayerNorm that doesn't center the data
# This makes it computationally more efficient while maintaining good performance

# Helper functions for the RMSNorm layer
# Apply activation function (or return input unchanged if activation is identity)
@inline __apply_activation(::typeof(identity), x) = x;
@inline __apply_activation(f, x) = f.(x);

@doc doc"""
    RMSNorm(shape::NTuple{N, Int}, activation=identity; epsilon=1f-5, dims=Colon(), affine::Bool=true, init_scale=ones32)

Computes root mean square normalization over the input array. Optionally applies an elementwise affine transformation afterwards.

Given an input array ``x``, this layer computes 
```math
y = \frac{x}{\sqrt{\mathbb{E}[x^2] + \epsilon}} * \gamma
```
where ``\gamma`` is a trainable parameter if `affine=true`.

## Arguments
- `shape`: Broadcastable shape of input array excluding the batch dimension.
- `activation`: After normalization, elementwise activation `activation` is applied.

## Keyword Arguments
- `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`
- `epsilon`: a value added to the denominator for numerical stability.
- `dims`: Dimensions to normalize the array over.
- If `affine=true`, it also applies a rescale to the input through a learnable scale parameter.
  + `init_scale`: Controls how the `scale` is initiliazed

## Inputs
- `x`: AbstractArray

## Returns
- `y`: Normalized Array
- Empty NamedTuple()

## Parameters
- `affine=false`: Empty `NamedTuple()`
- `affine=true`
  + `scale`: Scale of shape `(shape..., 1)`
"""
@concrete struct RMSNorm{affine, N} <: AbstractLuxLayer
    shape::NTuple{N, Int}
    activation
    epsilon
    init_scale
    dims
end

function RMSNorm(shape::NTuple{N, <:Int}, activation=identity; 
                epsilon::T=1.0f-5, dims=Colon(), affine::Bool=true, 
                init_scale=ones32, allow_fast_activation::Bool=true) where {N, T}
    # Apply fast activation if allowed (optimization for certain activation functions)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    # Create and return a new RMSNorm layer with the specified parameters
    return RMSNorm{affine, N}(shape, activation, epsilon, init_scale, dims)
end

# Check if the layer has a learnable scale parameter
@inline _affine(l::RMSNorm{A}) where {A} = A;

function Lux.initialparameters(rng::AbstractRNG, rn::RMSNorm)
    if _affine(rn)
        # If affine=true, create a scale parameter initialized with the provided function
        # This scale parameter will be learned during training
        # Use the compute type defined in nets.jl if possible, otherwise default to Float32
        compute_T = isdefined(@__MODULE__, :COMPUTE_TYPE) ? COMPUTE_TYPE : Float32
        scale = rn.init_scale(rng, compute_T, rn.shape..., 1) # Ensure scale is compute_T
        return (; scale=scale,) # Return NamedTuple consistent with Lux convention
    else
        # If affine=false, no learnable parameters are needed
        return NamedTuple() # Return empty NamedTuple consistent with Lux
    end
end

function (l::RMSNorm)(x::AbstractArray{T}, ps, st::NamedTuple) where T
    # Step 1: Compute the mean of squared values (mean square)
    mean_square = mean(abs2.(x); dims=l.dims)

    # Step 2: Calculate the scaling factor, casting epsilon to the input type
    epsilon_casted = T(l.epsilon)
    scale_factor = T(1.0) ./ sqrt.(mean_square .+ epsilon_casted)

    # Step 3: Apply the normalization by multiplying the input by the scale factor
    y = x .* scale_factor

    # Step 4: Apply the learnable scale parameter if affine=true
    if _affine(l)
        # Get the scale parameter
        scale_param = ps.scale # Assume ps is NamedTuple with :scale

        # Ensure scale_param has the correct number of dimensions corresponding to l.shape
        local scale_param_squeezed
        if ndims(scale_param) > length(l.shape) && size(scale_param)[end] == 1 && length(l.shape) > 0 # Avoid squeezing if shape is ()
             scale_param_squeezed = dropdims(scale_param; dims=ndims(scale_param))
        else
             scale_param_squeezed = scale_param
        end
         # Handle the case where l.shape is empty () which can happen if scale is just a scalar?
         expected_shape = isempty(l.shape) ? () : l.shape
         if size(scale_param_squeezed) != expected_shape
              error("RMSNorm scale parameter size $(size(scale_param_squeezed)) does not match expected shape $(expected_shape)")
         end


        # Determine target shape for broadcasting
        # Target shape should have size 1 in all dimensions EXCEPT those specified by l.shape
        # The dimensions specified in l.shape must correspond to the dimensions NOT listed in l.dims
        reshape_target = ones(Int, ndims(x))

        # Handle the common CNN case: shape = (C,), dims = (ChannelDim,)
        if length(l.shape) == 1 && l.dims isa NTuple{1, Int}
             channel_dim = l.dims[1]
             if ndims(x) >= channel_dim
                 reshape_target[channel_dim] = l.shape[1] # Set the channel dimension size
             else
                 error("Input dimensions ($(ndims(x))) less than specified normalization dimension ($(channel_dim))")
             end
        else
            # Fallback/General logic (potentially needs refinement for other use cases)
             shape_idx = 1
             dims_to_normalize = l.dims isa Colon ? (1:(ndims(x) - 1)) : Int.(collect(l.dims))
             # Iterate through all dimensions of x
             for d in 1:ndims(x)
                  if !(d in dims_to_normalize)
                      # If dimension is not normalized, try to assign size from l.shape
                      if shape_idx <= length(l.shape)
                          reshape_target[d] = l.shape[shape_idx]
                          shape_idx += 1
                      else
                          # Keep size 1 if l.shape doesn't cover this dim (e.g., batch)
                      end
                  end
                  # Otherwise (dimension is normalized), keep size 1
             end
             # Check if all shape dimensions were used
             if shape_idx <= length(l.shape) && !(l.dims isa Colon)
                  @warn "RMSNorm: Mismatch between l.shape $(l.shape) and non-normalized dimensions. Broadcasting might be incorrect."
             end
        end


        reshaped_scale = reshape(scale_param_squeezed, Tuple(reshape_target)...)

        # Apply the scale
        y = y .* reshaped_scale
    end

    # Step 5: Apply the activation function and return the result
    return __apply_activation(l.activation, y), st
end

function Base.show(io::IO, l::RMSNorm)
    print(io, "RMSNorm($(l.shape)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(_affine(l)), dims=$(l.dims)")
    return print(io, ")")
end