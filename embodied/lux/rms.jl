using Lux
using Lux: BoolType, has_affine, match_eltype, safe_getproperty, unwrapped_eltype, initialparameters, AbstractLuxLayer
using ConcreteStructs: @concrete
using Markdown: @doc_str
using Static: StaticBool, True, static
using Random: AbstractRNG
using NNlib
using Statistics: mean, std, var
using Random
using ChainRulesCore: ChainRulesCore, RuleConfig, HasReverseMode, rrule, NoTangent, ZeroTangent, unthunk, backing

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
    shape::NTuple{N, Int} # Shape of the learnable parameter (e.g., (Channels,))
    activation
    epsilon
    init_scale
    dims                  # Dimensions to normalize OVER
end

function RMSNorm(shape::NTuple{N, <:Int}, activation=identity; # Removed feature_dim arg
                epsilon::T=1.0f-5, dims=Colon(), affine::Bool=true,
                init_scale=ones32, allow_fast_activation::Bool=true) where {N, T}
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    return RMSNorm{affine, N}(shape, activation, epsilon, init_scale, dims) # Removed feature_dim
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
    rms_inv = one(T) ./ sqrt.(mean_square .+ epsilon_casted) # Use equivalent of rsqrt

    # Step 3: Apply the normalization by multiplying the input by the scale factor
    y_normalized = x .* rms_inv

    # Step 4: Apply the learnable scale parameter if affine=true
    if _affine(l)
        scale_param = ps.scale # Original scale parameter
        # --- ORIGINAL Mutable reshape logic --- 
        local scale_param_squeezed
        if ndims(scale_param) > length(l.shape) && size(scale_param)[end] == 1 && length(l.shape) > 0 # Avoid squeezing if shape is ()
             scale_param_squeezed = dropdims(scale_param; dims=ndims(scale_param))
        else
             scale_param_squeezed = scale_param
        end
        expected_shape = isempty(l.shape) ? () : l.shape
         if size(scale_param_squeezed) != expected_shape
              error("RMSNorm scale parameter size $(size(scale_param_squeezed)) does not match expected shape $(expected_shape)")
         end
        reshape_target = ones(Int, ndims(x))
        # Logic from before the immutable refactor
        if length(l.shape) == 1 && l.dims isa NTuple{1, Int}
            channel_dim = l.dims[1]
            if ndims(x) >= channel_dim
                reshape_target[channel_dim] = l.shape[1] # Set the channel dimension size
            else
                error("Input dimensions ($(ndims(x))) less than specified normalization dimension ($(channel_dim))")
            end
       else
           shape_idx = 1
           # Determine dimensions NOT being normalized
           dims_to_normalize = l.dims isa Colon ? (1:(ndims(x) - 1)) : Int.(collect(l.dims))
            for d in 1:ndims(x)
                 if !(d in dims_to_normalize)
                     if shape_idx <= length(l.shape)
                         reshape_target[d] = l.shape[shape_idx]
                         shape_idx += 1
                     end
                 end
            end
            # Optional: Warning check if needed
            # if shape_idx <= length(l.shape) && !(l.dims isa Colon)
            #      @warn "RMSNorm: Mismatch between l.shape $(l.shape) and non-normalized dimensions. Broadcasting might be incorrect."
            # end
       end
       reshaped_scale = reshape(scale_param_squeezed, Tuple(reshape_target)...)
       # --- End ORIGINAL Mutable reshape logic --- 

        y = y_normalized .* reshaped_scale
    else
        y = y_normalized
    end

    # Step 5: Apply the activation function and return the result
    return __apply_activation(l.activation, y), st
end

# Ensure NO rrule is defined here (it was commented out earlier)
# # function ChainRulesCore.rrule(...)
# # end

function Base.show(io::IO, l::RMSNorm)
    print(io, "RMSNorm($(l.shape)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(_affine(l)), dims=$(l.dims)")
    return print(io, ")")
end