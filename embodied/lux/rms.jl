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
    feature_dim::Int      # Which dimension index of the input does `shape` correspond to?
    activation
    epsilon
    init_scale
    dims                  # Dimensions to normalize OVER
end

function RMSNorm(shape::NTuple{N, <:Int}, feature_dim::Int, activation=identity;
                epsilon::T=1.0f-5, dims=Colon(), affine::Bool=true,
                init_scale=ones32, allow_fast_activation::Bool=true) where {N, T}
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    return RMSNorm{affine, N}(shape, feature_dim, activation, epsilon, init_scale, dims)
end

# Check if the layer has a learnable scale parameter
@inline _affine(l::RMSNorm{A}) where {A} = A;

function Lux.initialparameters(rng::AbstractRNG, rn::RMSNorm)
    if _affine(rn)
        scale_init_shape = rn.shape
        isempty(scale_init_shape) && (scale_init_shape = (1,))
        scale = rn.init_scale(rng, scale_init_shape...)
        return (; scale=scale,)
    else
        return NamedTuple()
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
        scale_param = ps.scale # Has shape l.shape, e.g. (F,) or (C,)
        # --- Construct reshape_target immutably based on feature_dim ---
        reshape_target_tuple = ntuple(ndims(x)) do d
            if d == l.feature_dim
                 # Use the parameter size for the specified feature dimension
                 @assert length(l.shape) == 1 "RMSNorm currently only supports 1D parameter shapes for scale"
                 l.shape[1]
            else
                 1 # Size 1 for all other dimensions
            end
         end

        # Reshape the scale parameter (which has shape l.shape) to the target broadcast shape
        reshaped_scale = reshape(scale_param, reshape_target_tuple)
        # --- End immutable construction ---
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
    print(io, "RMSNorm($(l.shape), feature_dim=$(l.feature_dim)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(_affine(l)), dims=$(l.dims)")
    return print(io, ")")
end