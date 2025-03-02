using Lux
using Lux: BoolType, has_affine, match_eltype, safe_getproperty, unwrapped_eltype, initialparameters, AbstractLuxLayer
using ConcreteStructs: @concrete
using Markdown: @doc_str
using Static: StaticBool, True, static
using Random: AbstractRNG
using NNlib
using Statistics: mean

@doc doc"""
    RMSNorm(shape::NTuple{N, Int}, activation=identity; epsilon=1f-5, dims=Colon(),
            affine=true, init_scale=ones32)

Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467) layer.

RMSNorm is a simplified variant of LayerNorm that only normalizes by the root mean square,
without centering. Given an input array ``x``, this layer computes:

```math
y = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} * \gamma
```

where ``\gamma`` is a trainable scale parameter if `affine=true`.

## Arguments
  - `shape`: Broadcastable shape of input array excluding the batch dimension.
  - `activation`: After normalization, elementwise activation `activation` is applied.

## Keyword Arguments
  - `epsilon`: a value added to the denominator for numerical stability.
  - `dims`: Dimensions to normalize the array over.
  - If `affine=true`, applies a learnable scale through parameter `scale`.
    + `init_scale`: Controls how the `scale` is initialized

## References

[1] Zhang, Biao, and Rico Sennrich. "Root mean square layer normalization." 
    Advances in Neural Information Processing Systems 32 (2019).
"""
@concrete struct RMSNorm <: AbstractLuxLayer
    shape
    activation
    epsilon
    init_scale
    dims
    affine <: StaticBool
end

function RMSNorm(shape, activation=identity; epsilon=1.0f-5, dims=Colon(),
        affine::BoolType=True(), init_scale=ones32)
    return RMSNorm(shape, activation, epsilon, init_scale, dims, static(affine))
end

function Lux.initialparameters(rng::AbstractRNG, rn::RMSNorm)
    if has_affine(rn)
        # If shape is a tuple, we need to handle it differently
        if rn.shape isa Tuple
            # For a tuple shape like (46, 46, 4), we typically want the scale
            # to be the size of the feature dimension (4 in this case)
            if length(rn.shape) == 3 && rn.dims == (1, 2, 3)
                # Initialize scale for each channel
                return (; scale=rn.init_scale(rng, rn.shape[3]))
            else
                # Default case - initialize with the full shape
                return (; scale=rn.init_scale(rng, rn.shape...))
            end
        else
            # Original behavior for scalar shape
            return (; scale=rn.init_scale(rng, rn.shape))
        end
    end
    return (;)
end

function (l::RMSNorm)(x::AbstractArray, ps, st::NamedTuple)
    x′ = match_eltype(l, ps, st, x)
    
    # Calculate RMS statistics
    ms = mean(abs2.(x′), dims=l.dims)
    rms = sqrt.(ms .+ convert(unwrapped_eltype(x′), l.epsilon))
    
    # Normalize and scale
    y = x′ ./ rms
    if has_affine(l)
        # For 4D input (H, W, C, B) with dims=(1,2,3)
        # We want to apply scale to each channel
        scale = safe_getproperty(ps, Val(:scale))
        
        # For a 4D input where we normalize across dims 1,2,3
        # The scale should be applied to each feature but be the same across batch
        if ndims(x′) == 4 && l.dims == (1, 2, 3)
            # Reshape to (1, 1, C, 1) for broadcasting
            scale = reshape(scale, (1, 1, length(scale), 1))
        else
            # For other cases, just reshape to a scalar for now
            # This is a fallback that might need to be expanded for other use cases
            scale = reshape(scale, 1)
        end
        
        y = y .* scale
    end
    
    # Apply activation
    y = NNlib.fast_act(l.activation, y)(y)
    return y, st
end

function Base.show(io::IO, l::RMSNorm)
    print(io, "RMSNorm($(l.shape)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(has_affine(l)), dims=$(l.dims)")
    return print(io, ")")
end
