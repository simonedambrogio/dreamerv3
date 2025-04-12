using Lux
using Lux: AbstractLuxLayer
using NNlib # Keep for consistency if other layers use it

@doc doc"""
    UpSample(mode::Symbol; factor::Int = 2, dims = (1, 2))

Upsamples specified dimensions of an input tensor.

Currently supports:
- `:nearest`: Nearest Neighbor Upsampling using `Base.repeat`.

## Arguments
- `mode::Symbol`: The upsampling mode. Currently only `:nearest` is supported.

## Keyword Arguments
- `factor::Int`: The integer factor by which to increase the specified dimensions (default: 2).
- `dims`: A tuple specifying which dimensions of the input tensor are the spatial
          dimensions to be upsampled (default: `(1, 2)`, assuming WH... format).

## Inputs
- `x`: Input tensor.

## Returns
- Upsampled tensor `y`.
- Empty `NamedTuple()` state.

## Parameters
- Empty `NamedTuple()`.

## States
- Empty `NamedTuple()`.
"""
struct UpSample{N} <: AbstractLuxLayer
    mode::Symbol
    factor::Int
    dims::NTuple{N, Int}

    # Constructor allowing dims to be specified
    function UpSample(mode::Symbol = :nearest; factor::Int = 2, dims::NTuple{N, Int} = (1, 2)) where N
        @assert mode === :nearest "Currently only :nearest mode is supported for UpSample."
        @assert factor > 0 "Upsampling factor must be positive."
        new{N}(mode, factor, dims)
    end
end

# Allow creating with just factor, defaulting dims=(1,2) and mode=:nearest
UpSample(factor::Int; mode::Symbol = :nearest, dims::NTuple{N, Int} = (1, 2)) where N = UpSample(mode; factor=factor, dims=dims)
# Default constructor
UpSample() = UpSample(:nearest; factor=2, dims=(1, 2))

# No parameters or states needed for nearest neighbor
Lux.initialparameters(rng::AbstractRNG, layer::UpSample) = NamedTuple()
Lux.initialstates(rng::AbstractRNG, layer::UpSample) = NamedTuple()

function (l::UpSample)(x::AbstractArray, ps, st::NamedTuple)
    if l.mode === :nearest
        # Create the 'inner' tuple for Base.repeat
        # It should have size 'factor' for dimensions specified in l.dims, and 1 otherwise.
        inner_tuple = ones(Int, ndims(x))
        for d in l.dims
            if d <= ndims(x) && d > 0
                inner_tuple[d] = l.factor
            else
                @warn "Upsample dimension $d is out of bounds (1:$(ndims(x))) for input. Skipping."
            end
        end
        y = repeat(x, inner=Tuple(inner_tuple))
        return y, st
    else
        # Placeholder for other modes like bilinear
        error("Unsupported upsampling mode: $(l.mode)")
    end
end

function Base.show(io::IO, l::UpSample)
    print(io, "UpSample(:$(l.mode), factor=$(l.factor), dims=$(l.dims))")
end
