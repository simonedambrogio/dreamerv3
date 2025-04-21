using Lux
using Lux: AbstractLuxLayer
using NNlib # Keep for consistency if other layers use it
using ChainRulesCore

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

# --- Define the Custom Adjoint (rrule) ---
function ChainRulesCore.rrule(l::UpSample, x::AbstractArray, ps, st::NamedTuple)
    # 1. Perform the forward pass
    y, _ = l(x, ps, st) # Ignore state return for rrule

    # 2. Define the pullback function (how to calculate gradient wrt x)
    function upsample_pullback(Δy_thunk)
        # Unthunk the incoming gradient w.r.t the forward pass output (y, st)
        Δy_raw = unthunk(Δy_thunk)

        # --- Extract the array gradient component corresponding to y ---
        local Δy::AbstractArray # Ensure Δy is correctly typed/assigned
        if Δy_raw isa NoTangent || Δy_raw isa ZeroTangent
            # If the gradient w.r.t. the output tuple is zero/non-existent, grad w.r.t x is zero
            return NoTangent(), zero(x), NoTangent(), NoTangent()
        # Try accessing the underlying structure using backing()
        elseif backing(Δy_raw) isa Tuple && length(backing(Δy_raw)) >= 1
            Δy_tuple = backing(Δy_raw)
            # Assume the first element of the tuple gradient corresponds to y
            Δy_component = Δy_tuple[1]
            if Δy_component isa AbstractArray
                Δy = Δy_component # Assign the actual array gradient
            else # Could be ZeroTangent etc.
                # If the gradient for y itself is No/ZeroTangent, propagate zero gradient
                return NoTangent(), zero(x), NoTangent(), NoTangent()
            end
        else
            # This case might occur if the layer was used differently, but erroring is safer
            error("Unexpected gradient structure in UpSample pullback: $(typeof(Δy_raw)) with backing: $(typeof(backing(Δy_raw)))")
        end
        # --- Finished extracting Δy ---

        # Calculate the gradient w.r.t. x (Δx)
        # This is the core non-mutating part: sum the gradient over repeated blocks.
        # We need to determine the slicing ranges for the summation.

        output_size = size(x)
        input_size = size(Δy) # Now use size on the extracted array gradient
        factor = l.factor
        dims_to_sum = l.dims

        # Initialize gradient for x with zeros (Non-mutating)
        Δx = zero(x) # Or fill!(similar(x), zero(eltype(x))) is also fine here

        # Use CartesianIndices to iterate over the original smaller tensor's indices
        # This helps structure the summation correctly.
        cartesian_indices_x = CartesianIndices(output_size)

        for I_x in cartesian_indices_x
            # Calculate the corresponding slice in the larger Δy tensor
            start_indices = ones(Int, ndims(x))
            end_indices = ones(Int, ndims(x))

            for d in 1:ndims(x)
                if d in dims_to_sum
                    # Calculate range for dimensions that were upsampled
                    start_indices[d] = (I_x[d] - 1) * factor + 1
                    end_indices[d] = I_x[d] * factor
                else
                    # Keep index the same for dimensions that were not upsampled
                    start_indices[d] = I_x[d]
                    end_indices[d] = I_x[d]
                end
            end

            # Define the slice ranges for Δy
            ranges = ntuple(d -> start_indices[d]:end_indices[d], ndims(x))

            # Sum the gradients in the corresponding block of Δy
            # and assign it to the correct element in Δx
            # Note: This assignment is okay, we are filling Δx, not mutating the incoming Δy
            Δx[I_x] = sum(view(Δy, ranges...))
        end

        # Return gradients: (∂L/∂self, ∂L/∂x, ∂L/∂ps, ∂L/∂st)
        # No gradient for layer struct, ps, or st
        return NoTangent(), Δx, NoTangent(), NoTangent()
    end

    # 3. Return the forward result and the pullback function
    return (y, st), upsample_pullback
end

function Base.show(io::IO, l::UpSample)
    print(io, "UpSample(:$(l.mode), factor=$(l.factor), dims=$(l.dims))")
end
