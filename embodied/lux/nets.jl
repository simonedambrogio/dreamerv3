using Lux, NNlib, Random
using BFloat16s
using ChainRulesCore # Need this for ZeroTangent, Tangent
using ForwardDiff # Import ForwardDiff to use its types

# Define the compute type globally
# const COMPUTE_TYPE = BFloat16
const COMPUTE_TYPE = Float32

function cast(x::Number)
    if x isa COMPUTE_TYPE
        return x
    elseif x isa Bool # Handle Bool explicitly
        return COMPUTE_TYPE(x)
    elseif isnothing(x) || isa(x, ChainRulesCore.ZeroTangent)
         return x # Pass through ZeroTangent/Nothing
    elseif !isfinite(x)
         @warn "Non-finite value encountered during cast: $x. Keeping original value." maxlog=1
         return x # Keep original Float32 Inf/NaN if needed
    else
        return COMPUTE_TYPE(x)
    end
end

function cast(x::AbstractArray)
     if isnothing(x) || isa(x, ChainRulesCore.ZeroTangent)
        return x
     end
    # Element-wise cast, applying the Number method's logic
    # Use broadcast `.` to handle potential ZeroTangent elements if the array itself isn't ZeroTangent
    return cast.(x)
end

# Cast for Tangents (handles NamedTuple backings)
function cast(t::Tangent{P, T}) where {P, T<:NamedTuple}
    # Check if all fields are ZeroTangent
    if all(val -> isa(val, ChainRulesCore.ZeroTangent), t.backing)
        return ChainRulesCore.ZeroTangent()
    end
    # Recursively cast the fields in the backing NamedTuple
    new_backing = cast(t.backing)
    # If casting resulted in all zero fields, return ZeroTangent
    if isa(new_backing, ChainRulesCore.ZeroTangent)
        return ChainRulesCore.ZeroTangent()
    end
    # Return new Tangent with casted backing
    # Ensure the primal type P is preserved
    return Tangent{P, typeof(new_backing)}(new_backing)
end

# Cast for ForwardDiff.Dual types introduced by AD
function cast(d::ForwardDiff.Dual)
    # Extract the value, cast it to COMPUTE_TYPE.
    # AD system handles the partials.
    return cast(ForwardDiff.value(d))
end

# Cast for regular NamedTuples (often used for gradients)
function cast(nt::NamedTuple)
     if all(val -> isa(val, ChainRulesCore.ZeroTangent), values(nt))
         return ChainRulesCore.ZeroTangent()
     end
     # Recursively cast each field
     casted_fields = map(values(nt)) do field
        cast(field)
    end
    # Check if all casted fields are now ZeroTangent
    if all(val -> isa(val, ChainRulesCore.ZeroTangent), casted_fields)
        return ChainRulesCore.ZeroTangent()
    end
    # Reconstruct NamedTuple with original keys and casted values
    return NamedTuple{keys(nt)}(casted_fields)
end

# Cast for ZeroTangent itself (identity)
function cast(z::ChainRulesCore.ZeroTangent)
    return z
end

# Cast for Nothing (identity)
function cast(n::Nothing)
    return n
end

function cast_zeros(rng::AbstractRNG, dims...)
    return cast(zeros(dims...))
end

cast_glorot_uniform(rng::AbstractRNG, dims...) = cast(glorot_uniform(rng, dims...))

cast_ones(rng::AbstractRNG, dims...) = cast(ones(dims...))

"""
    split(x::AbstractArray, n_splits::Integer, dim::Integer)

Splits an array `x` into `n_splits` equal views along the specified dimension `dim`.

Throws an assertion error if the size of the dimension `dim` is not evenly
divisible by `n_splits`.

Returns a Tuple of views.
"""
function Base.split(x::AbstractArray, n_splits::Integer, dim::Integer)
    N = ndims(x)
    @assert 1 <= dim <= N "Dimension $dim is out of bounds for array with $N dimensions"
    dim_size = size(x, dim)
    @assert dim_size % n_splits == 0 "Dimension $dim (size $dim_size) must be evenly divisible by n_splits ($n_splits)"
    slice_size = dim_size ÷ n_splits

    # Use ntuple to create the views efficiently
    return ntuple(i -> begin
        start = (i - 1) * slice_size + 1
        stop = i * slice_size
        slice_range = start:stop
        # Create the indices tuple for view()
        indices = ntuple(d -> d == dim ? slice_range : :, N)
        view(x, indices...)
    end, n_splits)
end
