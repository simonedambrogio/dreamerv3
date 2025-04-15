using BFloat16s


COMPUTE_TYPE = BFloat16

function cast(x::Number)
    if x isa BFloat16
        return x
    else
        return COMPUTE_TYPE(x)
    end
end

function cast(x::AbstractArray)
    return COMPUTE_TYPE.(x)
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
