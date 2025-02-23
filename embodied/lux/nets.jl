using BFloat16s


COMPUTE_TYPE = BFloat16

function cast(x::Number)
    if x isa BFloat16
        return x
    else
        return COMPUTE_TYPE(x)
    end
end

function cast(x::Array)
    return COMPUTE_TYPE.(x)
end

function cast_zeros(rng::AbstractRNG, dims...)
    return cast(zeros(dims...))
end

cast_glorot_uniform(rng::AbstractRNG, dims...) = cast(glorot_uniform(rng, dims...))

cast_ones(rng::AbstractRNG, dims...) = cast(ones(dims...))

