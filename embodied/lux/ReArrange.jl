struct ReArrange <: AbstractLuxLayer
    shape::Tuple
end

function (l::ReArrange)(x, ps, st)
    return reshape(x, l.shape), st
end

function Base.show(io::IO, l::ReArrange)
    print(io, "ReArrange($(l.shape))")
end
