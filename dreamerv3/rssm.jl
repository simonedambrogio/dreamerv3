using Lux, NNlib, Random

struct Encoder <: Lux.AbstractExplicitLayer
    units::Int
    norm::Symbol
    act::Symbol
    depth::Int
    mults::Tuple
    layers::Int
    kernel::Int
    symlog::Bool
    outer::Bool
    strided::Bool
    obs_space::Dict
    veckeys::Vector{String}
    imgkeys::Vector{String}
    depths::Tuple
    mlp_chain::Chain
    cnn_chain::Chain
end
