using Lux
using Lux: AbstractRecurrentCell # Explicitly import

struct ObserveCell <: AbstractRecurrentCell # Use the imported name
    rssm::Int
end
println("ObserveCell <: AbstractRecurrentCell: ", ObserveCell <: AbstractRecurrentCell)


struct TestCell{use_bias, train_state} <: AbstractRecurrentCell
    in_dims::Int
end

