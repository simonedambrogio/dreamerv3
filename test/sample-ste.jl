using Zygote
using Random
using StatsBase
using LinearAlgebra
using NNlib: logsoftmax, softmax # Ensure these are available
using OneHotArrays
include("../embodied/lux/nets.jl")

"""
    OneHotDist(logits, unimix)

Represents stoch_dim independent categorical distributions per batch item,
with unimix label smoothing. Sampling returns a one-hot encoded tensor.
"""
struct OneHotDist{T}
    logits::T # Shape (stoch_dim, classes_dim, Batch...)
    unimix::eltype(T) # Float, probability for uniform mixing
end

"""
    _dist(logits, unimix)

Helper function to create the OneHotDist object.
"""
function _dist(logits, unimix::Real)
    return OneHotDist(logits, unimix)
end


function sample_ste(rng::AbstractRNG, d::OneHotDist)
    logits = d.logits
    unimix = d.unimix
    stoch_dim, classes_dim = size(logits, 1), size(logits, 2)
    batch_dims = size(logits)[3:end]
    compute_T = eltype(logits)

    # --- Sampling based on smoothed probabilities --- 
    probs_raw = softmax(logits; dims=2)
    probs_smoothed = (1 - unimix) .* probs_raw .+ unimix / classes_dim
    probs_clipped = max.(probs_smoothed, zero(compute_T))

    num_distributions = stoch_dim * prod(batch_dims; init=1)
    probs_2d = reshape(permutedims(probs_clipped, (1, 3:ndims(probs_clipped)..., 2)), num_distributions, classes_dim)

    # Use a comprehension for functional creation of sampled_indices
    sampled_indices = [begin
        prob_view = view(probs_2d, i, :)
        prob_sum = one(eltype(prob_view))
        prob_weights = Weights(prob_view, prob_sum)
        StatsBase.sample(rng, 1:classes_dim, prob_weights)
    end for i in 1:num_distributions]

    # --- One-hot encode the sampled index (value for forward pass) ---
    indices_final_shape = (stoch_dim, batch_dims...)
    indices_reshaped = reshape(sampled_indices, indices_final_shape)

    # Use OneHotArrays.onehotbatch for non-mutating creation
    # onehotbatch creates shape (num_classes, shape_of_indices...)
    # Input indices_reshaped shape: (stoch_dim, batch_dims...)
    value_onehot_permuted = OneHotArrays.onehotbatch(indices_reshaped, 1:classes_dim)
    # Output shape: (classes_dim, stoch_dim, batch_dims...)

    # Permute to desired shape: (stoch_dim, classes_dim, batch_dims...)
    # Original index dims were (1, 2...), target index dims are (2, 1, 3...)
    perm = (2, 1, (3:ndims(value_onehot_permuted))...)
    value_onehot = permutedims(value_onehot_permuted, perm)

    # --- Apply Straight-Through Estimator (STE) ---
    # STE: sg(value) + (probs - sg(probs))
    # Use Zygote.@ignore to stop gradients for the discrete parts
    # Use the *unsmoothed* probs_raw for the gradient path
    ste_value = Zygote.@ignore(value_onehot) .+ (probs_raw .- Zygote.@ignore(probs_raw))

    # --- Cast final result to compute type ---
    return cast(ste_value)
end


rng = MersenneTwister(1234);
logits = rand(rng, Float32, 3, 10, 10);
dist = _dist(logits, 0.01f0);
sample_ste(rng, dist)

function loss(logits)
    d = _dist(logits, 0.01f0);
    stoch = sample_ste(rng, d)
    return mean(stoch)
end

println(loss(logits))

# Compute the gradient
grad = Zygote.gradient(loss, logits)

# Print the gradient (or parts of it)
println("\nGradient with respect to logits:")
println(size(grad[1])) # Print the size to confirm shape
println(grad[1][1, :, 1]) # Print a slice of the gradient

# Check if gradient is zero (it shouldn't be entirely zero if STE works)
println("\nIs gradient entirely zero? ", all(iszero, grad[1]))


