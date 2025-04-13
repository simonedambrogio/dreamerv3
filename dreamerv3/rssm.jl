using Lux, NNlib, Random, Tools, BFloat16s, YAML, Statistics, LuxCore, OneHotArrays
include("../embodied/lux/rms.jl");
include("../embodied/lux/nets.jl");
include("../embodied/lux/BlockLinear.jl");
include("../embodied/lux/ReArrange.jl");
# Note: We might need more includes later as we add specific layers

# Based on Python RSSM class attributes
@kwdef struct RSSM <: Lux.AbstractLuxLayer # Using @kwdef for easier construction
    deter_dim::Int = 4096
    hidden_dim::Int = 2048
    stoch_dim::Int = 32
    classes_dim::Int = 32
    act::Function = gelu
    unroll::Bool = false
    unimix::Float32 = 0.01f0
    imglayers::Int = 2
    obslayers::Int = 1
    dynlayers::Int = 1
    blocks::Int = 8
    free_nats::Float32 = 1.0f0
    act_space::Space
    # Sub-layers will be added here later
end

"""
    initial_state(rssm::RSSM, batch_size::Int, ::AbstractRNG)

Returns the initial recurrent state for the RSSM.
Output shape: (deter = (batch_size, deter_dim), stoch = (batch_size, stoch_dim, classes_dim))
"""
function LuxCore.initialstates(rng::AbstractRNG, rssm::RSSM)
    # This function is designed to be called via Lux.setup, which doesn't provide batch_size.
    # The actual state initialization with batch_size needs a separate function
    # or handling within the main call method when the first batch arrives.
    # For now, returning an empty state for setup purposes.
    # We'll define a separate function for runtime initialization.
    return NamedTuple() # LuxCore.initialstates expects states of sublayers
end

"""
    initial_carry(rssm::RSSM, batch_size::Int)

Returns the initial carry for the RSSM. Carry is the state of the RSSM.
Output shape: (deter = (batch_size, deter_dim), stoch = (batch_size, stoch_dim, classes_dim))
"""
function initial_carry(rssm::RSSM, batch_size::Int)
    # Ensure we use the correct compute type (e.g., BFloat16)
    compute_T = isdefined(@__MODULE__, :COMPUTE_TYPE) ? COMPUTE_TYPE : Float32
    # Let's stick to Lux convention for layers, but the state `carry`
    deter_init = fill!(similar(Array{compute_T}, rssm.deter_dim, batch_size), zero(compute_T))
    stoch_init = fill!(similar(Array{compute_T}, rssm.stoch_dim, rssm.classes_dim, batch_size), zero(compute_T))

    # Use LuxCore.initialstates to get states for any potential stateful sub-layers later
    # For now, the state only contains the carry-over tensors.
    return (; deter = deter_init, stoch = stoch_init)
end

function _observe(rssm::RSSM, carry::NamedTuple, tokens, action::AbstractVector{<:Integer}, reset::AbstractVector{Bool})
    # carry: NamedTuple with .deter and .stoch
    # tokens: Encoded observations, shape (token_dim, Batch) [Assuming single step for now]
    # action: Discrete action indices for the current step, shape (Batch,)
    # reset: Boolean vector for the current step, shape (Batch,)

    # --- 1. Apply reset mask to state ---
    # Create the inverted mask, ready for broadcasting
    keep_mask = .!reset # Shape: (Batch,)
    # Apply mask to deter state
    deter_mask = reshape(keep_mask, 1, :) # Shape: (1, Batch)
    deter_masked = carry.deter .* deter_mask
    # Apply mask to stoch state
    stoch_mask = reshape(keep_mask, 1, 1, :) # Shape: (1, 1, Batch)
    stoch_masked = carry.stoch .* stoch_mask

    # --- 2. Process Action ---
    # Assuming action is discrete and needs one-hot encoding
    # TODO: Handle continuous actions if necessary based on act_space
    @assert !isnothing(rssm.act_space) "RSSM requires act_space to process actions"
    num_actions = rssm.act_space.high # Assumes Space defines range [low, high)
    # Perform one-hot encoding. Note: NNlib.onehotbatch expects indices starting from 1.
    action_onehot = OneHotArrays.onehotbatch(action, 0:num_actions) # Shape: (num_actions, Batch)
    action_onehot_casted = cast(action_onehot) # Cast to COMPUTE_TYPE
    # Apply reset mask to the processed action
    action_processed = action_onehot_casted .* deter_mask # Broadcast (1, Batch) mask

    # --- 3. Core Recurrent Update (Transition Model) ---
    # TODO: Implement the _core function
    # deter_next = _core(rssm, deter_masked, stoch_masked, action_processed)
    # deter_next = deter_masked # Placeholder

    # --- 4. Observation Update (Posterior Calculation) ---
    # TODO: Implement observation update logic:
    # 4a. Combine deter_next and tokens
    # 4b. Pass through observation layers (Dense + Norm)
    # 4c. Calculate posterior logits using _logit layer
    # 4d. Sample new stochastic state from posterior distribution
    # stoch_next = stoch_masked # Placeholder
    # logit_posterior = similar(stoch_next) # Placeholder

    # --- 5. Prepare Outputs ---
    # carry_next = (; deter=deter_next, stoch=stoch_next)
    # feat = (; deter=deter_next, stoch=stoch_next, logit=logit_posterior) # Features include posterior logit
    # entry = (; deter=deter_next, stoch=stoch_next) # State entry for storage/logging

    # TODO: Add assertions for types/shapes if needed

    # Placeholder return
    # return carry_next, (entry, feat)
end

function _core(rssm::RSSM, deter::AbstractArray, stoch::AbstractArray, action::AbstractArray)
    # Combine classes and stoch
    stoch = reshape(stoch, (:, size(stoch,3)))  #
end

# TODO: Define sub-layers and constructor
# TODO: Update struct definition to be a ContainerLayer


