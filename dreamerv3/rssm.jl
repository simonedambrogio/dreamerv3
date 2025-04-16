using Lux, NNlib, Random, Tools, BFloat16s, YAML, Statistics, LuxCore, OneHotArrays
include("../embodied/lux/rms.jl");
include("../embodied/lux/nets.jl");
include("../embodied/lux/BlockLinear.jl");
include("../embodied/lux/ReArrange.jl");
# Note: We might need more includes later as we add specific layers

# Based on Python RSSM class attributes
struct RSSM{AS, CN} <: Lux.AbstractLuxContainerLayer{(:core,)} # Added AS for type stability
    deter_dim::Int
    hidden_dim::Int
    stoch_dim::Int
    classes_dim::Int
    act::Function
    unroll::Bool
    unimix::Float32
    imglayers::Int
    obslayers::Int
    dynlayers::Int
    blocks::Int
    free_nats::Float32
    act_space::AS # Use type parameter AS
    # Sub-layers defined in core
    core::CN
end

function RSSM(; # Constructor
    deter_dim::Int = 4096,
    hidden_dim::Int = 2048,
    stoch_dim::Int = 32,
    classes_dim::Int = 32,
    act::Function = gelu,
    unroll::Bool = false,
    unimix::Float32 = 0.01f0,
    imglayers::Int = 2,
    obslayers::Int = 1,
    dynlayers::Int = 1,
    blocks::Int = 8,
    free_nats::Float32 = 1.0f0,
    act_space::Space) # act_space is mandatory

    # --- Calculate Static Dimensions ---
    g = blocks
    @assert deter_dim % g == 0 "deter_dim must be divisible by blocks (g)"
    @assert (stoch_dim * classes_dim) % g == 0 "stoch_dim*classes_dim must be divisible by blocks (g)" # Might need this if stoch is used in BlockLinear
    @assert (3 * deter_dim) % g == 0 "3*deter_dim must be divisible by blocks (g)" # For gru_layer output
    num_actions = act_space.high + 1 # Assuming discrete Space

    # --- Define Core Layers --- 

    # Layers for initial context processing (deter, stoch, action)
    layer_deter = Chain(
        Dense(deter_dim => hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
        RMSNorm((hidden_dim,), act; dims=(1,), init_scale=cast_ones)
    )
    layer_stoch = Chain(
        Dense(stoch_dim * classes_dim => hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
        RMSNorm((hidden_dim,), act; dims=(1,), init_scale=cast_ones)
    )
    layer_action = Chain(
        Dense(num_actions => hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
        RMSNorm((hidden_dim,), act; dims=(1,), init_scale=cast_ones)
    )

    # Dynamic layers loop (dynhid + dynhidnorm)
    dyn_layers_list = []
    # Calculate input dim for the *first* dynamic layer
    h_deter_per_block = deter_dim ÷ g
    feat_concat_static = 3 * hidden_dim
    first_dyn_input_dim = (h_deter_per_block + feat_concat_static) * g
    current_dyn_input_dim = first_dyn_input_dim

    for _ in 1:dynlayers
        push!(dyn_layers_list, Chain(
            BlockLinear(current_dyn_input_dim, deter_dim, g; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
            RMSNorm((deter_dim,), act; dims=(1,), init_scale=cast_ones)
        ))
        current_dyn_input_dim = deter_dim # Input for subsequent layers is the output of the previous one
    end
    dyn_layers = Chain(dyn_layers_list...; name="dyn_layers_loop")

    # Final GRU layer (dyngru)
    gru_layer_input_dim = deter_dim # Output of the dyn_layers loop
    gru_layer_output_dim = 3 * deter_dim
    gru_layer = BlockLinear(gru_layer_input_dim, gru_layer_output_dim, g; init_weight=cast_glorot_uniform, init_bias=cast_zeros)

    # --- Store Layers in NamedTuple --- 
    core_layers = (
        layer_deter = layer_deter,
        layer_stoch = layer_stoch,
        layer_action = layer_action,
        dyn_layers = dyn_layers, # This is now a Chain
        gru_layer = gru_layer
    )

    # --- Return RSSM Instance --- 
    # Automatically determine types AS and CN
    return RSSM(deter_dim, hidden_dim, stoch_dim, classes_dim, act, unroll, unimix, 
                imglayers, obslayers, dynlayers, blocks, free_nats, 
                act_space, core_layers)
end


"""
    initial_state(rssm::RSSM, batch_size::Int, ::AbstractRNG)

Returns the initial recurrent state for the RSSM.
Output shape: (deter = (batch_size, deter_dim), stoch = (batch_size, stoch_dim, classes_dim))
"""
function LuxCore.initialstates(rng::AbstractRNG, rssm::RSSM)
    # Delegate state initialization to the sub-layers stored in rssm.core
    # This will recursively call initialstates on the layers within the core NamedTuple.
    return (; core = Lux.initialstates(rng, rssm.core))
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

rng = Random.default_rng()
ps, st = Lux.setup(rng, rssm)

function _core(rssm::RSSM, deter::AbstractArray, stoch::AbstractArray, action::AbstractArray, ps, st)
    # Combine classes and stoch
    stoch = reshape(stoch, (:, size(stoch,3)))  # Shape: (stoch_dim * classes_dim, Batch)
    
    # Apply dynamic layers
    deter, st.core.layer_deter = rssm.core.layer_deter(deter, ps.core.layer_deter, st.core.layer_deter)
    stoch, st.core.layer_stoch = rssm.core.layer_stoch(stoch, ps.core.layer_stoch, st.core.layer_stoch)
    action, st.core.layer_action = rssm.core.layer_action(action, ps.core.layer_action, st.core.layer_action)

    
end

# TODO: Define sub-layers and constructor
# TODO: Update struct definition to be a ContainerLayer


