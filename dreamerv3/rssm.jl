using Lux, NNlib, Random, Tools, BFloat16s, YAML, Statistics, LuxCore, OneHotArrays
using StatsBase: Weights # Added for sampling
using StatsBase
using OneHotArrays: onehot # Added for encoding
include("../embodied/lux/rms.jl");
include("../embodied/lux/nets.jl");
include("../embodied/lux/BlockLinear.jl");
include("../embodied/lux/ReArrange.jl");
# Note: We might need more includes later as we add specific layers

# Based on Python RSSM class attributes
struct RSSM{AS, CN, PO} <: Lux.AbstractLuxContainerLayer{(:core, :observation)} # Added AS for type stability
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
    token_dim::Int  # Added token dimension
    act_space::AS # Use type parameter AS
    # Sub-layers defined in core
    core::CN
    observation::PO
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
    token_dim::Int,        # Added token_dim (mandatory)
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
    gru_layers_list = []
    # Calculate input dim for the *first* dynamic layer
    h_deter_per_block = deter_dim ÷ g
    feat_concat_static = 3 * hidden_dim
    first_dyn_input_dim = (h_deter_per_block + feat_concat_static) * g
    current_dyn_input_dim = first_dyn_input_dim

    for _ in 1:dynlayers
        push!(gru_layers_list, Chain(
            BlockLinear(current_dyn_input_dim, deter_dim, g; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
            RMSNorm((deter_dim,), act; dims=(1,), init_scale=cast_ones)
        ))
        current_dyn_input_dim = deter_dim # Input for subsequent layers is the output of the previous one
    end
    
    # Final GRU layer (dyngru)
    gru_layer_input_dim = deter_dim # Output of the dyn_layers loop
    gru_layer_output_dim = 3 * deter_dim
    gru_layer = BlockLinear(gru_layer_input_dim, gru_layer_output_dim, g; init_weight=cast_glorot_uniform, init_bias=cast_zeros)
    push!(gru_layers_list, gru_layer)

    gru_layers = Chain(gru_layers_list...; name="gru_layers")

    # Posterior layers (obs + obsnorm)
    posterior_layers_list = []
    first_obs_input_dim = deter_dim + token_dim
    current_obs_input_dim = first_obs_input_dim

    for _ in 1:obslayers # Use the obslayers field
        push!(posterior_layers_list, Chain(
            Dense(current_obs_input_dim => hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
            RMSNorm((hidden_dim,), act; dims=(1,), init_scale=cast_ones)
        ))
        current_obs_input_dim = hidden_dim # Output of Norm is input to next Dense
    end
    posterior_layers = Chain(posterior_layers_list...; name="posterior_layers")

    # Prior layers (obs + obsnorm)
    prior_layers_list = []
    first_obs_input_dim = deter_dim + token_dim
    current_obs_input_dim = first_obs_input_dim

    for _ in 1:obslayers # Use the obslayers field
        push!(prior_layers_list, Chain(
            Dense(current_obs_input_dim => hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
            RMSNorm((hidden_dim,), act; dims=(1,), init_scale=cast_ones)
        ))
        current_obs_input_dim = hidden_dim # Output of Norm is input to next Dense
    end
    prior_layers = Chain(prior_layers_list...; name="prior_layers")

    # Logit layers (Dense + Reshape)
    logit_posterior = Chain(
        Dense(hidden_dim => stoch_dim * classes_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
        ReArrange((stoch_dim, classes_dim, :))
    )
    logit_prior = Chain(
        Dense(hidden_dim => stoch_dim * classes_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
        ReArrange((stoch_dim, classes_dim, :))
    )

    # --- Store Layers in NamedTuple --- 
    core_layers = (
        layer_deter = layer_deter,
        layer_stoch = layer_stoch,
        layer_action = layer_action,
        gru_layers = gru_layers
    )
    observation_layers = (
        posterior_layers = posterior_layers,
        prior_layers = prior_layers,
        logit_posterior = logit_posterior,
        logit_prior = logit_prior
    )

    # --- Return RSSM Instance --- 
    # Automatically determine types AS, CN, PO
    return RSSM(deter_dim, hidden_dim, stoch_dim, classes_dim, act, unroll, unimix, 
                imglayers, obslayers, dynlayers, blocks, free_nats, 
                token_dim, act_space, core_layers, observation_layers)
end


"""
    initial_state(rssm::RSSM, batch_size::Int, ::AbstractRNG)

Returns the initial recurrent state for the RSSM.
Output shape: (deter = (batch_size, deter_dim), stoch = (batch_size, stoch_dim, classes_dim))
"""
function LuxCore.initialstates(rng::AbstractRNG, rssm::RSSM)
    # Delegate state initialization to the sub-layers stored in rssm.core
    # This will recursively call initialstates on the layers within the core NamedTuple.
    return (; core = Lux.initialstates(rng, rssm.core), observation = Lux.initialstates(rng, rssm.observation))
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

function _observe(rssm::RSSM, carry::NamedTuple, tokens, action::AbstractVector{<:Integer}, reset::AbstractVector{Bool}, ps, st)
    # carry: NamedTuple with .deter and .stoch
    # tokens: Encoded observations, shape (token_dim, Batch) [Assuming single step for now]
    # action: Discrete action indices for the current step, shape (Batch,)
    # reset: Boolean vector for the current step, shape (Batch,)

    # --- 1. Apply reset mask to state ---
    # Create the inverted mask, ready for broadcasting
    keep_mask = .!reset # Shape: (Batch,)
    # Apply mask to deter state
    deter_mask = reshape(keep_mask, 1, :) # Shape: (1, Batch)
    deter = carry.deter .* deter_mask
    # Apply mask to stoch state
    stoch_mask = reshape(keep_mask, 1, 1, :) # Shape: (1, 1, Batch)
    stoch = carry.stoch .* stoch_mask

    # --- 2. Process Action ---
    # Assuming action is discrete and needs one-hot encoding
    # TODO: Handle continuous actions if necessary based on act_space
    @assert !isnothing(rssm.act_space) "RSSM requires act_space to process actions"
    num_actions = rssm.act_space.high # Assumes Space defines range [low, high)
    # Perform one-hot encoding. Note: NNlib.onehotbatch expects indices starting from 1.
    action_onehot = OneHotArrays.onehotbatch(action, 0:num_actions) # Shape: (num_actions, Batch)
    action_onehot_casted = cast(action_onehot) # Cast to COMPUTE_TYPE
    # Apply reset mask to the processed action
    action = action_onehot_casted .* deter_mask # Broadcast (1, Batch) mask

    # --- 3. Core Recurrent Update (Transition Model) ---
    deter_current, st_core = _core(rssm, deter, stoch, action, ps, st)
    
    # --- 4. Observation Update (Posterior Calculation) ---
    # 4.1 Combine current deter and tokens
    tokens = reshape(tokens, :, size(deter_current, ndims(deter_current)));
    x = vcat(deter_current, tokens)
    # 4.2 Apply Posterior Layers
    x_posterior, st_posterior = rssm.observation.posterior_layers(x, ps.observation.posterior_layers, st.observation.posterior_layers)
    logit_posterior, st_logit_posterior = rssm.observation.logit_posterior(x_posterior, ps.observation.logit_posterior, st.observation.logit_posterior)
    # 4d. Sample new stochastic state from posterior distribution
    dist_posterior = _dist(logit_posterior, rssm.unimix)
    stoch_current = rand(rng, dist_posterior) # Assuming rng is available

    # --- 5. Prepare Outputs ---
    carry = (; deter=deter_current, stoch=stoch_current)
    feat = (; deter=deter_current, stoch=stoch_current, logit=logit_posterior) # Features include posterior logit
    entry = (; deter=deter_current, stoch=stoch_current) # State entry for storage/logging

    @assert all(eltype(deter) == eltype(stoch) == eltype(logit_posterior)) "All variables must have the same element type"

    # Placeholder return
    return carry, (entry, feat)
end

flat2group(x, g) = reshape(x, :, g, size(x, ndims(x))); # Use ndims for robustness
group2flat(x) = reshape(x, :, size(x, ndims(x))); # Use ndims for robustness

function _core(rssm::RSSM, deter::AbstractArray, stoch::AbstractArray, action::AbstractArray, ps, st)
    # deter shape: (deter_dim, Batch)
    # stoch shape: (stoch_dim, classes_dim, Batch)
    # action shape: (num_actions, Batch)
    
    # --- Prepare Inputs ---
    B = size(deter, 2) # Get Batch size from deter
    # Combine classes and stoch
    stoch_flat = reshape(stoch, :, B)  # Shape: (stoch_dim * classes_dim, Batch)
    g = rssm.blocks

    # --- Create Context --- 
    # Transform deter, stoch, action using their respective layers
    # Note: We need the updated state from these layers, even if empty now
    _deter_ctx, st_deter_new = rssm.core.layer_deter(deter, ps.core.layer_deter, st.core.layer_deter)
    _stoch_ctx, st_stoch_new = rssm.core.layer_stoch(stoch_flat, ps.core.layer_stoch, st.core.layer_stoch)
    _action_ctx, st_action_new = rssm.core.layer_action(action, ps.core.layer_action, st.core.layer_action)
    
    # Combine context components
    # Using piping `|>` for readability, similar to the test script
    context = vcat(_deter_ctx, _stoch_ctx, _action_ctx) |>
    # Add dimension for broadcasting blocks & repeat
    ctx -> reshape(ctx, size(ctx, 1), 1, B) |>
    ctx_reshaped -> repeat(ctx_reshaped, 1, g, 1) |>
    # Concatenate with grouped original deter state and flatten
    ctx_repeated -> group2flat(
        vcat(flat2group(deter, g), ctx_repeated)
    );
    
    # --- Apply GRU Layers (Dynamic Layers + Final BlockLinear) ---
    # This computes the raw pre-gate values
    raw_gates, st_gru_new = rssm.core.gru_layers(context, ps.core.gru_layers, st.core.gru_layers)
    
    # --- Apply GRU Gating Mechanism --- 
    # Reshape raw_gates to be block-aware
    grouped_gates = flat2group(raw_gates, g) # Shape: (FeaturesPerBlock = 3*deter/g, Blocks=g, Batch=B)
    # Split into 3 gates along the FeaturesPerBlock dimension
    gates_split = split(grouped_gates, 3, 1) # Tuple of 3 tensors, each: (deter/g, g, B)
    # Flatten each gate back to (Features, Batch)
    reset_flat, cand_flat, update_flat = [group2flat(gate) for gate in gates_split] # Each: (deter_dim, B)
    
    # Apply activations
    reset = sigmoid.(reset_flat)
    cand = tanh.(reset .* cand_flat) # Apply reset gate to candidate pre-activation
    update = sigmoid.(update_flat .- cast(1))
    
    # Combine using update gate (GRU formula)
    # IMPORTANT: Uses the *original* deter passed into _core
    deter_next = update .* cand .+ (cast(1) .- update) .* deter 
    
    # --- Combine updated states --- 
    # Create the new state tuple, preserving the structure
    st_core_updated = (
        layer_deter = st_deter_new,
        layer_stoch = st_stoch_new,
        layer_action = st_action_new,
        gru_layers = st_gru_new
    )
    st_updated = (; core = st_core_updated) # Wrap in the top-level :core key

    return deter_next, st_updated
end

# TODO: Define sub-layers and constructor
# TODO: Update struct definition to be a ContainerLayer

# --- Distribution Helper ---

"""
    OneHotDist(logits, unimix)

Represents stoch_dim independent categorical distributions per batch item,
with unimix label smoothing. Sampling returns a one-hot encoded tensor.
"""
struct OneHotDist{T, U}
    logits::T # Shape (stoch_dim, classes_dim, Batch...)
    unimix::U # Float, probability for uniform mixing
end

"""
    _dist(logits, unimix)

Helper function to create the OneHotDist object.
"""
function _dist(logits, unimix::Real)
    return OneHotDist(logits, unimix)
end

"""
    Base.rand(rng::AbstractRNG, d::OneHotDist)

Sample from the OneHotDist. Applies unimix smoothing, samples independently
across stoch_dim and batch dimensions, and returns a one-hot encoded result.
Output shape: (stoch_dim, classes_dim, Batch...)
"""
function Base.rand(rng::AbstractRNG, d::OneHotDist)
    logits = d.logits
    unimix = d.unimix
    stoch_dim, classes_dim = size(logits, 1), size(logits, 2)
    batch_dims = size(logits)[3:end]
    compute_T = eltype(logits) # Use the compute type from logits

    # 1. Calculate smoothed probabilities
    probs_raw = softmax(logits; dims=2)
    probs_smoothed = (1 - unimix) .* probs_raw .+ unimix / classes_dim
    # Ensure probabilities are non-negative (can happen with float errors)
    probs_clipped = max.(probs_smoothed, zero(compute_T))

    # 2. Sample indices for each distribution independently
    # Reshape to 2D: (stoch_dim * product(batch_dims), classes_dim)
    num_distributions = stoch_dim * prod(batch_dims; init=1)
    # Permute dims so classes is last for reshape, then transpose
    probs_2d = reshape(permutedims(probs_clipped, (1, 3:ndims(probs_clipped)..., 2)), num_distributions, classes_dim)

    sampled_indices = similar(Array{Int}, num_distributions)
    for i in 1:num_distributions
        # sample needs probability weights summing to 1.
        # Ensure the sum value matches the element type.
        prob_view = view(probs_2d, i, :)
        prob_sum = one(eltype(prob_view))
        prob_weights = Weights(prob_view, prob_sum) # Use typed sum
        # Explicitly call StatsBase.sample to avoid conflict with Tools.sample
        sampled_indices[i] = StatsBase.sample(rng, 1:classes_dim, prob_weights)
    end

    # 3. One-hot encode the indices
    # Reshape indices back to (stoch_dim, batch_dims...)
    indices_final_shape = (stoch_dim, batch_dims...)
    indices_reshaped = reshape(sampled_indices, indices_final_shape)

    # Create one-hot output tensor
    stoch_onehot = similar(logits, Bool) # Shape: (stoch, classes, batch...)
    fill!(stoch_onehot, false)

    # Use CartesianIndices for efficient setting of true values
    cartesian_indices_batch = CartesianIndices(batch_dims)
    for i_batch in cartesian_indices_batch
        for s in 1:stoch_dim
            idx_class = indices_reshaped[s, i_batch] # Get the sampled class index
            stoch_onehot[s, idx_class, i_batch] = true
        end
    end

    # 4. Cast to compute type
    return cast(stoch_onehot) # Assumes cast function handles Bool -> compute_T
end

# --- End Distribution Helper ---


