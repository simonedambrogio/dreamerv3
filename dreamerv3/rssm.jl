using Lux, NNlib, Random, Tools, BFloat16s, YAML, Statistics, LuxCore, OneHotArrays
using StatsBase: Weights # Added for sampling
using NNlib: logsoftmax, softmax # Ensure these are available
using StatsBase
using OneHotArrays: onehot # Added for encoding
using Zygote
using SliceMap # Needed for preparing scan input
using Zygote: Buffer, @ignore # Import Buffer
using CUDA # Assuming CUDA.AbstractGPUArray is used for device check

# Based on Python RSSM class attributes
struct RSSM{AS, CN, PO, OI, T} <: Lux.AbstractLuxContainerLayer{(:core, :observation, :imagination)} # Added AS for type stability
    deter_dim::Int
    hidden_dim::Int
    stoch_dim::Int
    classes_dim::Int
    act::Function
    unimix::T
    imglayers::Int
    obslayers::Int
    dynlayers::Int
    blocks::Int
    free_nats::T
    token_dim::Int  # Added token dimension
    act_space::AS # Use type parameter AS
    # Sub-layers defined in core
    core::CN
    observation::PO
    imagination::OI
end

function RSSM(; # Constructor
    deter_dim::Int = 4096,
    hidden_dim::Int = 2048,
    stoch_dim::Int = 32,
    classes_dim::Int = 32,
    act::Function = gelu,
    imglayers::Int = 2,
    obslayers::Int = 1,
    dynlayers::Int = 1,
    blocks::Int = 8,
    unimix::T,
    free_nats::T,
    token_dim::Int,        # Added token_dim (mandatory)
    act_space::Space) where T<:Real # act_space is mandatory

    # --- Calculate Static Dimensions ---
    g = blocks
    @assert deter_dim % g == 0 "deter_dim must be divisible by blocks (g)"
    @assert (stoch_dim * classes_dim) % g == 0 "stoch_dim*classes_dim must be divisible by blocks (g)" # Might need this if stoch is used in BlockLinear
    @assert (3 * deter_dim) % g == 0 "3*deter_dim must be divisible by blocks (g)" # For gru_layer output
    num_actions = act_space.high# Assuming discrete Space

    # --- Define Core Layers --- 

    # Layers for initial context processing (deter, stoch, action)
    layer_deter = Chain(
        Dense(deter_dim => hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
        RMSNorm((hidden_dim,), 1, act; dims=(1,), init_scale=cast_ones)
    )
    layer_stoch = Chain(
        Dense(stoch_dim * classes_dim => hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
        RMSNorm((hidden_dim,), 1, act; dims=(1,), init_scale=cast_ones)
    )
    layer_action = Chain(
        Dense(num_actions => hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
        RMSNorm((hidden_dim,), 1, act; dims=(1,), init_scale=cast_ones)
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
            RMSNorm((deter_dim,), 1, act; dims=(1,), init_scale=cast_ones)
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
            RMSNorm((hidden_dim,), 1, act; dims=(1,), init_scale=cast_ones)
        ))
        current_obs_input_dim = hidden_dim # Output of Norm is input to next Dense
    end
    posterior_layers = Chain(posterior_layers_list...; name="posterior_layers")

    # Prior layers
    prior_layers_list = []
    # Prior only depends on deter_dim
    current_prior_input_dim = deter_dim 

    for _ in 1:imglayers # Use the imglayers field from RSSM struct
        push!(prior_layers_list, Chain(
            # Use current_prior_input_dim correctly
            Dense(current_prior_input_dim => hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
            RMSNorm((hidden_dim,), 1, act; dims=(1,), init_scale=cast_ones)
        ))
        current_prior_input_dim = hidden_dim # Output of Norm is input to next Dense
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
        logit_posterior = logit_posterior,
    )
    imagination_layers = (
        prior_layers = prior_layers,
        logit_prior = logit_prior
    )

    # --- Return RSSM Instance --- 
    # Automatically determine types AS, CN, PO
    return RSSM(deter_dim, hidden_dim, stoch_dim, classes_dim, act, unimix, 
                imglayers, obslayers, dynlayers, blocks, free_nats, 
                token_dim, act_space, core_layers, observation_layers, imagination_layers)
end


"""
    initial_state(rssm::RSSM, batch_size::Int, ::AbstractRNG)

Returns the initial recurrent state for the RSSM.
Output shape: (deter = (batch_size, deter_dim), stoch = (batch_size, stoch_dim, classes_dim))
"""
function LuxCore.initialstates(rng::AbstractRNG, rssm::RSSM)
    # Delegate state initialization to the sub-layers stored in rssm.core
    # This will recursively call initialstates on the layers within the core NamedTuple.
    return (; 
        core = Lux.initialstates(rng, rssm.core), 
        observation = Lux.initialstates(rng, rssm.observation),
        imagination = Lux.initialstates(rng, rssm.imagination)
    )
end

# --- ObserveCell Definition (Moved Inside) ---
struct ObserveCell <: Lux.AbstractRecurrentCell
    rssm::RSSM
end
# Parameters are those of the underlying RSSM
Lux.initialparameters(rng::AbstractRNG, cell::ObserveCell) = Lux.initialparameters(rng, cell.rssm)
# State is that of the underlying RSSM's components
Lux.initialstates(rng::AbstractRNG, cell::ObserveCell) = Lux.initialstates(rng, cell.rssm)

struct ObserveInput
    tokens::AbstractArray
    action::AbstractArray
    reset::AbstractArray
end

function (cell::ObserveCell)((x, carry)::Tuple, ps_rssm::NamedTuple, st_rssm::NamedTuple)
    carry_next, entry_t, feat_t = _observe(cell.rssm, carry, x.tokens, x.action, x.reset, ps_rssm, st_rssm)
    return ((entry_t, feat_t), carry_next), st_rssm # Pass st_rssm through unchanged
end

function (cell::ObserveCell)(x::ObserveInput, ps_rssm::NamedTuple, st_rssm::NamedTuple)
    batch_size = size(x.tokens, 2)
    carry = initial_carry(cell.rssm, batch_size) |> _device;
    carry_next, entry_t, feat_t = _observe(cell.rssm, carry, x.tokens, x.action, x.reset, ps_rssm, st_rssm)
    return ((entry_t, feat_t), carry_next), st_rssm # Pass st_rssm through unchanged
end

function StatefulRSSM(;
        deter_dim::Int = 4096,
        hidden_dim::Int = 2048,
        stoch_dim::Int = 32,
        classes_dim::Int = 32,
        act::Function = gelu,
        imglayers::Int = 2,
        obslayers::Int = 1,
        dynlayers::Int = 1,
        blocks::Int = 8,
        unimix::T,
        free_nats::T,
        token_dim::Int,        # Added token_dim (mandatory)
        act_space::Space
    ) where T<:Real
    rssm = RSSM(;
        deter_dim,
        hidden_dim,
        stoch_dim,
        classes_dim,
        act,
        unimix,
        imglayers,
        obslayers,
        dynlayers,
        blocks,
        free_nats,
        token_dim,
        act_space
    )

    l = ObserveCell(rssm);
    recurrent_observe = StatefulRecurrentCell(l);
    return recurrent_observe
end;

"""
    initial_carry(rssm::RSSM, batch_size::Int)

Returns the initial carry for the RSSM. Carry is the state of the RSSM.
Output shape: (deter = (batch_size, deter_dim), stoch = (batch_size, stoch_dim, classes_dim))
"""
function initial_carry(rssm::RSSM, batch_size::Int)
    # Ensure we use the correct compute type (e.g., BFloat16)
    compute_T = isdefined(@__MODULE__, :COMPUTE_TYPE) ? COMPUTE_TYPE : Float32
    # Let's stick to Lux convention for layers, but the state `carry`
    deter_init = zeros(compute_T, rssm.deter_dim, batch_size)
        stoch_init = zeros(compute_T, rssm.stoch_dim, rssm.classes_dim, batch_size)

    # Use LuxCore.initialstates to get states for any potential stateful sub-layers later
    # For now, the state only contains the carry-over tensors.
    return (; deter = deter_init, stoch = stoch_init)
end

function observe(recurrent_observe::StatefulRecurrentCell, tokens, action, reset, ps, st)
    # seq_tokens shape: (token_dim, T, B)
    # seq_actions shape: (T, B)
    # seq_resets shape: (T, B)

    T = size(tokens, 2) # Get sequence length
    B = size(tokens, 3) # Get batch size
    S, C = recurrent_observe.cell.rssm.stoch_dim, recurrent_observe.cell.rssm.classes_dim # Get stoch and classes dims
    D = recurrent_observe.cell.rssm.deter_dim # Get deter dim

    # --- Initialize for functional accumulation ---
    # Accumulated sequences will be stored as tuples of tensors
    acc_seq_deter = ()
    acc_seq_stoch = ()
    acc_seq_logit = () # Or whatever features you extract

    # The state for the Lux.AbstractRecurrentCell for the loop
    st_loop = st

    # --- Loop over time steps ---
    for t in 1:T
        # Get inputs for the current time step
        tokens_t = view(tokens, :, t, :) # Shape: (token_dim, B)
        action_t = view(action, t, :)   # Shape: (B,)
        reset_t = view(reset, t, :)     # Shape: (B,)
        current_step_input = ObserveInput(tokens_t, action_t, reset_t)

        # Call the recurrent cell for one step.
        # This uses ps and the current st_loop, and returns (output_for_step, new_recurrent_cell_state).
        (entry_t, feat_t), st_loop = recurrent_observe(current_step_input, ps, st_loop)
        
        # Accumulate outputs by creating new tuples (Zygote-friendly)
        # entry_t.deter, entry_t.stoch, feat_t.logit are expected to be on the GPU if inputs are.
        acc_seq_deter = (acc_seq_deter..., entry_t.deter) 
        acc_seq_stoch = (acc_seq_stoch..., entry_t.stoch) 
        acc_seq_logit = (acc_seq_logit..., feat_t.logit) # Assuming feat_t has a .logit field
    end

    # --- After loop, concatenate the tuples of CuArrays to form final sequence CuArrays ---
    
    # Determine dimensions for empty case from model config if possible
    D_dim = hasproperty(recurrent_observe, :cell) && hasproperty(recurrent_observe.cell, :rssm) ? recurrent_observe.cell.rssm.deter_dim : (isempty(acc_seq_deter) ? 0 : size(acc_seq_deter[1],1))
    stoch_size_S = hasproperty(recurrent_observe, :cell) && hasproperty(recurrent_observe.cell, :rssm) ? recurrent_observe.cell.rssm.stoch_dim : (isempty(acc_seq_stoch) ? 0 : size(acc_seq_stoch[1],1))
    stoch_size_C = hasproperty(recurrent_observe, :cell) && hasproperty(recurrent_observe.cell, :rssm) ? recurrent_observe.cell.rssm.classes_dim : (isempty(acc_seq_stoch) ? 0 : size(acc_seq_stoch[1],2))
    
    seq_deter_final = cat([reshape(d, size(d,1), 1, size(d,2)) for d in acc_seq_deter]...; dims=2)
    seq_stoch_final = cat([reshape(s, size(s,1), size(s,2), 1, size(s,3)) for s in acc_seq_stoch]...; dims=3)
    seq_logit_final = cat([reshape(l, size(l,1), size(l,2), 1, size(l,3)) for l in acc_seq_logit]...; dims=3)
    
    # Prepare final outputs as per your desired structure
    final_feat_combined = (; deter=seq_deter_final, stoch=seq_stoch_final, logit=seq_logit_final)
    final_entry_combined = (; deter=seq_deter_final, stoch=seq_stoch_final) # Subset of features

    # Return the combined sequences and the final state of the recurrent cell
    return (final_entry_combined, final_feat_combined), st_loop 
end

function _observe(rssm::RSSM, carry::NamedTuple, tokens::AbstractArray, action::AbstractVector{T}, reset::AbstractVector{B}, ps, st) where {T<:Integer, B<:Bool}
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
    action_onehot = OneHotArrays.onehotbatch(action, 1:num_actions) # Shape: (num_actions, Batch)
    action_onehot_casted = cast(action_onehot) # Cast to COMPUTE_TYPE
    # Apply reset mask to the processed action
    action = action_onehot_casted .* deter_mask # Broadcast (1, Batch) mask

    # --- 3. Core Recurrent Update (Transition Model) ---
    deter_current, _ = _core(rssm, deter, stoch, action, ps, st)
    
    # --- 4. Observation Update (Posterior Calculation) ---
    # 4.1 Combine current deter and tokens
    tokens = reshape(tokens, :, size(deter_current, ndims(deter_current)));
    x = vcat(deter_current, tokens)
    # 4.2 Apply Posterior Layers
    x_posterior, _ = rssm.observation.posterior_layers(x, ps.observation.posterior_layers, st.observation.posterior_layers)
    logit_posterior, _ = rssm.observation.logit_posterior(x_posterior, ps.observation.logit_posterior, st.observation.logit_posterior)
    # 4d. Sample new stochastic state from posterior distribution
    dist_posterior = _dist(logit_posterior, rssm.unimix)
    # Use sample_ste to get the one-hot state with straight-through gradients
    # Assuming rng is available in the scope (might need to be passed explicitly)
    # TODO: Ensure rng is available here. If not, pass it as an argument.
    rng = Random.default_rng() # Placeholder: Get default RNG if not passed
    stoch_current = sample_ste(rng, dist_posterior)

    # --- 5. Prepare Outputs ---
    carry = (; deter=deter_current, stoch=stoch_current)
    feat = (; deter=deter_current, stoch=stoch_current, logit=logit_posterior) # Features include posterior logit
    entry = (; deter=deter_current, stoch=stoch_current) # State entry for storage/logging

    @assert all(eltype(deter) == eltype(stoch) == eltype(logit_posterior)) "All variables must have the same element type"

    # Placeholder return
    return carry, entry, feat
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

function _prior(rssm::RSSM, deter_seq::AbstractArray, ps, st)
    # Helper to compute prior logits from deterministic sequence
    # Input deter_seq shape: (deter_dim, T, B)
    D, T, B = size(deter_seq)

    # Reshape input for layers: (D, T, B) -> (D, T*B)
    deter_flat = reshape(deter_seq, D, T * B)

    # Apply prior feature layers
    # Use imagination parameters and state
    prior_features_flat, _ = rssm.imagination.prior_layers(deter_flat, ps.imagination.prior_layers, st.imagination.prior_layers)

    # Apply prior logit layer
    prior_logits_flat, _ = rssm.imagination.logit_prior(prior_features_flat, ps.imagination.logit_prior, st.imagination.logit_prior)

    # Reshape output back: (S, C, T*B) -> (S, C, T, B)
    prior_logits_seq = reshape(prior_logits_flat, rssm.stoch_dim, rssm.classes_dim, T, B)

    # Note: Might need to return updated state if layers become stateful
    return prior_logits_seq
end

function loss(stateful_rssm::StatefulRecurrentCell, tokens, action, reset, ps, st)

    rssm = stateful_rssm.cell.rssm;
    # Get the final feature output
    (_, feat), st = observe(stateful_rssm, tokens, action, reset, ps, st);

    # Prior and Posterior Distributions Logits
    post_logits = feat.logit; # Shape: (S, C, T, B)
    prior_logits = _prior(rssm, feat.deter, ps, st.cell); # Shape: (S, C, T, B)

    # KL Divergence Losses
    post_dist = _dist(post_logits, rssm.unimix);
    prior_dist = _dist(prior_logits, rssm.unimix);

    dyn_elementwise = kl_divergence(_dist(dropgrad(post_logits), rssm.unimix), prior_dist) # Shape: (S, T, B)
    rep_elementwise = kl_divergence(post_dist, _dist(dropgrad(prior_logits), rssm.unimix)) # Shape: (S, T, B)

    # Sum over stochastic dimension (dim=1) to match Agg behavior
    dyn_summed = sum(dyn_elementwise; dims=1) |> # Shape: (1, T, B)
    x -> dropdims(x; dims=1) # Shape: (T, B)
    rep_summed = sum(rep_elementwise; dims=1) |> # Shape: (1, T, B)
    x -> dropdims(x; dims=1) # Shape: (T, B)

    dyn = max.(dyn_summed, rssm.free_nats)
    rep = max.(rep_summed, rssm.free_nats)

    # Store scalar losses (using NamedTuple for type stability)
    losses = (; dyn = dyn, rep = rep)

    return losses, feat, st
end

# --- Distribution Helper ---

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

"""
    sample_ste(rng::AbstractRNG, d::OneHotDist)

Sample from the OneHotDist and apply the Straight-Through Estimator (STE).
Uses smoothed probabilities for sampling but unsmoothed probabilities for gradient flow.
Output shape: (stoch_dim, classes_dim, Batch...)
"""
function sample_ste(rng::AbstractRNG, d::OneHotDist)
    logits = d.logits # Shape (stoch_dim, classes_dim, Batch...)
    unimix = d.unimix
    stoch_dim, classes_dim = size(logits, 1), size(logits, 2)
    batch_dims = size(logits)[3:end]
    compute_T = eltype(logits)

    # Determine the device function (e.g., gpu_device() or cpu_device())
    # based on the type of the input logits.
    dev_func = logits isa CUDA.AbstractGPUArray ? Lux.gpu_device() : Lux.cpu_device()

    # --- Sampling based on smoothed probabilities ---
    probs_raw = softmax(logits; dims=2) # Remains on original device (e.g., GPU)
    probs_smoothed = (1 - unimix) .* probs_raw .+ unimix / classes_dim
    probs_clipped = max.(probs_smoothed, zero(compute_T)) # Remains on original device

    num_distributions = stoch_dim * prod(batch_dims; init=1)

    # Reshape probs_clipped for easier iteration. It's still on the original device.
    # Permuted shape: (stoch_dim, batch_dims..., classes_dim)
    # Then reshaped to (num_distributions, classes_dim)
    probs_2d_device = reshape(permutedims(probs_clipped, (1, (3:ndims(probs_clipped))..., 2)), num_distributions, classes_dim)

    # --- Perform sampling on CPU to avoid scalar indexing on GPU ---
    # 1. Move all probability distributions to CPU at once.
    probs_2d_cpu = Array(probs_2d_device) # Transfer from GPU to CPU if on GPU

    # 2. Perform sampling on CPU data using a comprehension (Zygote-friendly)
    # This was the site of the previous Zygote error.
    sampled_indices_cpu = [begin
                               prob_view_cpu = view(probs_2d_cpu, i, :)
                               prob_weights = Weights(prob_view_cpu, one(eltype(prob_view_cpu))) # Ensure sum is typed correctly
                               StatsBase.sample(rng, 1:classes_dim, prob_weights)
                           end for i in 1:num_distributions]
    # sampled_indices_cpu is now a Vector{Int} created without in-place setindex!

    # --- One-hot encode the sampled index (value for forward pass) ---
    indices_final_shape = (stoch_dim, batch_dims...) # Target shape for indices
    indices_reshaped_cpu = reshape(sampled_indices_cpu, indices_final_shape)

    # 3. Move indices back to the original device (e.g., GPU) for subsequent operations.
    indices_reshaped_device = dev_func(indices_reshaped_cpu)

    # Use OneHotArrays.onehotbatch for non-mutating creation
    # Input indices_reshaped_device shape: (stoch_dim, batch_dims...)
    value_onehot_permuted = OneHotArrays.onehotbatch(indices_reshaped_device, 1:classes_dim)
    # Output shape of onehotbatch: (classes_dim, stoch_dim, batch_dims...)

    # Permute to desired shape: (stoch_dim, classes_dim, batch_dims...)
    perm = (2, 1, (3:ndims(value_onehot_permuted))...)
    value_onehot = permutedims(value_onehot_permuted, perm) # This will be on the original device

    # --- Apply Straight-Through Estimator (STE) ---
    # STE: sg(value) + (probs - sg(probs))
    # Use Zygote.@ignore to stop gradients for the discrete parts.
    # Use the *unsmoothed* probs_raw for the gradient path.
    ste_value = Zygote.@ignore(value_onehot) .+ (probs_raw .- Zygote.@ignore(probs_raw))

    return ste_value
end



"""
    sample_ste(rng::AbstractRNG, d::OneHotDist)

Sample from the OneHotDist and apply the Straight-Through Estimator (STE).
Uses smoothed probabilities for sampling but unsmoothed probabilities for gradient flow.
Output shape: (stoch_dim, classes_dim, Batch...)
"""
function sample_ste_old(rng::AbstractRNG, d::OneHotDist)
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
    return ste_value
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
    return stoch_onehot # Assumes cast function handles Bool -> compute_T
end


# --- End Distribution Helper ---
"""
    kl_divergence(posterior_dist::OneHotDist, prior_dist::OneHotDist)

Computes the KL divergence D_KL(posterior || prior) for each batch element,
summed over the classes dimension but kept separate for the stoch dimension.

Assumes logits represent the parameters of the *unsmoothed* distributions.
Uses logsoftmax for numerical stability.

Args:
    posterior_dist: OneHotDist representing the posterior distribution P.
    prior_dist: OneHotDist representing the prior distribution Q.

Returns:
    Tensor of KL divergences, shape (stoch_dim, Batch...)
"""
function kl_divergence(posterior_dist::OneHotDist, prior_dist::OneHotDist)
    # Extract the logits arrays from the distribution structs
    post_logits = posterior_dist.logits
    prior_logits = prior_dist.logits

    # Calculate log-probabilities using logsoftmax for numerical stability
    log_p = logsoftmax(post_logits; dims=2)
    log_q = logsoftmax(prior_logits; dims=2)

    # Calculate probabilities needed for the expectation E_p[...]
    p = softmax(post_logits; dims=2)

    # KL = sum_i p_i * (log p_i - log q_i)
    # Sum over the classes dimension (dim=2)
    kl_elementwise = p .* (log_p .- log_q)
    kl_summed_over_classes = sum(kl_elementwise; dims=2)

    # Remove the classes dimension (which is now size 1)
    kl_final = dropdims(kl_summed_over_classes; dims=2)

    return kl_final
end

"""
    entropy(d::OneHotDist)

Computes the entropy H(P) = - sum_i P(i) log P(i) for each distribution.

Assumes logits represent the parameters of the *unsmoothed* distributions.
Uses logsoftmax/softmax for numerical stability.

Args:
    d: OneHotDist representing the distribution P.

Returns:
    Tensor of entropies, shape (stoch_dim, Batch...)
"""
function entropy(d::OneHotDist)
    logits = d.logits # Shape: (S, C, Batch...)

    # Calculate log probabilities and probabilities using numerically stable functions
    log_p = logsoftmax(logits; dims=2) # log P(i)
    p = softmax(logits; dims=2)      # P(i)

    # Entropy = - sum_i p_i * log p_i
    # Sum over the classes dimension (dim=2)
    entropy_elementwise = -p .* log_p
    entropy_summed = sum(entropy_elementwise; dims=2) # Shape: (S, 1, Batch...)

    # Remove the classes dimension
    entropy_final = dropdims(entropy_summed; dims=2) # Shape: (S, Batch...)

    return entropy_final
end
