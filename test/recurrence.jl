using Test, Random, Lux, Zygote, NNlib, BFloat16s, YAML, Statistics, 
ChainRulesCore, OneHotArrays, SliceMap
using StatsBase: Weights # Added for sampling
using OneHotArrays: onehot # Added for encoding 
using Zygote: Buffer, @ignore # Import Buffer

# --- Workaround for AbstractRecurrentCell TypeError ---
# ------------------------------------------------------
const ANSI_GREEN = "\e[32m"
const ANSI_BLUE = "\e[34m"
const ANSI_ORANGE = "\e[33m"
const ANSI_VIOLET = "\e[35m"
const ANSI_RESET = "\e[0m"


# --- Include necessary components --- 
include("../embodied/lux/BlockLinear.jl");
include("../embodied/lux/ReArrange.jl");
include("../embodied/lux/rms.jl");
include("../embodied/lux/nets.jl"); # Defines COMPUTE_TYPE and cast
include("../dreamerv3/encoder.jl");
include("../dreamerv3/agents.jl");

test = 5
# --- Define RSSM struct ---
begin
    # Based on Python RSSM class attributes
    struct RSSM{AS, CN, PO, OI} <: Lux.AbstractLuxContainerLayer{(:core, :observation, :imagination)} # Added AS for type stability
        deter_dim::Int
        hidden_dim::Int
        stoch_dim::Int
        classes_dim::Int
        act::Function
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
        imagination::OI
    end
    
    function RSSM(; # Constructor
        deter_dim::Int = 4096,
        hidden_dim::Int = 2048,
        stoch_dim::Int = 32,
        classes_dim::Int = 32,
        act::Function = gelu,
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
    
    # --- ObserveCell Definition (Moved Inside) ---
    println("Type of Lux.AbstractRecurrentCell just before definition: ", typeof(Lux.AbstractRecurrentCell))
    struct ObserveCell <: Lux.AbstractRecurrentCell
        rssm::RSSM
    end
    # Parameters are those of the underlying RSSM
    Lux.initialparameters(rng::AbstractRNG, cell::ObserveCell) = Lux.initialparameters(rng, cell.rssm)
    # State is that of the underlying RSSM's components
    Lux.initialstates(rng::AbstractRNG, cell::ObserveCell) = Lux.initialstates(rng, cell.rssm)
    
    struct DynInput
        tokens::AbstractArray
        action::AbstractArray
        reset::AbstractArray
    end
    
    function (cell::ObserveCell)((x, carry)::Tuple, ps_rssm::NamedTuple, st_rssm::NamedTuple)
        carry_next, entry_t, feat_t = _observe(cell.rssm, carry, x.tokens, x.action, x.reset, ps_rssm, st_rssm)
        return ((entry_t, feat_t), carry_next), st_rssm # Pass st_rssm through unchanged
    end
    
    function (cell::ObserveCell)(x::DynInput, ps_rssm::NamedTuple, st_rssm::NamedTuple)
        batch_size = size(x.tokens, 2)
        carry = initial_carry(cell.rssm, batch_size)
        carry_next, entry_t, feat_t = _observe(cell.rssm, carry, x.tokens, x.action, x.reset, ps_rssm, st_rssm)
        return ((entry_t, feat_t), carry_next), st_rssm # Pass st_rssm through unchanged
    end
    
    function StatefulRSSM(;
            deter_dim::Int = 4096,
            hidden_dim::Int = 2048,
            stoch_dim::Int = 32,
            classes_dim::Int = 32,
            act::Function = gelu,
            unimix::Float32 = 0.01f0,
            imglayers::Int = 2,
            obslayers::Int = 1,
            dynlayers::Int = 1,
            blocks::Int = 8,
            free_nats::Float32 = 1.0f0,
            token_dim::Int,        # Added token_dim (mandatory)
            act_space::Space
        )
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
    
    """
        initial_carry(rssm::RSSM, batch_size::Int)
    
    Returns the initial carry for the RSSM. Carry is the state of the RSSM.
    Output shape: (deter = (batch_size, deter_dim), stoch = (batch_size, stoch_dim, classes_dim))
    """
    function initial_carry(rssm::RSSM, batch_size::Int)
        # Ensure we use the correct compute type (e.g., BFloat16)
        compute_T = isdefined(@__MODULE__, :COMPUTE_TYPE) ? COMPUTE_TYPE : Float32
        # Replace fill!(similar(...), zero) with zeros(T, dims...)
        deter_init = zeros(compute_T, rssm.deter_dim, batch_size)
        stoch_init = zeros(compute_T, rssm.stoch_dim, rssm.classes_dim, batch_size)
    
        # Use LuxCore.initialstates to get states for any potential stateful sub-layers later
        # For now, the state only contains the carry-over tensors.
        # --- Restore NamedTuple return ---
        return (; deter = deter_init, stoch = stoch_init)
    end
    
    function observe(rssm::RSSM, carry, tokens, action, reset, ps, st)
        # seq_tokens shape: (token_dim, T, B)
        # seq_actions shape: (T, B)
        # seq_resets shape: (T, B)
    
        T = size(tokens, 2) # Get sequence length
        B = size(tokens, 3) # Get batch size
    
        # --- Pre-allocate Output Arrays (Buffer Approach) ---
        # Determine element type from carry (assuming consistency)
        el_type = eltype(carry.deter)
    
        # Allocate arrays to store the full sequences
        # Shapes: (feature_dim, T, B) or (feature_dim1, feature_dim2, T, B)
        seq_deter_out = similar(tokens, el_type, rssm.deter_dim, T, B)
        # Assuming entry.stoch and feat.logit have same shape structure as initial carry.stoch
        stoch_dims = size(carry.stoch)[1:end-1] # Get stoch dims excluding Batch
        seq_stoch_out = similar(tokens, el_type, stoch_dims..., T, B)
        seq_logit_out = similar(tokens, el_type, stoch_dims..., T, B)
    
    
        current_carry = carry
        # --- Loop over time steps ---
        for t in 1:T
            # Get inputs for the current time step
            tokens_t = view(tokens, :, t, :) # Shape: (token_dim, B)
            action_t = view(action, t, :)   # Shape: (B,)
            reset_t = view(reset, t, :)     # Shape: (B,)
    
            # Call the single-step observe function
            carry_next, entry_t, feat_t = _observe(rssm, current_carry, tokens_t, action_t, reset_t, ps, st);
    
            # Store results for this time step directly into pre-allocated arrays
            view(seq_deter_out, :, t, :) .= entry_t.deter
            # Use ellipsis `..` for stoch/logit dimensions before T and B
            view(seq_stoch_out, :, :, t, :) .= entry_t.stoch # Shape (S, C, B)
            view(seq_logit_out, :, :, t, :) .= feat_t.logit  # Shape (S, C, B)
    
    
            # Update carry for the next iteration
            current_carry = carry_next
        end
    
        # --- Prepare final outputs ---
        final_carry = current_carry
        # The pre-allocated arrays now hold the full sequences
        final_feat = (; deter=seq_deter_out, stoch=seq_stoch_out, logit=seq_logit_out)
        final_entry = (; deter=seq_deter_out, stoch=seq_stoch_out)
    
        return final_carry, final_entry, final_feat
    end

    function _observe(rssm::RSSM, carry::NamedTuple, tokens::AbstractArray, action::AbstractVector{T}, reset::AbstractVector{B}, ps, st) where {T<:Integer, B<:Bool} # Use NamedTuple carry
        # carry: NamedTuple with (deter, stoch)
        # tokens: Encoded observations, shape (token_dim, Batch) [Assuming single step for now]
        # action: Discrete action indices for the current step, shape (Batch,)
        # reset: Boolean vector for the current step, shape (Batch,)
    
        # --- 1. Apply reset mask to state ---
        # Create the inverted mask, ready for broadcasting
        keep_mask = .!reset # Shape: (Batch,)
        # Apply mask to deter state
        deter_mask = reshape(keep_mask, 1, :) # Shape: (1, Batch)
        deter = carry.deter .* deter_mask # Use NamedTuple access
        # Apply mask to stoch state
        stoch_mask = reshape(keep_mask, 1, 1, :) # Shape: (1, 1, Batch)
        stoch = carry.stoch .* stoch_mask # Use NamedTuple access
    
        # --- 2. Process Action ---
        # Assuming action is discrete and needs one-hot encoding
        @assert !isnothing(rssm.act_space) "RSSM requires act_space to process actions"
        num_actions = rssm.act_space.high # Assumes Space defines range [low, high)
        # Perform one-hot encoding. Note: NNlib.onehotbatch expects indices starting from 1.
        action_onehot = OneHotArrays.onehotbatch(action, 0:num_actions) # Shape: (num_actions, Batch)
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
        # stoch_current = stoch # Assuming rng is available
        stoch_current = carry.stoch #Zygote.@ignore rand(rng, dist_posterior) # Using carry.stoch for testing grads
    
        # --- 5. Prepare Outputs ---
        # Calculate components first
        carry_out = (; deter=deter_current, stoch=stoch_current) # Return NamedTuple carry
        feat_out = (; deter=deter_current, stoch=stoch_current, logit=logit_posterior)
        entry_out = (; deter=deter_current, stoch=stoch_current)
    
        # Check types before returning the original tuples
        @assert all(eltype(deter_current) == eltype(stoch_current) == eltype(logit_posterior)) "All variables must have the same element type"
        
        # println("entry_out.stoch: ", size(entry_out.stoch))
        # Return the original tuples
        return carry_out, entry_out, feat_out
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
end;

# Inputs -------------------
begin
    
    # --- Configuration & Setup ---
    config = YAML.load_file("dreamerv3/configs.yaml");
    
    # Define compute type (should match nets.jl)
    @assert COMPUTE_TYPE == BFloat16 # Ensure consistency
    
    rng = Random.default_rng();
    Random.seed!(rng, 0);
    
    B = batch_size = config["debug"]["batch_size"];
    T = seq_length = config["debug"]["batch_length"];
    obs = Tools.Space(UInt8, (96, 96, 1));
    depth = config["debug"]["agent"][".*\\.depth"];
    units = config["debug"]["agent"][".*\\.units"];
    deter_dim = config["debug"]["agent"][".*\\.deter"];
    stoch_dim = config["debug"]["agent"][".*\\.stoch"];
    classes_dim = config["debug"]["agent"][".*\\.classes"];
    hidden_dim = config["debug"]["agent"][".*\\.hidden"];
    blocks = config["debug"]["agent"][".*\\.blocks"];
    
    act=gelu;
    mults=config["defaults"]["agent"]["enc"]["simple"]["mults"];
    kernel=config["defaults"]["agent"]["enc"]["simple"]["kernel"];
    
    rng = Random.default_rng();
    
    println(ANSI_GREEN, "\n----- Running Encoder Forward Pass -----", ANSI_RESET)
    enc = Encoder(; obs, act, mults, depth, kernel);
    ps, st = Lux.setup(rng, enc);
    dummy_image = rand(rng, UInt8, obs.size..., T, B);
    dummy_obs = (; image = dummy_image); # Use NamedTuple matching encoder input
    println("  - Input shape: ", size(dummy_obs[:image]), "     -> (W, H, C, T, B)")
    tokens, st_new = enc(dummy_obs, ps, st);
    println("  - Output shape: ", size(tokens),         "          -> (token_dim, T, B)")
    
    act_space = Tools.Space(Int32, low=0, high=18);
    
    # RSSM
    rssm_config = make_config("dyn", "debug");
    rssm = RSSM(
        deter_dim=rssm_config["deter"],
        hidden_dim=rssm_config["hidden"],
        stoch_dim=rssm_config["stoch"],
        classes_dim=rssm_config["classes"],
        blocks=rssm_config["blocks"],
        token_dim=size(tokens,1), # Get from encoder
        act_space=act_space
    );
    
    # Parameters and State
    ps_enc, st_enc = Lux.setup(rng, enc);
    ps_rssm, st_rssm = Lux.setup(rng, rssm);
    ps = (; encoder=ps_enc, rssm=ps_rssm);
    st = (; encoder=st_enc, rssm=st_rssm);
    
    # Convert parameters to COMPUTE_TYPE (BFloat16)
    ps = Lux.fmap(x -> x isa AbstractArray ? COMPUTE_TYPE.(x) : x, ps);
    
    println("  - Models Initialized")
    
    # --- Dummy Data Generation ---
    
    # Observation (example: image)
    obs_shape = (96, 96, 1, T, B);
    obs_image = rand(UInt8, obs_shape);
    obs = (; image = obs_image);
    
    # Actions (discrete, 0 to num_actions-1)
    # Action shape: (T, B)
    seq_actions = rand(rng, act_space.low:act_space.high, T, B);
    
    # Resets (boolean)
    # Reset shape: (T, B)
    seq_resets = rand(rng, Bool, T, B);
    
    println("  - Dummy Data Generated")
    
    # --- Encoder Forward Pass --- 
    # Convert obs to Float32 for encoder input processing like Python version
    tokens, _ = enc(dummy_obs, ps.encoder, st.encoder);
    
    println("  - Encoder Pass Completed. Token shape: ", size(tokens))
    
    # --- Initial Carry State --- 
    carry_init = initial_carry(rssm, B);
    println("  - Initial Carry Generated. Deter: ", size(carry_init.deter), ", Stoch: ", size(carry_init.stoch));
end


test == 1 && begin
    println("--- Test Script Finished ---") 
    model, carry0, tkns, acts, rsts, p, s = rssm, carry_init, tokens, seq_actions, seq_resets, ps.rssm, st.rssm
    function simplified_two_step_array_comprehension_objective(model::RSSM, carry0::NamedTuple, tkns, acts, rsts, p, s) # Use NamedTuple carry0
        println("    - Entering simplified_two_step_array_comprehension_objective...")
        
        array_comprehension = [
            _observe(model, carry0, tkns_t1, acts_t1, rsts_t1, p, s)
            for (tkns_t1, acts_t1, rsts_t1) in zip(eachslice(tkns, dims=2), eachslice(acts, dims=1), eachslice(rsts, dims=1))
        ];
        
        objective_value = sum( sum(step_t[2].deter) for step_t in  array_comprehension);
        
        return objective_value
    end

    val_3step, grads_3step = Zygote.withgradient(simplified_two_step_array_comprehension_objective, rssm, carry_init, tokens[:, 1:2, :], seq_actions[1:2, :], seq_resets[1:2, :], ps.rssm, st.rssm)
end

# --- Recurrence Test using StatefulRecurrentCell ---
test == 2 && begin 
    println("\n" * ANSI_VIOLET * "--- Running StatefulRecurrentCell Test ---" * ANSI_RESET)
    
    # 1. Create the Stateful Layer
    
    # 3. Prepare Input Sequence
    # Vector of Tuples: [(tokens_1, action_1, reset_1), (tokens_2, action_2, reset_2), ...]
    T_test = size(tokens, 2) # Use full sequence length T
    inputs_sequence = [
        # Ensure views are taken correctly
        (view(tokens, :, t, :), view(seq_actions, t, :), view(seq_resets, t, :))
        for t in 1:T_test
    ];
    println("  - Input sequence prepared. Length: ", length(inputs_sequence))
    
    # --- Test Manual Unrolling of ObserveCell ---
    println("\n" * ANSI_BLUE * "--- Running Manual Unroll Test ---" * ANSI_RESET)
    
    function (cell::ObserveCell)((x, carry), ps_rssm::NamedTuple, st_rssm::NamedTuple)
        tokens_t, action_t, reset_t = x
        carry_next, entry_t, feat_t = _observe(cell.rssm, carry, tokens_t, action_t, reset_t, ps_rssm, st_rssm)
        return ((entry_t, feat_t), carry_next), st_rssm # Pass st_rssm through unchanged
    end
    
    function (cell::ObserveCell)(x, ps_rssm::NamedTuple, st_rssm::NamedTuple)
        tokens_t, action_t, reset_t = x
        carry_next, entry_t, feat_t = _observe(cell.rssm, tokens_t, action_t, reset_t, ps_rssm, st_rssm)
        return ((entry_t, feat_t), carry_next), st_rssm # Pass st_rssm through unchanged
    end
    
    l = ObserveCell(rssm);
    x = (tokens[:, 1, :], seq_actions[1, :], seq_resets[1, :]);
    carry = initial_carry(rssm, B);
    ps, st = Lux.setup(rng, l);
    (out, carry), st_  = Lux.apply(l, x, ps, st);
    
    r = StatefulRecurrentCell(l);
    ps, st = Lux.setup(rng, r);
    out, st = r(x, ps, st)
end

test == 3 && begin
    println("\n" * ANSI_GREEN * "--- Running Iterative Recurrent Objective Gradient Test ---" * ANSI_RESET)

    # Define the objective function with proper iteration
    function recurrent_objective(r::StatefulRecurrentCell, x, ps, st)
        # Determine the element type for accumulation based on the expected output type
        el_type = eltype(x[1][1])
        objective_value = zero(el_type)
        # inputs_sequence should be like: [(tokens_1, action_1, reset_1), (tokens_2, ...), ...]
        # x_t = x[2]
        for x_t in x
            out, st = r(DynInput(x_t...), ps, st)
            objective_value += sum(out[1].deter)
        end

        return objective_value
    end

    # --- Prepare inputs for the objective ---
    T_test = 2 # Use 2 steps like the array comprehension test for comparison
    x = [
        (view(tokens, :, t, :), view(seq_actions, t, :), view(seq_resets, t, :))
        for t in 1:T_test
    ];
    println("  - Input sequence prepared. Length: ", length(x))


    l = ObserveCell(rssm);
    r = StatefulRecurrentCell(l);
    ps, st = Lux.setup(rng, r);

    # Wrap the objective call with Zygote.withgradient
    # We want gradients w.r.t. ps (parameters)
    println("  - Calling the objective function")
    val, grads = Zygote.withgradient(recurrent_objective, r, x, ps, st)

    println("  - Objective Value (StatefulRecurrentCell): ", val)

    # Check the gradients object
    grads_ps = grads[3] # Gradients are returned in a tuple matching args (r, x, ps, st)
    println("  - Gradients w.r.t. Parameters (ps) exist: ", grads_ps !== nothing)
end

test == 4 && begin
    println("\n" * ANSI_ORANGE * "--- Running Iterative Recurrent Objective Gradient Test w/ Output Collection (Test 4) ---" * ANSI_ORANGE)

    # Define the objective function with proper iteration and output collection
    function recurrent_objective_collect(r::StatefulRecurrentCell, x_sequence, ps_cell, initial_st)
        # Determine sequence length and element type
        T = length(x_sequence)
        el_type = eltype(x_sequence[1][1]) # Type from tokens

        # Extract RSSM and dimensions needed for buffers
        D = rssm.deter_dim
        S = rssm.stoch_dim
        C = rssm.classes_dim
        B = size(x_sequence[1][1], 2)

        objective_value = zero(el_type)

        # Initialize Zygote Buffers
        deter_buffer = Buffer(zeros(el_type, D, T, B))
        logit_buffer = Buffer(zeros(el_type, S, C, T, B))
        stoch_buffer = Buffer(zeros(el_type, S, C, T, B)) # Also collect stoch if needed

        current_st = initial_st
        println("    - Starting recurrent loop (T=$T)...")
        for t in 1:T
            x_t = x_sequence[t]

            out, next_st = r(DynInput(x_t...), ps_cell, current_st)
            entry_t, feat_t = out

            objective_value += sum(entry_t.deter)

            # Store results in buffers
            deter_buffer[:, t, :] = entry_t.deter
            stoch_buffer[:, :, t, :] = entry_t.stoch # Store stoch state
            logit_buffer[:, :, t, :] = feat_t.logit

            current_st = next_st
        end
        println("    - Finished recurrent loop.")

        # Convert buffers to regular arrays
        collected_deter = copy(deter_buffer)
        collected_stoch = copy(stoch_buffer)
        collected_logits = copy(logit_buffer)
        println("    - Buffers copied to arrays.")

        # Return objective, collected sequences, and final state
        # Returning stoch state as well, as it might be useful
        return objective_value, collected_deter, collected_stoch, collected_logits, current_st
    end

    # --- Prepare inputs for the objective ---
    T_test = 2
    x_t4 = [
        (view(tokens, :, t, :), view(seq_actions, t, :), view(seq_resets, t, :))
        for t in 1:T_test
    ];
    println("  - Input sequence prepared. Length: ", length(x_t4))

    # --- Setup Model and State ---
    l_t4 = ObserveCell(rssm);
    r_t4 = StatefulRecurrentCell(l_t4);
    ps_r_t4, st_r_t4 = Lux.setup(rng, r_t4);
    ps_r_t4 = Lux.fmap(x -> x isa AbstractArray ? COMPUTE_TYPE.(x) : x, ps_r_t4);
    println("  - Stateful Cell, Parameters, and State Initialized.")

    # --- Run Zygote ---
    println("  - Calling Zygote.withgradient...")
    (val_t4, deter_coll, stoch_coll, logits_coll, final_st_t4), grads_t4 = Zygote.withgradient(
        recurrent_objective_collect, # Use the collecting version
        r_t4,       # The stateful recurrent cell
        x_t4,       # Sequence of inputs
        ps_r_t4,    # Parameters of the cell
        st_r_t4     # Initial state of the cell
    )
    println("  - Zygote.withgradient finished.")

    # --- Use collected outputs ---
    println("  - Calculating Prior Logits using collected deter sequence...")
    # Use parameters (ps_r_t4) and the *internal state* of the ObserveCell (st_r_t4.st)
    prior_logits_t4 = _prior(rssm, deter_coll, ps_r_t4, st_r_t4.cell)
    println("  - Prior Logits Calculated. Shape: ", size(prior_logits_t4)) # Shape: (S, C, T, B)
    println("--- Iterative Recurrent Objective Test w/ Collection (Test 4) Finished ---")
end


function observe(recurrent_observe::StatefulRecurrentCell, tokens, action, reset, ps, st)
    # seq_tokens shape: (token_dim, T, B)
    # seq_actions shape: (T, B)
    # seq_resets shape: (T, B)

    T = size(tokens, 2) # Get sequence length
    B = size(tokens, 3) # Get batch size
    S, C = recurrent_observe.cell.rssm.stoch_dim, recurrent_observe.cell.rssm.classes_dim # Get stoch and classes dims
    D = recurrent_observe.cell.rssm.deter_dim # Get deter dim

    # --- Pre-allocate Output Arrays (Buffer Approach) ---
    # Determine element type from carry (assuming consistency)
    el_type = eltype(tokens)

    # Allocate arrays to store the full sequences
    # Shapes: (feature_dim, T, B) or (feature_dim1, feature_dim2, T, B)
    # seq_deter_out = similar(tokens, el_type, rssm.deter_dim, T, B)
    # Assuming entry.stoch and feat.logit have same shape structure as initial carry.stoch
    # stoch_dims = size(carry.stoch)[1:end-1] # Get stoch dims excluding Batch
    # seq_stoch_out = similar(tokens, el_type, stoch_dims..., T, B)
    # seq_logit_out = similar(tokens, el_type, stoch_dims..., T, B)

    seq_deter_out = Buffer(zeros(el_type, D, T, B))
    seq_stoch_out = Buffer(zeros(el_type, S, C, T, B))
    seq_logit_out = Buffer(zeros(el_type, S, C, T, B))

    # --- Loop over time steps ---
    for t in 1:T
        # Get inputs for the current time step
        tokens_t = view(tokens, :, t, :) # Shape: (token_dim, B)
        action_t = view(action, t, :)   # Shape: (B,)
        reset_t = view(reset, t, :)     # Shape: (B,)

        out, st = recurrent_observe(DynInput(tokens_t, action_t, reset_t), ps, st);
        entry_t, feat_t = out

        
        # Store results in buffers
        seq_deter_out[:, t, :] = entry_t.deter
        seq_stoch_out[:, :, t, :] = entry_t.stoch # Store stoch state
        seq_logit_out[:, :, t, :] = feat_t.logit
    end

    # --- Prepare final outputs ---
    # The pre-allocated arrays now hold the full sequences
    final_feat = (; deter=copy(seq_deter_out), stoch=copy(seq_stoch_out), logit=copy(seq_logit_out))
    final_entry = (; deter=copy(seq_deter_out), stoch=copy(seq_stoch_out))

    return (final_entry, final_feat), st
end

function loss(recurrent_observe::StatefulRecurrentCell, tokens, action, reset, ps, st)

    rssm = recurrent_observe.cell.rssm;
    # Get the final feature output
    (_, feat), st = observe(recurrent_observe, tokens, action, reset, ps, st);

    # Prior and Posterior Distributions Logits
    post_logits = feat.logit; # Shape: (S, C, T, B)
    prior_logits = _prior(rssm, feat.deter, ps, st.cell); # Shape: (S, C, T, B)

    # KL Divergence Losses
    post_dist = _dist(post_logits, rssm.unimix);
    prior_dist = _dist(prior_logits, rssm.unimix);

    dyn_elementwise = kl_divergence(_dist(dropgrad(post_logits), rssm.unimix), prior_dist)
    rep_elementwise = kl_divergence(post_dist, _dist(dropgrad(prior_logits), rssm.unimix))

    # Apply free_nats clamp (Shape: S, T, B)
    dyn_clamped = max.(dyn_elementwise, COMPUTE_TYPE(rssm.free_nats))
    rep_clamped = max.(rep_elementwise, COMPUTE_TYPE(rssm.free_nats))

    # Sum over stochastic dimension (dim=1) to match Agg behavior
    dyn_summed = sum(dyn_clamped; dims=1) # Shape: (1, T, B)
    rep_summed = sum(rep_clamped; dims=1) # Shape: (1, T, B)

    dyn = dropdims(dyn_summed; dims=1)
    rep = dropdims(rep_summed; dims=1)
    # Store scalar losses (using NamedTuple for type stability)
    losses = (; dyn = dyn, rep = rep)

    return mean(losses.dyn) + mean(losses.rep)
end

l = ObserveCell(rssm);
recurrent_observe = StatefulRecurrentCell(l);
ps, st = Lux.setup(rng, recurrent_observe);
loss(recurrent_observe, tokens, seq_actions, seq_resets, ps, st)

val, grads = Zygote.withgradient(
    loss,                # Use the collecting version
    recurrent_observe,   # The stateful recurrent cell
    tokens,              # Sequence of inputs
    seq_actions,         # Parameters of the cell
    seq_resets,          # Initial state of the cell
    ps,
    st
)
