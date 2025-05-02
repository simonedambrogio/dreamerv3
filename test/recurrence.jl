include("../dreamerv3/agents.jl");

# ------------------------------------------------------
const ANSI_GREEN = "\e[32m"
const ANSI_BLUE = "\e[34m"
const ANSI_ORANGE = "\e[33m"
const ANSI_VIOLET = "\e[35m"
const ANSI_RESET = "\e[0m"
test = 7
include("helper-gradient.jl")

# Inputs -------------------
test < 5 && begin
    
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
    rssm_config = make_config(config, "dyn", "debug");
    rssm = RSSM(
        deter_dim=rssm_config["deter"],
        hidden_dim=rssm_config["hidden"],
        stoch_dim=rssm_config["stoch"],
        classes_dim=rssm_config["classes"],
        blocks=rssm_config["blocks"],
        token_dim=size(tokens,1), # Get from encoder
        act_space=act_space,
        unimix=COMPUTE_TYPE(rssm_config["unimix"]),
        free_nats=COMPUTE_TYPE(rssm_config["free_nats"]),
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
    model, carry0, tkns, acts, rsts, p, s = rssm, carry_init, tokens, seq_actions, seq_resets, ps.rssm, st.rssm;
    # carry, tokens, action, reset, ps, st = carry_init, tokens[:, 1, :], seq_actions[1, :], seq_resets[1, :], ps.rssm, st.rssm;
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

test == 5 && begin
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
end

test == 6 && begin
    fullconfig = YAML.load_file("dreamerv3/configs.yaml");

    rng = MersenneTwister(1234);
    T, B = fullconfig["debug"]["batch_length"], fullconfig["debug"]["batch_size"];
    config = Dict(component => make_config(fullconfig, component, "debug") for component in ["enc", "dec", "dyn"]);
    spaces = Dict(:image => Tools.Space(UInt8, (96, 96, 1)), :action => Tools.Space(Int32, low=0, high=18));

    agent = WorldModelAgent(config, spaces);
    ps, st = Lux.setup(rng, agent);

    # Observation (example: image)
    obs_shape = (96, 96, 1, T, B);
    obs_image = rand(UInt8, obs_shape);
    obs = (; image = obs_image);
    seq_actions = rand(rng, spaces[:action].low:spaces[:action].high, T, B);
    seq_resets = rand(rng, Bool, T, B);
    tokens, _ = agent.encoder(obs, ps.encoder, st.encoder);


    loss_vale = loss(agent.rssm, tokens, seq_actions, seq_resets, ps.rssm, st.rssm)
    println("Loss Value: ", loss_vale)

    println("Starting Zygote Gradient: ")
    Zygote.withgradient(
        p_ -> loss(agent.rssm, tokens, seq_actions, seq_resets, p_, st.rssm),
        ps.rssm # Differentiate with respect to parameters
    );
    println("Zygote Gradient ended")
end

fullconfig = YAML.load_file("dreamerv3/configs.yaml");

rng = MersenneTwister(1234);
T, B = fullconfig["debug"]["batch_length"], fullconfig["debug"]["batch_size"];
config = Dict(component => make_config(fullconfig, component, "debug") for component in ["enc", "dec", "dyn"]);
spaces = Dict(:image => Tools.Space(UInt8, (96, 96, 1)), :action => Tools.Space(Int32, low=0, high=18));

agent = WorldModelAgent(config, spaces);
ps, st = Lux.setup(rng, agent);

# Observation (example: image)
obs_shape = (96, 96, 1, T, B);
obs_image = rand(UInt8, obs_shape);
obs = (; image = obs_image);
seq_actions = rand(rng, spaces[:action].low:spaces[:action].high, T, B);
seq_resets = rand(rng, Bool, T, B);
tokens, _ = agent.encoder(obs, ps.encoder, st.encoder);


loss_val, _ = loss(agent.rssm, tokens, seq_actions, seq_resets, ps.rssm, st.rssm);
println("Loss Value: ", loss_val)

println("Starting Zygote Gradient: ")
val, grad = Zygote.withgradient(
    p_ -> begin
        loss_val, _ = loss(agent.rssm, tokens, seq_actions, seq_resets, p_, st.rssm)
        mean(loss_val.dyn) + mean(loss_val.rep)
    end,
    ps.rssm # Differentiate with respect to parameters
);
println("Zygote Gradient ended")

print_grad_rssm(grad[1])

# --- Refactored Gradient Calculation for Test 7 ---
function _prior(rssm::RSSM, deter_t::AbstractArray, ps, st)
    # Helper to compute prior logits from a single step deterministic state
    # Input deter_t shape: (deter_dim, B)
    @assert ndims(deter_t) == 2 "Input deter_t must be 2D (deter_dim, B)"

    # Apply prior feature layers directly to the single step input
    # Use imagination parameters and state
    # Assumes prior_layers takes (Features, Batch) input
    prior_features, st_prior_layers = rssm.imagination.prior_layers(deter_t, ps.imagination.prior_layers, st.imagination.prior_layers)

    # Apply prior logit layer
    # Assumes logit_prior takes (Features, Batch) input and outputs (S, C, B)
    prior_logits, st_logit_prior = rssm.imagination.logit_prior(prior_features, ps.imagination.logit_prior, st.imagination.logit_prior)

    # TODO: Handle potential state updates from the layers if they become stateful
    # For now, we are ignoring the returned states st_prior_layers, st_logit_prior
    # If they need to be managed, the function signature and return value need adjustment.

    # Output shape: (S, C, B)
    return prior_logits
end


# Objective function with correct recurrence for Zygote
function rssm_kl_loss_objective(recurrent_layer, initial_st, ps_rssm, tokens_seq, actions_seq, resets_seq)
    rssm_model = recurrent_layer.cell.rssm # Get the underlying RSSM
    T = size(tokens_seq, 2)
    B = size(tokens_seq, 3)
    el_type = eltype(tokens_seq)

    total_dyn_loss = zero(el_type)
    total_rep_loss = zero(el_type)
    current_st = initial_st # Start with initial state

    # Loop through time
    for t in 1:T
        # Prepare input for this time step
        # Ensure action indices are Int for onehotbatch, reset is Bool
        input_t = ObserveInput(view(tokens_seq, :, t, :),
                               Int.(view(actions_seq, t, :)),
                               Bool.(view(resets_seq, t, :)))

        # Apply the stateful recurrent cell for one step
        (out_step, next_st) = recurrent_layer(input_t, ps_rssm, current_st)
        entry_t, feat_t = out_step # Unpack the relevant outputs

        # Update state for the next iteration
        current_st = next_st

        # -- Calculate KL for this step (needs prior based on deter_t) --
        prior_logits_t = _prior(rssm_model, entry_t.deter, ps_rssm, current_st.cell) # Prior depends on current deter
        post_logits_t = feat_t.logit

        post_dist_t = _dist(post_logits_t, rssm_model.unimix)
        prior_dist_t = _dist(prior_logits_t, rssm_model.unimix)

        dyn_elementwise_t = kl_divergence(_dist(Zygote.dropgrad(post_logits_t), rssm_model.unimix), prior_dist_t)
        rep_elementwise_t = kl_divergence(post_dist_t, _dist(Zygote.dropgrad(prior_logits_t), rssm_model.unimix))

        # Sum over stochastic dimension (dim=1) & drop dim
        dyn_summed_t = sum(dyn_elementwise_t; dims=1) |> x -> dropdims(x; dims=1) # Shape: (T, B) -> (B,) for step t
        rep_summed_t = sum(rep_elementwise_t; dims=1) |> x -> dropdims(x; dims=1) # Shape: (T, B) -> (B,) for step t

        # Apply free_nats clamp
        free_nats_val = convert(eltype(dyn_summed_t), rssm_model.free_nats)
        # TEMPORARILY REMOVE CLAMP for gradient testing
        loss_dyn_t = dyn_summed_t # max.(dyn_summed_t, free_nats_val)
        loss_rep_t = rep_summed_t # max.(rep_summed_t, free_nats_val)

        total_dyn_loss += mean(loss_dyn_t) # Accumulate mean loss per step
        total_rep_loss += mean(loss_rep_t) # Accumulate mean loss per step
    end

    # Return average loss across time
    return (total_dyn_loss + total_rep_loss) / T
end

# recurrent_layer, initial_st, ps_rssm, tokens_seq, actions_seq, resets_seq = agent.rssm, st.rssm, ps.rssm, tokens, seq_actions, seq_resets;

rssm_kl_loss_objective(agent.rssm, st.rssm, ps.rssm, tokens, seq_actions, seq_resets)

println("Starting Zygote Gradient (Test 7 - Refactored): ")
val, grad = Zygote.withgradient(
    (p, s) -> rssm_kl_loss_objective(agent.rssm, s, p, tokens, seq_actions, seq_resets),
    ps.rssm, # Parameters to differentiate w.r.t.
    st.rssm  # Initial state (should not get gradient)
);
println("Loss Value (Refactored): ", val)
println("Zygote Gradient ended (Test 7 - Refactored)")

# grad[1] should now contain non-zero gradients for ps.rssm
# grad[2] should be nothing (gradient w.r.t. initial state st.rssm)
if grad[1] !== nothing
    print_grad_rssm(grad[1]) # Use your helper function
else
    println("Gradients are still nothing!")
end

