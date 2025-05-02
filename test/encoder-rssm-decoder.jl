include("../dreamerv3/agents.jl");
using Optimisers

# Start Test ------------------------------------------------------------
fullconfig = YAML.load_file("dreamerv3/configs.yaml");

rng = MersenneTwister(1234);
T, B = fullconfig["debug"]["batch_length"], fullconfig["debug"]["batch_size"];
config = Dict(component => make_config(fullconfig, component, "debug") for component in ["enc", "dec", "dyn"]);
spaces = Dict(:image => Tools.Space(UInt8, (96, 96, 1)), :action => Tools.Space(Int32, low=0, high=18));

agent = WorldModelAgent(config, spaces);
ps, st = Lux.setup(rng, agent);

# Observation (example: image)
obs = (; 
    image = rand(rng, UInt8, spaces[:image].size..., T, B), 
    is_first = rand(rng, Bool, T, B),
    is_last = rand(rng, Bool, T, B),
    is_terminal = rand(rng, Bool, T, B),
    reward = rand(rng, Float32, T, B),
);
action = [Int16(Tools.sample(spaces[:action])) for _ in 1:T, _ in 1:B];
tokens, _ = agent.encoder(obs, ps.encoder, st.encoder);


function agent_loss(agent::WorldModelAgent, obs, action, ps, st)
    
    reset = obs[:is_first];
    batch_length, batch_size = size(reset);

    # --- Encoder ---
    tokens, st_enc = agent.encoder(obs, ps.encoder, st.encoder);

    # --- RSSM Loss (provides KL losses and features) ---
    # los.dyn and los.rep should have shape (T, B) based on rssm.loss implementation
    los, feat, st_rssm = loss(agent.rssm, tokens, action, reset, ps.rssm, st.rssm);

    # --- Decoder ---
    # recons should have shape (W, H, C, T, B)
    recons, st_dec = agent.decoder(feat, ps.decoder, st.decoder);

    # --- Reconstruction Loss ---
    target_image = cast.(obs.image) ./ cast(255); # Target shape (W, H, C, T, B), range [0, 1]
    # recon_loss should have shape (T, B)
    recon_loss = calculate_reconstruction_loss_sum(recons, target_image) # Use the sum version

    # --- Calculate Scalar Means ---
    mean_dyn = mean(los.dyn)
    mean_rep = mean(los.rep)
    mean_recon = mean(recon_loss)

    # --- Get Loss Scales (Example - adapt to your config) ---
    # TODO: Implement proper loading of scales from agent.config
    scales = get(agent.config, "loss_scales", Dict("dyn" => 1.0, "rep" => 1.0, "image" => 1.0))
    scale_dyn = cast(scales["dyn"])
    scale_rep = cast(scales["rep"])
    scale_recon = cast(scales["image"]) # Assuming key 'image' for recon scale

    # --- Calculate Total Weighted Loss ---
    total_loss = scale_dyn * mean_dyn + scale_rep * mean_rep + scale_recon * mean_recon

    # --- Combine Final States ---
    st_new = (encoder=st_enc, rssm=st_rssm, decoder=st_dec) # NOTE: Using st.rssm as loss doesn't return state update yet

    # --- Prepare Auxiliary Output ---
    # Store both per-element and scalar losses, plus other info
    kl_losses_scalar = (; dyn = mean_dyn, rep = mean_rep)
    aux_losses = (; kl=los, kl_scalar=kl_losses_scalar, recon=recon_loss, recon_scalar=mean_recon)
    aux_full = (losses=aux_losses, repfeat=feat) # Include repfeat if needed later

    return total_loss, aux_full, st_new
end

# --- Minimal Training Loop --- 
println("\nStarting minimal training loop...")
# Optimizer Setup
learning_rate = 1e-4;
clip_threshold = 5.0f0; # Define the clipping threshold
opt = OptimiserChain(Optimisers.ClipGrad(clip_threshold), Optimisers.Adam(learning_rate));
opt_state = Optimisers.setup(opt, ps);

# Training Hyperparameters
num_steps = 100;
print_every = 10;

# Training Loop
mutable_ps = deepcopy(ps); # Use a mutable copy for updates
mutable_st = deepcopy(st); # Use a mutable copy for state updates

for step in 1:num_steps
    # Calculate loss and gradients
    # We need the loss value and the gradients for the update
    (loss_val, aux, current_st), grad = Zygote.withgradient(
        (p, s) -> agent_loss(agent, obs, action, p, s),
        mutable_ps, mutable_st
    );

    # Update parameters and optimizer state
    # Zygote returns nothing for non-differentiable args (like st)
    # We only need the gradient w.r.t parameters (grad[1])
    if grad[1] !== nothing
        global opt_state, mutable_ps # Indicate update of global vars
        # Apply gradient clipping
        # The Optimisers.Chain now handles clipping implicitly during update
        opt_state, mutable_ps = Optimisers.update(opt_state, mutable_ps, grad[1])
        # Update the state if it changed (optional, depends if agent_loss modifies st)
        global mutable_st = current_st
    else
        println("Warning: Gradient was nothing at step $step")
    end

    # Print loss periodically
    if step % print_every == 0 || step == 1
        println("Step: $step, Loss: $loss_val")
    end
end

println("Training loop finished.")