# test/train-wm.jl
include("../dreamerv3/WorldModel.jl");
include("../embodied/envs/custom/generalization.jl");
include("../embodied/core/replay.jl");
using Optimisers, Random, Statistics, YAML, Zygote
using Dates, JLD2

println("--- World Model Training Test ---")
if CUDA.functional()
    println("CUDA is functional")
    _device = gpu_device()
else
    println("CUDA is not functional")
    _device = cpu_device()
end

# --- Configuration ---
config_filepath = joinpath(@__DIR__, "..", "dreamerv3", "configs.yaml");
fullconfig = YAML.load_file(config_filepath);
config = make_config(fullconfig, "debug");


# --- Helper Functions ---
function experience_replay(env, replay, batch_size, num_steps, spaces)
    println("\n--- Starting Warmup Phase ($num_steps steps) ---")
    obs = env.fixation_cross # Use the passed initial observation
    for warmup_step in 1:num_steps
        action = rand(replay.rng, spaces[:action].low:spaces[:action].high) # Random action
        next_obs, reward, done = step!(env, action);

        step_data = Dict(
            :image => obs,       # Use :image key
            :action => Int32(action),
            :reward => Float32(reward),
            :is_first => (warmup_step == 1), # Technically only true for the very first step overall
            :is_last => done,
            :is_terminal => done
        )
        add!(replay, step_data, 0) # Add to worker 0 stream

        obs = next_obs # Update local obs
        # if done
        #     obs = reset!(env) # Reset env and update local obs
        # end

        if warmup_step % 200 == 0 || warmup_step == num_steps
            println("Warmup Step: $warmup_step / $num_steps, Replay items: $(length(replay))")
        end
    end

    if length(replay) < batch_size
        error("Replay buffer has only $(length(replay)) items after warmup, less than batch size $batch_size. Increase warmup steps.")
    end
    println("Warmup complete. Replay buffer size: $(length(replay)) items.")
    return obs # Return the final observation state
end;

function run_training_loop(
        agent, 
        replay, 
        num_steps, 
        batch_size, 
        log_every,
        logdir,
        save_every,
        initial_ps, 
        initial_st, 
        initial_opt_st
    )

    println("\n--- Starting Training Phase ($num_steps steps) ---")
    # Make ps and st mutable copies for updates within the function
    mutable_ps = deepcopy(initial_ps);
    mutable_st = deepcopy(initial_st);
    opt_st = initial_opt_st; # Use the passed optimizer state

    # Accumulators are local to this function
    total_loss_acc = 0.0f0;
    dyn_loss_acc = 0.0f0;
    rep_loss_acc = 0.0f0;
    recon_loss_acc = 0.0f0;

    for train_step in 1:num_steps
        # Sample batch
        batch = sample(replay, batch_size);

        (loss_val, aux, current_st), grad = Zygote.withgradient(
            (p, s) -> loss(agent, batch, p, s),
            mutable_ps, mutable_st
        );

        # Accumulate losses for logging
        total_loss_acc += loss_val
        dyn_loss_acc += aux.losses.kl_scalar.dyn
        rep_loss_acc += aux.losses.kl_scalar.rep
        recon_loss_acc += aux.losses.recon_scalar

        # Update parameters and optimizer state (no global needed)
        if grad !== nothing && grad[1] !== nothing
            # Update happens on the function-local copies
            opt_st, mutable_ps = Optimisers.update!(opt_st, mutable_ps, grad[1])
            mutable_st = current_st # Update the function-local state
        else
            println("Warning: Gradient was nothing at train step $train_step")
        end

        # Log every log_every steps
        if train_step % log_every == 0 || train_step == 1
            count = (train_step == 1) ? 1 : (train_step % log_every == 0 ? log_every : train_step % log_every)
            avg_total_loss = total_loss_acc / count
            avg_dyn_loss = dyn_loss_acc / count
            avg_rep_loss = rep_loss_acc / count
            avg_recon_loss = recon_loss_acc / count

            println("Train Step: $train_step, Avg Loss: $(round(avg_total_loss, digits=4)) [D:$(round(avg_dyn_loss, digits=4)), R:$(round(avg_rep_loss, digits=4)), Img:$(round(avg_recon_loss, digits=4))]")
            # Reset local accumulators
            total_loss_acc = 0.0f0
            dyn_loss_acc = 0.0f0
            rep_loss_acc = 0.0f0
            recon_loss_acc = 0.0f0

            # log_reconstruction(replay, agent, batch_size, mutable_ps, mutable_st, lg)
        end

        # Save every save_every steps
        if train_step % save_every == 0 || train_step == 1
            # Create the "checkpoints" directory if it doesn't exist
            checkpoint_dir = joinpath(logdir, "checkpoints");
            !isdir(checkpoint_dir) && mkdir(checkpoint_dir);
            @save joinpath(checkpoint_dir, "ckpt_$(train_step).jld2") mutable_ps mutable_st
        end
    end

    println("--- Training loop finished. ---")
    return mutable_ps, mutable_st, opt_st # Return the final states
end;

# --- Main Script Execution ---
# Initialization
println("Initializing Environment, Replay Buffer, and Agent...")
rng = MersenneTwister(config["run"]["seed"]);
env = GeneralizationEnv(; num_trials = 1_000, seed=config["run"]["seed"], verbose=false);
replay = Replay(; 
    length=config["run"]["batch_size"], 
    capacity=config["replay"]["size"],
    chunksize=config["replay"]["chunksize"], 
    seed=config["run"]["seed"]
);

spaces = Dict(
    :image => Tools.Space(UInt8, (64, 64, 1)),
    :action => Tools.Space(Int32; low=1, high=2) # Assuming actions 1 and 2
);

agent = WorldModelAgent(config, spaces);
initial_ps, initial_st = Lux.setup(rng, agent);
initial_ps = initial_ps |> _device
initial_st = initial_st |> _device
opt = Adam( 
    config["agent"]["opt"]["lr"],
    (config["agent"]["opt"]["beta1"], config["agent"]["opt"]["beta2"]),
    config["agent"]["opt"]["eps"]
);
initial_opt_st = Optimisers.setup(opt, initial_ps);

println("Initialization complete.")

# Run Warmup
initial_obs = reset!(env);
final_obs_after_warmup = experience_replay(env, replay, config["run"]["batch_size"], config["agent"]["opt"]["warmup"], spaces);

# Run Training Loop
final_ps, final_st, final_opt_st = run_training_loop(
    agent, 
    replay, 
    config["run"]["steps"], 
    config["run"]["batch_size"], 
    config["run"]["log_every"], 
    config["logdir"],
    config["run"]["save_every"],
    initial_ps, 
    initial_st, 
    initial_opt_st
);
# You can now use final_ps, final_st if needed for further steps
println("--- Script finished training ---")