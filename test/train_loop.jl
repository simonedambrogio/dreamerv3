# test/train_loop.jl

using Lux, Optimisers, Zygote, Random, Tools, BFloat16s, YAML, Statistics, Test, OneHotArrays

# --- Include Embodied Files Once ---
include("../embodied/lux/nets.jl");
include("../embodied/lux/rms.jl");
include("../embodied/lux/BlockLinear.jl");
include("../embodied/lux/ReArrange.jl");
# Add any other direct embodied includes needed by dreamer files

# --- Include necessary files ---
# Assuming nets.jl, rms.jl etc. are included via these files
include("../dreamerv3/encoder.jl");
include("../dreamerv3/rssm.jl");
include("../dreamerv3/decoder.jl");
include("../dreamerv3/agents.jl"); 

# --- Config Loading and Setup ---
println("--- Loading Config ---")
fullconfig = YAML.load_file("dreamerv3/configs.yaml");
batch_length, batch_size = fullconfig["debug"]["batch_length"], fullconfig["debug"]["batch_size"];
config = Dict(component => make_config(component, "debug") for component in ["enc", "dec", "dyn"]);
debug_config = fullconfig["debug"]

# --- Parameters ---
learning_rate = 1e-4 # Example learning rate
num_steps = 10     # Number of optimization steps

# --- Spaces ---
println("--- Setting up Spaces ---")
obs_space = Dict(:image => Tools.Space(UInt8, (96, 96, 1)))
act_space = Dict(:action => Tools.Space(Int32; low=0, high=18))
spaces = merge(obs_space, act_space)

# --- RNG ---
rng = MersenneTwister(1234); # Use a fixed seed for reproducibility

# --- Instantiate Agent and Setup ---
println("--- Instantiating Agent ---")
agent = WorldModelAgent(config, spaces);
ps, st = Lux.setup(rng, agent);
println("Agent instantiated successfully.")

# --- Optimizer Setup ---
println("--- Setting up Optimizer ---")
opt = Optimisers.Adam(learning_rate)
opt_state = Optimisers.setup(opt, ps);
println("Optimizer set up.")

# --- Prepare Dummy Inputs (Consistent across loop) ---
println("--- Preparing Dummy Inputs ---")
# Initial carry needs state structure matching agent layers
initial_dyn_carry = initial_carry(agent.rssm, batch_size);
current_carry = initial_dyn_carry # Start with initial RSSM carry

obs = (;
    image = rand(rng, UInt8, spaces[:image].size..., batch_length, batch_size),
    is_first = zeros(Bool, batch_length, batch_size), # Start with no resets for simplicity
    # Add other potential obs keys if needed by encoder
);
obs.is_first[1, :] .= true; # Set first step for all batches

actions = [Int16(Tools.sample(spaces[:action])) for _ in 1:batch_length, _ in 1:batch_size];
println("Inputs prepared.")

# --- Training Loop ---
println("\n--- Starting Training Loop (", num_steps, " steps) ---")

# Need to define the loss function for Zygote correctly
# It should only take parameters `p` as input for gradient calculation
# State `s`, carry `c`, obs `o`, and actions `a` are treated as constants for the grad
function loss_for_grad(p, s, c, o, a)
    loss_val, aux = agent_loss(agent, c, o, a, p, s)
    return loss_val, aux # Return aux to get updated state/carry
end

loss_for_grad(ps, st, current_carry, obs, actions)

for i in 1:num_steps
    # Calculate loss and gradients
    # Pass current state `st` and current carry `current_carry`
    (loss_val, aux_full), grads = Zygote.withgradient(
        p -> loss_for_grad(p, st, current_carry, obs, actions),
        ps
    )

    # Update optimizer state and parameters
    opt_state, ps = Optimisers.update(opt_state, ps, grads[1]); # grads is a tuple

    # Update the state and carry for the next iteration using aux_full
    st = aux_full.st
    current_carry = aux_full.final_carry

    # Print loss
    println("Step: ", i, " / ", num_steps, ", Loss: ", loss_val)

    # Basic check for NaN/Inf
    if isnan(loss_val) || isinf(loss_val)
        println("Loss is NaN or Inf. Stopping training.")
        break
    end
end

println("--- Training Loop Finished ---") 