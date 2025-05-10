# test/train-wm.jl
include("../dreamerv3/WorldModel.jl");
using LuxCUDA

println("--- Testing Encoder GPU ---")
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
rng = MersenneTwister(config["run"]["seed"]);
spaces = Dict(
    :image => Tools.Space(UInt8, (64, 64, 1)),
    :action => Tools.Space(Int32; low=1, high=2)
);

enc = WorldModelAgent(config, spaces).encoder;
ps, st = Lux.setup(rng, enc);
ps, st = _device(ps), _device(st);

# Observation (example: image)
T, B = config["run"]["batch_length"], config["run"]["batch_size"];
obs_image = rand(UInt8, (spaces[:image].size..., T, B));
obs = (; image = _device(obs_image));
tokens, _ = enc(obs, ps, st);

println("--- Testing Encoder GPU ---")
println("Tokens shape: $(size(tokens))")
println("Tokens type: $(eltype(tokens))")
println("Tokens device: $(device(tokens))")

# Minimal test to check if the gradient is computed ------
println("--- Testing Zygote Gradient ---")
using Zygote
# Define a simple loss function
function loss_fn(p, s, current_obs)
    _tokens, _st = enc(current_obs, p, s)
    return sum(abs2, _tokens), _st # Ensure the loss is a scalar
end

# Compute gradients
(loss_val, _), back = Zygote.pullback(p -> loss_fn(p, st, obs), ps)
grads = back((one(loss_val), nothing))[1]

