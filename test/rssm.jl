using Lux, NNlib, Random, Tools, BFloat16s, YAML, Statistics, Test
include("../embodied/lux/rms.jl");
include("../embodied/lux/nets.jl");
include("../embodied/lux/BlockLinear.jl");
include("../embodied/lux/ReArrange.jl");
include("../embodied/lux/UpSample.jl");
include("../dreamerv3/encoder.jl");
include("../dreamerv3/rssm.jl");
config = YAML.load_file("dreamerv3/configs.yaml");

const ANSI_GREEN = "\e[32m"
const ANSI_BLUE = "\e[34m"
const ANSI_ORANGE = "\e[33m"
const ANSI_VIOLET = "\e[35m"
const ANSI_RESET = "\e[0m"

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
mults=(2, 3, 4, 4);
kernel=5;

rng = Random.default_rng();

println(ANSI_GREEN, "\n----- Running Encoder Forward Pass -----", ANSI_RESET)
enc = Encoder(; obs, act, mults, depth, kernel);
ps, st = Lux.setup(rng, enc);
dummy_image = rand(rng, UInt8, obs.size..., T, B);
dummy_obs = (; image = dummy_image); # Use NamedTuple matching encoder input
println("  - Input shape: ", size(dummy_obs[:image]), "     -> (W, H, C, T, B)")
tokens, st_new = enc(dummy_obs, ps, st);
println("  - Output shape: ", size(tokens),         "          -> (token_dim, T, B)")

println(ANSI_GREEN, "\n----- Running RSSM Initial State -----", ANSI_RESET)
act_space = Tools.Space(Int32, low=0, high=18);
rssm = RSSM(; act_space, deter_dim, stoch_dim, classes_dim, hidden_dim, blocks);
ps, st = Lux.setup(rng, rssm);
carry = initial_carry(rssm, B);
println("  - Deterministic part: ", size(carry.deter), "          -> (deter_dim, B)")
println("  - Stochastic part: ",    size(carry.stoch), "          -> (stoch_dim, classes_dim, B)")

println(ANSI_GREEN, "\n----- Running RSSM _observe -----", ANSI_RESET)
action = [Int16(sample(act_space)) for _ in 1:T, _ in 1:B];
reset = rand(rng, Bool, T, B);
t = 1

# 
reset = reset[t,:]
action = action[t,:]
tokens = tokens[:,t,:]
# _observe(rssm, carry, tokens, action, reset)


println(ANSI_BLUE, "\t Applying reset mask to state", ANSI_RESET)
# Create the inverted mask, ready for broadcasting
keep_mask = .!reset # Shape: (Batch,)
# Apply mask to deter state
deter_mask = reshape(keep_mask, 1, :) # Shape: (1, Batch)
deter = carry.deter .* deter_mask
# Apply mask to stoch state
stoch_mask = reshape(keep_mask, 1, 1, :) # Shape: (1, 1, Batch)
stoch = carry.stoch .* stoch_mask

println(ANSI_BLUE, "\t Processing Action", ANSI_RESET)
# Assuming action is discrete and needs one-hot encoding
# TODO: Handle continuous actions if necessary based on act_space
@assert !isnothing(rssm.act_space) "RSSM requires act_space to process actions"
num_actions = rssm.act_space.high # Assumes Space defines range [low, high)
# Perform one-hot encoding. Note: NNlib.onehotbatch expects indices starting from 1.
action_onehot = OneHotArrays.onehotbatch(action, 0:num_actions) # Shape: (num_actions, Batch)
action_onehot_casted = cast(action_onehot) # Cast to COMPUTE_TYPE
# Apply reset mask to the processed action
action = action_onehot_casted .* deter_mask # Broadcast (1, Batch) mask

println(ANSI_BLUE, "\t Compute Core RSSM", ANSI_RESET)
println(ANSI_VIOLET, "\t  - Reshape stoch", ANSI_RESET)
stoch = reshape(stoch, (:, size(stoch,3)))  # stoch x classes x B -> (stoch * classes, B)
                                            # e.g. 2, 4, 8 -> (8, 8) 


# carry, tokens, action, reset = _observe(rssm, carry, tokens[:,t,:], action, reset[t,:]);
