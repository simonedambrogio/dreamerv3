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
num_actions = rssm.act_space.high + 1 # Assumes Space defines range [low, high)
# Perform one-hot encoding. Note: NNlib.onehotbatch expects indices starting from 1.
action_onehot = OneHotArrays.onehotbatch(action, 0:(num_actions-1)) # Shape: (num_actions, Batch)
action_onehot_casted = cast(action_onehot) # Cast to COMPUTE_TYPE
# Apply reset mask to the processed action
action = action_onehot_casted .* deter_mask # Broadcast (1, Batch) mask

println(ANSI_BLUE, "\t Compute Core RSSM", ANSI_RESET)

g = rssm.blocks # Get number of blocks from rssm instance


text = "\t    - Reshape stoch"
print(ANSI_VIOLET, text, ANSI_RESET)
size_input = size(stoch)
stoch = reshape(stoch, (:, size(stoch,3)))  # stoch x classes x B -> (stoch * classes, B) # e.g. 2, 4, 8 -> (8, 8) 
size_output = size(stoch)
padded_size_str = "$(" "^(35 - length(text) )) $size_input -> $size_output"
print(ANSI_VIOLET, padded_size_str, ANSI_RESET)
println(ANSI_VIOLET, " "^(70 - (length(text) + length(padded_size_str))), "stoch_dim, classes_dim, B -> stoch_dim*classes_dim, B", ANSI_RESET, "\n")

text = "\t    - Norm(Dense(deter), act)"
print(ANSI_VIOLET, text, ANSI_RESET)
l = Chain(
    Dense(stoch_dim * classes_dim, hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
    RMSNorm((hidden_dim,), act; dims = (1,), init_scale=cast_ones)
);
ps, st = Lux.setup(rng, l);
size_input = size(deter)
x0, st = l(deter, ps, st);
size_output = size(x0)
padded_size_str = "$(" "^(35 - length(text) )) $size_input -> $size_output"
print(ANSI_VIOLET, padded_size_str, ANSI_RESET)
println(ANSI_VIOLET, " "^(70 - (length(text) + length(padded_size_str))), "deter_dim, B -> hidden_dim, B", ANSI_RESET)


text = "\t    - Norm(Dense(stoch), act)"
print(ANSI_VIOLET, text, ANSI_RESET)
l = Chain(
    Dense(stoch_dim * classes_dim, hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
    RMSNorm((hidden_dim,), act; dims = (1,), init_scale=cast_ones)
);
ps, st = Lux.setup(rng, l);
size_input = size(stoch)
x1, st = l(stoch, ps, st);
size_output = size(x1)
padded_size_str = "$(" "^(35 - length(text) )) $size_input -> $size_output"
print(ANSI_VIOLET, padded_size_str, ANSI_RESET)
println(ANSI_VIOLET, " "^(70 - (length(text) + length(padded_size_str))), "stoch_dim*classes_dim, B -> hidden_dim, B", ANSI_RESET)


text = "\t    - Norm(Dense(action), act)"
print(ANSI_VIOLET, text, ANSI_RESET)
l = Chain(
    Dense(num_actions, hidden_dim; init_weight=cast_glorot_uniform, init_bias=cast_zeros),
    RMSNorm((hidden_dim,), act; dims = (1,), init_scale=cast_ones)
);
ps, st = Lux.setup(rng, l);
size_input = size(action)
x2, st = l(action, ps, st);
size_output = size(x2)
padded_size_str = "$(" "^(35 - length(text) )) $size_input -> $size_output"
print(ANSI_VIOLET, padded_size_str, ANSI_RESET)
println(ANSI_VIOLET, " "^(70 - (length(text) + length(padded_size_str))), "num_actions, B -> hidden_dim, B", ANSI_RESET)

println(ANSI_RESET)

text = "\t    - Define context concatenating the output features from the three branches (deter, stoch, action)"
print(ANSI_VIOLET, text, ANSI_RESET)
xc = vcat(x0, x1, x2) 
size_output = size(xc)
padded_size_str = "$(" "^(110 - length(text) )) $(size_output)"
println(ANSI_VIOLET, padded_size_str, ANSI_RESET)

text = "\t    - Reshape context"
print(ANSI_VIOLET, text, ANSI_RESET)
feat_concat, B_in = size(xc)
reshaped_xc = reshape(xc, feat_concat, 1, B_in)
size_output = size(reshaped_xc)
padded_size_str = "$(" "^(110 - length(text) )) $(size_output)"
println(ANSI_VIOLET, padded_size_str, ANSI_RESET)

text = "\t    - Repeat context"
print(ANSI_VIOLET, text, ANSI_RESET)
repeated_xc = repeat(reshaped_xc, outer=(1, g, 1))
size_output = size(repeated_xc)
padded_size_str = "$(" "^(110 - length(text) )) $(size_output)"
println(ANSI_VIOLET, padded_size_str, ANSI_RESET)

text = "\t    - Group deter (flat2group)"
print(ANSI_VIOLET, text, ANSI_RESET)
flat2group(x, g) = reshape(x, :, g, size(x,2))
grouped_deter = flat2group(deter, g)
size_output = size(grouped_deter)
padded_size_str = "$(" "^(110 - length(text) )) $(size_output)"
println(ANSI_VIOLET, padded_size_str, ANSI_RESET)

text = "\t    - Concatenate context and deter and Flatten (group2flat)"
print(ANSI_VIOLET, text, ANSI_RESET)
group2flat(x) = reshape(x, :, size(x,3))
x = group2flat(vcat(grouped_deter, repeated_xc))
size_output = size(x)
padded_size_str = "$(" "^(110 - length(text) )) $(size_output)"
println(ANSI_VIOLET, padded_size_str, ANSI_RESET)

println(ANSI_RESET)


# The variable 'repeated_xc' now holds the result equivalent to the Python line
# x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
# but with Julia's (Features, Blocks, Batch) dimension order.

# Next step will be the flat2group/concatenate/group2flat part
