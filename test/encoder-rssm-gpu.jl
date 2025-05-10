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
T, B = config["run"]["batch_length"], config["run"]["batch_size"];
obs_image = rand(UInt8, (spaces[:image].size..., T, B));
obs = (; image = _device(obs_image));
action = [Int16(Tools.sample(spaces[:action])) for _ in 1:T, _ in 1:B] |> _device;
reset = rand(Bool, (T, B)) |> _device;

println("--- Instantiating World Model ---")
wm = WorldModelAgent(config, spaces);

println("--- Encoder ---")
enc = wm.encoder;
ps_enc, st_enc = Lux.setup(rng, enc);
ps_enc, st_enc = _device(ps_enc), _device(st_enc);
tokens, _ = enc(obs, ps_enc, st_enc);

println("--- RSSM ---")
rssm = wm.rssm;
ps_rssm, st_rssm = Lux.setup(rng, rssm);
ps_rssm, st_rssm = _device(ps_rssm), _device(st_rssm);

observe(rssm, tokens, action, reset, ps_rssm, st_rssm)

# test zygote ------------------------------------------------------------
function rssm_gradient_test_loss_fn(p_rssm_local, st_rssm_local, tokens_local, action_local, reset_local, rssm_model)
    # Call the main observe function
    # Assumed return: (sequence_deter, sequence_stoch, sequence_logit, final_recurrent_state)
    (final_entry, final_feat), final_st = observe(rssm_model, tokens_local, action_local, reset_local, p_rssm_local, st_rssm_local)
    
    # A simple scalar loss: sum of squares of all elements in the output sequences
    # This ensures all outputs are part of the computation graph
    loss = sum(abs2, final_entry.deter) + sum(abs2, final_entry.stoch) + sum(abs2, final_feat.logit)
    return loss, final_st # Zygote.pullback expects the primary value and the state
end
rssm_gradient_test_loss_fn(ps_rssm, st_rssm, tokens, action, reset, rssm)


println("--- Zygote Gradient ---")
(loss_val_rssm, state_after_loss_rssm), back_rssm = Zygote.pullback(
    p -> rssm_gradient_test_loss_fn(p, st_rssm, tokens, action, reset, rssm), # rssm is captured
    ps_rssm
)

println("running back function")
grads_rssm = back_rssm((one(loss_val_rssm), nothing))[1]
println("done")

# # sample_ste ------------------------------------------------------------
# d = OneHotDist(rand(rng, 10, 10, 10) |> _device, 0.0)
# sample_ste(rng, d)


# # observe ------------------------------------------------------------
# T = size(tokens, 2) # Get sequence length
# B = size(tokens, 3) # Get batch size
# S, C = rssm.cell.rssm.stoch_dim, rssm.cell.rssm.classes_dim # Get stoch and classes dims
# D = rssm.cell.rssm.deter_dim # Get deter dim
# el_type = eltype(tokens)
# t=1
# tokens_t = view(tokens, :, t, :) # Shape: (token_dim, B)
# action_t = view(action, t, :)   # Shape: (B,)
# reset_t = view(reset, t, :)     # Shape: (B,)
# x = ObserveInput(tokens_t, action_t, reset_t)
# out, st = rssm(x, ps_rssm, st_rssm);

# seq_deter_out = Buffer(zeros(el_type, D, T, B))
# seq_stoch_out = Buffer(zeros(el_type, S, C, T, B))
# seq_logit_out = Buffer(zeros(el_type, S, C, T, B))

# entry_t, feat_t = out
# list_deter_t = Vector{Any}()
# push!(list_deter_t, entry_t.deter) # entry_t.deter is (D, B)
# seq_deter_out = cat([reshape(d, size(d, 1), 1, size(d, 2)) for d in list_deter_t]...; dims=2)

# # # ObserveCell ------------------------------------------------------------
# # batch_size = size(x.tokens, 2)
# # carry = initial_carry(rssm.cell.rssm, batch_size) |> _device;
# # carry_next, entry_t, feat_t = _observe(rssm.cell.rssm, carry, x.tokens, x.action, x.reset, ps_rssm, st_rssm.cell)

# # # _observe ------------------------------------------------------------
# # keep_mask = .!x.reset # Shape: (Batch,)
# # deter_mask = reshape(keep_mask, 1, :) # Shape: (1, Batch)
# # deter = carry.deter .* deter_mask
# # stoch_mask = reshape(keep_mask, 1, 1, :) # Shape: (1, 1, Batch)
# # stoch = carry.stoch .* stoch_mask
# # num_actions = rssm.cell.rssm.act_space.high # Assumes Space defines range [low, high)
# # action_onehot = OneHotArrays.onehotbatch(x.action, 1:num_actions) # Shape: (num_actions, Batch)
# # action_onehot_casted = cast(action_onehot) # Cast to COMPUTE_TYPE
# # action = action_onehot_casted .* deter_mask # Broadcast (1, Batch) mask
# # deter_current, _ = _core(rssm.cell.rssm, deter, stoch, action, ps_rssm, st_rssm.cell)

# # #  _core ------------------------------------------------------------
# # B = size(deter, 2) # Get Batch size from deter
# # stoch_flat = reshape(stoch, :, B)  # Shape: (stoch_dim * classes_dim, Batch)
# # g = rssm.cell.rssm.blocks
# # _deter_ctx, st_deter_new = rssm.cell.rssm.core.layer_deter(deter, ps_rssm.core.layer_deter, st_rssm.cell.core.layer_deter)
# # _stoch_ctx, st_stoch_new = rssm.cell.rssm.core.layer_stoch(stoch_flat, ps_rssm.core.layer_stoch, st_rssm.cell.core.layer_stoch)
# # _action_ctx, st_action_new = rssm.cell.rssm.core.layer_action(action, ps_rssm.core.layer_action, st_rssm.cell.core.layer_action)
# # context = vcat(_deter_ctx, _stoch_ctx, _action_ctx) |>
# # ctx -> reshape(ctx, size(ctx, 1), 1, B) |>
# # ctx_reshaped -> repeat(ctx_reshaped, 1, g, 1) |>
# # ctx_repeated -> group2flat(
# #     vcat(flat2group(deter, g), ctx_repeated)
# # );
# # raw_gates, st_gru_new = rssm.cell.rssm.core.gru_layers(context, ps_rssm.core.gru_layers, st_rssm.cell.core.gru_layers)
# # grouped_gates = flat2group(raw_gates, g) # Shape: (FeaturesPerBlock = 3*deter/g, Blocks=g, Batch=B)
# # gates_split = split(grouped_gates, 3, 1) # Tuple of 3 tensors, each: (deter/g, g, B)
# # reset_flat, cand_flat, update_flat = [group2flat(gate) for gate in gates_split] # Each: (deter_dim, B)
# # reset = sigmoid.(reset_flat)
# # cand = tanh.(reset .* cand_flat) # Apply reset gate to candidate pre-activation
# # update = sigmoid.(update_flat .- cast(1))
# # # Combine using update gate (GRU formula)
# # # IMPORTANT: Uses the *original* deter passed into _core
# # deter_next = update .* cand .+ (cast(1) .- update) .* deter 







# acc_seq_deter = ()
# acc_seq_stoch = ()
# acc_seq_logit = () # Or whatever features you extract

# # The state for the Lux.AbstractRecurrentCell for the loop
# st_loop = st_rssm

# # --- Loop over time steps ---
# for t in 1:T
#     # Get inputs for the current time step
#     tokens_t = view(tokens, :, t, :) # Shape: (token_dim, B)
#     action_t = view(action, t, :)   # Shape: (B,)
#     reset_t = view(reset, t, :)     # Shape: (B,)
#     current_step_input = ObserveInput(tokens_t, action_t, reset_t)

#     # Call the recurrent cell for one step.
#     # This uses ps and the current st_loop, and returns (output_for_step, new_recurrent_cell_state).
#     (entry_t, feat_t), st_loop = rssm(current_step_input, ps_rssm, st_loop)
    
#     # Accumulate outputs by creating new tuples (Zygote-friendly)
#     # entry_t.deter, entry_t.stoch, feat_t.logit are expected to be on the GPU if inputs are.
#     acc_seq_deter = (acc_seq_deter..., entry_t.deter) 
#     acc_seq_stoch = (acc_seq_stoch..., entry_t.stoch) 
#     acc_seq_logit = (acc_seq_logit..., feat_t.logit) # Assuming feat_t has a .logit field
# end
