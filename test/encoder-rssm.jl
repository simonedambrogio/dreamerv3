include("../dreamerv3/agents.jl");

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


# loss_val, _ = loss_fn(agent.encoder, obs, ps.encoder, st.encoder)
# println("  - Loss value: ", loss_val)

# loss_val_zygote, grad_zygote = Zygote.withgradient(
#     p_ -> begin # p_ represents the parameters (ps)
#         loss_val, _ = loss_fn(agent.encoder, obs, p_, st.encoder) # Pass enc, obs, p_, st
#         loss_val
#     end,
#     ps.encoder # Differentiate with respect to parameters
# );



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


# Function to calculate the full world model loss
function agent_loss(agent::WorldModelAgent, obs, action, ps, st)
    # enc_carry, dyn_carry, dec_carry = carry;
    reset = obs[:is_first];
    batch_length, batch_size = size(reset);

    # --- Encoder ---
    tokens, st_enc = agent.encoder(obs, ps.encoder, st.encoder);

    # --- RSSM Loss (provides KL losses and features) ---
    # los.dyn and los.rep should have shape (T, B) based on rssm.loss implementation
    los, st_rssm = loss(agent.rssm, tokens, action, reset, ps.rssm, st.rssm);

    # --- Calculate Scalar Means ---
    mean_dyn = mean(los.dyn)
    mean_rep = mean(los.rep)

    # --- Get Loss Scales (Example - adapt to your config) ---
    # TODO: Implement proper loading of scales from agent.config
    scales = get(agent.config, "loss_scales", Dict("dyn" => 1.0, "rep" => 1.0))
    scale_dyn = cast(scales["dyn"])
    scale_rep = cast(scales["rep"])

    # --- Calculate Total Weighted Loss ---
    total_loss = scale_dyn * mean_dyn + scale_rep * mean_rep

    # --- Combine Final States ---
    st_new = (encoder=st_enc, rssm=st.rssm) # NOTE: Using st.rssm as loss doesn't return state update yet

    return total_loss, st_new
end

loss_val, _ = agent_loss(agent, obs, action, ps, st);
println("Loss Value: ", loss_val)

g = Zygote.withgradient(
    p_ -> agent_loss(agent, obs, action, p_, st),
    ps # Differentiate with respect to parameters
);
println("Zygote Gradient ended")

