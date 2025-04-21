include("../dreamerv3/encoder.jl");
include("../dreamerv3/rssm.jl");
include("../dreamerv3/decoder.jl");
include("../dreamerv3/agents.jl");

fullconfig = YAML.load_file("dreamerv3/configs.yaml");

function make_config(component::String, config_type::String)
    # Input validation
    if !(component in ["enc", "dec", "dyn"])
        error("Invalid component: $component. Must be one of 'enc', 'dec', 'dyn'.")
    end
    if !(config_type in ["default", "debug"])
        error("Invalid config_type: $config_type. Must be 'default' or 'debug'.")
    end

    # Get default component type (e.g., "simple" for "enc", "rssm" for "dyn")
    default_component_base = config["defaults"]["agent"][component]
    default_component_type = default_component_base["typ"]
    # Get default component config
    default_config = default_component_base[default_component_type]

    if config_type == "default"
        # Return a copy to prevent accidental modification of the global config
        return deepcopy(default_config)
    else # config_type == "debug"
        # Get debug overrides
        debug_agent_config = config["debug"]["agent"]

        # Create a mutable copy of the default config
        merged_config = deepcopy(default_config)

        # Apply debug overrides by matching regex keys
        for (debug_key, debug_value) in debug_agent_config
            # Match regex like ".*\\\\.key" -> "key"
            m = match(r"\.\*\\\.(.*)", debug_key) # Updated regex
            if m !== nothing
                actual_key = m.captures[1]
                # Check if the extracted key exists in the component's default config
                if haskey(merged_config, actual_key)
                    merged_config[actual_key] = debug_value
                end
            end
        end
        return merged_config
    end
end;

function mse(recons, target; dims=(1, 2, 3), agg=sum)
    # 1. Calculate element-wise squared error
    elementwise_sq_error = (recons .- target).^2 # Shape: (W, H, C, T, B)

    # 2. Sum over spatial and channel dimensions (dims 1, 2, 3)
    recon_loss_summed_spatial = agg(elementwise_sq_error; dims=dims) # Shape: (1, 1, 1, T, B)

    # 3. Remove singleton dimensions to get shape (T, B)
    # This is the equivalent of Python's losses['image']
    recon_loss_per_TB = dropdims(recon_loss_summed_spatial; dims=dims) # Shape: (T, B)
    return recon_loss_per_TB
end

rng = MersenneTwister(1234);
batch_length, batch_size = fullconfig["debug"]["batch_length"], fullconfig["debug"]["batch_size"];
config = Dict(component => make_config(component, "debug") for component in ["enc", "dec", "dyn"]);
spaces = Dict(:image => Tools.Space(UInt8, (96, 96, 1)), :action => Tools.Space(Int32, low=0, high=18));

agent = WorldModelAgent(config, spaces);
ps, st = Lux.setup(rng, agent);

encoder = Encoder(; obs=spaces[:image], act=eval(Meta.parse(config["enc"]["act"])), mults=config["enc"]["mults"], depth=config["enc"]["depth"], kernel=config["enc"]["kernel"]);
token_dim = calculate_encoder_output_dim(config["enc"], spaces[:image])

# Instantiate RSSM
rssm = RSSM(;
    deter_dim=config["dyn"]["deter"],
    hidden_dim=config["dyn"]["hidden"],
    stoch_dim=config["dyn"]["stoch"],
    classes_dim=config["dyn"]["classes"],
    act=eval(Meta.parse(config["dyn"]["act"])),
    token_dim=token_dim,
    imglayers=config["dyn"]["imglayers"],
    obslayers=config["dyn"]["obslayers"],
    dynlayers=config["dyn"]["dynlayers"],
    blocks=config["dyn"]["blocks"],
    act_space=spaces[:action], # Assuming act_space has an :action key
);

# Instantiate Decoder
decoder = Decoder(;
    obs=spaces[:image],
    deter_dim=config["dyn"]["deter"], # Use deter_dim from dyn_config
    units=config["dec"]["units"],
    stoch_dim=config["dyn"]["stoch"], # Use stoch_dim from dyn_config
    classes_dim=config["dyn"]["classes"], # Use classes_dim from dyn_config
    act=eval(Meta.parse(config["dec"]["act"])),
    mults=config["dec"]["mults"],
    depth=config["dec"]["depth"],
    kernel=config["dec"]["kernel"],
    bspace=config["dyn"]["blocks"] # Assuming bspace uses RSSM blocks
);


# Loss function ---------------------------------------------------------------

# Prepare inputs
carry = [NamedTuple(), initial_carry(rssm, batch_size), NamedTuple()];
obs = (; 
    image = rand(rng, UInt8, spaces[:image].size..., batch_length, batch_size), 
    is_first = rand(rng, Bool, batch_length, batch_size),
    is_last = rand(rng, Bool, batch_length, batch_size),
    is_terminal = rand(rng, Bool, batch_length, batch_size),
    reward = rand(rng, Float32, batch_length, batch_size),
);
action = [Int16(Tools.sample(spaces[:action])) for _ in 1:batch_length, _ in 1:batch_size];
agent = WorldModelAgent(config, spaces);
ps, st = Lux.setup(rng, agent);

# Start loss function
enc_carry, dyn_carry, dec_carry = carry;
reset = obs[:is_first];
batch_length, batch_size = size(reset);
tokens, st_enc = agent.encoder(obs, ps.encoder, st.encoder);

tokens, _ = agent.encoder(obs, ps.encoder, st.encoder);
dyn_carry, dyn_entries, los, repfeat, mets = loss(agent.rssm, dyn_carry, tokens, action, reset, ps.rssm, st.rssm);
# update loss and metrics ...
recons, _ = agent.decoder(repfeat, ps.decoder, st.decoder);

space, value = spaces[:image], obs[:image];
target = Float32.(value) ./ 255f0;

recon_loss = mse(recons, target; dims=(1, 2, 3), agg=sum)

mean(los.dyn)
mean(los.rep)
mean(recon_loss)
