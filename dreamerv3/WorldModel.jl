using Lux, Statistics

# Loading Agent's components
include("encoder.jl");
include("rssm.jl");
include("decoder.jl");

# Define Agent's struct for the World Model part
struct WorldModelAgent{E, R, D} <: Lux.AbstractLuxContainerLayer{(:encoder, :rssm, :decoder)}
    encoder::E
    rssm::R
    decoder::D
    config::Dict # Optional: Store config dictionary
end

function WorldModelAgent(config::Dict, spaces::Dict)
    # Ensure obs_space and act_space are likely Dicts of Space objects
    # Convert Dict to NamedTuple of Space if needed, or adjust accessors
    # For simplicity, assume direct access works for now

    encoder = Encoder(; 
        obs=spaces[:image], 
        act=eval(Meta.parse(config["agent"]["enc"]["act"])), 
        mults=config["agent"]["enc"]["mults"], 
        depth=config["agent"]["enc"]["depth"], 
        kernel=config["agent"]["enc"]["kernel"]
    ) # Use eval for act string
    token_dim = calculate_encoder_output_dim(config["agent"]["enc"], spaces[:image])

    # Instantiate RSSM
    rssm = StatefulRSSM(;
        deter_dim=config["agent"]["dyn"]["deter"],
        hidden_dim=config["agent"]["dyn"]["hidden"],
        stoch_dim=config["agent"]["dyn"]["stoch"],
        classes_dim=config["agent"]["dyn"]["classes"],
        act=eval(Meta.parse(config["agent"]["dyn"]["act"])),
        token_dim=token_dim,
        imglayers=config["agent"]["dyn"]["imglayers"],
        obslayers=config["agent"]["dyn"]["obslayers"],
        dynlayers=config["agent"]["dyn"]["dynlayers"],
        blocks=config["agent"]["dyn"]["blocks"],
        act_space=spaces[:action], # Assuming act_space has an :action key
        free_nats=config["agent"]["dyn"]["free_nats"] |> cast,
        unimix=config["agent"]["dyn"]["unimix"] |> cast
    )

    # Instantiate Decoder
    decoder = Decoder(;
        obs=spaces[:image],
        deter_dim=config["agent"]["dyn"]["deter"], # Use deter_dim from dyn_config
        units=config["agent"]["dec"]["units"],
        stoch_dim=config["agent"]["dyn"]["stoch"], # Use stoch_dim from dyn_config
        classes_dim=config["agent"]["dyn"]["classes"], # Use classes_dim from dyn_config
        act=eval(Meta.parse(config["agent"]["dec"]["act"])),
        mults=config["agent"]["dec"]["mults"],
        depth=config["agent"]["dec"]["depth"],
        kernel=config["agent"]["dec"]["kernel"],
        bspace=config["agent"]["dyn"]["blocks"] # Assuming bspace uses RSSM blocks
    )

    # Return the constructed Agent
    return WorldModelAgent(encoder, rssm, decoder, config)
end

function loss(agent::WorldModelAgent, obs, ps, st)
    action = obs[:action];
    reset = obs[:is_first];
    
    # --- Encoder ---
    tokens, st_enc = agent.encoder(obs, ps.encoder, st.encoder);

    # --- RSSM Loss (provides KL losses and features) ---
    # los.dyn and los.rep should have shape (T, B) based on rssm.loss implementation
    los, feat, st_rssm = loss(agent.rssm, tokens, action, reset, ps.rssm, st.rssm);

    # --- Decoder ---
    # recons should have shape (W, H, C, T, B)
    recons, st_dec = agent.decoder(feat, ps.decoder, st.decoder);

    # --- Reconstruction Loss ---
    target_image = cast.(obs[:image]) ./ cast(255); # Access Dict with Symbol
    # recon_loss should have shape (T, B)
    recon_loss = calculate_reconstruction_loss_sum(recons, target_image) # Use the sum version

    # --- Calculate Scalar Means ---
    mean_dyn = Statistics.mean(los.dyn)
    mean_rep = Statistics.mean(los.rep)
    mean_recon = Statistics.mean(recon_loss)

    # --- Get Loss Scales (Ensure defaults are used if keys are missing) ---
    scale_dyn, scale_rep, scale_recon = 1.0f0, 0.1f0, 1.0f0;
    
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


# Helper function for reconstruction loss (sum version)
function calculate_reconstruction_loss_sum(recons, target; dims=(1, 2, 3), agg=sum)
    # recons, target shape: (W, H, C, T, B)
    elementwise_sq_error = (recons .- target).^2
    recon_loss_summed_spatial = agg(elementwise_sq_error; dims=dims) # Shape: (1, 1, 1, T, B)
    recon_loss_per_TB = dropdims(recon_loss_summed_spatial; dims=dims) # Shape: (T, B)
    return recon_loss_per_TB
end

function make_config(fullconfig::Dict, component::String, config_type::String)
    if component == "agent"
        return make_config_agent(fullconfig, config_type)
    elseif component == "replay"
        return make_config_replay(fullconfig, config_type)
    elseif component == "run"
        return make_config_run(fullconfig, config_type)
    end
end;

function make_config(fullconfig::Dict, config_type::String)    
    return Dict(
        "agent" => make_config_agent(fullconfig, config_type),
        "replay" => make_config_replay(fullconfig, config_type),
        "run" => make_config_run(fullconfig, config_type),
        "logdir" => "~/logs"
    )
end;

function make_config_replay(fullconfig::Dict, config_type::String)
    replay_config = deepcopy(fullconfig["defaults"]["replay"])  
    if config_type == "debug"
        replay_config["size"] = Int64(fullconfig["debug"]["replay.size"])
    else
        replay_config["size"] = Int64(replay_config["size"])
    end
    return replay_config
end;


function make_config_agent_typ(fullconfig::Dict, component::String, config_type::String)

    default_component_base = fullconfig["defaults"]["agent"][component]
    default_component_type = default_component_base["typ"]
    default_config = default_component_base[default_component_type]

    if config_type == "defaults"
        return deepcopy(default_config)
    end
    
    # Get debug overrides
    debug_agent_config = fullconfig["debug"]["agent"]
        
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
end;

function make_config_agent(fullconfig::Dict, config_type::String)


    agent_config = deepcopy(fullconfig["defaults"]["agent"])  
    agent_config["enc"] = make_config_agent_typ(fullconfig, "enc", config_type)
    agent_config["dec"] = make_config_agent_typ(fullconfig, "dec", config_type)
    agent_config["dyn"] = make_config_agent_typ(fullconfig, "dyn", config_type)

    return agent_config

    return Dict(
        "enc" => enc_dec_dyn[1],
        "dec" => enc_dec_dyn[2],
        "dyn" => enc_dec_dyn[3]
    )
end;

function make_config_run(fullconfig::Dict, config_type::String)
    return deepcopy(fullconfig[config_type]["run"])
end;
