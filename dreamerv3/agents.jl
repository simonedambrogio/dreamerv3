using Lux

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

    encoder = Encoder(; obs=spaces[:image], act=eval(Meta.parse(config["enc"]["act"])), mults=config["enc"]["mults"], depth=config["enc"]["depth"], kernel=config["enc"]["kernel"]) # Use eval for act string
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
    )

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
    )

    # Return the constructed Agent
    return WorldModelAgent(encoder, rssm, decoder, config)
end

# Function to calculate the full world model loss
function agent_loss(agent::WorldModelAgent, carry, obs, action, ps, st)
    # enc_carry, dyn_carry, dec_carry = carry;
    reset = obs[:is_first];
    batch_length, batch_size = size(reset);

    # --- Encoder ---
    tokens, st_enc = agent.encoder(obs, ps.encoder, st.encoder);

    # --- RSSM Loss (provides KL losses and features) ---
    # los.dyn and los.rep should have shape (T, B) based on rssm.loss implementation
    final_dyn_carry, dyn_entries, los, repfeat, mets = loss(agent.rssm, carry, tokens, action, reset, ps.rssm, st.rssm);

    # --- Decoder ---
    # recons should have shape (W, H, C, T, B)
    recons, st_dec = agent.decoder(repfeat, ps.decoder, st.decoder);

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
    st_new = (encoder=st_enc, rssm=st.rssm, decoder=st_dec) # NOTE: Using st.rssm as loss doesn't return state update yet

    # --- Prepare Auxiliary Output ---
    # Store both per-element and scalar losses, plus other info
    kl_losses_scalar = (; dyn = mean_dyn, rep = mean_rep)
    aux_losses = (; kl=los, kl_scalar=kl_losses_scalar, recon=recon_loss, recon_scalar=mean_recon)
    aux_full = (final_carry=final_dyn_carry, losses=aux_losses, metrics=mets, st=st_new, repfeat=repfeat) # Include repfeat if needed later

    return total_loss, aux_full
end

# Helper function for reconstruction loss (sum version)
function calculate_reconstruction_loss_sum(recons, target; dims=(1, 2, 3), agg=sum)
    # recons, target shape: (W, H, C, T, B)
    elementwise_sq_error = (recons .- target).^2
    recon_loss_summed_spatial = agg(elementwise_sq_error; dims=dims) # Shape: (1, 1, 1, T, B)
    recon_loss_per_TB = dropdims(recon_loss_summed_spatial; dims=dims) # Shape: (T, B)
    return recon_loss_per_TB
end




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
