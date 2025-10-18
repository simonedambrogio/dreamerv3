using GLMakie, Lux, JLD2, CSV, DataFrames
include("../dreamerv3/WorldModel.jl");
include("../embodied/envs/custom/generalization.jl");
include("../embodied/core/replay.jl");

# imputs ------------------------
foldername="I-II-O";
ckp="ckp105480.0";
# -------------------------------


path2log= joinpath("logs", "defaults", foldername);
# Look at learning curve
begin
    
    losses = CSV.read(joinpath(path2log, "losses.csv"), DataFrame) |> filter(r -> r.train_step > 300);
    lw = 5
    fig = Figure(size=(900, 600), fontsize=25)
    ax = GLMakie.Axis(
        fig[1, 1], xlabel="Step", ylabel="Loss", 
        xticks = (1:100:nrow(losses), string.(losses.train_step[1:100:end])),
    )
    lines!(ax, losses.total_loss, label="total", linewidth=lw)
    lines!(ax, losses.dyn_loss, label="dyn", linewidth=lw)
    lines!(ax, losses.rep_loss, label="rep", linewidth=lw)
    lines!(ax, losses.recon_loss, label="recon", linewidth=lw)
    # Add legend
    axislegend(ax, orientation=:horizontal, title="Losses")

    display(fig)
end

# Look at reconstruction quality (original vs reconstructed)
input = load_object(joinpath(path2log, ckp, "encoder_input.jld2"));
output = load_object(joinpath(path2log, ckp, "decoder_output.jld2"));
begin
    sizeimage = 150
    nimages = 12
    
    function plot(fig::Figure, input::AbstractArray, output::AbstractArray; t::Int)
        # remove all decorations
        ax1 = GLMakie.Axis(fig[t, 1], aspect=DataAspect())
        hidedecorations!(ax1)
        hidespines!(ax1)
        original = input[:,:,1,t,1]
        image!(ax1, rotr90(original), colormap=:grays, colorrange = (0, 255), label="original"); 
        ax2 = GLMakie.Axis(fig[t, 2], aspect=DataAspect())
        hidedecorations!(ax2)
        hidespines!(ax2)
        reconstructed = output[:,:,1,t,1]
        image!(ax2, rotr90(reconstructed), colormap=:grays, colorrange = (0, 255), label="reconstructed"); 

        return ax1, ax2
    end

    function plot(fig::Figure, images::AbstractArray; t::Array{Int}, row::Int)
        # remove all decorations
        for i in t
            ax = GLMakie.Axis(fig[row, i], aspect=DataAspect())
            hidedecorations!(ax)
            hidespines!(ax)
            image!(ax, rotr90(images[:,:,1,i,1]), colormap=:grays, colorrange = (0, 255), label="original"); 
        end
    end

    # Plot original and reconstructed images
    fig = Figure(size=(sizeimage*nimages, sizeimage*2), fontsize=25)
    plot(fig, input; t=collect(1:nimages), row=1)
    plot(fig, output; t=collect(1:nimages), row=2)
    display(fig)
end

# Check model's ability to dream
begin
    # Load parameters and state
    @load joinpath(path2log, ckp, "ckpt.jld2") ps_cpu st_cpu
    
    
end

# Instantiate agent
# --- Configuration ---
config_filepath = joinpath(@__DIR__, "..", "dreamerv3", "configs.yaml");
fullconfig = YAML.load_file(config_filepath);
config = make_config(fullconfig, "defaults");
spaces = Dict(
    :image => Tools.Space(UInt8, (64, 64, 1)),
    :action => Tools.Space(Int32; low=1, high=2) # Assuming actions 1 and 2
);
agent = WorldModelAgent(config, spaces);

# 2. Encode the batch (using GPU batch)
tokens, _ = agent.encoder(Dict(:image => input), ps_cpu.encoder, st_cpu.encoder);
# 3. Run the RSSM observe step (using GPU batch for actions/is_first)
(_, feat_seq), _ = observe(agent.rssm, tokens, recon_batch_gpu[:action], recon_batch_gpu[:is_first], mutable_ps.rssm, mutable_st.rssm)
# 4. Decode the features
recons, _ = agent.decoder(feat_seq, mutable_ps.decoder, mutable_st.decoder) # recons will be on GPU
# recons shape: (W, H, C, T, B=1)

# 5. Select images (first time step, first batch element)
# t_idx = 1 # Select the first time step
b_idx = 1 # Select the first (only) batch element

# Original image from CPU batch
original_image_uint8_cpu = recon_batch_cpu[:image][:, :, :, :, b_idx] # Shape (W, H, C)
# Reconstructed image - move to CPU and then process
reconstructed_image_gpu = recons[:, :, :, :, b_idx]
reconstructed_image_cpu = reconstructed_image_gpu |> cpu_device()
reconstructed_image_processed = reconstructed_image_cpu .* cast(255)
