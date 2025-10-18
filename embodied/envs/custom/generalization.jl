using YAML
using Images
using Random
using StatsBase: sample # For sampling option pairs
using LuxCore: AbstractRNG # For type hinting RNG

if gethostname()=="epsymac58.psy.ox.ac.uk"
    const config_path = "/Volumes/PROJECTS/Ongoing/Exploration/dreamerv3-lux/embodied/envs/custom/configs.yaml";
else
    const config_path = "/users/rushworth/gwr089/scratch/Explore/dreamerv3/embodied/envs/custom/configs.yaml";
end

getnode() = gethostname()=="epsymac58.psy.ox.ac.uk" ? "local" : "cluster";

# Helper function to create a fixation cross
function create_fixation_cross(W::Int, H::Int, cross_thickness::Int=2, cross_length::Int=10, bg_color::UInt8=0xff, cross_color::UInt8=0x00)
    img = fill(bg_color, W, H, 1) # Gray background
    center_w, center_h = W ÷ 2, H ÷ 2
    half_len = cross_length ÷ 2
    half_thick = cross_thickness ÷ 2

    # Horizontal bar
    row_start = max(1, center_w - half_thick)
    row_end = min(W, center_w + half_thick)
    col_start = max(1, center_h - half_len)
    col_end = min(H, center_h + half_len)
    img[row_start:row_end, col_start:col_end, 1] .= cross_color

    # Vertical bar
    row_start = max(1, center_w - half_len)
    row_end = min(W, center_w + half_len)
    col_start = max(1, center_h - half_thick)
    col_end = min(H, center_h + half_thick)
    img[row_start:row_end, col_start:col_end, 1] .= cross_color

    return img
end;

# Define the structure for the environment
mutable struct GeneralizationEnv
    # Stimuli (UInt8 arrays)
    fixation_cross::Array{UInt8, 3}        # (W, H, 1)
    options::Vector{Array{UInt8, 3}}       # 3 x (W, H/2, 1) - Good, Medium, Bad
    directions::Matrix{Array{UInt8, 3}}    # 3x2 x (W, H, 1) - Good/Med/Bad x Left/Right
    outcomes::Vector{Array{UInt8, 3}}      # 4 x (W, H, 1) - Correct Good, Med, Bad, Incorrect

    # Current state variables
    current_state_idx::Int                 # 1: Fixation, 2: Option choice, 3: Direction choice, 4: Outcome
    current_option_indices::Vector{Int}    # Indices (1-3) of the two options currently shown [left, right]
    current_stimulus_indices::Vector{Int}  # Indices (1-3) used to generate stimuli [left, right]
    chosen_option_idx::Int                 # Index (1-3) of the option chosen in State 2
    correct_direction::Int                 # 0: Left, 1: Right - The correct direction for the current trial
    current_observation::Array{UInt8, 3}   # Current observation (W, H, 1)
    current_trial::Int                     # Current trial number
    current_task::Int                      # Current task number
    option_idxs_used::Vector{Int}          # Indices (1-584) of the three options used in this task

    # Configuration
    image_dims::Tuple{Int, Int}            # (W, H)
    images::UnitRange{Int}                 # Indices (1-584) of the images used in this task
    num_trials::Int                        # Number of trials per task
    rng::AbstractRNG
end

function load_options(
        W::Int, H::Int, rng::AbstractRNG, images::Vector{Int}, verbose::Bool=false,
        config_path::String=config_path
    )
    cfg = YAML.load_file(config_path)
    stimuli_path = cfg["paths"][getnode()]["grays"] # Use grayscale images

    # Load Options (Good, Medium, Bad) - Expected size W x H/2
    options = Vector{Array{UInt8, 3}}(undef, 3)
    # Choose 3 distinct images for the options semi-permanently for this env instance
    # This differs from env.jl which sampled dynamically per task
    option_idxs_used = sample(rng, images, 3, replace=false) # Assuming 584 images exist as in env.jl
    for i in 1:3
        path = joinpath(stimuli_path, "$(option_idxs_used[i]).png")
        img = load(path)
        # Ensure image is grayscale and correct size W x H/2
        img_gray = Gray.(img)
        # Resize the original img_gray directly
        img_final_resized = imresize(img_gray, (W, H ÷ 2))
        # Convert to UInt8 (0-255) and add channel dimension
        # Important: Reshape needs dimensions in (Width, Height, Channels) order
        options[i] = reshape(UInt8.(round.(Float32.(img_final_resized) .* 255)), W, H ÷ 2, 1)
        verbose && println("Loaded option $i: size=$(size(options[i])), eltype=$(eltype(options[i]))")
    end
    return options, option_idxs_used
end;

function load_directions(W::Int, H::Int, verbose::Bool=false, config_path::String=config_path)
    cfg = YAML.load_file(config_path)
    stimuli_path = cfg["paths"][getnode()]["grays"] # Use grayscale images

    directions = Matrix{Array{UInt8, 3}}(undef, 3, 2)
    dir_names = ["left", "right"]
    for i in 1:3 # Good, Medium, Bad
        for j in 1:2 # Left, Right
            path = joinpath(stimuli_path, "direction", "$(i)-$(dir_names[j]).png")
            img = load(path)
            img_gray = Gray.(img)
             if size(img_gray) != (H, W)
                 verbose && println("Warning: Resizing direction image $(i)-$(dir_names[j]) from $(size(img_gray)) to ($H, $W)")
                 img_resized = imresize(img_gray, (H, W))
             else
                 img_resized = img_gray
             end
            # Convert to UInt8 and add channel dimension
            directions[i, j] = reshape(UInt8.(round.(Float32.(img_resized) .* 255)), W, H, 1)
            verbose && println("Loaded direction $(i)-$(dir_names[j]): size=$(size(directions[i,j])), eltype=$(eltype(directions[i,j]))")
        end
    end
    return directions
end

function GeneralizationEnv(;
        config_path::String=config_path, 
        W::Int=64, 
        H::Int=64,
        images=1:584,
        num_trials::Int=100,
        seed::Int=42, 
        verbose::Bool=false
    )

    rng = Random.MersenneTwister(seed)
    cfg = YAML.load_file(config_path)
    stimuli_path = cfg["paths"][getnode()]["grays"] # Use grayscale images
    image_dims = (W, H)

    # --- Create Fixation Cross ---
    fixation_cross = create_fixation_cross(W, H)
    verbose && println("Created fixation cross: size=$(size(fixation_cross)), eltype=$(eltype(fixation_cross))")

    # --- Load Stimuli ---
    verbose && println("Loading stimuli...")

    # Load Options (Good, Medium, Bad) - Expected size W x H/2
    options, option_idxs_used = load_options(W, H, rng, collect(images), verbose)
   
    # Load Directions (Good, Medium, Bad) x (Left, Right) - Expected size W x H
    directions = load_directions(W, H, verbose)

    # Load Outcomes (Correct Good, Med, Bad, Incorrect) - Expected size W x H
    outcomes = Vector{Array{UInt8, 3}}(undef, 4)
    outcome_names = ["1-correct", "2-correct", "3-correct", "incorrect"]
    for i in 1:4
        path = joinpath(stimuli_path, "outcome", "$(outcome_names[i]).png")
        img = load(path)
        img_gray = Gray.(img)
        if size(img_gray) != (H, W)
             verbose && println("Warning: Resizing outcome image $(outcome_names[i]) from $(size(img_gray)) to ($H, $W)")
             img_resized = imresize(img_gray, (H, W))
        else
             img_resized = img_gray
        end
        # Convert to UInt8 and add channel dimension
        outcomes[i] = reshape(UInt8.(round.(Float32.(img_resized) .* 255)), W, H, 1)
        verbose && println("Loaded outcome $(outcome_names[i]): size=$(size(outcomes[i])), eltype=$(eltype(outcomes[i]))")

    end
    verbose && println("Stimuli loading complete.")

    # --- Initialize State ---
    # Create dummy observation first
    current_observation = Array{UInt8, 3}(undef, W, H, 1)
    current_state_idx = 1 # Start in dummy fixation state so reset works correctly
    current_option_indices = [0, 0]
    current_stimulus_indices = [0, 0]
    chosen_option_idx = 0
    correct_direction = 0
    current_trial = 1
    current_task = 1
    
    env = GeneralizationEnv(fixation_cross, options, directions, outcomes,
              current_state_idx, current_option_indices, current_stimulus_indices,
              chosen_option_idx, correct_direction, current_observation,
              current_trial, current_task, option_idxs_used,
              image_dims, images, num_trials, rng)

    # Call reset! to set the proper initial state
    reset!(env)
    return env
end

# Helper to create the option observation by combining two stimuli
function _generate_option_observation!(env::GeneralizationEnv)
    # Randomly assign the 3 options to left/right positions
    env.current_stimulus_indices = sample(env.rng, 1:3, 2, replace=false)
    left_idx, right_idx = env.current_stimulus_indices
    env.current_option_indices = [left_idx, right_idx] # Store which index is left/right

    left_img = env.options[left_idx]    # Should be (W, H/2, 1) = (64, 32, 1)
    right_img = env.options[right_idx]   # Should be (W, H/2, 1) = (64, 32, 1)

    # Concatenate horizontally (along dimension 2)
    env.current_observation = hcat(left_img, right_img) # Result should be (W, H/2 + H/2, 1) = (64, 64, 1)

    # Check against env.image_dims = (W, H)
    if size(env.current_observation) != (env.image_dims[1], env.image_dims[2], 1)
         error("Generated option observation has wrong size: $(size(env.current_observation)), expected $((env.image_dims[1], env.image_dims[2], 1))")
    end
end

# Reset the environment to the start of a new episode (State 1 - Fixation)
function reset!(env::GeneralizationEnv)
    env.chosen_option_idx = 0 # Reset chosen option
    env.current_option_indices = [0, 0]
    env.current_stimulus_indices = [0, 0]
    # Set observation to fixation cross
    # env.current_state_idx = 1 # Start at Fixation state
    # env.current_observation = env.fixation_cross
    
    env.current_trial = 1
    env.current_state_idx = 2
    _generate_option_observation!(env)

    # Sample the correct direction for the *next* trial (which starts after the wait in State 1)
    # We sample it here to ensure consistency if reset is called mid-trial
    env.correct_direction = rand(env.rng, 1:2) # 1 for left, 2 for right

    return env.current_observation
end

# Environment step function
function step!(env::GeneralizationEnv, action::Int, verbose::Bool=false)
    # action: 0 for left, 1 for right, 2 for wait
    reward = 0.0f0
    done = false

    @assert action in [1, 2, 3] "Invalid action: $action. Action must be 1 (left), 2 (right), or 3 (wait)."

    # --- State Transitions based on generalization.txt mapping ---
    if env.current_state_idx == 1 # Currently in Fixation state (State 1)
        if action == 3 # Wait action
            # Transition to Option state (State 2)
            env.current_state_idx = 2
            _generate_option_observation!(env) # Generate and set the new observation
            # Note: correct_direction for this trial was already sampled in reset! or the previous state 4 transition
            reward = 0.0f0
        else # Left or Right action
            # Stay in State 1
            env.current_observation = env.fixation_cross # Ensure observation remains fixation
            reward = 0.0f0
        end

    elseif env.current_state_idx == 2 # Currently in Option state (State 2)
        if action == 1 || action == 2 # Left or Right action
            # Agent chose an option
            env.chosen_option_idx = env.current_option_indices[action] # Get the index (1, 2, or 3) of the chosen option
            # Transition to Direction state (State 3)
            env.current_state_idx = 3
            # The direction stimulus shown depends on the CHOSEN option and the CORRECT direction
            env.current_observation = env.directions[env.chosen_option_idx, env.correct_direction]
            reward = 0.0f0
        else # Wait action
            # Stay in State 2 (observation doesn't change)
            reward = 0.0f0
        end

    elseif env.current_state_idx == 3 # Currently in Direction state (State 3)
        if action == 1 || action == 2 # Left or Right action
            # Agent chose a direction
            chosen_direction = action
            # Transition to Outcome state (State 4)
            env.current_state_idx = 4
            if chosen_direction == env.correct_direction
                # Correct choice: Show correct outcome stimulus (V)
                env.current_observation = env.outcomes[env.chosen_option_idx] # Outcome depends on the option chosen in State 2
                # Assign reward immediately based on chosen option
                reward = Float32(env.chosen_option_idx)
            else
                # Incorrect choice: Show incorrect outcome stimulus (X)
                env.current_observation = env.outcomes[4] # Incorrect outcome stimulus
                # Assign zero reward immediately
                reward = 0.0f0
            end
        else # Wait action
             # Stay in State 3 (observation doesn't change)
             reward = 0.0f0
        end

    elseif env.current_state_idx == 4 # Currently in Outcome state (State 4)
        # Any action transitions back to Fixation state (State 1)
        # Reward for this transition is always 0, as the reward was given upon entering State 4.
        reward = 0.0f0
        # Transition back to Fixation state (State 1)
        # env.current_state_idx = 1
        # env.current_observation = env.fixation_cross
        env.current_state_idx = 2
        _generate_option_observation!(env)
        env.correct_direction = rand(env.rng, 1:2) # 1 for left, 2 for right
        env.current_trial += 1

        if env.current_trial > env.num_trials
            env.current_task += 1
            println("Moving to next task: task $(env.current_task)")
            env.current_trial = 1
            done = true
            # Load new options
            new_bag = filter(i -> i ∉ env.option_idxs_used, env.images)
            env.options, new_option_idxs_used = load_options(env.image_dims..., env.rng, new_bag, verbose)
            env.option_idxs_used = vcat(env.option_idxs_used, new_option_idxs_used)
        end
    else
        error("Invalid state index: $(env.current_state_idx)")
    end

    # Return observation, reward, done status
    return env.current_observation, reward, done #, Dict() # Optional info dictionary
end

function plot_obs(obs)
    f = Figure(); 
    ax = GLMakie.Axis(f[1, 1], aspect=DataAspect()); 
    heatmap!(ax, rotr90(obs[:,:,1]), colormap=:grays, colorrange = (0, 255)); 
    display(f)
end
