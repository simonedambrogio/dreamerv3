using ArcadeLearningEnvironment
using Images
using Statistics, Random
using ImageTransformations: imresize
using Interpolations: BSpline, Linear, Lanczos4OpenCV
using GLMakie

"""
Atari environment
    Example:
    ```julia
        using YAML, Tools
        config = YAML.load_file("dreamerv3/configs.yaml") |> symbolize;
        include("atari.jl")
        env = Atari(; name = "pong", config[:defaults][:env][:atari]...);
        obs = step!(env, Dict("action" => 1, "reset" => false))
        plot(env)
        plot(obs["image"])
    ```
"""
mutable struct Atari
    ale::Ptr{ArcadeLearningEnvironment.ALEInterface}
    repeat::Int64
    size::Tuple{Int64,Int64}
    gray::Bool
    noops::Int64
    lives::Symbol  # :unused, :discount, :reset
    sticky::Bool
    length::Int64
    pooling::Int64
    aggregate::Symbol  # :max, :mean
    resize::Symbol    # :opencv, :pillow
    autostart::Bool
    clip_reward::Bool
    actionset::Vector{Int64}
    buffers::Vector{Array{UInt8,3}}
    prevlives::Int64
    duration::Int64
    done::Bool
    rng::AbstractRNG
    ACTION_MEANING::Vector{String}
    WEIGHTS::Array{Float32,3}
end

function Atari(;
    name::String="pong",
    repeat::Int64=4,
    size::Vector{Int64}=[84, 84],
    gray::Bool=true,
    noops::Int64=0,
    lives::Symbol=:unused, # :unused, :discount, :reset
    sticky::Bool=true,
    actions::Symbol=:all, # :all, :needed
    max_num_frames_per_episode::Int64=108_000,
    pooling::Int64=2,
    aggregate::Symbol=:max, # :max, :mean
    resize::Symbol=:pillow, # :opencv, :pillow
    autostart::Bool=false,
    clip_reward::Bool=false,
    seed=nothing,
    log_level=:error) # :error, info, :warning 

    # reference 1: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/main/src/ReinforcementLearningEnvironments/src/environments/3rd_party/atari.jl
    # reference 2: https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl/blob/master/src/aleinterface.jl

    size = Tuple(size)
    # Initialize ALE
    ale = ALE_new() # reference 2
    setLoggerMode!(log_level) # reference 2
    if isnothing(seed) # reference 1
        rng = Random.default_rng()
    else
        setInt(ale, "random_seed", Int32(seed % typemax(Int32)))
        rng = MersenneTwister(hash(seed + 1))
    end
    loadROM(ale, name) # reference 2
    setFloat(ale, "repeat_action_probability", sticky ? 0.25 : 0.0) # reference 2
    actionset = if actions == :all
        getLegalActionSet(ale) # reference 2
    else
        getMinimalActionSet(ale)
    end
    
    setInt(ale, "frame_skip", Int32(1))  # reference 1 !!! do not use internal frame_skip here, we need to apply max-pooling for the latest two frames, so we need to manually implement the mechanism.
    setBool(ale, "color_averaging", false) # reference 1

    # Set sticky actions
    setFloat(ale, "repeat_action_probability", sticky ? 0.25 : 0.0) # reference 2
    
    # Initialize buffers for frame pooling
    W, H = getScreenWidth(ale), getScreenHeight(ale);
    buffers = [zeros(UInt8, W, H, 3) for _ in 1:pooling];

    action_meaning =[
        "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
        "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
        "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"
    ];
    weights = reshape(Float32[0.299, 0.587, 1 - (0.299 + 0.587)], 1, 1, 3);

    Atari(
        ale,
        repeat,
        size,
        gray,
        noops,
        lives,
        sticky,
        max_num_frames_per_episode,
        pooling,
        aggregate,
        resize,
        autostart,
        clip_reward,
        actionset,
        buffers,
        0,
        0,
        true,
        Random.MersenneTwister(seed),
        action_meaning,
        weights
    )
end


@doc """
    step!(env::Atari, action::Dict)

Advance the Atari environment by executing an action.

The function:
1. Handles episode reset if requested or if episode is done
2. Repeats the action for `env.repeat` frames (frame skipping)
3. Renders and pools only the last `env.pooling` frames
4. Tracks lives and episode termination conditions
5. Returns a dictionary with the processed observation and episode information

Returns:
    Dict with:
    - "image": Processed frame(s)
    - "reward": Accumulated reward
    - "is_first": Start of episode
    - "is_last": End of episode
    - "is_terminal": True terminal state (game over)
"""
function step!(env::Atari, action::Dict)
    if action["reset"] || env.done
        reset!(env)
        env.prevlives = ArcadeLearningEnvironment.lives(env.ale)
        env.duration = 0
        env.done = false
        return _obs(env, 0f0, is_first=true)
    end

    reward = 0f0
    is_terminal = false
    is_last = false
    
    _act = env.actionset[action["action"]]

    """
    The repeat option (also called "frame skipping") is a common technique in 
    reinforcement learning to speed up the learning process by reducing the number 
    of actions taken. 
    Example: With repeat=4, agent makes decisions every 4 frames instead of every frame
    """
    for repeat in 1:env.repeat
        reward += act(env.ale, _act)
        env.duration += 1
        
        """
        # Get new Observation:
            Processing pipeline:
            Raw Frame → Frame Pooling → Resize → (Optional) Grayscale → Agent Observation
                ↑
            _render!() is the function that gets the Raw Frames.
            (All the subsequent processing (pooling, resize, grayscale) happens in the _obs() function when we actually need to give an observation to the agent.)

            # Frame Pooling:
            --------------- Visual example ---------------
            repeat=4, pooling=2:
            Frame:    1    2    3    4
            Action:   A    A    A    A
            Render:   ✗    ✗    ✓    ✓    Only render frames needed for pooling
            Buffer:   -    -   [B1]  [B2]  These two frames will be pooled
            ----------------------------------------------
        """
        if repeat >= env.repeat - env.pooling
            _render!(env)
        end
                
        if game_over(env.ale)
            is_terminal = true
            is_last = true
        end

        if env.duration >= env.length
            is_last = true
        end
        
        current_lives = ArcadeLearningEnvironment.lives(env.ale)
        if env.lives == :discount && 0 < current_lives < env.prevlives
            is_terminal = true
        end
        if env.lives == :reset && 0 < current_lives < env.prevlives
            is_terminal = true
            is_last = true
        end
        env.prevlives = current_lives
        
        if is_terminal || is_last
            break
        end
    end
    
    env.done = is_last
    return _obs(env, reward; is_last, is_terminal)
end

function reset!(env::Atari)
    reset_game(env.ale)
    
    """
    # Handle no-operations (noops)
        When noops > 0, the environment will execute a random number (between 0 and noops) 
        of "do nothing" actions at the start of each episode

        This creates different starting conditions for each episode. It helps prevent the 
        agent from memorizing exact action sequences from fixed starting points.
    """
    for _ in 1:rand(env.rng, 0:env.noops)
        noop_action = findfirst(==("NOOP"), env.ACTION_MEANING)
        act(env.ale, noop_action)  # NOOP action
        if game_over(env.ale)
            reset_game(env.ale)
        end
    end
    
    """
    Handle autostart if needed
        The FIRE and UP sequence is specifically designed for games like Pong where:
        FIRE: Starts the game/round
        In Pong, you need to press FIRE to start playing
        Similar to pressing "Start" on the Atari console
        UP: Moves the paddle up initially
        In Pong, this moves your paddle away from the starting position
        Helps create a more standardized starting state
        Prevents the paddle from being stuck at the bottom
        Here's a visual of what happens in Pong:

        --------------- Visual example ---------------
        Initial State:     After FIRE:        After UP:
        |          |      |          |      |          |
        |          |      |          |      |    P     |  <- Paddle moved up
        |          |      |          |      |          |
        |P         |      |P    •    |      |          |
        (waiting)         (ball appears)     (ready to play)
        ----------------------------------------------
    """
    if env.autostart
        # find the index of FIRE action
        fire_action = findfirst(==("FIRE"), env.ACTION_MEANING)
        if !isnothing(fire_action)
            act(env.ale, fire_action)
            if game_over(env.ale)
                reset_game(env.ale)
            end
            up_action = findfirst(==("UP"), env.ACTION_MEANING)
            if !isnothing(up_action)
                act(env.ale, up_action)
                if game_over(env.ale)
                    reset_game(env.ale)
                end
            end
        end
    end
    """
    Initialize frame buffers at reset
        Get the first frame into the first buffer position
        Copy that first frame to all other buffer positions

        --------------- Visual example ---------------
        After _render!:
        buffers[1] = [First Frame]
        buffers[2] = [Empty/Random]
        buffers[3] = [Empty/Random]

        After copying:
        buffers[1] = [First Frame]
        buffers[2] = [First Frame]  <- Copied
        buffers[3] = [First Frame]  <- Copied
        --------------------------------------------
    """
    _render!(env) # Get first frame into buffers[1]
    for i in 2:length(env.buffers) # Copy first frame to all other buffers
        copyto!(env.buffers[i], env.buffers[1])
    end
end

function _render!(env::Atari)
    # Rotate buffers
    circshift!(env.buffers, -1)
    
    # Get RGB screen and reshape to WHC format
    w, h = getScreenWidth(env.ale), getScreenHeight(env.ale)
    screen_data = reshape(getScreenRGB(env.ale), 3, w, h) |> 
    x -> permutedims(x, (2,3,1))
    
    # No need for permutedims since we're already in WHC format
    copyto!(view(env.buffers[1], 1:w, 1:h, :), screen_data)
end;

function _obs(env::Atari, reward; is_first=false, is_last=false, is_terminal=false)
    if env.clip_reward
        reward = sign(reward)
    end
    
    # Aggregate frames
    image = if env.aggregate == :max
        maximum(cat(env.buffers..., dims=4); dims=4) |>
        x -> dropdims(x, dims=4)
    else  # :mean
        mean(env.buffers)
    end
    
    # Resize image based on method
    if env.resize == :opencv
        # OpenCV-like resizing
        image = imresize(image, env.size, method=Lanczos4OpenCV()) .|> 
        round |> x -> clamp.(x, 0, 255) .|> UInt8
    else 
        # PIL BILINEAR equivalent
        image = imresize(image, env.size, method=BSpline(Linear())) .|> 
        round |> x -> clamp.(x, 0, 255) .|> UInt8
    end

    
    # Convert to grayscale if needed
    if env.gray
        image = sum(Float32.(image) .* env.WEIGHTS, dims=3) .|>
        round |> x -> clamp.(x, 0, 255) .|> UInt8
    end

    Dict(
        "image" => image,
        "reward" => Float32(reward),
        "is_first" => is_first,
        "is_last" => is_last,
        "is_terminal" => is_terminal
    )
end

# Clean up when done
function Base.close(env::Atari)
    ALE_del(env.ale)
end

"""
    plot(env::Atari, action::Int)

Plot the game screen after taking an action. 
It shows the observation outputted by the step! function, which 
performs all the necessary processing (frame pooling, resizing, grayscale)
"""
function plot!(env::Atari, action::Int)
    w, h = getScreenWidth(env.ale), getScreenHeight(env.ale)
    fig = Figure(size = (w*4, h*4))
    ax = GLMakie.Axis(fig[1,1])
    obs = step!(env, Dict("action" => action, "reset" => false))
    GLMakie.heatmap!(ax, dropdims(obs["image"] ./ 255, dims = 3), colormap = :gray1)
    display(fig)
end

@doc """
    plot(observation::Array{UInt8, 3}; width::Int64=160, height::Int64=210, colormap::Symbol=:grays)

Plot the game screen after processing (step! function output).
"""
function plot(observation::Array{UInt8, 3}; width::Int64=160, height::Int64=210, colormap::Symbol=:grays)
    fig = Figure(size = (width*4, height*4))
    ax = GLMakie.Axis(fig[1,1])
    GLMakie.heatmap!(ax, dropdims(observation ./ 255, dims = 3), colormap = colormap)
    display(fig)
end;

"""
    plot(env::Atari)

Plot the game screen from the first buffer.
This is the raw screen data, before any processing is applied.
"""
function plot(env::Atari)
    img = env.buffers[1] ./ 255
    rgb_image = [RGB(img[i,j,:]...) for i in axes(img,1), j in axes(img,2)]
    f = Figure(size= (size(rgb_image)[2]*4, size(rgb_image)[1]*4))
    ax = GLMakie.Axis(f[1,1], aspect = DataAspect())
    GLMakie.image!(ax, rgb_image)  # Removed rotr90
    display(f)
end
