
using ReinforcementLearningEnvironments
using ArcadeLearningEnvironment
using ReinforcementLearningEnvironments: AtariEnv, list_atari_rom_names
using ReinforcementLearningBase
using ImageTransformations: imresize

# 1. List available games
println("Available Atari games:")
games = list_atari_rom_names()
println(first(games, 5), "...")


# 2. Create environment with specific settings
# atari: {size: [96, 96], repeat: 4, sticky: True, gray: True, actions: all, lives: unused, noops: 30, autostart: False, pooling: 2, aggregate: max, resize: pillow, clip_reward: False}
env = AtariEnv(
    name="pong",
    grayscale_obs=true,        # gray: True
    frame_skip=4,              # repeat: 4
    noop_max=30,              # noops: 30
    terminal_on_life_loss=false,  # lives: unused
    max_num_frames_per_episode=108000,  # default max frames
    repeat_action_probability=0.25,  # sticky: True (25% chance to repeat)
    full_action_space=true,    # actions: all
    color_averaging=false      # pooling related
)
# Then resize observations directly
env = StateTransformedEnv(env, state_mapping=s -> imresize(s, (96, 96)))



# 3. Environment information
println("\nEnvironment Info:")
println("Action Space: ", action_space(env))
println("Observation Space: ", RLBase.state_space(env))

# 4. Run a short episode
println("\nRunning a short episode:")
state = RLBase.reset!(env)
total_reward = 0.0

for step in 1:100
    # Get state info
    println("\nStep $step:")
    println("State shape: ", size(RLBase.state(env)))
    
    # Take random action
    action = rand(action_space(env))
    println("Action taken: ", action)
    
    # Step environment
    RLBase.act!(env, action)
    reward = RLBase.reward(env)
    done = RLBase.is_terminated(env)
    
    total_reward += reward
    println("Reward: $reward")
    println("Done: $done")
    
    if done
        println("\nEpisode finished after $step steps")
        break
    end
end

println("\nTotal reward: $total_reward")
