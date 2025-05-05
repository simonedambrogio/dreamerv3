using GLMakie
include("generalization.jl")

env = GeneralizationEnv();
obs = reset!(env);
println("Initial State: $(env.current_state_idx), Obs size: $(size(obs))")

# State 1
env = GeneralizationEnv();
obs_1 = reset!(env);
plot_obs(obs_1)

# State 1 -> State 2
println("--- Trial $(env.current_trial) ---");
println("Current option indices: $(env.current_option_indices)");
action = 2;
obs_2, rew, done = step!(env, action);
plot_obs(obs_2); println("Rew: $rew"); println("Direction: $(env.correct_direction)")

# State 2 -> State 3
action = 2;
obs_3, rew, done = step!(env, action);
plot_obs(obs_3); println("Rew: $rew");

# State 3 -> State 1
action = 1;
obs_1, rew, done = step!(env, action);
plot_obs(obs_1); println("Rew: $rew");