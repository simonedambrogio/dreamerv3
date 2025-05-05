# test/replay.jl
include("../embodied/envs/custom/generalization.jl")
include("../embodied/core/replay.jl")
using Test

@testset "Replay Buffer Tests" begin

    # --- Setup ---
    seq_length = 10
    capacity = 50
    chunksize = 20 # Keep small for testing chunk completion
    batch_size = 4

    replay = Replay(length=seq_length, capacity=capacity, chunksize=chunksize)
    env = GeneralizationEnv(; num_trials=10, verbose=false) # 10 trials * 4 steps/trial = 40 steps needed

    @test length(replay) == 0
    @test isempty(replay.chunks)

    # --- Test Add ---
    println("\n--- Testing Add ---")
    obs = reset!(env)
    total_steps = 0
    max_steps = 65 # Enough to fill multiple chunks and exceed capacity slightly

    # Keep track of the first step added to compare later if needed
    first_step_data = Dict()

    # step_num = 1
    for step_num in 1:max_steps
        total_steps += 1
        action = rand(replay.rng, 1:3) # Random action (1, 2, or 3)
        next_obs, reward, done = step!(env, action)

        # Create step dictionary (mimicking driver output)
        step_data = Dict(
            :observation => obs, # Observation *before* action
            :action => action,
            :reward => Float32(reward),
            :is_first => (step_num == 1 && total_steps == 1), # Only true for the very first step overall
            :is_last => done,
            :is_terminal => done # Assuming is_last means terminal here
        )

        if step_num == 1 && total_steps == 1
            first_step_data = deepcopy(step_data) # Store for potential later comparison
        end

        # We use worker 0 for simplicity
        add!(replay, step_data, 0)

        obs = next_obs # Prepare for next iteration

        # Check internal state periodically
        if step_num == 1
            @test length(replay.chunks) == 1
            @test replay.current_index[0] == 1
        end
        if step_num == chunksize + 1
             println("Second chunk should have started.")
             @test length(replay.chunks) == 2
             @test replay.current_index[0] == 1
         end
        if step_num >= seq_length
            expected_items = min(capacity, total_steps - seq_length + 1)
            @test length(replay) == expected_items # Check number of insertable sequences
        end

        if done
            println("Environment episode finished at step $step_num, resetting.")
            obs = reset!(env)
            # In a real scenario, the first step after reset would have :is_first = true
            # For this test, we simplify and only mark the very first step overall as is_first
        end
    end

    println("\n--- Final State After Add ---")
    println("Total steps added: $total_steps")
    println("Number of items (sequences) in replay: $(length(replay))")
    @test length(replay) == capacity # Should have hit capacity
    println("Number of chunks: $(length(replay.chunks))")
    expected_chunks = ceil(Int, total_steps / chunksize)
    println("Expected chunks based on steps: $expected_chunks")
    @test length(replay.chunks) >= expected_chunks

    println("\nBasic Add tests completed.")

    # --- Test Sample ---
    println("\n--- Testing Sample ---")
    @test length(replay) > 0 # Ensure buffer is not empty before sampling

    batch = sample(replay, batch_size)

    @test batch isa Dict{Symbol, AbstractArray}
    # Check necessary keys are present
    expected_keys = [:observation, :action, :reward, :is_first, :is_last, :is_terminal]
    @test all(k -> haskey(batch, k), expected_keys)

    # Check shapes and types
    W, H = env.image_dims
    @test size(batch[:observation]) == (W, H, 1, seq_length, batch_size)
    @test eltype(batch[:observation]) == UInt8

    @test size(batch[:action]) == (seq_length, batch_size)
    @test eltype(batch[:action]) == Int

    @test size(batch[:reward]) == (seq_length, batch_size)
    @test eltype(batch[:reward]) == Float32

    @test size(batch[:is_first]) == (seq_length, batch_size)
    @test eltype(batch[:is_first]) == Bool

    @test size(batch[:is_last]) == (seq_length, batch_size)
    @test eltype(batch[:is_last]) == Bool

    @test size(batch[:is_terminal]) == (seq_length, batch_size)
    @test eltype(batch[:is_terminal]) == Bool

    println("Sampled batch structure:")
    for (k, v) in batch
        println("  Key: $k, Type: $(typeof(v)), Shape: $(size(v))")
    end

    # Optional: Sample multiple times
    println("Sampling multiple batches...")
    for i in 1:3
        batch_i = sample(replay, batch_size)
        @test size(batch_i[:observation]) == (W, H, 1, seq_length, batch_size)
    end
    println("Multiple samples successful.")

    println("\nSampling tests completed.")

end # End Testset


# batch_n = 1

# t = 11
# begin
#     plot_obs(batch[:observation][:,:,1,t,batch_n])
#     println("Action: $(batch[:action][t,batch_n])")
#     println("Reward: $(batch[:reward][t,batch_n])")
#     println("Is first: $(batch[:is_first][t,batch_n])")
#     println("Is last: $(batch[:is_last][t,batch_n])")
#     println("Is terminal: $(batch[:is_terminal][t,batch_n])")
# end
