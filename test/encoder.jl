using Test
using Lux
using Random
using Tools
using BFloat16s # Assuming COMPUTE_TYPE is BFloat16s based on nets.jl
using Statistics
using Printf # Added for formatting
using LinearAlgebra # Added for norm
using Zygote

# Include necessary files relative to the test script location
include("../dreamerv3/encoder.jl")
include("../embodied/lux/nets.jl") # Already included by encoder.jl if needed there
include("../embodied/lux/rms.jl") # Need this for _affine if RMSNorm is used directly

# --- Helper Function for Readable Model Summary ---
const ANSI_GREEN = "\e[32m"
const ANSI_RESET = "\e[0m"

# Define _affine locally if needed, mirroring the one in rms.jl
@inline _affine(l::RMSNorm{A}) where {A} = A

function print_model_summary(model::Lux.Chain)
    # Start Green
    
    println("Model: Chain")
    layers = model.layers # Access the field directly
    # total_params = Lux.parameterlength(model) # Get total params accurately

    for name in keys(layers)
        layer = layers[name]
        layer_params = Lux.parameterlength(layer)

        print("  ($(name)): \n\t")
        layer_type_full = typeof(layer)
        layer_type_name = layer_type_full.name.name
        print(layer_type_name)

        # Add specific parameters based on type
        if layer isa Conv
            k = layer.kernel_size
            c_in = layer.in_chs # Or layer.in_channels depending on Lux version
            c_out = layer.out_chs # Or layer.out_channels
            pad = layer.pad # Correct field name for padding
            stride = layer.stride
            act_str = layer.activation == identity ? "" : ", $(layer.activation)"
            # Use string interpolation with Printf for formatting numbers if needed
            print("(k=$k, $c_in => $c_out, stride=$stride, pad=$pad$act_str)")
        elseif layer isa MaxPool
            mode = layer.layer.mode
            k = mode.kernel_size
            s = mode.stride
            pad = mode.pad
            print("(k=$k, stride=$s, pad=$pad)")
        elseif layer isa RMSNorm
            shape = layer.shape
            dims = layer.dims
            affine_flag = _affine(layer)
            act_str = layer.activation == identity ? "" : ", $(layer.activation)"
            print("(shape=$shape, dims=$dims, affine=$affine_flag$act_str)")
        # Add more layer types as needed (Dense, etc.)
        else
            # Generic print for other layers
            print("(...)")
        end

        # Print parameter count for the layer
        @printf "\n\t# Params: %d" layer_params
        println() # Newline for next layer
    end
    # println("Total Parameters: ", total_params)

    # Reset Color at the end
end # End of print_model_summary function

# --- RMSNorm Test Helper Functions ---

# 1. Check the RMS of the normalized output along the normalized dimensions
function check_rms_normalized(x, expected_rms, dims; atol=nothing)
    # Adjust tolerance based on type
    T = eltype(x)
    test_atol = isnothing(atol) ? (T == BFloat16 ? 1e-2 : 1e-4) : atol

    rms_actual = sqrt.(mean(abs2.(x); dims=dims))
    # Calculate mean in Float32 for potentially better precision/stability
    mean_rms = mean(Float32.(rms_actual))
    min_rms, max_rms = extrema(rms_actual) # Extrema still on original type
    println("    RMS along dim $dims: Mean(F32)=$mean_rms, Min=$min_rms, Max=$max_rms (Expected: ≈ $expected_rms, atol=$test_atol)")
    @test isapprox(mean_rms, expected_rms, atol=test_atol) # Compare F32 mean
end

# 2. Check re-scaling invariance property (Approximate due to epsilon)
function check_rescaling_invariance(norm_layer, ps, st, x, scale_factor=2.0; atol=nothing)
    # Adjust tolerance based on type
    T = eltype(ps.scale) # Get compute type from parameters
    # Relaxing default tolerance slightly for this check due to epsilon
    default_atol = T == BFloat16 ? 5e-2 : 1e-4 # Increased default tolerance
    test_atol = isnothing(atol) ? default_atol : atol

    # Ensure input is correct type
    x_typed = T.(x)
    # Get output with original input
    y1, _ = norm_layer(x_typed, ps, st)

    # Get output with scaled input
    scaled_x = x_typed .* T(scale_factor)
    y2, _ = norm_layer(scaled_x, ps, st)

    # Check if outputs are approximately equal
    max_diff = maximum(abs.(y1 .- y2))
    println("    Max difference after rescaling input by $scale_factor: $max_diff (Expected: ≈ 0.0, atol=$test_atol)")
    @test isapprox(max_diff, 0.0, atol=test_atol)
end

# 3. Check that mean is NOT normalized (unlike LayerNorm)
function check_mean_not_normalized(normalized_output, original_input, dims; threshold=0.01)
    mean_out = mean(normalized_output; dims=dims)
    mean_in = mean(original_input; dims=dims)
    # Calculate std dev of means across the non-normalized dimensions
    # If mean *was* normalized, std dev of means would be close to 0
    std_means = std(mean_out[:]; corrected=false) # Use population std dev
    println("    Std Dev of Mean values across non-normalized dims: $std_means (Expected: > $threshold)")
    # Check if the standard deviation is significantly larger than zero
    @test std_means > threshold
end

# --- Test Setup ---
rng = Random.default_rng()
COMPUTE_TYPE = BFloat16 # Or Float32 if you prefer

# Define a sample observation space (adjust as needed)
obs_space = Dict(
    :image => Tools.Space(UInt8, (48, 48, 1)),
    # Add other obs types if your encoder handles them
)
img_space = obs_space[:image];

# Define sample parameters for the encoder
enc_kwargs = (
    obs = img_space,
    act = gelu,
    mults = [2, 3, 4, 4], # Example mults
    depth = 8,           # Smaller depth for faster testing
    kernel = 5
)

println(ANSI_GREEN, "\n--- Parameters used for testing ---", ANSI_RESET)
println(enc_kwargs)

# --- Test Suite ---
@testset "Encoder Tests" begin

    @testset "1. Structure and Initialization" begin

        println(ANSI_GREEN, "\n--- Testing Encoder Structure ---", ANSI_RESET)
        # Instantiate the Encoder
        enc = Encoder(; enc_kwargs...)
        @test enc isa Encoder
        @test enc.convolve isa Chain

        # Show the structure
        println("\nEncoder network summary:") # Changed label
        # println(enc.net) # Removed default print
        print_model_summary(enc.convolve) # Use helper function

        # Test Lux setup
        ps, st = Lux.setup(rng, enc.convolve)
        @test ps isa NamedTuple
        @test st isa NamedTuple
        # println("\nParameter structure (keys): ", keys(ps))
        # println("State structure (keys): ", keys(st))
        println("---------------------------------\n")
    end

    @testset "2. Forward Pass" begin
        println(ANSI_GREEN, "\n--- Testing Encoder Forward Pass ---", ANSI_RESET)
        # Instantiate Encoder and setup Lux
        enc = Encoder(; enc_kwargs...)
        ps, st = Lux.setup(rng, enc)

        # Define batch and sequence length
        T = 10 # Sequence Length
        B = 4  # Batch Size

        # Create dummy input data matching obs_space
        # Input shape: (Width, Height, Channels, Sequence Length, Batch Size)
        dummy_image = rand(rng, UInt8, img_space.size..., T, B);
        dummy_obs = (; image = dummy_image); # Use NamedTuple matching encoder input

        # Run forward pass
        # Note: The Encoder forward pass takes (obs, params, state)
        output, st_new = enc(dummy_obs, ps, st)

        # Calculate expected output dimensions
        final_depth = enc_kwargs.depth * enc_kwargs.mults[end]
        downsample_factor = 2^length(enc_kwargs.mults)
        final_W = img_space.size[1] ÷ downsample_factor
        final_H = img_space.size[2] ÷ downsample_factor
        expected_embedding_dim = final_W * final_H * final_depth

        # Check output type
        @test output isa AbstractArray{COMPUTE_TYPE}

        # Check output shape: (embedding_dim, T, B)
        @test size(output) == (expected_embedding_dim, T, B)
        # println("Input image shape: ", size(dummy_image))
        # println("Output embedding shape: ", size(output), " (Expected: ($expected_embedding_dim, $T, $B))")

        # println("State unchanged after forward pass: ", st == st_new)
        println("---------------------------------\n")

    end

    # --- Add more tests below ---
    @testset "3. Check that RMSNorm layer works properly" begin
        println(ANSI_GREEN, "\n--- Testing RMSNorm Properties ---", ANSI_RESET)

        # Test parameters
        in_shape = (24, 24, 8) # Example input shape (W, H, C)
        batch_size = 4
        norm_dims = (3,) # Normalize over channels (typical for CNN)
        cnn_channels = in_shape[3]
        correct_param_shape = (cnn_channels,)
        correct_feature_dim = 3 # Channel dim is 3

        # a) Test with default scale (1.0)
        println("  Testing RMSNorm with scale = 1.0")
        rmsn_layer = RMSNorm(correct_param_shape, correct_feature_dim, identity; dims=norm_dims, init_scale=cast_ones)
        ps_rmsn, st_rmsn = Lux.setup(rng, rmsn_layer)
        dummy_x = rand(rng, Float32, in_shape..., batch_size) # Start with Float32
        T_compute = eltype(ps_rmsn.scale)
        dummy_x_typed = T_compute.(dummy_x);

        # Forward pass
        y, _ = rmsn_layer(dummy_x_typed, ps_rmsn, st_rmsn);

        # Perform checks
        check_rms_normalized(y, 1.0, norm_dims)
        check_rescaling_invariance(rmsn_layer, ps_rmsn, st_rmsn, dummy_x)
        check_mean_not_normalized(y, dummy_x_typed, norm_dims)

        # b) Test with custom scale (e.g., 0.5)
        println("\n  Testing RMSNorm with scale = 0.5")
        custom_init_scale(rng, dims...) = fill!(similar(rand(rng, Float32, dims...)), 0.5f0) # Use Float32 for init then cast
        rmsn_layer_custom = RMSNorm(correct_param_shape, correct_feature_dim, identity; dims=norm_dims, init_scale=custom_init_scale)
        ps_rmsn_custom, st_rmsn_custom = Lux.setup(rng, rmsn_layer_custom)
        T_compute_custom = eltype(ps_rmsn_custom.scale)
        dummy_x_custom_typed = T_compute_custom.(dummy_x);

        # Forward pass
        y_custom, _ = rmsn_layer_custom(dummy_x_custom_typed, ps_rmsn_custom, st_rmsn_custom)

        # Perform checks with expected_rms = 0.5
        check_rms_normalized(y_custom, 0.5, norm_dims);
        check_rescaling_invariance(rmsn_layer_custom, ps_rmsn_custom, st_rmsn_custom, dummy_x);
        check_mean_not_normalized(y_custom, dummy_x_custom_typed, norm_dims);

        println("---------------------------------")
    end

end # End Encoder Tests


# Test Gradient Computation
println(ANSI_GREEN, "\n--- Testing Encoder Gradient Computation ---", ANSI_RESET)

enc = Encoder(; enc_kwargs...);
ps, st = Lux.setup(rng, enc);

# Define loss function
function loss_fn(m, x_obs, p, s)
    output, st_new = m(x_obs, p, s) # Use m for model, p for params, s for state
    return sum(output), st_new # Return state to avoid grad issues if loss depends on it (though not here)
end;

# Create dummy input data again for the gradient test
T = 10 # Sequence Length
B = 4  # Batch Size
dummy_image_grad = rand(rng, UInt8, img_space.size..., T, B);
dummy_obs_grad = (; image = dummy_image_grad);

# Test gradient computation
println("Attempting Zygote.gradient for Encoder...")
try
    # Gradient calculation: Zygote needs model *params* as the argument for differentiation
    # Loss function needs to be adapted slightly for Zygote's common pattern
    loss_val_zygote, grad_zygote = Zygote.withgradient(
        p_ -> begin # p_ represents the parameters (ps)
            loss_val, _ = loss_fn(enc, dummy_obs_grad, p_, st) # Pass enc, obs, p_, st
            loss_val
        end,
        ps # Differentiate with respect to parameters
    )

    println("Encoder Gradient calculation successful!")
    println("Loss value from Zygote: ", loss_val_zygote)
    # Optional: Check if gradients are Nothing or contain actual values
    # println("Gradient structure (keys): ", keys(grad_zygote[1]))
    @test grad_zygote[1] isa NamedTuple
    @test !isnothing(grad_zygote[1].convolve.layer_1.weight) # Check a specific grad

catch e
    println("ERROR during Encoder gradient calculation:")
    showerror(stdout, e)
    println()
    # Optionally print stacktrace
    Base.show_backtrace(stdout, catch_backtrace())
    @test false # Fail the test explicitly on error
end

println("---------------------------------")
