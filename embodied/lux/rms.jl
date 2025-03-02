using Lux
using Lux: BoolType, has_affine, match_eltype, safe_getproperty, unwrapped_eltype, initialparameters, AbstractLuxLayer
using ConcreteStructs: @concrete
using Markdown: @doc_str
using Static: StaticBool, True, static
using Random: AbstractRNG
using NNlib
using Statistics: mean, std, var
using Random

# RMSNorm is a normalization technique that scales inputs by their root mean square (RMS)
# It's a simpler alternative to LayerNorm that doesn't center the data
# This makes it computationally more efficient while maintaining good performance

# Helper functions for the RMSNorm layer
# Apply activation function (or return input unchanged if activation is identity)
@inline __apply_activation(::typeof(identity), x) = x;
@inline __apply_activation(f, x) = f.(x);

@doc doc"""
    RMSNorm(shape::NTuple{N, Int}, activation=identity; epsilon=1f-5, dims=Colon(), affine::Bool=true, init_scale=ones32)

Computes root mean square normalization over the input array. Optionally applies an elementwise affine transformation afterwards.

Given an input array ``x``, this layer computes 
```math
y = \frac{x}{\sqrt{\mathbb{E}[x^2] + \epsilon}} * \gamma
```
where ``\gamma`` is a trainable parameter if `affine=true`.

## Arguments
- `shape`: Broadcastable shape of input array excluding the batch dimension.
- `activation`: After normalization, elementwise activation `activation` is applied.

## Keyword Arguments
- `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`
- `epsilon`: a value added to the denominator for numerical stability.
- `dims`: Dimensions to normalize the array over.
- If `affine=true`, it also applies a rescale to the input through a learnable scale parameter.
  + `init_scale`: Controls how the `scale` is initiliazed

## Inputs
- `x`: AbstractArray

## Returns
- `y`: Normalized Array
- Empty NamedTuple()

## Parameters
- `affine=false`: Empty `NamedTuple()`
- `affine=true`
  + `scale`: Scale of shape `(shape..., 1)`
"""
@concrete struct RMSNorm{affine, N} <: AbstractLuxLayer
    shape::NTuple{N, Int}
    activation
    epsilon
    init_scale
    dims
end

function RMSNorm(shape::NTuple{N, <:Int}, activation=identity; 
                epsilon::T=1.0f-5, dims=Colon(), affine::Bool=true, 
                init_scale=ones32, allow_fast_activation::Bool=true) where {N, T}
    # Apply fast activation if allowed (optimization for certain activation functions)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    # Create and return a new RMSNorm layer with the specified parameters
    return RMSNorm{affine, N}(shape, activation, epsilon, init_scale, dims)
end

# Check if the layer has a learnable scale parameter
@inline _affine(l::RMSNorm{A}) where {A} = A;

function Lux.initialparameters(rng::AbstractRNG, rn::RMSNorm)
    if _affine(rn)
        # If affine=true, create a scale parameter initialized with the provided function
        # This scale parameter will be learned during training
        scale = rn.init_scale(rng, rn.shape..., 1)
        return scale  # Return just the scale parameter
    else
        # If affine=false, no learnable parameters are needed
        return Float32[]  # Return an empty array
    end
end

function (l::RMSNorm)(x::AbstractArray, ps, st::NamedTuple)
    # Step 1: Compute the mean of squared values (mean square)
    mean_square = mean(abs2.(x); dims=l.dims)
    
    # Step 2: Calculate the scaling factor
    scale_factor = 1.0 ./ sqrt.(mean_square .+ l.epsilon)
    
    # Step 3: Apply the normalization by multiplying the input by the scale factor
    y = x .* scale_factor
    
    # Step 4: Apply the learnable scale parameter if affine=true
    if _affine(l)
        # Get the scale parameter
        scale_param = ps isa NamedTuple ? ps.scale : ps
        
        # Print debug information
        println("Input shape: ", size(x))
        println("Scale param shape: ", size(scale_param))
        println("l.shape: ", l.shape)
        println("l.dims: ", l.dims)
        
        # For a specific case where shape=(C,) and dims=(3,)
        # We need to reshape scale_param to have 1s in all dims except dim 3
        if length(l.shape) == 1 && l.dims == (3,)
            # Special case for normalizing along channel dimension
            # Reshape to [1, 1, C, 1] for a 4D input
            if ndims(x) == 4
                reshaped_scale = reshape(scale_param, (1, 1, l.shape[1], 1))
                println("Reshaped scale to: ", size(reshaped_scale))
                y = y .* reshaped_scale
            else
                # Handle other dimensionalities
                println("Unsupported input dimensionality: ", ndims(x))
            end
        else
            # More general approach for other cases
            # Create a reshape pattern with ones except at the dimensions we want to scale
            reshape_pattern = ones(Int, ndims(x))
            
            # If dims is Colon(), we need to handle it differently
            if l.dims isa Colon
                # In this case, shape should match the non-batch dimensions
                for i in 1:length(l.shape)
                    reshape_pattern[i] = l.shape[i]
                end
            else
                # For specific dims, place the shape values at those dimensions
                for (i, dim) in enumerate(l.dims)
                    if i <= length(l.shape)
                        reshape_pattern[dim] = l.shape[i]
                    end
                end
            end
            
            println("Reshape pattern: ", reshape_pattern)
            reshaped_scale = reshape(scale_param, Tuple(reshape_pattern))
            println("Reshaped scale to: ", size(reshaped_scale))
            
            # Apply the scale
            y = y .* reshaped_scale
        end
    end
    
    # Step 5: Apply the activation function and return the result
    return __apply_activation(l.activation, y), st
end

function Base.show(io::IO, l::RMSNorm)
    print(io, "RMSNorm($(l.shape)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(_affine(l)), dims=$(l.dims)")
    return print(io, ")")
end


#=
# Test RMSNorm =============================================================
x = rand(Float32, 46, 46, 4, 80);
rng = Random.default_rng();
rms = RMSNorm((4,), dims=(3,));
ps, st = Lux.setup(rng, rms);
y, st = rms(x, ps, st);
size(y)

# 1. Check the RMS of the normalized output along the normalized dimensions
# This should be close to 1.0 if scale parameter is 1.0
function check_rms(x, dims)
    # Calculate RMS along the specified dimensions
    rms_value = sqrt.(mean(abs2.(x); dims=dims))
    # The mean of these RMS values should be close to 1.0
    mean_rms = mean(rms_value)
    println("Mean RMS of normalized output: ", mean_rms)
    # Should be close to 1.0 (with some small epsilon deviation)
    # EXPECT: ~1.0 because RMSNorm explicitly normalizes by the RMS value
    # This is the core property of RMSNorm as defined in the paper
    println("Close to 1.0? ", isapprox(mean_rms, 1.0, atol=1e-4))
    return mean_rms
end

# 2. Check re-scaling invariance property
# If we scale the input, the output should remain the same
function check_rescaling_invariance(norm_layer, ps, st, x, scale_factor=2.0)
    # Get output with original input
    y1, _ = norm_layer(x, ps, st)
    
    # Get output with scaled input
    scaled_x = x .* scale_factor
    y2, _ = norm_layer(scaled_x, ps, st)
    
    # Check if outputs are approximately equal
    diff = maximum(abs.(y1 .- y2))
    println("Maximum difference after rescaling input by $(scale_factor): ", diff)
    # EXPECT: Very small difference (close to 0) because RMSNorm should be invariant to input scaling
    # This is a key property mentioned in the paper - the re-scaling invariance
    # The RMS of scaled input is exactly scale_factor times the RMS of original input,
    # so the normalization should cancel out the scaling completely
    println("Rescaling invariant? ", isapprox(diff, 0.0, atol=1e-5))
    return diff
end

# 3. Check that mean is NOT normalized (unlike LayerNorm)
function check_mean_not_normalized(x, original_x, dims)
    # Calculate mean of original input
    original_mean = mean(original_x; dims=dims)
    
    # Calculate mean of normalized output
    normalized_mean = mean(x; dims=dims)
    
    # The ratio of means should follow the same pattern as the normalization
    ratio = normalized_mean ./ (original_mean .+ 1e-10)  # avoid division by zero
    
    # The standard deviation of this ratio should NOT be close to zero
    # (if it were normalized, all means would be zero and ratio would be constant)
    std_ratio = std(ratio[:])
    println("Standard deviation of mean ratios: ", std_ratio)
    # EXPECT: std_ratio > 0.01 because RMSNorm doesn't center the data
    # This is a key difference from LayerNorm - RMSNorm only scales but doesn't shift
    # If the means were normalized, all normalized means would be 0 and the ratio would be 0
    # For random data, we expect some variation in the ratio, indicating means aren't normalized
    println("Means NOT normalized? ", std_ratio > 0.01)
    return std_ratio
end

# 4. Check that variance is normalized
function check_variance_normalized(x, dims)
    # Calculate variance along the specified dimensions
    var_value = var(x; dims=dims, corrected=false)
    
    # The mean of these variances should be close to 1.0
    mean_var = mean(var_value)
    println("Mean variance of normalized output: ", mean_var)
    # EXPECT: For random data with mean ~0.5, we expect variance ~0.25, not 1.0
    # This is because RMSNorm normalizes by RMS, not variance
    # For data with mean μ and variance σ², the RMS² = μ² + σ²
    # If RMS = 1 and μ ≈ 0.5, then σ² ≈ 0.75
    # However, for uniform random data in [0,1], the variance is 1/12 ≈ 0.083
    # So the expected normalized variance depends on the input distribution
    println("Variance normalized? ", isapprox(mean_var, 1.0, atol=0.1))
    return mean_var
end

# Run all tests
println("\n=== Testing RMSNorm ===")
rms_value = check_rms(y, (1, 2, 3))
rescale_diff = check_rescaling_invariance(rms, ps, st, x)
mean_std = check_mean_not_normalized(y, x, (1, 2, 3))
var_mean = check_variance_normalized(y, (1, 2, 3))

# 5. Bonus: Test with different scale values
# Create a RMSNorm with custom scale initialization
custom_init_scale(rng, dims...) = fill(0.5f0, dims...)
rms_custom = RMSNorm((46, 46, 4), dims=(1, 2, 3), init_scale=custom_init_scale);
ps_custom, st_custom = Lux.setup(rng, rms_custom);
y_custom, _ = rms_custom(x, ps_custom, st_custom);

println("\n=== Testing RMSNorm with custom scale (0.5) ===")
rms_value_custom = check_rms(y_custom, (1, 2, 3))
# EXPECT: ~0.5 because we've set the scale parameter to 0.5
# The scale parameter directly multiplies the normalized values,
# so the RMS of the output should be scaled by the same factor
println("RMS with scale=0.5 should be ~0.5: ", isapprox(rms_value_custom, 0.5, atol=0.05))
=#
