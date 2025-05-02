# Minimal Training Loop for Dense Layer with BFloat16

using Lux
using Optimisers
using Zygote
using Random
using BFloat16s
using Statistics # For mean

# 1. Define Compute Type
const COMPUTE_TYPE_TEST = BFloat16
println("Using Compute Type: ", COMPUTE_TYPE_TEST)

# Helper cast function (minimal version for this test)
cast(x::AbstractArray) = COMPUTE_TYPE_TEST.(x)
cast(x::Number) = COMPUTE_TYPE_TEST(x)
cast_zeros(rng, dims...) = cast(zeros(dims...))
cast_glorot_uniform(rng, dims...) = cast(Lux.glorot_uniform(rng, dims...))


# 2. Define Model
model = Dense(10 => 5; init_weight=cast_glorot_uniform, init_bias=cast_zeros)
println("Model Created: ", model)

# 3. Setup Model & Convert Params
rng = MersenneTwister(123)
ps, st = Lux.setup(rng, model)
println("Initial PS type: ", typeof(ps.weight))

# Ensure parameters are BFloat16 (should be handled by init_weight/bias now)
# ps = Lux.fmap(x -> x isa AbstractArray ? cast(x) : x, ps)
# println("Converted PS type: ", typeof(ps.weight))


# 4. Generate Data
batch_size = 4
x_data = cast(randn(Float32, 10, batch_size))
y_target = cast(randn(Float32, 5, batch_size))
println("Data Type: x=", typeof(x_data), ", y=", typeof(y_target))

# 5. Define Loss Function
function mse_loss(model, x, y, ps, st)
    y_pred, st_new = model(x, ps, st)
    loss = mean(abs2.(y_pred .- y))
    return loss, st_new
end

# 6. Setup Optimizer
learning_rate = 1e-3 # Use a slightly smaller LR for BFloat16
opt = Adam(learning_rate)
opt_state = Optimisers.setup(opt, ps)
println("Optimizer Setup: ", opt)

# 7. Training Loop
num_steps = 50
println("\nStarting Training Loop (", num_steps, " steps)...")

mutable_ps = deepcopy(ps) # Work with copies
mutable_st = deepcopy(st)

(loss_val, current_st), back = Zygote.pullback(
    (p, s, m, x, y) -> mse_loss(m, x, y, p, s),
    mutable_ps, mutable_st, model, x_data, y_target
)
grad = back((one(loss_val), nothing))[1] # Get gradients for ps
println("Grad weight: ", grad.weight)
println("Grad bias: ", grad.bias)
opt_state, mutable_ps = Optimisers.update(opt_state, mutable_ps, grad)
println("After Update weight: ", mutable_ps.weight)
println("After Update bias: ", mutable_ps.bias)
