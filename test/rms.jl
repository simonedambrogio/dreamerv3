# test/rms.jl

# Imports
using Lux
using Zygote
using Random
using Statistics
# Include necessary custom layers
include("../embodied/lux/nets.jl");
include("../embodied/lux/rms.jl");

println("--- RMSNorm Dense Zygote Test ---")

# --- Dense Test Case --- 
feature_dim = 10
batch_size = 1

model_dense = Chain(
    Dense(feature_dim => feature_dim), 
    # RMSNorm: shape=(10,), feature_dim=1, normalize over dims=(1,)
    RMSNorm((feature_dim,), 1, identity; dims=(1,), affine=true) # Added feature_dim=1
)

println("Model (Dense): ", model_dense)

rng = MersenneTwister(12345);
x_dense_in = randn(rng, Float32, feature_dim, batch_size);
ps_dense, st_dense = Lux.setup(rng, model_dense);

println("RMSNorm scale parameter shape (Dense): ", size(ps_dense.layer_2.scale));
println("Input shape (Dense): ", size(x_dense_in));

function loss_fn_dense(m, x_in, p, s)
    y, st_new = m(x_in, p, s);
    loss = sum(y);
    return loss
end;

println("Testing forward pass (Dense)...")
y_pred_dense, st_final_dense = model_dense(x_dense_in, ps_dense, st_dense)
println("Final output shape (Dense): ", size(y_pred_dense))
loss_val_dense = loss_fn_dense(model_dense, x_dense_in, ps_dense, st_dense)
println("Loss value (Dense): ", loss_val_dense)

println("Testing RMSNorm normalization properties (Dense)...")
y_intermediate_dense, _ = model_dense.layers.layer_1(x_dense_in, ps_dense.layer_1, st_dense.layer_1)
rms_layer_dense = model_dense.layers.layer_2
mean_square_intermediate = mean(abs2.(y_intermediate_dense); dims=rms_layer_dense.dims)
epsilon_casted = eltype(y_intermediate_dense)(rms_layer_dense.epsilon)
rms_inv = one(eltype(y_intermediate_dense)) ./ sqrt.(mean_square_intermediate .+ epsilon_casted)
y_normalized_manual = y_intermediate_dense .* rms_inv
mean_square_normalized = mean(abs2.(y_normalized_manual); dims=rms_layer_dense.dims)
@assert all(isapprox.(mean_square_normalized, 1.0f0; atol=1f-4)) "Dense Mean square check failed!"
rms_normalized = sqrt.(mean_square_normalized)
@assert all(isapprox.(rms_normalized, 1.0f0; atol=1f-4)) "Dense RMS check failed!"
println("Dense Normalization checks passed.")

println("Attempting Zygote.gradient (Dense)...")
try
    grads_dense = Zygote.gradient(loss_fn_dense, model_dense, x_dense_in, ps_dense, st_dense);
    println("Dense Gradient calculation successful!");
catch e
    println("ERROR during Dense gradient calculation:")
    showerror(stdout, e); println()
end

println("\n--- RMSNorm CNN-like Zygote Test ---")

# --- CNN-like Test Case --- 
W, H, C, B_cnn = 6, 6, 4, 2 # Example CNN dimensions

# Model: Just the RMSNorm layer with CNN config
cmodel_cnn = RMSNorm((C,), 3; dims=(3,), affine=true) # shape=(C,), feature_dim=3, dims=(3,)

println("Model (CNN): ", cmodel_cnn)

x_cnn_in = randn(rng, Float32, W, H, C, B_cnn);
ps_cnn, st_cnn = Lux.setup(rng, cmodel_cnn); # ps_cnn will be like (; scale = ...) 

println("RMSNorm scale parameter shape (CNN): ", size(ps_cnn.scale));
println("Input shape (CNN): ", size(x_cnn_in));

function loss_fn_cnn(m, x_in, p, s)
    y, st_new = m(x_in, p, s);
    loss = sum(y);
    return loss
end;

println("Testing forward pass (CNN)...")
y_pred_cnn, st_final_cnn = cmodel_cnn(x_cnn_in, ps_cnn, st_cnn)
println("Final output shape (CNN): ", size(y_pred_cnn))
loss_val_cnn = loss_fn_cnn(cmodel_cnn, x_cnn_in, ps_cnn, st_cnn)
println("Loss value (CNN): ", loss_val_cnn)

println("Testing RMSNorm normalization properties (CNN)...")
y_intermediate_cnn = x_cnn_in # Input to RMSNorm is just the input here
rms_layer_cnn = cmodel_cnn
mean_square_intermediate_cnn = mean(abs2.(y_intermediate_cnn); dims=rms_layer_cnn.dims)
epsilon_casted_cnn = eltype(y_intermediate_cnn)(rms_layer_cnn.epsilon)
rms_inv_cnn = one(eltype(y_intermediate_cnn)) ./ sqrt.(mean_square_intermediate_cnn .+ epsilon_casted_cnn)
y_normalized_manual_cnn = y_intermediate_cnn .* rms_inv_cnn
mean_square_normalized_cnn = mean(abs2.(y_normalized_manual_cnn); dims=rms_layer_cnn.dims)
# Need to check mean square approx 1 for each element along non-normalized dims
println("Mean square of normalized output shape (CNN - target ≈ 1): ", size(mean_square_normalized_cnn))
@assert all(isapprox.(mean_square_normalized_cnn, 1.0f0; atol=1f-3)) "CNN Mean square check failed!"
rms_normalized_cnn = sqrt.(mean_square_normalized_cnn)
@assert all(isapprox.(rms_normalized_cnn, 1.0f0; atol=1f-3)) "CNN RMS check failed!"
println("CNN Normalization checks passed.")

println("Attempting Zygote.gradient (CNN)...")
try
    # Gradient needs to be w.r.t. (loss_fn, model, input, params, state)
    # Note: ps_cnn is the NamedTuple (; scale = ...), not nested like ps_dense
    grads_cnn = Zygote.gradient(loss_fn_cnn, cmodel_cnn, x_cnn_in, ps_cnn, st_cnn);
    println("CNN Gradient calculation successful!");
    # println("CNN scale grad shape: ", size(grads_cnn[3].scale))
catch e
    println("ERROR during CNN gradient calculation:")
    showerror(stdout, e); println()
end

println("--- Test Finished ---")
