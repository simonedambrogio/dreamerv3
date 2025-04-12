using Lux, NNlib, Random, Tools, BFloat16s, YAML, Statistics, Test
include("../embodied/lux/rms.jl");
include("../embodied/lux/nets.jl");
include("../embodied/lux/BlockLinear.jl");
include("../embodied/lux/ReArrange.jl");
include("../embodied/lux/UpSample.jl");
include("../dreamerv3/decoder.jl");
config = YAML.load_file("dreamerv3/configs.yaml");

const ANSI_GREEN = "\e[32m"
const ANSI_BLUE = "\e[34m"
const ANSI_ORANGE = "\e[33m"
const ANSI_RESET = "\e[0m"

B = batch_size = config["debug"]["batch_size"];
T = seq_length = config["debug"]["batch_length"];
deter_dim = config["debug"]["agent"][".*\\.deter"];
obs = Tools.Space(UInt8, (96, 96, 1));
depth = config["debug"]["agent"][".*\\.depth"];
units = config["debug"]["agent"][".*\\.units"];
stoch_vars = config["debug"]["agent"][".*\\.stoch"];
classes_per_vars = config["debug"]["agent"][".*\\.classes"];

act=gelu;
mults=(2, 3, 4, 4);
kernel=5;
bspace=8;

rng = Random.default_rng();

feat = Dict(
    "deter" => cast(rand(rng, Float32, deter_dim, seq_length, batch_size)),
    "stoch" => cast(rand(rng, Float32, stoch_vars, classes_per_vars, seq_length, batch_size)),
);
reset = rand(rng, Bool, seq_length, batch_size);


# Test Forward Pass ------------------------------------------------------------
@testset "Forward Pass" begin
    println(ANSI_GREEN, "\n----- Testing Forward Pass... -----", ANSI_RESET)
    dec = Decoder(; obs, deter_dim, units, stoch_vars, classes_per_vars, mults, depth, kernel, bspace);
    ps, state = Lux.setup(rng, dec);
    out, state_new = dec(feat, ps, state);
    @testset "Output Type" begin
        @test out isa AbstractArray{COMPUTE_TYPE}
    end
    @testset "Output Size" begin
        @test size(out) == (obs.size..., T, B)
    end
end
# ----------------------------------------------------------------------------

depths = [depth * m for m in mults];
# Workout the final spatial dimensions after all CNN layers: 
# 1. Calculate the total downsampling factor
# Example: if self.depths = [64, 128, 256, 512] and self.outer=False
# factor = 2^4 = 16 (image will be downsampled by 16)
factor = 2 ^ length(depths);
# 2. Calculate the minimum spatial dimensions after all CNN layers
minres = [Integer(x // factor) for x in obs.size[1:2]];
# If input image is 96x96 and factor=16
# minres = [96//16, 96//16] = [6, 6]
# 3. Assert the final resolution is reasonable
@assert 3 <= minres[1] <= 16 "minres[1] must be between 3 and 16 $minres"
@assert 3 <= minres[2] <= 16 "minres[2] must be between 3 and 16 $minres"
# 4. Final shape includes the number of channels from last layer
shape = [minres..., depths[end]]; # final spatial dimensions after all CNN layers

u, g = prod(shape), bspace;
w, h, c = shape;

println(ANSI_GREEN, "\n----- Input shape -----", ANSI_RESET)
println("Width: ", w, "\nHeight: ", h, "\nChannels: ", c)
println("Deterministic part shape: ", deter_dim)
println("Stochastic part shape: ", stoch_vars, " x ", classes_per_vars)
println("Sequence length: ", T, "\nBatch size: ", B)

println(ANSI_GREEN, "\n----- Summary of the Steps -----", ANSI_RESET)
println("1. Spatialize the Deterministic part")
println(ANSI_BLUE, "\t BlockLinear layer", ANSI_RESET)
println(ANSI_BLUE, "\t ReArrange layer", ANSI_RESET)
println("2. Spatialize the Stochastic part")
println(ANSI_BLUE, "\t Linear layer", ANSI_RESET)
println(ANSI_BLUE, "\t RMSNorm layer", ANSI_RESET)
println(ANSI_BLUE, "\t Dense + Reshape layer", ANSI_RESET)
println("3. Combine the Deterministic and Stochastic parts")
println(ANSI_BLUE, "\t 3X[", ANSI_RESET)
println(ANSI_BLUE, "\t      RMSNorm layer (Combine Step)", ANSI_RESET)
println(ANSI_BLUE, "\t      Upsampling step (using `UpSample` layer)", ANSI_RESET)
println(ANSI_BLUE, "\t      Conv2D Layer (Upsampling Path)", ANSI_RESET)
println(ANSI_BLUE, "\t      RMSNorm Layer (After Conv)", ANSI_RESET)
println(ANSI_BLUE, "\t ]", ANSI_RESET)
println(ANSI_BLUE, "\t UpSample", ANSI_RESET)
println(ANSI_BLUE, "\t Conv2D Layer (Upsampling Path)", ANSI_RESET)
println(ANSI_BLUE, "\t RMSNorm Layer (After Conv)", ANSI_RESET)

# Spatialize the Deterministic part =========================================
println(ANSI_GREEN, "\n----- 1. Spatialize the Deterministic part -----", ANSI_RESET)

# Test BlockLinear --------------------------------------------------------
bl = BlockLinear(deter_dim, u, g)
ps, st = Lux.setup(rng, bl);
x = rand32(deter_dim, seq_length * batch_size);
out_bl, st = bl(x, ps, st);

println(ANSI_BLUE, "BlockLinear layer", ANSI_RESET)
print(ANSI_ORANGE) # Start color
println("  - Goal: Perform linear transformation efficiently with fewer parameters than Dense.")
println("  - Mechanism: Divides input features (", deter_dim, ") and output features (", u, ") into ", g, " blocks.")
println("               Each block is processed independently using smaller weight matrices.")
println("               Creates structured sparsity (no connections between blocks).")
println("  - Input Features : ", deter_dim)
println("  - Output Features: ", u, " ( = w * h * c = ", w, " * ", h, " * ", c, ")")
println("  - Blocks (g)   : ", g)
println("  - Input Shape: ", size(x))
println("  - Output Shape: ", size(out_bl))
print(ANSI_RESET) # End color

# Test ReArrange -----------------------------------------------------------
r = ReArrange((w, h, c, :))
ps, st = Lux.setup(rng, r);
out_deter, st = r(out_bl, ps, st);
size(out_deter)

println(ANSI_BLUE, "ReArrange layer", ANSI_RESET)
print(ANSI_ORANGE) # Start color
println("  - Goal: Change the shape (layout) of a tensor without changing its elements.")
println("  - Mechanism: Uses Julia's `reshape` function internally.")
println("  - Purpose here: Converts the flat feature vector from BlockLinear into")
println("                  a spatial tensor (Width, Height, Channels, Seq*Batch)")
println("                  suitable for input to convolutional layers.")
println("  - Input Shape : (", u, ", ", seq_length * batch_size, ") -> (Features, Seq*Batch)")
println("  - Output Shape: (", w, ", ", h, ", ", c, ", ", seq_length * batch_size, ") -> (W, H, C, Seq*Batch)")
println(ANSI_RESET) # End color


# Spatialize the Stochastic part =========================================
println(ANSI_GREEN, "\n----- 2. Spatialize the Stochastic part -----", ANSI_RESET)

# Test Linear -------------------------------------------------------------
l = Dense(stoch_vars * classes_per_vars, 2units, act)
ps, st = Lux.setup(rng, l);
x = rand32(stoch_vars * classes_per_vars, seq_length * batch_size);
out_l, st = l(x, ps, st);
size(out_l)

println(ANSI_BLUE, "Linear layer", ANSI_RESET)
print(ANSI_ORANGE) # Start color
println("  - Goal: Perform a standard linear transformation (fully connected).")
println("  - Mechanism: Multiplies input features by a weight matrix and adds a bias.")
println("               Every input feature connects to every output feature.")
println("  - Purpose here: Project the flattened stochastic state features")
println("                  (", stoch_vars * classes_per_vars, ") into an intermediate representation of size (", 2 * units, ").")
println("  - Input Shape: ", size(x))
println("  - Output Shape: ", size(out_l))
print(ANSI_RESET) # End color


# Test RMSNorm -------------------------------------------------------------
rn = RMSNorm((2units,), act; dims = (1,), init_scale=cast_ones)
ps, st = Lux.setup(rng, rn);
out_rn, st = rn(out_l, ps, st);
size(out_rn)

println(ANSI_BLUE, "RMSNorm layer", ANSI_RESET)
print(ANSI_ORANGE) # Start color
println("  - Goal: Normalize the output of a linear layer to improve training stability.")
println("  - Mechanism: Calculates the Root Mean Square (RMS) of the preceding layer's output features")
println("               (across the feature dimension, dim=", rn.dims, "). Divides the features by their RMS.")
println("               Optionally applies a learnable scaling factor ('gamma') afterwards.")
println("  - Purpose here: Stabilize training by controlling the magnitude (specifically, the RMS)")
println("                  of the activations flowing into the next layer, without altering the mean.")
println("  - Input Shape: ", size(out_l))
println("  - Output Shape: ", size(out_rn))
print(ANSI_RESET) # End color


# Test Dense + Reshape -----------------------------------------------------
l = Chain(Dense(16 => u, act), ReArrange((w, h, c, :)))
ps, st = Lux.setup(rng, l);
out_stoch, st = l(out_rn, ps, st);
size(out_stoch)

println(ANSI_BLUE, "Dense + Reshape layer", ANSI_RESET)
print(ANSI_ORANGE) # Start color
println("  - Goal: Combine a Dense layer with a ReArrange layer.")
println("  - Mechanism: First applies a Dense layer to the input, then reshapes the output.")
println("  - Purpose here: Convert the normalized output of the RMSNorm layer")
println("                  into a spatial tensor (Width, Height, Channels, Seq*Batch).")
println("  - Input Shape: ", size(out_rn))
println("  - Output Shape: ", size(out_stoch))
print(ANSI_RESET) # End color


# Combine the Deterministic and Stochastic parts =========================================
println(ANSI_GREEN, "\n----- 3. Combine the Deterministic and Stochastic parts -----", ANSI_RESET)

# Test combine -------------------------------------------------------------
# Input x has shape (W, H, C, Seq*Batch) = (6, 6, 8, 80)
x = out_deter + out_stoch
# Normalize over Channel dimension (dim=3)
# Scale parameter shape should match the normalized dim size: (C,) = (8,)
rn = RMSNorm((c,), act; dims=(3,), init_scale=cast_ones) # Corrected shape
ps, st = Lux.setup(rng, rn);
out_combine, st = rn(x, ps, st);
size(out_combine)

println(ANSI_BLUE, "RMSNorm layer (Combine Step)", ANSI_RESET)
print(ANSI_ORANGE) # Start color
println("  - Goal: Normalize the combined spatial features (deterministic + stochastic).")
println("  - Mechanism: Calculates the Root Mean Square (RMS) of the combined features")
println("               (across the channel dimension, dim=", rn.dims, "). Divides the features by their RMS.")
println("  - Purpose here: Stabilize the activations before feeding them into the")
println("                  main convolutional upsampling layers of the decoder.")
println("  - Input Shape (deter+stoch): ", size(x))
println("  - Output Shape: ", size(out_combine))
print(ANSI_RESET) # End color



# Apply decoder convolutions =========================================
println(ANSI_GREEN, "\n----- 4. Apply decoder convolutions -----", ANSI_RESET)

# Starting convolution layers ----------------------------------------------
x = out_combine; # Input shape (W, H, C, Seq*Batch)

# Test UpSample Layer ------------------------------------------------------
# Use default factor=2, dims=(1, 2) which matches the previous repeat logic
upsample_layer = UpSample(factor=2, dims=(1, 2))
ps_up, st_up = Lux.setup(rng, upsample_layer)
x_upsampled, st_up_new = upsample_layer(x, ps_up, st_up)
# apply upsampling (nearest neighbor via repeat)
# Repeat 2x along dim 1 (W) and dim 2 (H), 1x along others
# x_upsampled = repeat(x, inner=(2, 2, 1, 1)) # Replaced by UpSample layer test
println(ANSI_BLUE, "Upsampling step (using `UpSample` layer)", ANSI_RESET) # Updated title
print(ANSI_ORANGE) # Start color
println("  - Goal: Double the spatial dimensions (Width and Height) of the feature map.")
println("  - Mechanism: Uses the custom `UpSample(:nearest)` layer, which internally uses `Base.repeat`.")
println("               `inner=(2, 2, 1, 1)` means:")
println("                 - Dim 1 (Width): Repeat each element twice consecutively.")
println("                 - Dim 2 (Height): Repeat each element twice consecutively.")
println("                 - Dim 3 (Channels): Repeat each element once (no change).")
println("                 - Dim 4 (Seq*Batch): Repeat each element once (no change).")
println("               This effectively performs nearest-neighbor upsampling by 2x.")
println("  - Example (on a 2x2 slice):")
println("      Input: [a b]")
println("             [c d]")
println("      Output with inner=(2, 2):")
println("             [a a b b]")
println("             [a a b b]")
println("             [c c d d]")
println("             [c c d d]")
println("  - Purpose here: Increase the spatial resolution before applying the next convolution,")
println("                  part of the process to reconstruct the full-size image.")
println("  - Input Shape: ", size(x), " -> (W, H, C, Seq*Batch)")
println("  - Output Shape: ", size(x_upsampled), " -> (2W, 2H, C, Seq*Batch)")
print(ANSI_RESET) # End color

# Test Conv2D -------------------------------------------------------------
# Corresponds to first iteration of reversed loop:
# x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
# where depth is depths[-2]

depth_in = c # Channels from previous step
depth_out = depths[end-1] # Output channels for this conv layer
k_size = (kernel, kernel)
conv_layer = Conv(k_size, depth_in => depth_out; pad=SamePad(), init_weight=cast_glorot_uniform, init_bias=cast_zeros)

ps_conv, st_conv = Lux.setup(rng, conv_layer)
out_conv, st_conv_new = conv_layer(x_upsampled, ps_conv, st_conv)

println(ANSI_BLUE, "Conv2D Layer (Upsampling Path)", ANSI_RESET)
print(ANSI_ORANGE) # Start color
println("  - Goal: Apply convolution after upsampling.")
println("  - Mechanism: Standard 2D convolution with kernel size ", k_size, ", input channels ", depth_in, ", output channels ", depth_out, ".")
println("               `pad=SamePad()` ensures output spatial dimensions match input.")
println("  - Purpose here: Process the upsampled features, preparing for further upsampling or final output.")
println("  - Input Shape: ", size(x_upsampled), " -> (W, H, C_in, Seq*Batch)")
println("  - Output Shape: ", size(out_conv), " -> (W, H, C_out, Seq*Batch)")
print(ANSI_RESET) # End color

# Test Normalization -------------------------------------------------------------
#  x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))

norm_layer = RMSNorm((depth_out,), act; dims=(3,), init_scale=cast_ones)
ps_norm, st_norm = Lux.setup(rng, norm_layer)
out_norm, st_norm_new = norm_layer(out_conv, ps_norm, st_norm)

println(ANSI_BLUE, "RMSNorm Layer (After Conv)", ANSI_RESET)
print(ANSI_ORANGE) # Start color
println("  - Goal: Normalize the output of the Conv2D layer.")
println("  - Mechanism: Applies RMS Normalization across the channel dimension (dim=", norm_layer.dims, ").")
println("               Includes the activation function: ", norm_layer.activation)
println("  - Purpose here: Stabilize activations after convolution before the next upsampling/conv block.")
println("  - Input Shape: ", size(out_conv), " -> (W, H, C, Seq*Batch)")
println("  - Output Shape: ", size(out_norm), " -> (W, H, C, Seq*Batch)")
print(ANSI_RESET) # End color

println(ANSI_BLUE, "(Repeat Upsampling, Conv2D, Norm $(length(depths)-3) times)", ANSI_RESET)

# Test UpSample Layer ------------------------------------------------------
# Use default factor=2, dims=(1, 2) which matches the previous repeat logic
x = rand32(48, 48, 4, 80)
upsample_layer = UpSample()
ps_up, st_up = Lux.setup(rng, upsample_layer)
x_upsampled, st_up_new = upsample_layer(x, ps_up, st_up)
# apply upsampling (nearest neighbor via repeat)
# Repeat 2x along dim 1 (W) and dim 2 (H), 1x along others
# x_upsampled = repeat(x, inner=(2, 2, 1, 1)) # Replaced by UpSample layer test
println(ANSI_BLUE, "Upsampling step (using `UpSample` layer)", ANSI_RESET) # Updated title
print(ANSI_ORANGE) # Start color
println("  - Goal: Double the spatial dimensions (Width and Height) of the feature map.")
println("  - Purpose here: Increase the spatial resolution before applying the next convolution,")
println("                  part of the process to reconstruct the full-size image.")
println("  - Input Shape: ", size(x), " -> (W, H, C, Seq*Batch)")
println("  - Output Shape: ", size(x_upsampled), " -> (2W, 2H, C, Seq*Batch)")
print(ANSI_RESET) # End color


# Test Cond2D -------------------------------------------------------------
# x = x.repeat(2, -2).repeat(2, -3)
# kw = dict(**self.kw, outscale=self.outscale)
# print("outscale: ", self.outscale)
# print("x before conv2d: ", x.shape)
# x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
# print("x after conv2d: ", x.shape)
# outscale:  1.0
# x before conv2d:  (80, 96, 96, 4)
# x after conv2d:  (80, 96, 96, 1)
# x after reshape:  (8, 10, 96, 96, 1)
imgdep = obs.size[3] # Get target image channels (should be 1)
depth_in_final = size(x_upsampled, 3) # Get input channels from upsampled tensor (should be 4)
k_size_final = (kernel, kernel)

# Define the final convolution layer
final_conv_layer = Conv(k_size_final, depth_in_final => imgdep; pad=SamePad(), init_weight=cast_glorot_uniform, init_bias=cast_zeros)

# Setup and apply
ps_fconv, st_fconv = Lux.setup(rng, final_conv_layer)
out_fconv, st_fconv_new = final_conv_layer(x_upsampled, ps_fconv, st_fconv)

println(ANSI_BLUE, "Final Conv2D Layer", ANSI_RESET)
print(ANSI_ORANGE) # Start color
println("  - Goal: Map features to the final image channel dimension.")
println("  - Mechanism: Standard 2D convolution with kernel size ", k_size_final, ", input channels ", depth_in_final, ", output channels ", imgdep, ".")
println("               `pad=SamePad()` maintains spatial dimensions.")
println("  - Purpose here: Produce the final pre-activation output with the correct number of channels.")
println("  - Input Shape: ", size(x_upsampled), " -> (W, H, C_in, Seq*Batch)")
println("  - Output Shape: ", size(out_fconv), " -> (W, H, C_out, Seq*Batch)")
print(ANSI_RESET) # End color

println(ANSI_BLUE, "Apply sigmoid function", ANSI_RESET)
out_sigmoid = sigmoid.(out_fconv)

# Test ReArrange -----------------------------------------------------------
r = ReArrange((obs.size..., T, B))
ps, st = Lux.setup(rng, r);
out_rearrange, st = r(out_sigmoid, ps, st);

println(ANSI_BLUE, "ReArrange layer", ANSI_RESET)
print(ANSI_ORANGE) # Start color
println("  - Goal: Reshape the output tensor to separate the sequence (T) and batch (B) dimensions.")
println("  - Input Shape : ", size(out_sigmoid))
println("  - Output Shape: ", size(out_rearrange), " -> (W, H, C, T, B)")
println(ANSI_RESET) # End color


print("\n")


# Note: Python code applies sigmoid activation *after* this layer
# final_output = sigmoid.(out_fconv) # Example if needed later

