using YAML, Tools
config = YAML.load_file("dreamerv3/configs.yaml") |> symbolize;
include("atari.jl")
env = Atari(; name = "pong", config[:defaults][:env][:atari]...);


using GLMakie

function plot(env::Atari, action::Int)
    w, h = getScreenWidth(env.ale), getScreenHeight(env.ale)
    fig = Figure(size = (w*4, h*4))
    ax = GLMakie.Axis(fig[1,1])
    obs = step!(env, Dict("action" => action, "reset" => false))
    GLMakie.heatmap!(ax, dropdims(obs["image"] ./ 255, dims = 3), colormap = :gray1)
    display(fig)
end


function plot(env::Atari)
    img = env.buffers[1] ./ 255
    rgb_image = [RGB(img[i,j,:]...) for i in axes(img,1), j in axes(img,2)]
    f = Figure()
    ax = GLMakie.Axis(f[1,1], aspect = DataAspect())
    GLMakie.image!(ax, rgb_image)  # Removed rotr90
    display(f)
end



# 1. Create environment with autostart=true
env = Atari(; name="pong", config[:defaults][:env][:atari]..., autostart=false);

# 2. Or manually perform the start sequence:
# First reset
step!(env, Dict("action" => 1, "reset" => true))

# Then FIRE to start the game
step!(env, Dict("action" => findfirst(==("FIRE"), env.ACTION_MEANING), "reset" => false))

# Then optionally UP to move paddle
step!(env, Dict("action" => findfirst(==("UP"), env.ACTION_MEANING), "reset" => false))

step!(env, Dict("action" => findfirst(==("UP"), env.ACTION_MEANING), "reset" => false))

# Display
plot(env)
plot(env, 1)

step!(env, Dict("action" => findfirst(==("UP"), env.ACTION_MEANING), "reset" => false))


using ArcadeLearningEnvironment

getROMList()

episodes = 50

ale = ALE_new()
loadROM(ale, "seaquest")

S = zeros(Int64, episodes)
TR = zeros(episodes)
for ei = 1:episodes
    ctr = 0.0

    fc = 0
    while game_over(ale) == false
        actions = getLegalActionSet(ale)
        ctr += act(ale, actions[rand(1:length(actions))])
        fc += 1
    end
    reset_game(ale)
    println("Game $ei ended after $fc frames with total reward $(ctr).")

    S[ei] = fc
    TR[ei] = ctr
end
ALE_del(ale)



function plot(ale::Ptr)
    w, h = getScreenWidth(ale), getScreenHeight(ale)
    screen_data = reshape(getScreenRGB(ale), w, h, 3)  # WHC format directly

    # Convert UInt8 RGB data to proper color format
    im = permutedims(screen_data, (2,1,3)) ./ 255  # Normalize to 0-1
    rgb_image = [RGB(im[i,j,:]...) for i in axes(im,1), j in axes(im,2)]
    f = Figure(size = (w*4, h*4))
    ax = GLMakie.Axis(f[1,1], aspect = DataAspect())
    GLMakie.image!(ax, rotr90(rgb_image))  # Removed rotr90
    display(f)
end;

plot(ale)