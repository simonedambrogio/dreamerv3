using YAML, Tools
config = YAML.load_file("dreamerv3/configs.yaml") |> symbolize;
include("atari.jl")
Atari(; name = "pong", config[:defaults][:env][:atari]...);

