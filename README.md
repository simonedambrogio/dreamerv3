# DreamerV3 in Julia/Lux.jl

A Julia implementation of DreamerV3, the world model-based reinforcement learning agent from ["Mastering Diverse Domains through World Models"](https://arxiv.org/abs/2301.04104) (Hafner et al., 2023).

> **Status:** 🚧 In active development. Core world model components are functional, but the full agent pipeline is still being built out.

## What is DreamerV3?

DreamerV3 is a reinforcement learning algorithm that learns to predict the future by building a world model. Instead of learning directly from environment interactions, it learns a compressed representation of the world and uses this model to imagine future scenarios, making it highly sample-efficient across diverse domains like Atari games, robotic control, and more.

## What's Implemented

This repository implements the **world model** portion of DreamerV3 in Julia using the [Lux.jl](https://github.com/LuxDL/Lux.jl) framework for neural networks. Here's what's currently working:

### Core Components
- **Encoder**: CNN-based visual encoder that compresses observations into latent representations
- **RSSM (Recurrent State-Space Model)**: The dynamics model that predicts how states evolve over time, combining deterministic and stochastic components
- **Decoder**: Deconvolutional network that reconstructs observations from latent states
- **World Model**: Integration of encoder, RSSM, and decoder with training losses

### Custom Lux Layers
- `RMSNorm`: Root Mean Square normalization layer
- `BlockLinear`: Sparse block-structured linear layers for efficient computation
- `ReArrange`: Tensor reshaping layer
- `UpSample`: Upsampling layer for the decoder
- `MultiLinear`: Multiple parallel linear transformations

### Infrastructure
- Atari environment integration via ArcadeLearningEnvironment.jl
- Custom visual generalization environments
- Replay buffer for experience storage
- Training loops with GPU support (CUDA)
- Weights & Biases integration for experiment tracking

## Project Structure

```
dreamerv3/          # Core world model implementation
  ├── encoder.jl    # Visual encoder
  ├── rssm.jl       # Recurrent State-Space Model
  ├── decoder.jl    # Image reconstruction decoder
  └── WorldModel.jl # Combined model and loss functions

embodied/           # Infrastructure and environments
  ├── lux/          # Custom Lux layers
  ├── envs/         # Environment wrappers (Atari, custom)
  └── core/         # Replay buffer and training utilities

test/               # Development and testing scripts
```

## Dependencies

Key packages used in this project:
- [Lux.jl](https://github.com/LuxDL/Lux.jl) - Neural network framework
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) - GPU acceleration
- [ArcadeLearningEnvironment.jl](https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl) - Atari environments
- [Zygote.jl](https://github.com/FluxML/Zygote.jl) - Automatic differentiation
- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) - Optimization algorithms

See `Project.toml` for the complete dependency list.


## Development Notes

This is a learning project and research implementation. The goal is to understand DreamerV3 deeply by rebuilding it in Julia, taking advantage of Julia's performance and composability. The implementation stays close to the original Python version for validation purposes, with saved checkpoints and training logs available in the `logs/` and `wandb/` directories.

## Citation

If you use this work or find it helpful, please cite the original DreamerV3 paper:

```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```

---

*This is an independent implementation and is not affiliated with the original authors.*
