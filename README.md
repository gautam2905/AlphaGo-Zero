# AlphaGo Zero Implementation

A clean implementation of DeepMind's AlphaGo Zero algorithm for 9x9 Go board.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Training
```bash
python train.py
```

### Play Against AI
```bash
python play_gui_working.py
```

## Structure

- `alpha_zero/` - Core AlphaGo Zero implementation
  - `core/` - MCTS, Neural Network, Replay Buffer
  - `envs/` - Go environment and GUI
  - `utils/` - Utilities and transformations
- `train.py` - Main training script
- `config.py` - Configuration settings
- `play_gui_working.py` - GUI interface to play against trained model

## Key Features

- Self-play reinforcement learning
- Monte Carlo Tree Search (MCTS)
- ResNet-based neural network
- Multi-GPU support
- Human vs AI gameplay interface

## Configuration

All parameters are in `config.py`:
- Board size: 9x9 (default)
- MCTS simulations: 800
- Neural network: 20 ResNet blocks
- Training iterations: 100

## Requirements

- Python 3.8+
- PyTorch 1.12.0+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for full dependencies.