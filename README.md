# AlphaGo Zero Implementation

A comprehensive implementation of DeepMind's AlphaGo Zero algorithm with multi-GPU support, extensive logging, and real-time visualization.

## Overview

This project implements the AlphaGo Zero algorithm described in the 2017 DeepMind paper. The implementation includes:

- **Monte Carlo Tree Search (MCTS)** with neural network guidance
- **ResNet-based neural network** with policy and value heads  
- **Self-play training loop** with experience replay
- **Multi-GPU support** using PyTorch DataParallel
- **Comprehensive evaluation tools** and plotting utilities
- **Modular, extensible architecture**

## Features

- ✅ Complete AlphaGo Zero training pipeline
- ✅ Multi-GPU training support
- ✅ Experience replay with compression
- ✅ Configurable hyperparameters
- ✅ Interactive evaluation against random players
- ✅ SGF game record support
- ✅ Training visualization and plotting
- ✅ Proper package structure with type hints

## Requirements

- Python 3.8+
- PyTorch 1.12.0+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd alphago-zero
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPUs (optimized for 2x H100)
- Required packages: `pip install -r requirements.txt`

### Basic Usage
```bash
python train.py
```

## 📋 Configuration Parameters

All training parameters are defined in `config.py`. Below is a comprehensive documentation of each parameter:

### Environment Settings

| Parameter | Default | Description | Notes |
|-----------|---------|-------------|-------|
| `board_size` | 9 | Size of the Go board (9x9, 13x13, 19x19) | Start with 9x9 for faster training |
| `komi` | 7.5 | Compensation points for white player | Standard Go rule |
| `num_stack` | 8 | Number of board history planes to stack | AlphaGo Zero uses 8 |

### Self-Play Settings

| Parameter | Default | Description | Recommended for Production |
|-----------|---------|-------------|---------------------------|
| `num_parallel_games` | 500 | Number of parallel self-play games | Scale with available CPUs |
| `num_mcts_simulations` | 800 | MCTS simulations per move | AlphaGo Zero standard |
| `c_puct` | 1.0 | PUCT exploration constant | Balance exploration/exploitation |
| `dirichlet_epsilon` | 0.25 | Root noise mixing parameter | For exploration diversity |
| `dirichlet_alpha` | 0.03 | Dirichlet noise concentration | Lower = more diverse |
| `temperature` | 1.0 | Temperature for move selection | Higher = more random |
| `temperature_drop` | 30 | Move number to reduce temperature | Reduces randomness over time |

### Neural Network Architecture

| Parameter | Default | Description | Production Recommendation |
|-----------|---------|-------------|---------------------------|
| `num_res_blocks` | 20 | Number of residual blocks | 19-40 for full strength |
| `num_hidden` | 256 | Hidden units per residual block | 256 is AlphaGo Zero standard |
| `learning_rate` | 0.01 | Initial learning rate | Will decay automatically |
| `weight_decay` | 1e-4 | L2 regularization strength | Prevents overfitting |
| `momentum` | 0.9 | SGD momentum | Standard value |
| `batch_size` | 2048 | Training batch size | Optimized for H100 GPUs |

### Training Schedule

| Parameter | Default | Description | Time Estimate |
|-----------|---------|-------------|---------------|
| `num_iterations` | 100 | Total training iterations | ~10-12 hours on 2x H100 |
| `num_episodes_per_iteration` | 500 | Self-play games per iteration | Reduced for faster training |
| `num_epochs` | 100 | Training epochs per iteration | Balance quality vs speed |
| `checkpoint_interval` | 10 | Save/evaluate every N iterations | Frequent checkpoints |

### Evaluation Settings

| Parameter | Default | Description | Notes |
|-----------|---------|-------------|-------|
| `num_eval_games` | 40 | Games for model evaluation | Reduced for faster iteration |
| `eval_win_threshold` | 0.55 | Win rate to update best model | 55% threshold |
| `eval_temperature` | 0.1 | Lower temperature for evaluation | More deterministic play |

### Memory Management

| Parameter | Default | Description | Memory Usage |
|-----------|---------|-------------|--------------|
| `replay_buffer_size` | 500000 | Max positions in replay buffer | ~2-4GB RAM |
| `min_replay_size` | 10000 | Min positions before training | Reduced for quick start |

### Multi-GPU Configuration

| Parameter | Default | Description | H100 Optimization |
|-----------|---------|-------------|-------------------|
| `use_gpu` | True | Enable GPU acceleration | Required for training |
| `num_gpus` | 2 | Number of GPUs to use | Set to 2 for dual H100 |
| `num_workers` | 16 | Data loading workers | 8 per GPU recommended |

## 🏗️ Architecture Overview

### Model Architecture
- **Backbone**: ResNet with configurable depth (20 blocks default)
- **Policy Head**: Convolutional layers → Fully connected → Softmax
- **Value Head**: Convolutional layers → Fully connected → Tanh
- **Input**: 8-plane board history + 1 color plane

### Training Pipeline
1. **Self-Play**: Generate games using current best model
2. **Training**: Update neural network on self-play data
3. **Evaluation**: Test new model against previous best
4. **Update**: Replace best model if win rate > threshold

## 📊 Monitoring and Logging

### Real-time Metrics
- Loss curves (policy, value, total)
- Win rates vs different opponents
- Training efficiency (samples/sec)
- System utilization (GPU, memory)
- Game statistics (length, outcomes)

### Generated Plots
- `comprehensive_training_dashboard.png`: Overview of all metrics
- `epoch_losses_iter_X.png`: Per-epoch loss progression
- `training_progression_iter_X.png`: Smoothed loss curves
- `win_rate_progression.png`: Model strength over time
- `performance_vs_time.png`: Efficiency metrics

### Log Files
- `training_metrics.json`: Complete numerical data
- `training_report.md`: Human-readable progress report
- `game_histories.json`: Detailed game records

## 🎯 Performance Optimization

### H100 GPU Optimizations
- **Mixed Precision Training**: Automatic FP16/FP32 mixing
- **TensorFloat-32**: Enabled for faster matrix operations  
- **CUDA Optimizations**: Benchmark mode and memory optimization
- **DataParallel**: Efficient multi-GPU scaling

### Memory Optimizations
- **Gradient Scaling**: Prevents underflow in mixed precision
- **Replay Buffer Compression**: Reduces memory footprint
- **Efficient Data Loading**: Pin memory and multiple workers

## 📈 Expected Performance

### Training Timeline (2x H100, 10-12 hour training)
- **Hours 1-2**: Learning basic patterns, ~55% vs random
- **Hours 2-4**: Developing tactics, ~65% vs random  
- **Hours 4-8**: Strategic understanding, ~75% vs random
- **Hours 8-12**: Refined play, amateur level strength

### Resource Requirements
- **GPU Memory**: ~40GB per H100 (80GB total)
- **System RAM**: 64GB+ recommended
- **Storage**: ~50GB for logs and checkpoints
- **Training Time**: 10-12 hours total

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory
```python
# Reduce batch size in config.py
self.batch_size = 1024  # Instead of 2048
```

#### Slow Training
```python
# Increase number of workers
self.num_workers = 24  # If you have many CPU cores
```

#### Poor Convergence
```python
# Adjust learning rate schedule
self.learning_rate = 0.001  # Lower initial LR
```

### Hardware Requirements
- **Minimum**: 1x RTX 3090, 32GB RAM
- **Recommended**: 2x H100, 128GB RAM  
- **Optimal**: 4x H100, 256GB RAM

## 📚 Advanced Usage

### Custom Evaluation
```python
# Add your own evaluation opponents
from expert_evaluator import evaluate_model_vs_experts
results = evaluate_model_vs_experts(model, config, num_games=100)
```

### Hyperparameter Sweeps
```python
# Modify config for different experiments  
configs = [
    {'learning_rate': 0.01, 'num_res_blocks': 20},
    {'learning_rate': 0.001, 'num_res_blocks': 40},
]
```

### Multi-Node Training
```python
# For distributed training across multiple machines
# Use DistributedDataParallel instead of DataParallel
```

## 📊 Benchmarks

### Performance Targets
- **MCTS Speed**: >50 searches/second
- **Training Speed**: >1000 samples/second  
- **GPU Utilization**: >85%
- **Memory Efficiency**: <90% GPU memory usage

### Quality Metrics
- **vs Random**: >95% after 1 week
- **vs Amateur**: >70% after 1 month
- **vs Professional**: >55% after 3 months

## Architecture

### Core Components

- **`alpha_zero/core/network.py`**: ResNet neural network with policy and value heads
- **`alpha_zero/core/mcts.py`**: Monte Carlo Tree Search implementation  
- **`alpha_zero/core/replay.py`**: Experience replay buffer with compression
- **`alpha_zero/envs/go.py`**: Go game environment using OpenAI Gym API

### Training Pipeline

1. **Self-Play**: Generate games using current best model + MCTS
2. **Data Collection**: Store (state, policy, value) tuples in replay buffer
3. **Neural Network Training**: Train on collected self-play data
4. **Model Evaluation**: Test new model against previous best
5. **Model Update**: Replace best model if win rate > 55%

### Multi-GPU Support

The implementation automatically detects and uses multiple GPUs:

```python
# In train.py
if self.num_devices > 1:
    self.model = DataParallel(self.model)
```

Configure GPU usage in `config.py`:
```python
use_gpu = True
num_gpus = -1  # Use all available GPUs
```

## File Structure

```
alphago-zero/
├── alpha_zero/
│   ├── core/               # Core AlphaGo Zero components
│   │   ├── mcts.py        # Monte Carlo Tree Search
│   │   ├── network.py     # Neural network architecture
│   │   ├── replay.py      # Experience replay buffer
│   │   └── rating.py      # Elo rating system
│   ├── envs/              # Game environments  
│   │   ├── base.py        # Base board game environment
│   │   ├── go.py          # Go game implementation
│   │   └── go_engine.py   # Go rules engine (from Minigo)
│   ├── utils/             # Utility functions
│   │   ├── transformation.py  # Data augmentation
│   │   ├── util.py        # General utilities
│   │   └── sgf_wrapper.py # SGF file handling
│   └── eval_plays/        # Evaluation scripts
├── train.py               # Main training script
├── plot_go.py            # Plotting utilities
├── config.py             # Configuration
├── requirements.txt      # Dependencies
└── setup.py             # Package setup
```

## Training Tips

### For Faster Development:
- Start with 9x9 board size
- Use fewer MCTS simulations (25-50)
- Reduce neural network size (6-8 ResNet blocks)

### For Production Training:
- Use 19x19 board size
- Increase MCTS simulations (400-1600)
- Deeper neural networks (20+ ResNet blocks)
- More self-play games per iteration

### Multi-GPU Scaling:
- Each GPU can run ~20 self-play games in parallel
- Monitor GPU memory usage and adjust batch sizes
- Use distributed training for very large scale

## Known Limitations

1. **Self-play parallelization**: Currently sequential, could be improved with multiprocessing
2. **Dead stone detection**: Uses simplified Tromp-Taylor rules
3. **Opening book**: No opening book integration
4. **Time controls**: No time management system

## Future Improvements

- [ ] Parallel self-play with multiprocessing/distributed training
- [ ] More sophisticated evaluation metrics
- [ ] Integration with existing Go engines (GTP protocol)
- [ ] Support for different board sizes in single model
- [ ] Curriculum learning strategies

## References

- [Mastering the Game of Go without Human Knowledge (AlphaGo Zero paper)](https://www.nature.com/articles/nature24270)
- [Minigo Project](https://github.com/tensorflow/minigo) - Go rules engine
- [The Art of Reinforcement Learning](https://github.com/michaelnny/The-Art-of-Reinforcement-Learning) - Reference implementation

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.