"""Debug training script to test if the code is working with small parameters."""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from collections import deque
import multiprocessing as mp
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

from alpha_zero.envs import GoEnv
from alpha_zero.core import ResNet, MCTS, UniformReplay
from alpha_zero.core.replay import Transition
from alpha_zero.utils import (
    set_random_seed, save_checkpoint, load_checkpoint, 
    get_device, AverageMeter, board_augment
)
from debug_config import DebugConfig


def main():
    """Debug main function with minimal parameters."""
    print("="*60)
    print("DEBUG MODE - Testing AlphaGo Zero Implementation")
    print("="*60)
    
    config = DebugConfig()
    
    # Print debug configuration
    print(f"MCTS simulations per move: {config.num_mcts_simulations}")
    print(f"Games per iteration: {config.num_episodes_per_iteration}")
    print(f"Total iterations: {config.num_iterations}")
    print(f"Neural network: {config.num_res_blocks} ResBlocks, {config.num_hidden} hidden")
    print(f"Batch size: {config.batch_size}")
    print("="*60)
    
    # Initialize environment
    env = GoEnv(komi=config.komi, num_stack=config.num_stack)
    print(f"Environment initialized: {config.board_size}x{config.board_size} board")
    
    # Initialize device
    device = get_device()
    num_devices = config.get_device_count()
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Number of CUDA devices detected: {torch.cuda.device_count()}")
    print(f"Using primary device: {device}")
    print(f"Configured to use {num_devices} devices")
    
    # Print GPU information
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Initialize model
    model = ResNet(env, config.num_res_blocks, config.num_hidden, device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Multi-GPU setup
    if torch.cuda.is_available() and num_devices > 1:
        print(f"Setting up DataParallel for {num_devices} GPUs")
        device_ids = list(range(num_devices))
        model = DataParallel(model, device_ids=device_ids)
        print(f"DataParallel setup complete")
    
    # Test single forward pass
    print("\nTesting neural network...")
    test_input = torch.randn(1, env.observation_space.shape[0], 
                           config.board_size, config.board_size).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        policy, value = model(test_input)
        forward_time = time.time() - start_time
        
    print(f"Forward pass successful: {forward_time:.4f}s")
    print(f"Policy shape: {policy.shape}, Value shape: {value.shape}")
    
    # Test MCTS
    print("\nTesting MCTS...")
    mcts = MCTS(env, config.get_mcts_args(), model)
    
    start_time = time.time()
    pi_probs = mcts.search()
    mcts_time = time.time() - start_time
    
    print(f"MCTS search successful: {mcts_time:.2f}s for {config.num_mcts_simulations} simulations")
    print(f"Policy probabilities shape: {pi_probs.shape}")
    
    # Test one complete game
    print("\nTesting complete self-play game...")
    game_start = time.time()
    
    # Reset environment
    env = GoEnv(komi=config.komi, num_stack=config.num_stack)
    mcts = MCTS(env, config.get_mcts_args(), model)
    
    moves = 0
    while not env.is_game_over() and moves < 20:  # Limit to 20 moves for debug
        pi_probs = mcts.search()
        action = np.argmax(pi_probs)
        env.step(action)
        moves += 1
    
    game_time = time.time() - game_start
    print(f"Game completed: {moves} moves in {game_time:.2f}s")
    print(f"Average time per move: {game_time/moves:.2f}s")
    
    # Check GPU memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"GPU {i} memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
    print("\n" + "="*60)
    print("DEBUG TEST COMPLETED SUCCESSFULLY!")
    print("The code is working. Main bottleneck is MCTS simulations.")
    print("For full training, consider reducing num_mcts_simulations.")
    print("="*60)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()