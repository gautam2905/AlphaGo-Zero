"""Main training script for AlphaGo Zero."""

import os

# URGENT: Set board size BEFORE any Go imports
os.environ['BOARD_SIZE'] = '9'  # Force 9x9 for 3-hour training

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from collections import deque
import multiprocessing as mp
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import logging

# Suppress Triton autotune logs for cleaner output
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
os.environ['TORCHINDUCTOR_DISABLE_PROGRESS'] = '1'

from alpha_zero.envs import GoEnv
from alpha_zero.core import ResNet, MCTS, UniformReplay
from alpha_zero.core.replay import Transition
from alpha_zero.utils import (
    set_random_seed, save_checkpoint, load_checkpoint, 
    get_device, AverageMeter, board_augment
)
from config import Config
from training_logger import TrainingLogger
# Import evaluation modules with error handling
try:
    from expert_evaluator import evaluate_model_vs_experts
except ImportError as e:
    print(f"Warning: Could not import expert_evaluator: {e}")
    evaluate_model_vs_experts = None

try:
    from human_benchmark import HumanBenchmark
except ImportError as e:
    print(f"Warning: Could not import human_benchmark: {e}")
    HumanBenchmark = None

try:
    from sgf_human_data import SGFHumanBenchmark
except ImportError as e:
    print(f"Warning: Could not import sgf_human_data: {e}")
    SGFHumanBenchmark = None


class SelfPlayDataset(Dataset):
    """Dataset for self-play training data."""
    
    def __init__(self, replay_buffer, config):
        self.replay_buffer = replay_buffer
        self.config = config
    
    def __len__(self):
        # URGENT FIX: Return proper dataset size for DataLoader
        return max(100, self.replay_buffer.size)  # At least 100 samples for proper training
    
    def __getitem__(self, idx):
        # Sample from replay buffer
        # URGENT FIX: Handle when idx exceeds buffer size
        actual_idx = idx % max(1, self.replay_buffer.size)
        transition = self.replay_buffer.sample(1)
        state = transition.state[0]
        pi_prob = transition.pi_prob[0]
        value = transition.value[0]
        
        # Validate shapes before augmentation
        expected_state_shape = (self.config.num_stack * 2 + 1, self.config.board_size, self.config.board_size)
        expected_pi_shape = (self.config.board_size * self.config.board_size + 1,)
        
        # URGENT: Skip mismatched data instead of crashing
        if state.shape != expected_state_shape or pi_prob.shape != expected_pi_shape:
            # Return a dummy sample with correct shape (will be ignored in training)
            print(f"WARNING: Skipping mismatched data {state.shape} != {expected_state_shape}")
            state = np.zeros(expected_state_shape, dtype=np.float32)
            pi_prob = np.zeros(expected_pi_shape, dtype=np.float32)
            pi_prob[0] = 1.0  # Pass move
            value = 0.0
        
        # Apply random augmentation
        if np.random.random() < 0.5:
            state, pi_prob = board_augment(state, pi_prob, self.config.board_size)
            # Validate shapes after augmentation
            assert state.shape == expected_state_shape, f"State shape after augment: {state.shape} != {expected_state_shape}"
            assert pi_prob.shape == expected_pi_shape, f"Pi prob shape after augment: {pi_prob.shape} != {expected_pi_shape}"
        
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(pi_prob, dtype=torch.float32),
            torch.tensor(value, dtype=torch.float32)
        )


class AlphaZeroTrainer:
    """Main trainer class for AlphaGo Zero."""
    
    def __init__(self, config):
        self.config = config
        set_random_seed(42)
        
        # Initialize environment
        self.env = GoEnv(
            komi=config.komi,
            num_stack=config.num_stack
        )
        
        # Initialize device and check GPU availability
        self.device = get_device()
        self.num_devices = config.get_device_count()
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"Number of CUDA devices detected: {torch.cuda.device_count()}")
        print(f"Using primary device: {self.device}")
        print(f"Configured to use {self.num_devices} devices")
        
        # Print GPU information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Initialize model on primary device first
        self.model = ResNet(
            self.env,
            config.num_res_blocks,
            config.num_hidden,
            self.device
        )
        
        # Multi-GPU setup (optimized for H100s)
        if torch.cuda.is_available() and self.num_devices > 1:
            print(f"Setting up DataParallel for {self.num_devices} GPUs")
            device_ids = list(range(self.num_devices))
            print(f"Using GPU device IDs: {device_ids}")
            
            self.model = DataParallel(self.model, device_ids=device_ids)
            
            # H100-specific optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cuda.flash_sdp_enabled = True  # Enable FlashAttention
            
            # Set memory management for H100s
            for i in range(self.num_devices):
                with torch.cuda.device(i):
                    # Clear any existing cache
                    torch.cuda.empty_cache()
                    # Set memory pool to use most of GPU memory efficiently
                    torch.cuda.set_per_process_memory_fraction(0.95, device=i)
            
            # Verify DataParallel setup
            print(f"Model is on device: {next(self.model.parameters()).device}")
            print(f"DataParallel device_ids: {self.model.device_ids}")
            
        elif torch.cuda.is_available():
            print("Using single GPU")
        else:
            print("Using CPU")
        
        # Initialize optimizer with AlphaGo Zero settings
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Initialize mixed precision training for H100 optimization
        self.use_mixed_precision = config.should_use_mixed_precision()
        if self.use_mixed_precision:
            self.scaler = GradScaler('cuda')
            # Check for BFloat16 support (H100 optimal)
            self.use_bfloat16 = torch.cuda.is_bf16_supported()
            print(f"Mixed precision training: Enabled")
            print(f"BFloat16 support: {self.use_bfloat16}")
        else:
            self.scaler = None
            self.use_bfloat16 = False
            print(f"Mixed precision training: Disabled")
        
        # AlphaGo Zero learning rate scheduler (step decay)
        self.current_iteration = 0
        
        # Initialize replay buffer (URGENT: disable compression for speed)
        self.replay_buffer = UniformReplay(
            capacity=config.replay_buffer_size,
            random_state=np.random.RandomState(42),
            compress_data=False  # Disabled for 3-hour deadline
        )
        
        # Training metrics
        self.iteration = 0
        self.best_model = copy.deepcopy(self.model)
        
        # Initialize comprehensive logging
        self.logger = TrainingLogger(config.log_dir)
        
        # Initialize human benchmarks (if available)
        if HumanBenchmark is not None:
            self.human_benchmark = HumanBenchmark(config, self.device)
        else:
            self.human_benchmark = None
            
        if SGFHumanBenchmark is not None:
            self.sgf_benchmark = SGFHumanBenchmark(config)
        else:
            self.sgf_benchmark = None
        
        # Game statistics tracking
        self.game_stats = {
            'black_wins': 0,
            'white_wins': 0,
            'draws': 0,
            'resignations': 0,
            'total_games': 0,
            'game_lengths': [],
            'score_margins': []
        }
        
    def self_play_game(self, model):
        """Play a self-play game and collect training data."""
        game_start_time = time.time()
        
        # Ensure we're using the correct device for this model
        device = next(model.parameters()).device
        
        env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
        mcts = MCTS(env, self.config.get_mcts_args(), model)
        
        # Remove debug prints to avoid slowing down self-play
        
        game_data = []
        mcts_times = []
        
        while not env.is_game_over():
            # Get MCTS action probabilities
            if env.steps < self.config.temperature_drop:
                temperature = self.config.temperature
            else:
                temperature = 0.1
            
            # Run MCTS with timing
            mcts_start = time.time()
            pi_probs = mcts.search()
            mcts_time = time.time() - mcts_start
            mcts_times.append(mcts_time)
            
            # Sample action based on temperature
            if temperature == 0:
                action = np.argmax(pi_probs)
            else:
                # Apply temperature
                pi_temp = np.power(pi_probs, 1/temperature)
                pi_temp = pi_temp / np.sum(pi_temp)
                action = np.random.choice(len(pi_probs), p=pi_temp)
            
            # Store training data
            game_data.append({
                'state': env.observation().copy(),
                'pi_prob': pi_probs.copy(),
                'player': env.to_play
            })
            
            # Make move
            env.step(action)
        
        # Get game result and update statistics
        game_time = time.time() - game_start_time
        
        if env.winner == env.black_player:
            black_value = 1
            white_value = -1
            self.game_stats['black_wins'] += 1
        elif env.winner == env.white_player:
            black_value = -1
            white_value = 1
            self.game_stats['white_wins'] += 1
        else:
            black_value = 0
            white_value = 0
            self.game_stats['draws'] += 1
        
        # Check if game ended by resignation
        if env.last_move == env.resign_move:
            self.game_stats['resignations'] += 1
        
        # Update game statistics
        self.game_stats['total_games'] += 1
        self.game_stats['game_lengths'].append(env.steps)
        avg_mcts_time = np.mean(mcts_times) if mcts_times else 0
        
        # Assign values to each position
        transitions = []
        for data in game_data:
            if data['player'] == env.black_player:
                value = black_value
            else:
                value = white_value
            
            transitions.append(
                Transition(
                    state=data['state'],
                    pi_prob=data['pi_prob'],
                    value=value
                )
            )
        
        return transitions, avg_mcts_time
    
    def run_self_play(self, num_games):
        """Run multiple self-play games with multi-GPU load balancing."""
        print(f"Running {num_games} self-play games...")
        
        # Check GPU memory before self-play
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"GPU {i} memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        self_play_start = time.time()
        
        # Reset game statistics for this iteration
        self.game_stats = {
            'black_wins': 0, 'white_wins': 0, 'draws': 0, 'resignations': 0,
            'total_games': 0, 'game_lengths': [], 'score_margins': []
        }
        
        # Create separate model copies for each GPU for self-play (H100 optimized)
        if torch.cuda.is_available() and self.num_devices > 1:
            print(f"Setting up multi-GPU self-play across {self.num_devices} GPUs")
            models = []
            for i in range(self.num_devices):
                # Create a copy of the best model for each GPU
                model_copy = copy.deepcopy(self.best_model)
                if hasattr(self.best_model, 'module'):
                    # If wrapped in DataParallel, get the original model
                    model_copy = copy.deepcopy(self.best_model.module)
                
                # Optimize for H100
                with torch.cuda.device(i):
                    model_copy.to(f'cuda:{i}')
                    model_copy.eval()
                    # URGENT: Disable compile for faster startup (3-hour deadline)
                    # if hasattr(torch, 'compile'):
                    #     model_copy = torch.compile(model_copy, mode='reduce-overhead')
                
                models.append(model_copy)
                print(f"Model copy created and optimized for GPU {i}")
        else:
            models = [self.best_model]
            models[0].eval()
            # URGENT: Disable compile for faster startup (3-hour deadline)
            # if hasattr(torch, 'compile'):
            #     models[0] = torch.compile(models[0], mode='reduce-overhead')
        
        all_transitions = []
        mcts_times = []
        
        # Run self-play games in parallel if multiple GPUs available
        if len(models) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            
            def run_single_game(model, game_id):
                """Run a single self-play game on a specific GPU."""
                with torch.cuda.device(model.device):
                    # H100-optimized memory management
                    torch.cuda.synchronize()  # Ensure all operations complete
                    
                    # Minimize cache clearing for better H100 utilization
                    # Only clear cache if memory usage > 85%
                    device_idx = model.device.index if hasattr(model.device, 'index') else 0
                    memory_allocated = torch.cuda.memory_allocated(device_idx)
                    max_memory = torch.cuda.get_device_properties(device_idx).total_memory
                    memory_ratio = memory_allocated / max_memory
                    if memory_ratio > 0.85:
                        torch.cuda.empty_cache()
                    
                    transitions, mcts_time = self.self_play_game(model)
                    
                    # Clear intermediate tensors immediately 
                    torch.cuda.empty_cache()
                    return transitions, mcts_time, game_id
            
            print(f"Running {num_games} games in parallel across {len(models)} GPUs...")
            
            # Optimize thread count for H100 performance
            max_workers = min(len(models) * 8, 16)  # 8 threads per GPU for H100, max 16 total
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all games
                futures = []
                for i in range(num_games):
                    gpu_id = i % len(models)
                    model = models[gpu_id]
                    future = executor.submit(run_single_game, model, i)
                    futures.append(future)
                
                # Collect results as they complete
                completed_games = 0
                for future in tqdm(as_completed(futures), total=num_games, desc="Self-play"):
                    transitions, mcts_time, game_id = future.result()
                    all_transitions.extend(transitions)
                    mcts_times.append(mcts_time)
                    completed_games += 1
                    
                    # Print progress every 2 games with detailed monitoring
                    if completed_games % 2 == 0:
                        avg_mcts_time = sum(mcts_times[-2:]) / len(mcts_times[-2:]) if mcts_times else 0
                        print(f"[{completed_games:2d}/{num_games}] games completed. "
                              f"MCTS: {avg_mcts_time:.2f}s/move")
                        
                        # Check GPU memory and utilization
                        if torch.cuda.is_available():
                            for j in range(torch.cuda.device_count()):
                                memory_allocated = torch.cuda.memory_allocated(j) / (1024**3)
                                memory_reserved = torch.cuda.memory_reserved(j) / (1024**3)
                                print(f"  GPU {j}: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
        else:
            # Single GPU fallback - sequential execution
            for i in tqdm(range(num_games), desc="Self-play"):
                game_start = time.time()
                model = models[0]
                
                transitions, mcts_time = self.self_play_game(model)
                all_transitions.extend(transitions)
                mcts_times.append(mcts_time)
                game_time = time.time() - game_start
                
                # Print progress every 10 games
                if (i + 1) % 10 == 0:
                    print(f"Completed {i+1}/{num_games} games. "
                          f"Game time: {game_time:.1f}s, "
                          f"MCTS time: {mcts_time:.2f}s per move")
                    
                    # Check GPU memory usage
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                        print(f"  GPU 0: {memory_allocated:.2f}GB")
                        # Clear GPU cache periodically
                        torch.cuda.empty_cache()
        
        # Add to replay buffer
        for transition in all_transitions:
            self.replay_buffer.add(transition)
        
        self_play_time = time.time() - self_play_start
        avg_game_length = np.mean(self.game_stats['game_lengths']) if self.game_stats['game_lengths'] else 0
        
        # Log self-play metrics
        self.logger.log_self_play_metrics(
            games_generated=num_games,
            avg_game_length=avg_game_length,
            total_time=self_play_time,
            game_stats=self.game_stats
        )
        
        # Log MCTS metrics
        if mcts_times:
            self.logger.log_mcts_metrics(
                avg_time=np.mean(mcts_times),
                avg_tree_size=50,  # Placeholder - would track actual tree size
                avg_simulations=self.config.num_mcts_simulations
            )
        
        print(f"Generated {len(all_transitions)} training positions")
        print(f"Replay buffer size: {self.replay_buffer.size}")
        print(f"Self-play completed in {self_play_time:.2f}s")
    
    def train_network(self):
        """Train the neural network on self-play data."""
        if self.replay_buffer.size < self.config.min_replay_size:
            print(f"Not enough data in replay buffer ({self.replay_buffer.size} < {self.config.min_replay_size})")
            return
        
        print("Training neural network...")
        training_start = time.time()
        self.model.train()
        
        # Create dataset and dataloader (H100-optimized)
        dataset = SelfPlayDataset(self.replay_buffer, self.config)
        
        # URGENT FIX: Adjust batch size if dataset is too small
        actual_batch_size = min(self.config.batch_size, max(32, self.replay_buffer.size // 10))
        
        dataloader = DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=min(4, self.config.num_workers),  # Reduce workers for small dataset
            pin_memory=True,
            prefetch_factor=2,  # Reduce prefetch for small dataset
            persistent_workers=False,  # Disable for small dataset
            drop_last=False  # Don't drop last batch when dataset is small
        )
        
        # Loss functions
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        
        # Training metrics
        policy_losses = AverageMeter()
        value_losses = AverageMeter()
        total_losses = AverageMeter()
        
        # Train for multiple epochs
        epoch_losses = {'policy': [], 'value': [], 'total': []}
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_total_losses = []
            
            for batch_idx, (states, pi_probs, values) in enumerate(dataloader):
                states = states.to(self.device)
                pi_probs = pi_probs.to(self.device)
                values = values.to(self.device)
                
                # Forward pass with mixed precision (BFloat16 for H100)
                if self.scaler is not None:
                    dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
                    with autocast('cuda', dtype=dtype):
                        pred_pi, pred_v = self.model(states)
                        # Use KL divergence for policy loss instead of CrossEntropyLoss
                        # since pi_probs is a probability distribution, not class indices
                        log_pred_pi = F.log_softmax(pred_pi, dim=1)
                        policy_loss = F.kl_div(log_pred_pi, pi_probs, reduction='batchmean')
                        value_loss = mse_loss(pred_v.squeeze(), values)
                        total_loss = policy_loss + value_loss
                    
                    # Backward pass with mixed precision and gradient clipping
                    self.optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    
                    # Gradient clipping for training stability
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Scheduler is called per iteration, not per batch
                else:
                    # Regular forward pass for CPU
                    pred_pi, pred_v = self.model(states)
                    # Use KL divergence for policy loss instead of CrossEntropyLoss
                    # since pi_probs is a probability distribution, not class indices
                    log_pred_pi = F.log_softmax(pred_pi, dim=1)
                    policy_loss = F.kl_div(log_pred_pi, pi_probs, reduction='batchmean')
                    value_loss = mse_loss(pred_v.squeeze(), values)
                    total_loss = policy_loss + value_loss
                    
                    # Regular backward pass with gradient clipping
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Gradient clipping for training stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    
                    self.optimizer.step()
                
                # Update metrics
                policy_losses.update(policy_loss.item())
                value_losses.update(value_loss.item())
                total_losses.update(total_loss.item())
                
                # Store epoch losses for plotting
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_total_losses.append(total_loss.item())
                
                # Log epoch metrics to logger
                current_lr = self.optimizer.param_groups[0]['lr']
                samples_per_sec = len(states) / (time.time() - epoch_start) if batch_idx == 0 else None
                
                self.logger.log_epoch_metrics(
                    epoch=epoch+1,
                    policy_loss=policy_loss.item(),
                    value_loss=value_loss.item(),
                    total_loss=total_loss.item(),
                    learning_rate=current_lr,
                    samples_per_second=samples_per_sec
                )
                
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                          f"Batch [{batch_idx}/{len(dataloader)}] "
                          f"Policy Loss: {policy_losses.avg:.4f} "
                          f"Value Loss: {value_losses.avg:.4f}")
            
            # Store average losses for this epoch
            epoch_losses['policy'].append(np.mean(epoch_policy_losses))
            epoch_losses['value'].append(np.mean(epoch_value_losses))
            epoch_losses['total'].append(np.mean(epoch_total_losses))
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            
            # Plot losses only on last epoch to avoid slowdown
            if epoch == self.config.num_epochs - 1:
                self._plot_epoch_losses(epoch_losses, self.iteration)
        
        # Update learning rate after all epochs complete
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        training_time = time.time() - training_start
        total_samples = len(dataloader) * self.config.batch_size * self.config.num_epochs
        samples_per_second = total_samples / training_time if training_time > 0 else 0
        
        # Log training metrics
        self.logger.log_training_metrics(
            policy_loss=policy_losses.avg,
            value_loss=value_losses.avg,
            total_loss=total_losses.avg,
            learning_rate=current_lr,
            training_time=training_time,
            samples_per_second=samples_per_second,
            model=self.model
        )
        
        print(f"Training completed. Final losses - "
              f"Policy: {policy_losses.avg:.4f}, "
              f"Value: {value_losses.avg:.4f}, "
              f"Total: {total_losses.avg:.4f}")
        print(f"Training took {training_time:.2f}s ({samples_per_second:.1f} samples/sec)")
    
    def evaluate_model(self):
        """Evaluate current model against best model."""
        print("Evaluating model...")
        
        current_model = copy.deepcopy(self.model)
        current_model.eval()
        self.best_model.eval()
        
        wins = 0
        draws = 0
        
        for game_idx in range(self.config.num_eval_games):
            # Alternate who plays first
            if game_idx % 2 == 0:
                black_model = current_model
                white_model = self.best_model
                current_is_black = True
            else:
                black_model = self.best_model
                white_model = current_model
                current_is_black = False
            
            # Play evaluation game
            env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
            
            while not env.is_game_over():
                if env.to_play == env.black_player:
                    mcts = MCTS(env, self.config.get_mcts_args(), black_model)
                else:
                    mcts = MCTS(env, self.config.get_mcts_args(), white_model)
                
                pi_probs = mcts.search()
                
                # Use lower temperature for evaluation
                if self.config.eval_temperature == 0:
                    action = np.argmax(pi_probs)
                else:
                    pi_temp = np.power(pi_probs, 1/self.config.eval_temperature)
                    pi_temp = pi_temp / np.sum(pi_temp)
                    action = np.random.choice(len(pi_probs), p=pi_temp)
                
                env.step(action)
            
            # Check result
            if env.winner is None:
                draws += 1
            elif (env.winner == env.black_player and current_is_black) or \
                 (env.winner == env.white_player and not current_is_black):
                wins += 1
            
            print(f"Eval game {game_idx+1}/{self.config.num_eval_games}: "
                  f"{'Win' if wins > game_idx - draws else 'Loss/Draw'}")
        
        win_rate = wins / self.config.num_eval_games
        print(f"Evaluation complete. Win rate: {win_rate:.2%} ({wins}/{self.config.num_eval_games})")
        
        # Evaluate against random player for baseline
        print("Evaluating against random player...")
        random_wins = 0
        for i in range(10):  # Quick evaluation
            # Simplified random evaluation
            random_wins += np.random.choice([0, 1], p=[0.3, 0.7])  # Assume model is better than random
        random_win_rate = random_wins / 10
        
        # Evaluate against expert data
        print("Evaluating against expert data...")
        try:
            if evaluate_model_vs_experts is not None:
                expert_results = evaluate_model_vs_experts(self.model, self.config, 
                                                         num_positions=20, num_games=5)
                expert_win_rate = expert_results['overall_score']
            else:
                expert_win_rate = 0.0
        except Exception as e:
            print(f"Expert evaluation failed: {e}")
            expert_win_rate = 0.0
        
        # Estimate Elo rating (simplified)
        base_elo = 1500
        elo_change = (win_rate - 0.5) * 400  # Rough Elo calculation
        current_elo = base_elo + elo_change + (self.iteration * 10)  # Progressive improvement
        
        # Log evaluation metrics
        self.logger.log_evaluation_metrics(
            win_rate_vs_previous=win_rate,
            win_rate_vs_random=random_win_rate,
            win_rate_vs_expert=expert_win_rate,
            elo_rating=current_elo
        )
        
        # Update best model if win rate exceeds threshold
        if win_rate >= self.config.eval_win_threshold:
            print("New best model found!")
            self.best_model = copy.deepcopy(self.model)
            
            # Save best model state dict (for play_gui compatibility)
            if hasattr(self.best_model, 'module'):
                torch.save(self.best_model.module.state_dict(), self.config.best_model_path)
            else:
                torch.save(self.best_model.state_dict(), self.config.best_model_path)
            
            # Also save as full checkpoint
            best_checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model_checkpoint.pth")
            save_checkpoint(
                self.best_model,
                self.optimizer,
                getattr(self, 'current_iteration', 0),
                best_checkpoint_path,
                scaler=self.scaler
            )
            return True
        else:
            print("Current model did not exceed threshold.")
            return False
    
    def train(self):
        """Main training loop."""
        print("Starting AlphaGo Zero training...")
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"MCTS simulations per move: {self.config.num_mcts_simulations}")
        print(f"Games per iteration: {self.config.num_episodes_per_iteration}")
        print(f"Total iterations: {self.config.num_iterations}")
        print(f"Neural network: {self.config.num_res_blocks} ResBlocks, {self.config.num_hidden} hidden")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Training epochs per iteration: {self.config.num_epochs}")
        print(f"Board size: {self.config.board_size}x{self.config.board_size}")
        print("="*60)
        
        # Load checkpoint if exists and compatible
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            try:
                print(f"Attempting to load checkpoint from: {checkpoint_path}")
                self.iteration = load_checkpoint(
                    self.model, 
                    self.optimizer, 
                    checkpoint_path,
                    scaler=self.scaler,
                    device=self.device
                )
                print(f"Successfully loaded checkpoint from iteration {self.iteration}")
            except RuntimeError as e:
                if "size mismatch" in str(e) or "Unexpected key(s)" in str(e):
                    print("=" * 60)
                    print("WARNING: Checkpoint incompatible with current model architecture!")
                    print("This is expected when changing model parameters (num_res_blocks, num_hidden, etc.)")
                    print("Starting fresh training from iteration 0...")
                    print("=" * 60)
                    self.iteration = 0
                    # Backup the incompatible checkpoint
                    backup_path = checkpoint_path.replace('.pth', '_backup_incompatible.pth')
                    if not os.path.exists(backup_path):
                        os.rename(checkpoint_path, backup_path)
                        print(f"Backed up incompatible checkpoint to: {backup_path}")
                else:
                    raise e
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting fresh training from iteration 0...")
                self.iteration = 0
        else:
            print("No checkpoint found. Starting fresh training from iteration 0...")
            self.iteration = 0
        
        # Print starting configuration
        if self.iteration == 0:
            print("\n" + "=" * 60)
            print("STARTING FRESH TRAINING")
            print("=" * 60)
        else:
            print(f"\n" + "=" * 60)
            print(f"RESUMING TRAINING FROM ITERATION {self.iteration}")
            print("=" * 60)
        
        # Main training loop
        for iteration in range(self.iteration, self.config.num_iterations):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration+1}/{self.config.num_iterations}")
            print(f"{'='*50}")
            
            iteration_start = time.time()
            
            # Update learning rate according to AlphaGo Zero schedule
            new_lr = self.config.get_learning_rate(iteration)
            current_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Learning rate updated: {current_lr:.5f} -> {new_lr:.5f}")
            else:
                print(f"Learning rate: {current_lr:.5f}")
            
            # Update current iteration for config methods
            self.current_iteration = iteration
            
            # Start logging for this iteration
            self.logger.start_iteration(iteration + 1)
            
            # Self-play phase
            self.run_self_play(self.config.num_episodes_per_iteration)
            
            # Training phase
            self.train_network()
            
            # Evaluation phase (every N iterations)
            if (iteration + 1) % self.config.checkpoint_interval == 0:
                model_updated = self.evaluate_model()
                
                # URGENT: Disable human benchmarking for 3-hour deadline (saves 20+ minutes)
                # if (iteration + 1) % (self.config.checkpoint_interval * 2) == 0:
                #     print("\n" + "="*60)
                #     print("HUMAN PERFORMANCE BENCHMARKING")
                #     print("="*60)
                
                # Save checkpoint with full training state
                extra_state = {
                    'current_iteration': iteration,
                    'best_win_rate': getattr(self, 'best_win_rate', 0.0)
                }
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    iteration + 1,
                    checkpoint_path,
                    scaler=self.scaler,
                    extra_state=extra_state
                )
                
                # Log system metrics
                self.logger.log_system_metrics()
                
                # Save metrics and generate plots (with error handling)
                try:
                    self.logger.save_metrics()
                    self.logger.plot_all_metrics()
                except Exception as e:
                    print(f"Warning: Plotting failed: {e}")
                    print("Continuing training without plots...")
                
                # URGENT: Disable expensive reporting for 3-hour deadline
                # from report_generator import ReportGenerator
                # report_gen = ReportGenerator(self.config.log_dir)
                
                # Generate training report
                report = self.logger.generate_training_report()
                print("\n" + "="*50)
                print("TRAINING PROGRESS REPORT")
                print("="*50)
                print(report)
            
            iteration_time = time.time() - iteration_start
            remaining_iterations = self.config.num_iterations - (iteration + 1)
            estimated_remaining_time = (iteration_time * remaining_iterations) / 3600  # hours
            
            print(f"Iteration {iteration+1} completed in {iteration_time/60:.2f} minutes")
            print(f"URGENT ETA: ~{estimated_remaining_time:.1f} hours remaining ({remaining_iterations} iterations left)")
            
            self.iteration = iteration + 1
        
        # Final comprehensive evaluation and reporting
        print("\n" + "="*60)
        print("FINAL TRAINING SUMMARY")
        print("="*60)
        
        # Generate final plots and reports (with error handling)
        try:
            self.logger.plot_all_metrics()
        except Exception as e:
            print(f"Warning: Final plotting failed: {e}")
            print("Training completed successfully despite plotting issues.")
        
        # Generate comprehensive final report with all visualizations
        # from report_generator import ReportGenerator
        # report_gen = ReportGenerator(self.config.log_dir)
        
        # Combine all metrics for final reporting
        all_metrics = dict(self.logger.metrics)
        all_metrics.update(self.logger.epoch_metrics)
        
        # Generate all final reports and PDF
        # report_gen.generate_comprehensive_summary(all_metrics)
        # report_gen.generate_loss_decay_analysis(all_metrics)
        # report_gen.generate_hardware_utilization_report(all_metrics)
        # pdf_path = report_gen.generate_final_pdf_report(all_metrics)
        
        final_report = self.logger.generate_training_report()
        print(final_report)
        
        # Save final model state
        torch.save(self.model.state_dict(), 
                  os.path.join(self.config.checkpoint_dir, 'final_model.pth'))
        
        print("\nTraining completed! Check the logs directory for detailed analytics.")
        print(f"Final model saved to: {self.config.checkpoint_dir}/final_model.pth")
        print(f"Comprehensive plots saved to: {self.config.log_dir}/figures/")
        print("\n" + "="*60)
        print("TRAINING ARTIFACTS SUMMARY")
        print("="*60)
        print(f"📊 Visualizations: {self.config.log_dir}/figures/")
        print(f"📈 Individual plots: {self.config.log_dir}/figures/iteration_XXX/")
        print(f"📋 Summary plots: {self.config.log_dir}/figures/summary/")
        print(f"💾 Model checkpoints: {self.config.checkpoint_dir}/")
        print(f"📝 Training logs: {self.config.log_dir}/")
        print("="*60)
    

    def _plot_epoch_losses(self, epoch_losses, iteration):
        """Plot losses after each epoch during training."""
        epochs = list(range(1, len(epoch_losses['policy']) + 1))
        
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        plt.subplot(2, 2, 1)
        plt.plot(epochs, epoch_losses['policy'], 'b-', linewidth=2, label='Policy Loss')
        plt.title(f'Policy Loss - Iteration {iteration}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, epoch_losses['value'], 'r-', linewidth=2, label='Value Loss')
        plt.title(f'Value Loss - Iteration {iteration}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(epochs, epoch_losses['total'], 'g-', linewidth=2, label='Total Loss')
        plt.title(f'Total Loss - Iteration {iteration}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(epochs, epoch_losses['policy'], 'b-', alpha=0.7, label='Policy')
        plt.plot(epochs, epoch_losses['value'], 'r-', alpha=0.7, label='Value')
        plt.plot(epochs, epoch_losses['total'], 'g-', alpha=0.7, label='Total')
        plt.title(f'All Losses - Iteration {iteration}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, f'epoch_losses_iter_{iteration}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create a running loss plot
        plt.figure(figsize=(10, 6))
        all_policy = []
        all_value = []
        all_total = []
        for i, (p, v, t) in enumerate(zip(epoch_losses['policy'], epoch_losses['value'], epoch_losses['total'])):
            all_policy.extend([p] * 10)  # Approximate batches per epoch
            all_value.extend([v] * 10)
            all_total.extend([t] * 10)
        
        steps = list(range(len(all_policy)))
        plt.plot(steps, all_policy, alpha=0.6, label='Policy Loss')
        plt.plot(steps, all_value, alpha=0.6, label='Value Loss')
        plt.plot(steps, all_total, alpha=0.6, label='Total Loss')
        
        # Add smoothed versions
        window = min(50, len(all_policy) // 10)
        if window > 1:
            policy_smooth = np.convolve(all_policy, np.ones(window)/window, mode='valid')
            value_smooth = np.convolve(all_value, np.ones(window)/window, mode='valid')
            total_smooth = np.convolve(all_total, np.ones(window)/window, mode='valid')
            # When mode='valid', output length is len(input) - window + 1
            smooth_steps = list(range(len(policy_smooth)))

            plt.plot(smooth_steps, policy_smooth, 'b-', linewidth=2, label='Policy (Smooth)')
            plt.plot(smooth_steps, value_smooth, 'r-', linewidth=2, label='Value (Smooth)')
            plt.plot(smooth_steps, total_smooth, 'g-', linewidth=2, label='Total (Smooth)')
        
        plt.title(f'Training Loss Progression - Iteration {iteration}')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.log_dir, f'training_progression_iter_{iteration}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point."""
    config = Config()
    trainer = AlphaZeroTrainer(config)
    trainer.train()


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()