"""Configuration for AlphaGo Zero training."""

import os


class Config:
    """AlphaGo Zero Production Configuration - 19x19 Board.
    
    Production-ready configuration for professional-level AlphaGo Zero:
    - Full 19x19 Go board (competition standard)
    - Network: 40 ResNet blocks, 256 filters (~23M parameters)
    - MCTS: 800 simulations per move (original paper specification)
    - Batch size: 2048 (original AlphaGo Zero specification)
    - Optimized for dual H100 GPUs with actor-learner architecture
    
    Expected training profile:
    - 150-1000 iterations for professional strength
    - Progressive improvement with self-play
    - Efficient GPU utilization (>90%)
    """
    
    def __init__(self):
        # Environment settings - PRODUCTION 19x19
        self.board_size = 19  # Full 19x19 board for production
        
        # Set environment variable for Go board size
        import os
        os.environ['BOARD_SIZE'] = str(self.board_size)
        self.komi = 7.5  # Standard komi for 19x19
        self.num_stack = 8  # 8 board history planes (original paper)
        
        # MCTS settings - EXACT ALPHAGO ZERO PAPER PARAMETERS
        self.num_mcts_simulations = 1600  # Paper: 1600 simulations per move
        self.c_puct = 1.0  # Paper mentions PUCT constant
        self.c_puct_base = 19652  # UCT exploration base
        self.c_puct_init = 1.25   # UCT exploration init
        self.dirichlet_epsilon = 0.25  # Paper: ε=0.25 for root noise
        self.dirichlet_alpha = 0.03  # Paper: α=0.03 for 19x19 Go  
        self.temperature = 1.0  # Paper: τ=1 for first 30 moves
        self.temperature_drop = 30  # Paper: drop to τ≈0 after move 30
        
        # Self-play settings - PRODUCTION SCALE
        self.num_parallel_games = 100  # Parallel games for efficiency
        self.num_actors_per_gpu = 50  # Actors per GPU
        
        # Neural network settings - EXACT ALPHAGO ZERO PAPER
        self.num_res_blocks = 20  # Paper: either 20 or 40 blocks (20 for faster training)
        self.num_hidden = 256     # Paper: 256 filters in conv layers
        self.learning_rate = 0.2  # Paper: initial learning rate 0.2
        self.learning_rate_schedule = [0.2, 0.02, 0.002, 0.0002]  # Paper schedule
        self.lr_schedule_steps = [100000, 200000, 300000]  # Paper: decay at 100k, 200k, 300k
        self.weight_decay = 1e-4  # Paper: L2 weight regularization c=10^-4
        self.momentum = 0.9       # Paper: SGD with momentum 0.9
        self.batch_size = 32      # Paper: mini-batch size of 32
        
        # Training settings - EXACT ALPHAGO ZERO PAPER
        self.num_iterations = 700000  # Paper: 700k iterations total
        self.num_episodes_per_iteration = 25000  # Paper: 25,000 self-play games per iteration
        self.num_epochs = 1000  # Paper: sample positions for training
        self.checkpoint_interval = 100  # Save more frequently for server crashes (was 1000)
        self.save_interval = 10  # Auto-save every 10 iterations for safety
        
        # Evaluation settings - EXACT ORIGINAL PAPER
        self.num_eval_games = 400  # Original: 400 games for evaluation
        self.eval_win_threshold = 0.55  # Original: 55% win rate to update
        self.eval_temperature = 0.1  # Original: low temperature for evaluation
        
        # RESIGNATION - EXACT ORIGINAL PAPER  
        self.resign_threshold = -0.9  # Original: resign when value < -0.9
        self.resign_disabled_ratio = 0.1  # Original: 10% games without resignation
        
        # Memory settings - EXACT ALPHAGO ZERO PAPER
        self.replay_buffer_size = 500000  # Paper: 500,000 most recent self-play positions
        self.min_replay_size = 10000  # Start training after initial games
        self.gradient_accumulation_steps = 1  # No accumulation with small batch size
        
        # Multi-GPU settings - OPTIMIZED FOR 2x H100 90GB
        self.use_gpu = True
        self.num_gpus = 2  # Use both H100 90GB GPUs
        self.use_bfloat16 = True        # H100 native BF16 precision
        self.use_mixed_precision = True  # Enable autocast for tensor cores
        self.compile_model = True       # PyTorch 2.0 compilation for H100
        self.gpu_memory_fraction = 0.95  # Can use more with 90GB VRAM
        self.num_workers = 32  # 16 workers per H100 GPU
        self.prefetch_factor = 8  # More prefetching with 90GB memory
        self.persistent_workers = True  # Keep workers alive between epochs
        self.pin_memory = True  # Faster CPU-GPU transfers with NVLink
        
        # Multiprocessing settings for self-play
        self.use_multiprocessing = True  # Enable true multiprocessing for self-play
        self.max_processes = 32  # Maximum concurrent processes (16 per GPU)
        
        # Actor-Learner Pipeline settings - OPTIMIZED FOR 2x H100 90GB
        self.num_actors = 16      # 16 actors across 2 H100s (8 per GPU)
        self.actors_per_gpu = 8   # 8 actors per H100 GPU
        self.mcts_batch_size = 16  # Larger batch for H100 tensor cores
        self.num_parallel_mcts = 16  # More parallel MCTS for H100
        self.max_game_length = 722  # Paper: games truncated at 722 moves (19×19×2)
        self.total_games = 25000  # Paper: 25,000 games per iteration
        self.checkpoint_path = None  # Initial checkpoint to load
        
        # Mixed precision training for H100
        self.use_mixed_precision = True  # Enable FP16 training
        self.gradient_clip_norm = 1.0  # Gradient clipping for stability
        
        # Paths - All data saved to current D drive directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory on D drive
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.data_dir = os.path.join(self.base_dir, "data")  # For training data
        self.temp_dir = os.path.join(self.base_dir, "temp")   # For temporary files
        self.expert_games_dir = os.path.join(self.base_dir, "expert_games")
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        
        # Create all directories on D drive
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.expert_games_dir, exist_ok=True)
        
        # Set environment variables to use D drive for caching
        os.environ['TORCH_HOME'] = os.path.join(self.base_dir, 'torch_cache')
        os.environ['TMPDIR'] = self.temp_dir
        os.environ['TEMP'] = self.temp_dir
        os.environ['TMP'] = self.temp_dir
    
    def get_mcts_args(self):
        """Get arguments for MCTS."""
        return {
            'num_searches': self.num_mcts_simulations,
            'C': self.c_puct,
            'dirichlet_epsilon': self.dirichlet_epsilon,
            'dirichlet_alpha': self.dirichlet_alpha,
        }
    
    def get_device_count(self):
        """Get number of devices to use."""
        import torch
        if not self.use_gpu or not torch.cuda.is_available():
            return 1
        
        available_gpus = torch.cuda.device_count()
        if self.num_gpus == -1:
            return available_gpus
        else:
            return min(self.num_gpus, available_gpus)
    
    def get_learning_rate(self, iteration):
        """Get learning rate for current iteration using step decay schedule."""
        for i, step in enumerate(self.lr_schedule_steps):
            if iteration < step:
                return self.learning_rate_schedule[i]
        # If past all schedule steps, use the last learning rate
        return self.learning_rate_schedule[-1]
    
    def should_use_mixed_precision(self):
        """Check if mixed precision should be used based on GPU capabilities."""
        import torch
        if not self.use_mixed_precision:
            return False
        
        if not torch.cuda.is_available():
            return False
            
        # Check for H100 or other Ampere/Hopper GPUs that benefit from mixed precision
        gpu_name = torch.cuda.get_device_name(0)
        return any(gpu_type in gpu_name.upper() for gpu_type in 
                  ['H100', 'A100', 'RTX 30', 'RTX 40', 'V100', 'T4'])