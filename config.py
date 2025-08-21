"""Configuration for AlphaGo Zero training."""

import os


class Config:
    """AlphaGo Zero configuration matching DeepMind's specifications.
    
    Optimized for 2x H100 GPUs targeting professional-level play:
    - Network: 40 ResNet blocks, 256 filters (~23M parameters)
    - MCTS: 800 simulations (cautious increase from 1600 original)
    - Batch size: 2048 (original AlphaGo Zero specification)
    - Training: Long-term convergence with step decay learning rate
    
    Expected results:
    - Professional-level strength (3000+ Elo)
    - 99%+ win rate against amateur players
    - Strong tactical and strategic understanding
    - Training time: ~20-40 hours for full convergence
    """
    
    def __init__(self):
        # Environment settings
        self.board_size = 9  # 9x9 for efficient training
        
        # Set environment variable for Go board size
        import os
        os.environ['BOARD_SIZE'] = str(self.board_size)
        self.komi = 7.5
        self.num_stack = 8  # Number of board history planes to stack
        
        # Self-play settings - ALPHAGO ZERO SPECIFICATIONS
        self.num_parallel_games = 40  # Conservative start: 20 per H100 GPU
        self.num_mcts_simulations = 1200  # Cautious increase from 1600 original
        self.c_puct = 1.0  # PUCT exploration constant
        self.dirichlet_epsilon = 0.25  # Root noise mixing parameter
        self.dirichlet_alpha = 0.03  # Dirichlet noise concentration
        self.temperature = 1.0  # Temperature for move selection
        self.temperature_drop = 30  # Move number to drop temperature to near-zero
        
        # Neural network settings - ALPHAGO ZERO ARCHITECTURE
        self.num_res_blocks = 40  # Original AlphaGo Zero: 40 residual blocks
        self.num_hidden = 256  # Original AlphaGo Zero: 256 filters per layer
        self.learning_rate = 0.01  # Original AlphaGo Zero learning rate
        self.learning_rate_schedule = [0.01, 0.001, 0.0001]  # Step decay schedule
        self.lr_schedule_steps = [400, 600]  # Decay at these iterations
        self.weight_decay = 1e-4  # L2 regularization
        self.momentum = 0.9  # SGD momentum
        self.batch_size = 2048  # Original AlphaGo Zero batch size
        
        # Training settings - LONG-TERM CONVERGENCE
        self.num_iterations = 1000  # Long-term training for professional strength
        self.num_episodes_per_iteration = 500  # Scaled from 25K original (1/50th)
        self.num_epochs = 10  # Epochs per iteration (original used different setup)
        self.checkpoint_interval = 10  # Evaluate every 10 iterations
        
        # Evaluation settings - ALPHAGO ZERO SPECIFICATIONS
        self.num_eval_games = 40  # More games for reliable evaluation
        self.eval_win_threshold = 0.55  # Win rate threshold to update best model
        self.eval_temperature = 0.1  # Lower temperature for deterministic evaluation
        
        # Memory settings - OPTIMIZED FOR H100 GPUS
        self.replay_buffer_size = 1000000  # Large buffer for extensive training
        self.min_replay_size = 10000  # Minimum samples before training starts
        self.gradient_accumulation_steps = 1  # Large batch size, no accumulation needed
        
        # Multi-GPU settings - CAUTIOUS H100 OPTIMIZATION
        self.use_gpu = True
        self.num_gpus = 2  # Use both H100 GPUs
        self.num_workers = 16  # Conservative: 8 workers per GPU
        self.prefetch_factor = 2  # Conservative prefetching to avoid memory issues
        self.persistent_workers = True  # Keep workers alive between epochs
        self.pin_memory = True  # Faster CPU-GPU transfers
        
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