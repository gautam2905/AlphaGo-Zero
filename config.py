"""Configuration for AlphaGo Zero training."""

import os


class Config:
    """AlphaGo Zero configuration.
    
    Optimized for 10-12 hour training on 2x H100 GPUs with practical parameters.
    
    Key reductions from original AlphaGo Zero for single-machine training:
    - MCTS simulations: 100 (vs 800) - still strong but much faster
    - Training iterations: 100 (vs 700k)
    - Games per iteration: 100 (vs 25k)
    - Training epochs: 100 (vs 1000)
    - Evaluation games: 40 (vs 400)
    
    Expected timeline:
    - Total training time: 10-12 hours
    - ~6 minutes per iteration (100 iterations total)
    - Both H100 GPUs utilized for self-play
    - Reaches amateur level play strength
    """
    
    def __init__(self):
        # Environment settings
        self.board_size = 9  # Start with 9x9 for faster training
        self.komi = 7.5
        self.num_stack = 8  # Number of board history to stack
        
        # Self-play settings
        self.num_parallel_games = 200  # Number of parallel self-play games (reduced for faster iterations)
        self.num_mcts_simulations = 100  # Number of MCTS simulations per move (reduced for practical training)
        self.c_puct = 1.0  # PUCT constant
        self.dirichlet_epsilon = 0.25  # Exploration noise at root
        self.dirichlet_alpha = 0.03  # Dirichlet noise parameter
        self.temperature = 1.0  # Temperature for first 30 moves
        self.temperature_drop = 30  # Move number to drop temperature to 0
        
        # Neural network settings
        self.num_res_blocks = 20  # Number of residual blocks (AlphaGo Zero uses 19-40)
        self.num_hidden = 256  # Number of hidden units in res blocks (AlphaGo Zero uses 256)
        self.learning_rate = 0.01  # Initial learning rate
        self.weight_decay = 1e-4  # L2 regularization
        self.momentum = 0.9
        self.batch_size = 2048  # Training batch size (larger for H100s)
        
        # Training settings (optimized for 10-12 hour training)
        self.num_iterations = 100  # Number of training iterations (reduced from 700k)
        self.num_episodes_per_iteration = 100  # Self-play games per iteration (reduced for practical training)
        self.num_epochs = 100  # Training epochs per iteration (reduced from 1000)
        self.checkpoint_interval = 10  # Save checkpoint every N iterations
        
        # Evaluation settings
        self.num_eval_games = 40  # Number of games for evaluation (reduced for faster iteration)
        self.eval_win_threshold = 0.55  # Win rate threshold to update best model
        self.eval_temperature = 0.1  # Lower temperature for evaluation games
        
        # Memory settings
        self.replay_buffer_size = 500000  # Max number of positions to store (reduced)
        self.min_replay_size = 10000  # Min positions before training starts (reduced)
        
        # Multi-GPU settings (optimized for 2 H100s)
        self.use_gpu = True
        self.num_gpus = 2  # Explicitly use 2 H100 GPUs
        self.num_workers = 16  # Number of workers for data loading (8 per GPU)
        
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