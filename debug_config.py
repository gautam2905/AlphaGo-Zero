"""Debug configuration for AlphaGo Zero to test if code is working."""

import os

class DebugConfig:
    """Debug configuration with very small parameters to test functionality."""
    
    def __init__(self):
        # Environment settings
        self.board_size = 9  # Start with 9x9 for faster training
        self.komi = 7.5
        self.num_stack = 8  # Number of board history to stack
        
        # Self-play settings (HEAVILY REDUCED FOR TESTING)
        self.num_parallel_games = 10  # Much smaller for debug
        self.num_mcts_simulations = 25  # Much smaller for debug (vs 800)
        self.c_puct = 1.0  # PUCT constant
        self.dirichlet_epsilon = 0.25  # Exploration noise at root
        self.dirichlet_alpha = 0.03  # Dirichlet noise parameter
        self.temperature = 1.0  # Temperature for first 30 moves
        self.temperature_drop = 15  # Move number to drop temperature to 0
        
        # Neural network settings (REDUCED FOR TESTING)
        self.num_res_blocks = 4  # Much smaller for debug (vs 20)
        self.num_hidden = 64  # Much smaller for debug (vs 256)
        self.learning_rate = 0.01  # Initial learning rate
        self.weight_decay = 1e-4  # L2 regularization
        self.momentum = 0.9
        self.batch_size = 32  # Much smaller for debug (vs 2048)
        
        # Training settings (MINIMAL FOR TESTING)
        self.num_iterations = 3  # Just 3 iterations for debug
        self.num_episodes_per_iteration = 5  # Just 5 games for debug (vs 500)
        self.num_epochs = 2  # Just 2 epochs for debug (vs 100)
        self.checkpoint_interval = 1  # Save every iteration for debug
        
        # Evaluation settings
        self.num_eval_games = 2  # Just 2 games for debug
        self.eval_win_threshold = 0.55  # Win rate threshold to update best model
        self.eval_temperature = 0.1  # Lower temperature for evaluation games
        
        # Memory settings
        self.replay_buffer_size = 10000  # Much smaller for debug
        self.min_replay_size = 50  # Much smaller for debug
        
        # Multi-GPU settings (optimized for 2 H100s)
        self.use_gpu = True
        self.num_gpus = 2  # Explicitly use 2 H100 GPUs
        self.num_workers = 4  # Fewer workers for debug
        
        # Paths - All data saved to current D drive directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory on D drive
        self.checkpoint_dir = os.path.join(self.base_dir, "debug_checkpoints")
        self.log_dir = os.path.join(self.base_dir, "debug_logs")
        self.data_dir = os.path.join(self.base_dir, "debug_data")  # For training data
        self.temp_dir = os.path.join(self.base_dir, "debug_temp")   # For temporary files
        self.expert_games_dir = os.path.join(self.base_dir, "debug_expert_games")
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