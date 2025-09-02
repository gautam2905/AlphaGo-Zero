import os
import json
import datetime
import random
import numpy as np
import torch


def get_time_stamp():
    """Get current timestamp string."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_directory(path: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(model, optimizer, iteration, filepath, scaler=None, extra_state=None):
    """Save model checkpoint with full training state.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        iteration: Current training iteration
        filepath: Path to save checkpoint
        scaler: GradScaler for mixed precision (optional)
        extra_state: Additional state dict to save (optional)
    """
    # Handle DataParallel models - save the underlying model
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Save mixed precision scaler state if available
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Save any extra training state
    if extra_state is not None:
        checkpoint.update(extra_state)
    
    # Create backup of existing checkpoint
    if os.path.exists(filepath):
        backup_path = filepath.replace('.pth', '_backup.pth')
        os.rename(filepath, backup_path)
    
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved at iteration {iteration} to {filepath}")


def load_checkpoint(model, optimizer, filepath, scaler=None, device=None):
    """Load model checkpoint with full training state.
    
    Args:
        model: The model to load into
        optimizer: The optimizer to load into
        filepath: Path to load checkpoint from
        scaler: GradScaler for mixed precision (optional)
        device: Device to map tensors to (optional)
        
    Returns:
        iteration: The iteration number from checkpoint, or 0 if failed
    """
    if not os.path.isfile(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0
    
    try:
        # Load checkpoint with proper device mapping
        map_location = device if device else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Handle different checkpoint formats
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Handle DataParallel models - the checkpoint has no 'module.' prefix
        # but the current model might be wrapped in DataParallel
        if hasattr(model, 'module'):
            # Current model is DataParallel, but checkpoint might not have 'module.' prefix
            try:
                model.module.load_state_dict(model_state_dict)
            except RuntimeError:
                # Try loading directly if keys don't match
                model.load_state_dict(model_state_dict)
        else:
            # Current model is not DataParallel
            # Remove 'module.' prefix from checkpoint if it exists
            if any(key.startswith('module.') for key in model_state_dict.keys()):
                cleaned_state_dict = {}
                for key, value in model_state_dict.items():
                    new_key = key[7:] if key.startswith('module.') else key
                    cleaned_state_dict[new_key] = value
                model.load_state_dict(cleaned_state_dict)
            else:
                model.load_state_dict(model_state_dict)
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load mixed precision scaler state if available
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        iteration = checkpoint.get('iteration', 0)
        print(f"✓ Checkpoint loaded from iteration {iteration}")
        return iteration
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise e


def save_config(config, filepath):
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(filepath):
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')  # Use first GPU as primary
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_all_gpu_devices():
    """Get all available GPU devices."""
    if torch.cuda.is_available():
        return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return []


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def board_augment(board_state, pi_prob, board_size):
    """Apply random board augmentation (rotation/reflection)."""
    # Randomly choose augmentation
    aug_type = np.random.randint(8)
    
    # Reshape for easier manipulation
    board = board_state.reshape(-1, board_size, board_size)
    policy = pi_prob.reshape(board_size, board_size)
    
    # Apply augmentation
    if aug_type == 0:  # No change
        pass
    elif aug_type == 1:  # Rotate 90 degrees
        board = np.rot90(board, 1, axes=(1, 2))
        policy = np.rot90(policy, 1)
    elif aug_type == 2:  # Rotate 180 degrees
        board = np.rot90(board, 2, axes=(1, 2))
        policy = np.rot90(policy, 2)
    elif aug_type == 3:  # Rotate 270 degrees
        board = np.rot90(board, 3, axes=(1, 2))
        policy = np.rot90(policy, 3)
    elif aug_type == 4:  # Horizontal flip
        board = np.flip(board, axis=2)
        policy = np.flip(policy, axis=1)
    elif aug_type == 5:  # Vertical flip
        board = np.flip(board, axis=1)
        policy = np.flip(policy, axis=0)
    elif aug_type == 6:  # Diagonal flip
        board = np.swapaxes(board, 1, 2)
        policy = np.transpose(policy)
    elif aug_type == 7:  # Anti-diagonal flip
        board = np.flip(np.swapaxes(board, 1, 2), axis=2)
        policy = np.flip(np.transpose(policy), axis=1)
    
    return board.reshape(board_state.shape), policy.reshape(pi_prob.shape)