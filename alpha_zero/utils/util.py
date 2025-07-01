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


def save_checkpoint(model, optimizer, iteration, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at iteration {iteration}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']
        print(f"Checkpoint loaded from iteration {iteration}")
        return iteration
    else:
        print(f"No checkpoint found at {filepath}")
        return 0


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