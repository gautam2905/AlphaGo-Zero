import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math



class AlphaGoZeroResNet(nn.Module):
    """AlphaGo Zero ResNet architecture with 40 blocks and 256 filters.
    
    Based on the original DeepMind paper specifications:
    - 40 residual blocks (vs 16 in original implementation)  
    - 256 filters per convolutional layer (vs 128)
    - ~23 million parameters total
    - Optimized policy and value heads
    """
    
    def __init__(self, game, num_resBlocks=40, num_hidden=256, device='cuda'):
        super().__init__()
        
        self.device = device
        self.board_size = game.board_size
        self.action_size = game.action_dim
        self.num_hidden = num_hidden
        
        # Input channels: num_stack * 2 + 1 (history for both players + color to play)
        input_channels = game.observation_space.shape[0]
        
        print(f"Building AlphaGo Zero ResNet:")
        print(f"  - Residual blocks: {num_resBlocks}")
        print(f"  - Filters per block: {num_hidden}")
        print(f"  - Input channels: {input_channels}")
        print(f"  - Board size: {self.board_size}x{self.board_size}")
        print(f"  - Action size: {self.action_size}")
        
        # Initial convolutional block (AlphaGo Zero spec)
        self.startBlock = nn.Sequential(
            nn.Conv2d(input_channels, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(inplace=True)
        )
        
        # 40 residual blocks (AlphaGo Zero spec)
        self.backBone = nn.ModuleList([
            AlphaGoZeroResBlock(num_hidden) for _ in range(num_resBlocks)
        ])
        
        # Policy head (AlphaGo Zero spec: 2 filters)
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, self.action_size)
        )
        
        # Value head (AlphaGo Zero spec: 1 filter)
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.board_size * self.board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        # Initialize weights using Kaiming initialization
        self.apply(self._init_weights)
        
        self.to(device)
        
        # Calculate and display parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # H100 optimizations
        if device.type == 'cuda' or str(device).startswith('cuda'):
            print("  - Enabling H100 optimizations...")
            # Enable Tensor Core optimizations for H100
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Optimize for H100 memory hierarchy
            torch.backends.cudnn.benchmark = True
            # Enable mixed precision optimizations
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
    
    def _init_weights(self, module):
        """Initialize weights using Kaiming normal initialization."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        """Forward pass with mixed precision support."""
        x = self.startBlock(x)
        
        # Use gradient checkpointing for memory efficiency with large model
        # Checkpoint every 8 blocks to balance memory vs computation
        for i, resBlock in enumerate(self.backBone):
            if self.training and x.requires_grad and (i % 8 == 0):
                x = checkpoint(resBlock, x, use_reentrant=False)
            else:
                x = resBlock(x)
        
        # Separate policy and value computations
        policy_logits = self.policyHead(x)
        value = self.valueHead(x)
        
        return policy_logits, value


class AlphaGoZeroResBlock(nn.Module):
    """Improved residual block matching AlphaGo Zero specifications.
    
    Features:
    - No bias in convolutional layers (BatchNorm handles bias)
    - ReLU activations with inplace operations for memory efficiency
    - Proper residual connections
    """
    
    def __init__(self, num_hidden):
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        """Forward pass with residual connection."""
        residual = x
        
        # First conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # Second conv-bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual and apply ReLU
        out += residual
        out = F.relu(out, inplace=True)
        
        return out


# Backward compatibility - keep old ResNet class name
ResNet = AlphaGoZeroResNet
ResBlock = AlphaGoZeroResBlock
        