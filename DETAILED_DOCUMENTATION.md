# Detailed Code Documentation - AlphaGo Zero Implementation

This document provides line-by-line explanations of every file and function in the AlphaGo Zero implementation.

## Table of Contents
1. [Project Structure Overview](#project-structure-overview)
2. [Core Components](#core-components)
3. [Game Environments](#game-environments)
4. [Utilities](#utilities)
5. [Training and Evaluation](#training-and-evaluation)
6. [Configuration](#configuration)

---

## Project Structure Overview

```
alphago-zero/
├── alpha_zero/                    # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── core/                     # Core AlphaGo Zero algorithms
│   │   ├── __init__.py          # Core module exports
│   │   ├── mcts.py              # Monte Carlo Tree Search implementation
│   │   ├── network.py           # Neural network architecture (ResNet)
│   │   ├── replay.py            # Experience replay buffer
│   │   └── rating.py            # Elo rating system
│   ├── envs/                    # Game environments
│   │   ├── __init__.py          # Environment module exports
│   │   ├── base.py              # Base board game environment
│   │   ├── coords.py            # Coordinate conversion utilities
│   │   ├── go.py                # Go game implementation
│   │   ├── go_engine.py         # Go rules engine (from Minigo)
│   │   └── gui.py               # GUI components
│   ├── eval_plays/              # Evaluation scripts
│   │   ├── __init__.py          # Evaluation module exports
│   │   └── eval_agent_go.py     # Go agent evaluation
│   └── utils/                   # Utility functions
│       ├── __init__.py          # Utils module exports
│       ├── sgf_wrapper.py       # SGF file handling
│       ├── transformation.py    # Data augmentation
│       └── util.py              # General utilities
├── train.py                     # Main training script
├── plot_go.py                   # Training visualization
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
└── README.md                    # Project documentation
```

---

# Core Components

## alpha_zero/core/network.py - Neural Network Architecture

### Class: ResNet
**Purpose**: Implements the neural network architecture from AlphaGo Zero paper - a ResNet with policy and value heads.

```python
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
```
**Parameters**:
- `game`: GoEnv instance to get board dimensions and action space
- `num_resBlocks`: Number of residual blocks (typically 10-20)
- `num_hidden`: Hidden units per layer (typically 128-256)
- `device`: CUDA device for GPU training

**Architecture Components**:

1. **Input Processing**:
   ```python
   input_channels = game.observation_space.shape[0]  # History planes + color to play
   ```
   - Calculates input channels: `num_stack * 2 + 1` planes
   - Each plane is board_size × board_size

2. **Start Block**:
   ```python
   self.startBlock = nn.Sequential(
       nn.Conv2d(input_channels, num_hidden, kernel_size=3, padding=1),
       nn.BatchNorm2d(num_hidden),
       nn.ReLU()
   )
   ```
   - Initial convolution to project input to hidden dimension
   - Batch normalization for stable training
   - ReLU activation

3. **Backbone (Residual Blocks)**:
   ```python
   self.backBone = nn.ModuleList([ResBlock(num_hidden) for i in range(num_resBlocks)])
   ```
   - Stack of residual blocks for feature extraction
   - Each block preserves spatial dimensions

4. **Policy Head**:
   ```python
   self.policyHead = nn.Sequential(
       nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
       nn.BatchNorm2d(32),
       nn.ReLU(),
       nn.Flatten(),
       nn.Linear(32 * self.board_size * self.board_size, self.action_size)
   )
   ```
   - Outputs move probabilities for each board position + pass
   - Flattened to 1D vector of action probabilities

5. **Value Head**:
   ```python
   self.valueHead = nn.Sequential(
       nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
       nn.BatchNorm2d(3),
       nn.ReLU(),
       nn.Flatten(),
       nn.Linear(3 * self.board_size * self.board_size, 1),
       nn.Tanh()
   )
   ```
   - Outputs single value in [-1, 1] representing win probability
   - Tanh ensures output is bounded

### Class: ResBlock
**Purpose**: Individual residual block with skip connections.

```python
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out
```
- **Skip connection**: Adds input to output to enable deep network training
- **Two conv layers**: Each with batch norm and ReLU
- **Preserves dimensions**: Input and output have same shape

---

## alpha_zero/core/mcts.py - Monte Carlo Tree Search

### Class: Node
**Purpose**: Represents a single node in the MCTS tree.

```python
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0, to_play=None):
```

**Attributes**:
- `game`: Reference to game environment
- `args`: MCTS configuration parameters
- `state`: Board observation at this node
- `parent`: Parent node in tree
- `action_taken`: Action that led to this node
- `prior`: Prior probability from neural network
- `to_play`: Which player's turn it is
- `children`: List of child nodes
- `visit_count`: Number of times node was visited
- `value_sum`: Sum of backup values

**Methods**:

1. **is_fully_expanded()**:
   ```python
   def is_fully_expanded(self):
       return len(self.children) > 0
   ```
   - Returns True if node has children (has been expanded)
   - Simple check: any children means expanded

2. **select()**:
   ```python
   def select(self):
       best_child = None
       best_ucb = -np.inf
       for child in self.children:
           ucb = self.get_ucb(child)
           if ucb > best_ucb:
               best_child = child
               best_ucb = ucb
       return best_child
   ```
   - Selects child with highest UCB (Upper Confidence Bound) value
   - Balances exploitation (high value) vs exploration (high uncertainty)

3. **get_ucb(child)**:
   ```python
   def get_ucb(self, child):
       if child.visit_count == 0:
           q_value = 0
       else:
           q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
       return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
   ```
   - **Q-value**: Average reward from this child
   - **U-value**: Exploration bonus based on visit counts and prior
   - **C parameter**: Controls exploration vs exploitation balance

4. **expand(policy, game_env)**:
   ```python
   def expand(self, policy, game_env):
       for action, prob in enumerate(policy):
           if prob > 0:
               child = Node(self.game, self.args, None, self, action, prob, to_play=game_env.opponent_player)
               self.children.append(child)
   ```
   - Creates child nodes for all legal actions
   - Each child gets prior probability from neural network policy

### Class: MCTS
**Purpose**: Implements Monte Carlo Tree Search algorithm.

```python
class MCTS:
    def __init__(self, game, args, model):
        self.game = game          # GoEnv instance
        self.args = args          # MCTS parameters
        self.model = model        # Neural network
        self.device = next(model.parameters()).device  # GPU/CPU device
```

**Main Method: search()**:
```python
@torch.no_grad()
def search(self, state=None):
```

**Step 1: Initialize Root**:
```python
if state is None:
    state = self.game.observation()
root = Node(self.game, self.args, state, visit_count=1, to_play=self.game.to_play)
```
- Creates root node from current game state
- Sets visit count to 1 to avoid division by zero

**Step 2: Get Root Policy**:
```python
state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
policy, _ = self.model(state_tensor)
policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
```
- Convert state to tensor and run through neural network
- Apply softmax to get probability distribution
- Convert back to numpy for processing

**Step 3: Add Dirichlet Noise**:
```python
policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
    * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_dim)
```
- Adds exploration noise to root node
- Encourages trying different moves during self-play

**Step 4: Mask Invalid Moves**:
```python
valid_moves = self.game.legal_actions
policy *= valid_moves
policy_sum = np.sum(policy)
if policy_sum > 0:
    policy /= policy_sum
```
- Zeros out probabilities for illegal moves
- Renormalizes to ensure valid probability distribution

**Step 5: MCTS Simulations**:
```python
for search in range(self.args['num_searches']):
    node = root
    game_copy = cp.deepcopy(self.game)
    path = [node]
```

**Selection Phase**:
```python
while node.is_fully_expanded():
    node = node.select()
    path.append(node)
    if node.action_taken is not None:
        game_copy.step(node.action_taken)
```
- Traverse tree using UCB selection
- Apply moves to game copy to track board state

**Terminal Check**:
```python
is_terminal = game_copy.is_game_over()
if is_terminal:
    if game_copy.winner == game_copy.to_play:
        value = 1
    elif game_copy.winner is None:
        value = 0  # Draw
    else:
        value = -1
```
- If game is over, get exact value from game result

**Expansion and Evaluation**:
```python
if not is_terminal:
    current_state = game_copy.observation()
    state_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device).unsqueeze(0)
    policy, value = self.model(state_tensor)
    # ... process policy and expand node
    node.expand(policy, game_copy)
```
- Use neural network to evaluate position and get policy
- Expand node with new children

**Backpropagation**:
```python
for i, node in enumerate(reversed(path)):
    if i == 0:
        node.value_sum += value
        node.visit_count += 1
    else:
        value = -value  # Flip for opponent
        node.value_sum += value
        node.visit_count += 1
```
- Propagate value up the tree
- Flip value sign for alternating players

**Step 6: Extract Action Probabilities**:
```python
action_probs = np.zeros(self.game.action_dim)
for child in root.children:
    action_probs[child.action_taken] = child.visit_count
action_probs /= np.sum(action_probs)
return action_probs
```
- Convert visit counts to probability distribution
- More visited actions get higher probability

---

## alpha_zero/core/replay.py - Experience Replay

### Class: Transition
**Purpose**: Named tuple storing a single training example.

```python
class Transition(NamedTuple):
    state: Optional[np.ndarray]    # Board observation
    pi_prob: Optional[np.ndarray]  # MCTS policy (action probabilities)
    value: Optional[float]         # Game outcome value
```

### Compression Functions

**compress_array(array)**:
```python
def compress_array(array):
    return snappy.compress(array), array.shape, array.dtype
```
- Uses Snappy compression to reduce memory usage
- Stores shape and dtype for reconstruction

**uncompress_array(compressed)**:
```python
def uncompress_array(compressed):
    compressed_array, shape, dtype = compressed
    byte_string = snappy.uncompress(compressed_array)
    return np.frombuffer(byte_string, dtype=dtype).reshape(shape)
```
- Reconstructs numpy array from compressed data

### Class: UniformReplay
**Purpose**: Circular buffer for storing and sampling training data.

```python
class UniformReplay:
    def __init__(self, capacity: int, random_state: np.random.RandomState, compress_data: bool = True):
```

**Attributes**:
- `capacity`: Maximum number of transitions to store
- `random_state`: Random number generator for sampling
- `compress_data`: Whether to compress state arrays
- `storage`: List storing transitions
- `num_games_added`: Counter for games added
- `num_samples_added`: Counter for samples added

**Methods**:

1. **add_game(game_seq)**:
   ```python
   def add_game(self, game_seq: Sequence[Transition]) -> None:
       for transition in game_seq:
           self.add(transition)
       self.num_games_added += 1
   ```
   - Adds entire game sequence to replay buffer
   - Increments game counter

2. **add(transition)**:
   ```python
   def add(self, transition: Any) -> None:
       index = self.num_samples_added % self.capacity
       self.storage[index] = self.encoder(transition)
       self.num_samples_added += 1
   ```
   - Adds single transition using circular indexing
   - Overwrites oldest data when buffer is full

3. **sample(batch_size)**:
   ```python
   def sample(self, batch_size: int) -> Transition:
       if self.size < batch_size:
           return
       indices = self.random_state.randint(low=0, high=self.size, size=batch_size)
       samples = self.get(indices)
       transposed = zip(*samples)
       stacked = [np.stack(xs, axis=0) for xs in transposed]
       return type(self.structure)(*stacked)
   ```
   - Samples random batch from buffer
   - Stacks individual samples into batch tensors

4. **encoder/decoder**:
   ```python
   def encoder(self, transition: Transition) -> Transition:
       if self.compress_data:
           return transition._replace(state=compress_array(transition.state))
       return transition
   ```
   - Compresses state arrays to save memory
   - Decoder reverses the process for sampling

---

## alpha_zero/core/rating.py - Elo Rating System

### Class: EloRating
**Purpose**: Tracks relative strength of different model versions.

```python
class EloRating:
    def __init__(self, initial_rating=1500, k_factor=32):
        self.ratings = {}           # Player ID -> rating
        self.initial_rating = initial_rating
        self.k_factor = k_factor    # Learning rate for rating updates
```

**Methods**:

1. **get_rating(player_id)**:
   ```python
   def get_rating(self, player_id):
       return self.ratings.get(player_id, self.initial_rating)
   ```
   - Returns current rating or initial rating for new players

2. **update_ratings(player1_id, player2_id, result)**:
   ```python
   def update_ratings(self, player1_id, player2_id, result):
       rating1 = self.get_rating(player1_id)
       rating2 = self.get_rating(player2_id)
       
       expected1 = 1 / (1 + 10**((rating2 - rating1) / 400))
       expected2 = 1 - expected1
       
       new_rating1 = rating1 + self.k_factor * (result - expected1)
       new_rating2 = rating2 + self.k_factor * ((1 - result) - expected2)
   ```
   - Updates both player ratings based on game result
   - Uses standard Elo formula with expected outcomes

---

# Game Environments

## alpha_zero/envs/base.py - Base Board Game Environment

### Class: PlayerMove
**Purpose**: Named tuple for storing game history.

```python
class PlayerMove(namedtuple('PlayerMove', ['color', 'move'])):
```
- `color`: 'B' for black, 'W' for white
- `move`: Action index or special moves (pass/resign)

### Class: BoardGameEnv
**Purpose**: Base class following OpenAI Gym interface for board games.

**Initialization**:
```python
def __init__(self, board_size=15, num_stack=8, black_player_id=1, white_player_id=2, 
             has_pass_move=False, has_resign_move=False, id=''):
```

**Key Attributes**:
- `board_size`: Dimensions of the board (15x15, 19x19, etc.)
- `num_stack`: Number of history planes to stack
- `black_player`/`white_player`: Player IDs (1 and 2)
- `observation_space`: Gym space defining observation format
- `action_space`: Gym space defining valid actions
- `board`: Current board state as numpy array
- `legal_actions`: Binary mask of legal moves
- `board_deltas`: Deque storing board history
- `history`: List of PlayerMove objects

**Core Methods**:

1. **reset()**:
   ```python
   def reset(self, **kwargs) -> np.ndarray:
       super().reset(**kwargs)
       self.board = np.zeros_like(self.board)
       self.legal_actions = np.ones_like(self.legal_actions, dtype=np.int8).flatten()
       self.to_play = self.black_player
       # Reset all counters and history
       return self.observation()
   ```
   - Resets game to initial state
   - Clears board and history
   - Returns initial observation

2. **observation()**:
   ```python
   def observation(self) -> np.ndarray:
       # Create stacked feature planes
       features = np.zeros((self.num_stack * 2, self.board_size, self.board_size), dtype=np.int8)
       deltas = np.array(self.board_deltas)
       
       # Current player first, then opponent
       features[::2] = deltas == self.to_play
       features[1::2] = deltas == self.opponent_player
       
       # Color to play plane
       color_to_play = np.zeros((1, self.board_size, self.board_size), dtype=np.int8)
       if self.to_play == self.black_player:
           color_to_play += 1
       
       return np.concatenate([features, color_to_play], axis=0)
   ```
   - Creates input tensor for neural network
   - Stacks history planes: [current_player, opponent, current_player_t-1, opponent_t-1, ...]
   - Adds color-to-play plane (all 1s for black, all 0s for white)

3. **Coordinate Conversion Methods**:
   ```python
   def action_to_coords(self, action: int) -> Tuple[int, int]:
       return self.cc.from_flat(action)
   
   def coords_to_action(self, coords: Tuple[int, int]) -> int:
       return self.cc.to_flat(coords)
   
   def action_to_gtp(self, action: int) -> str:
       return self.cc.to_gtp(self.cc.from_flat(action))
   ```
   - Converts between different coordinate systems
   - GTP format: Standard Go notation (A1, B2, etc.)

---

## alpha_zero/envs/go.py - Go Game Implementation

### Class: GoEnv
**Purpose**: Complete Go game implementation inheriting from BoardGameEnv.

**Initialization**:
```python
def __init__(self, komi=7.5, num_stack=8, max_steps=go.N * go.N * 2):
    super().__init__(
        id='Go',
        board_size=go.N,              # Usually 19
        num_stack=num_stack,
        black_player_id=go.BLACK,     # 1
        white_player_id=go.WHITE,     # 2
        has_pass_move=True,
        has_resign_move=True,
    )
    self.position = go.Position(komi=self.komi)  # Go rules engine
```

**Core Methods**:

1. **step(action)**:
   ```python
   def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
       # Validation
       if self.is_game_over():
           raise RuntimeError('Game is over, call reset before using step method.')
       
       # Handle resignation
       if action == self.resign_move:
           self.winner = self.opponent_player
           return self.observation(), -1, True, {}
       
       # Record move in history
       self.add_to_history(self.last_player, self.last_move)
       
       # Apply move to go.Position
       self.position = self.position.play_move(
           c=self.cc.from_flat(action), 
           color=self.to_play, 
           mutate=True
       )
       
       # Update board representation
       self.board = self.position.board
       self.legal_actions = self.position.all_legal_moves()
       
       # Check for game end
       done = self.is_game_over()
       if done:
           # Calculate final score and determine winner
           result_str = self.get_result_string()
           # Set self.winner based on result
       
       return self.observation(), reward, done, {}
   ```
   - Handles move validation and application
   - Updates internal game state
   - Calculates rewards and game termination

2. **is_game_over()**:
   ```python
   def is_game_over(self) -> bool:
       if self.last_move == self.resign_move:
           return True
       if self.steps >= self.max_steps:
           return True
       # Two consecutive passes end the game
       if len(self.history) >= 2 and \
          self.history[-1].move == self.pass_move and \
          self.history[-2].move == self.pass_move:
           return True
       return False
   ```
   - Checks multiple termination conditions
   - Resignation, max moves, or two consecutive passes

3. **get_result_string()**:
   ```python
   def get_result_string(self) -> str:
       if self.last_move == self.resign_move:
           string = 'B+R' if self.winner == self.black_player else 'W+R'
       else:
           string = self.position.result_string()  # Uses go_engine scoring
       return string
   ```
   - Returns SGF-format result string
   - Handles resignation and scoring

---

## alpha_zero/envs/coords.py - Coordinate Conversion

### Class: CoordsConvertor
**Purpose**: Converts between different coordinate systems used in Go.

```python
class CoordsConvertor:
    def __init__(self, board_size):
        self.board_size = board_size
```

**Coordinate Systems**:
1. **Flat**: Single integer index (0 to board_size² - 1)
2. **Minigo**: (row, col) tuples
3. **GTP**: String format like "D4"
4. **SGF**: String format for game records

**Key Methods**:

1. **from_flat(flat_coord) -> (row, col)**:
   ```python
   def from_flat(self, flat_coord):
       if flat_coord >= self.board_size * self.board_size:
           return None  # Pass move
       row = flat_coord // self.board_size
       col = flat_coord % self.board_size
       return (row, col)
   ```

2. **to_flat((row, col)) -> int**:
   ```python
   def to_flat(self, coords):
       row, col = coords
       return row * self.board_size + col
   ```

3. **to_gtp((row, col)) -> str**:
   ```python
   def to_gtp(self, coords):
       row, col = coords
       col_letter = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'[col]  # Skip 'I'
       row_number = self.board_size - row
       return f"{col_letter}{row_number}"
   ```

---

# Utilities

## alpha_zero/utils/util.py - General Utilities

### Time and Randomness

1. **get_time_stamp()**:
   ```python
   def get_time_stamp():
       return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
   ```
   - Returns formatted timestamp string for file naming

2. **set_random_seed(seed)**:
   ```python
   def set_random_seed(seed: int):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       if torch.cuda.is_available():
           torch.cuda.manual_seed(seed)
           torch.cuda.manual_seed_all(seed)
   ```
   - Sets seeds for all random number generators
   - Ensures reproducible experiments

### File and Directory Operations

3. **create_directory(path)**:
   ```python
   def create_directory(path: str):
       if not os.path.exists(path):
           os.makedirs(path)
   ```
   - Creates directory if it doesn't exist

### Model Checkpointing

4. **save_checkpoint(model, optimizer, iteration, filepath)**:
   ```python
   def save_checkpoint(model, optimizer, iteration, filepath):
       checkpoint = {
           'iteration': iteration,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
       }
       torch.save(checkpoint, filepath)
   ```
   - Saves model, optimizer state, and training iteration
   - Enables resuming training from any point

5. **load_checkpoint(model, optimizer, filepath)**:
   ```python
   def load_checkpoint(model, optimizer, filepath):
       if os.path.isfile(filepath):
           checkpoint = torch.load(filepath)
           model.load_state_dict(checkpoint['model_state_dict'])
           optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
           iteration = checkpoint['iteration']
           return iteration
       return 0
   ```
   - Loads saved checkpoint and returns iteration number
   - Returns 0 if no checkpoint found

### Configuration Management

6. **save_config(config, filepath)**:
   ```python
   def save_config(config, filepath):
       with open(filepath, 'w') as f:
           json.dump(config, f, indent=4)
   ```
   - Saves configuration dictionary to JSON file

7. **load_config(filepath)**:
   ```python
   def load_config(filepath):
       with open(filepath, 'r') as f:
           return json.load(f)
   ```
   - Loads configuration from JSON file

### Training Utilities

8. **class AverageMeter**:
   ```python
   class AverageMeter:
       def __init__(self):
           self.reset()
       
       def reset(self):
           self.val = 0      # Current value
           self.avg = 0      # Running average
           self.sum = 0      # Sum of all values
           self.count = 0    # Number of updates
       
       def update(self, val, n=1):
           self.val = val
           self.sum += val * n
           self.count += n
           self.avg = self.sum / self.count
   ```
   - Tracks running averages of training metrics
   - Useful for loss tracking during training

9. **get_device()**:
   ```python
   def get_device():
       if torch.cuda.is_available():
           return torch.device('cuda')
       elif torch.backends.mps.is_available():
           return torch.device('mps')  # Apple Silicon
       else:
           return torch.device('cpu')
   ```
   - Returns best available device for computation

10. **count_parameters(model)**:
    ```python
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    ```
    - Counts trainable parameters in a model

---

## alpha_zero/utils/sgf_wrapper.py - SGF File Handling

### Class: SGFWrapper
**Purpose**: Creates and parses SGF (Smart Game Format) files for Go games.

**Initialization**:
```python
class SGFWrapper:
    def __init__(self, board_size: int = 19, komi: float = 7.5):
        self.board_size = board_size
        self.komi = komi
        self.moves = []
        self.metadata = {
            'GM': '1',                    # Game type (1 = Go)
            'FF': '4',                    # File format version
            'CA': 'UTF-8',                # Character encoding
            'SZ': str(board_size),        # Board size
            'KM': str(komi),              # Komi
            'DT': datetime.now().strftime('%Y-%m-%d'),  # Date
            'AP': 'AlphaGoZero:1.0',      # Application name
        }
```

**Methods**:

1. **add_move(color, coords)**:
   ```python
   def add_move(self, color: str, coords: Tuple[int, int]):
       if coords is None:
           move_str = f"{color}[]"  # Pass move
       else:
           row, col = coords
           sgf_col = chr(ord('a') + col)  # Convert to SGF coordinates
           sgf_row = chr(ord('a') + row)
           move_str = f"{color}[{sgf_col}{sgf_row}]"
       self.moves.append(move_str)
   ```
   - Adds move to game record in SGF format
   - Handles pass moves as empty coordinates

2. **to_sgf()**:
   ```python
   def to_sgf(self) -> str:
       sgf = "(;"
       # Add metadata
       for key, value in self.metadata.items():
           sgf += f"{key}[{value}]"
       # Add moves
       for move in self.moves:
           sgf += f";{move}"
       sgf += ")"
       return sgf
   ```
   - Converts game to SGF format string
   - Includes metadata and all moves

3. **from_sgf(sgf_content)**:
   ```python
   @classmethod
   def from_sgf(cls, sgf_content: str) -> 'SGFWrapper':
       wrapper = cls()
       # Parse metadata
       metadata_pattern = r'(\w+)\[([^\]]*)\]'
       metadata_matches = re.findall(metadata_pattern, sgf_content)
       # Parse moves
       move_pattern = r';([BW])\[([^\]]*)\]'
       move_matches = re.findall(move_pattern, sgf_content)
       # Convert coordinates back to (row, col) format
       return wrapper
   ```
   - Parses SGF content into SGFWrapper instance
   - Uses regex to extract metadata and moves

**Utility Function**:
```python
def moves_to_sgf(moves, board_size=19, komi=7.5, winner=None, 
                player_black="AlphaGoZero", player_white="AlphaGoZero"):
```
- Converts list of moves to complete SGF string
- Includes player names and game result

---

## alpha_zero/utils/transformation.py - Data Augmentation

### Purpose
Data augmentation increases training data diversity by applying symmetries to board positions.

**Key Function: board_augment(board, policy, board_size)**:
```python
def board_augment(board, policy, board_size):
    # Randomly choose transformation
    transform_id = np.random.randint(0, 8)
    
    transformations = [
        lambda x: x,                           # Identity
        lambda x: np.rot90(x, 1, axes=(1, 2)), # Rotate 90°
        lambda x: np.rot90(x, 2, axes=(1, 2)), # Rotate 180°
        lambda x: np.rot90(x, 3, axes=(1, 2)), # Rotate 270°
        lambda x: np.flip(x, axis=1),          # Horizontal flip
        lambda x: np.flip(x, axis=2),          # Vertical flip
        lambda x: np.transpose(x, (0, 2, 1)),  # Transpose
        # Combined transformations...
    ]
    
    # Apply transformation to board
    augmented_board = transformations[transform_id](board)
    
    # Transform policy to match board transformation
    policy_2d = policy[:-1].reshape((board_size, board_size))  # Exclude pass move
    augmented_policy_2d = transformations[transform_id](policy_2d)
    augmented_policy = augmented_policy_2d.flatten()
    
    # Add pass move probability back
    augmented_policy = np.append(augmented_policy, policy[-1])
    
    return augmented_board, augmented_policy
```

**Transformations Applied**:
1. **Rotations**: 90°, 180°, 270° rotations
2. **Reflections**: Horizontal and vertical flips
3. **Transpose**: Diagonal reflection
4. **Combinations**: Multiple transformations combined

**Why Augmentation Works**:
- Go board has 8-fold symmetry (4 rotations × 2 reflections)
- Same position can appear in 8 different orientations
- Training on all orientations improves generalization

---

# Training and Evaluation

## train.py - Main Training Script

### Class: SelfPlayDataset
**Purpose**: PyTorch Dataset for self-play training data.

```python
class SelfPlayDataset(Dataset):
    def __init__(self, replay_buffer, config):
        self.replay_buffer = replay_buffer
        self.config = config
    
    def __getitem__(self, idx):
        # Sample from replay buffer
        transition = self.replay_buffer.sample(1)
        state = transition.state[0]
        pi_prob = transition.pi_prob[0]
        value = transition.value[0]
        
        # Apply random augmentation
        if np.random.random() < 0.5:
            state, pi_prob = board_augment(state, pi_prob, self.config.board_size)
        
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(pi_prob, dtype=torch.float32),
            torch.tensor(value, dtype=torch.float32)
        )
```
- Samples random transitions from replay buffer
- Applies data augmentation with 50% probability
- Returns tensors ready for training

### Class: AlphaZeroTrainer
**Purpose**: Main training orchestrator implementing the AlphaGo Zero algorithm.

**Initialization**:
```python
def __init__(self, config):
    self.config = config
    set_random_seed(42)
    
    # Initialize environment
    self.env = GoEnv(komi=config.komi, num_stack=config.num_stack)
    
    # Setup device and multi-GPU
    self.device = get_device()
    self.num_devices = config.get_device_count()
    
    # Initialize model
    self.model = ResNet(self.env, config.num_res_blocks, config.num_hidden, self.device)
    if self.num_devices > 1:
        self.model = DataParallel(self.model)  # Multi-GPU support
    
    # Initialize optimizer and scheduler
    self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, 
                              momentum=config.momentum, weight_decay=config.weight_decay)
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                   milestones=[100, 200, 300], gamma=0.1)
    
    # Initialize replay buffer
    self.replay_buffer = UniformReplay(capacity=config.replay_buffer_size,
                                      random_state=np.random.RandomState(42))
```

**Core Training Methods**:

1. **self_play_game(model)**:
   ```python
   def self_play_game(self, model):
       env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
       mcts = MCTS(env, self.config.get_mcts_args(), model)
       
       game_data = []
       
       while not env.is_game_over():
           # Apply temperature for exploration
           if env.steps < self.config.temperature_drop:
               temperature = self.config.temperature
           else:
               temperature = 0.1
           
           # Run MCTS
           pi_probs = mcts.search()
           
           # Sample action with temperature
           if temperature == 0:
               action = np.argmax(pi_probs)
           else:
               pi_temp = np.power(pi_probs, 1/temperature)
               pi_temp = pi_temp / np.sum(pi_temp)
               action = np.random.choice(len(pi_probs), p=pi_temp)
           
           # Store training data
           game_data.append({
               'state': env.observation().copy(),
               'pi_prob': pi_probs.copy(),
               'player': env.to_play
           })
           
           env.step(action)
       
       # Assign final values based on game outcome
       # Return transitions for replay buffer
   ```
   - Plays complete self-play game using MCTS
   - Applies temperature for exploration/exploitation balance
   - Collects training examples at each move

2. **run_self_play(num_games)**:
   ```python
   def run_self_play(self, num_games):
       model = self.best_model
       model.eval()
       
       all_transitions = []
       for i in tqdm(range(num_games), desc="Self-play"):
           transitions = self.self_play_game(model)
           all_transitions.extend(transitions)
       
       # Add to replay buffer
       for transition in all_transitions:
           self.replay_buffer.add(transition)
   ```
   - Generates multiple self-play games
   - Uses current best model for game generation
   - Adds all transitions to replay buffer

3. **train_network()**:
   ```python
   def train_network(self):
       if self.replay_buffer.size < self.config.min_replay_size:
           return
       
       self.model.train()
       dataset = SelfPlayDataset(self.replay_buffer, self.config)
       dataloader = DataLoader(dataset, batch_size=self.config.batch_size, 
                              shuffle=True, num_workers=self.config.num_workers)
       
       # Loss functions
       mse_loss = nn.MSELoss()       # Value loss
       ce_loss = nn.CrossEntropyLoss()  # Policy loss
       
       for epoch in range(self.config.num_epochs):
           for batch_idx, (states, pi_probs, values) in enumerate(dataloader):
               states = states.to(self.device)
               pi_probs = pi_probs.to(self.device)
               values = values.to(self.device)
               
               # Forward pass
               pred_pi, pred_v = self.model(states)
               
               # Calculate losses
               policy_loss = ce_loss(pred_pi, pi_probs)
               value_loss = mse_loss(pred_v.squeeze(), values)
               total_loss = policy_loss + value_loss
               
               # Backward pass
               self.optimizer.zero_grad()
               total_loss.backward()
               self.optimizer.step()
   ```
   - Trains neural network on self-play data
   - Uses combined policy and value loss
   - Applies gradient descent updates

4. **evaluate_model()**:
   ```python
   def evaluate_model(self):
       current_model = copy.deepcopy(self.model)
       wins = 0
       
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
           
           # Play evaluation game with both models
           # ... game playing logic ...
           
           # Check result and update win count
       
       win_rate = wins / self.config.num_eval_games
       
       # Update best model if win rate exceeds threshold
       if win_rate >= self.config.eval_win_threshold:
           self.best_model = copy.deepcopy(self.model)
           return True
       return False
   ```
   - Evaluates current model against previous best
   - Plays games with models alternating colors
   - Updates best model if win rate > 55%

5. **train() - Main Training Loop**:
   ```python
   def train(self):
       for iteration in range(self.iteration, self.config.num_iterations):
           # Phase 1: Self-play data generation
           self.run_self_play(self.config.num_episodes_per_iteration)
           
           # Phase 2: Neural network training
           self.train_network()
           
           # Phase 3: Model evaluation (periodic)
           if (iteration + 1) % self.config.checkpoint_interval == 0:
               model_updated = self.evaluate_model()
               save_checkpoint(self.model, self.optimizer, iteration + 1, checkpoint_path)
   ```
   - Implements complete AlphaGo Zero training cycle
   - Self-play → Training → Evaluation → Repeat

---

## alpha_zero/eval_plays/eval_agent_go.py - Agent Evaluation

### Class: AlphaGoEvaluator
**Purpose**: Evaluates trained AlphaGo agents against baselines and humans.

**Initialization**:
```python
def __init__(self, model_path, config=None):
    # Load trained model
    self.model = ResNet(self.env, config.num_res_blocks, config.num_hidden, self.device)
    checkpoint = torch.load(model_path, map_location=self.device)
    self.model.load_state_dict(checkpoint)
    self.model.eval()
    
    # Initialize MCTS
    self.mcts = MCTS(self.env, config.get_mcts_args(), self.model)
```

**Evaluation Methods**:

1. **play_game_vs_random(agent_plays_black)**:
   ```python
   def play_game_vs_random(self, agent_plays_black=True):
       env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
       
       while not env.is_game_over():
           if (env.to_play == env.black_player and agent_plays_black) or \
              (env.to_play == env.white_player and not agent_plays_black):
               # Agent's turn - use MCTS
               mcts = MCTS(env, self.config.get_mcts_args(), self.model)
               pi_probs = mcts.search()
               action = np.argmax(pi_probs)
           else:
               # Random player's turn
               valid_actions = np.where(env.legal_actions == 1)[0]
               action = np.random.choice(valid_actions)
           
           env.step(action)
       
       # Return result from agent's perspective
       return determine_winner(env, agent_plays_black)
   ```
   - Plays game between agent and random player
   - Agent uses MCTS, random player chooses legal moves randomly

2. **evaluate_vs_random(num_games)**:
   ```python
   def evaluate_vs_random(self, num_games=100):
       wins = draws = losses = 0
       
       for i in range(num_games):
           agent_plays_black = (i % 2 == 0)  # Alternate colors
           result = self.play_game_vs_random(agent_plays_black)
           
           if result == 1: wins += 1
           elif result == 0: draws += 1
           else: losses += 1
       
       win_rate = wins / num_games
       return win_rate
   ```
   - Runs multiple games against random player
   - Alternates colors to ensure fair evaluation

3. **play_interactive_game()**:
   ```python
   def play_interactive_game(self):
       env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
       
       while not env.is_game_over():
           env.render()
           
           if env.to_play == env.black_player:
               # Human's turn
               move_str = input("Your move: ").strip().upper()
               if move_str == 'PASS':
                   action = env.pass_move
               else:
                   action = env.gtp_to_action(move_str)
           else:
               # Agent's turn
               mcts = MCTS(env, self.config.get_mcts_args(), self.model)
               pi_probs = mcts.search()
               action = np.argmax(pi_probs)
               print(f"Agent plays: {env.action_to_gtp(action)}")
           
           env.step(action)
       
       env.render()
       print(f"Game over! Result: {env.get_result_string()}")
   ```
   - Allows human to play against trained agent
   - Human enters moves in GTP format
   - Agent uses MCTS for move selection

---

# Configuration

## config.py - Configuration Management

### Class: Config
**Purpose**: Centralized configuration for all hyperparameters and settings.

**Categories of Settings**:

1. **Environment Settings**:
   ```python
   self.board_size = 9           # Start with 9x9 for faster training
   self.komi = 7.5              # Compensation points for white
   self.num_stack = 8           # History planes to stack
   ```

2. **Self-play Settings**:
   ```python
   self.num_parallel_games = 100        # Parallel self-play games
   self.num_mcts_simulations = 50       # MCTS simulations per move
   self.c_puct = 1.0                   # UCB exploration constant
   self.dirichlet_epsilon = 0.25        # Root exploration noise
   self.dirichlet_alpha = 0.03          # Dirichlet concentration
   self.temperature = 1.0               # Move selection temperature
   self.temperature_drop = 30           # When to reduce temperature
   ```

3. **Neural Network Settings**:
   ```python
   self.num_res_blocks = 10      # ResNet depth
   self.num_hidden = 128         # Hidden units per layer
   self.learning_rate = 0.01     # Initial learning rate
   self.weight_decay = 1e-4      # L2 regularization
   self.batch_size = 256         # Training batch size
   ```

4. **Training Settings**:
   ```python
   self.num_iterations = 1000               # Total training iterations
   self.num_episodes_per_iteration = 100    # Self-play games per iteration
   self.num_epochs = 10                     # Training epochs per iteration
   self.checkpoint_interval = 10            # Save frequency
   ```

5. **Multi-GPU Settings**:
   ```python
   self.use_gpu = True
   self.num_gpus = -1           # -1 = use all available GPUs
   self.num_workers = 4         # DataLoader workers
   ```

**Key Methods**:

1. **get_mcts_args()**:
   ```python
   def get_mcts_args(self):
       return {
           'num_searches': self.num_mcts_simulations,
           'C': self.c_puct,
           'dirichlet_epsilon': self.dirichlet_epsilon,
           'dirichlet_alpha': self.dirichlet_alpha,
       }
   ```
   - Returns MCTS configuration dictionary

2. **get_device_count()**:
   ```python
   def get_device_count(self):
       if not self.use_gpu or not torch.cuda.is_available():
           return 1
       available_gpus = torch.cuda.device_count()
       if self.num_gpus == -1:
           return available_gpus
       else:
           return min(self.num_gpus, available_gpus)
   ```
   - Determines number of GPUs to use

---

# Visualization

## plot_go.py - Training Visualization

### Class: TrainingPlotter
**Purpose**: Creates visualizations for training analysis and debugging.

**Plotting Methods**:

1. **plot_training_loss()**:
   ```python
   def plot_training_loss(self, loss_file="training_losses.json"):
       with open(loss_path, 'r') as f:
           losses = json.load(f)
       
       fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
       
       # Policy loss
       ax1.plot(iterations, losses['policy_loss'])
       ax1.set_title('Policy Loss')
       
       # Value loss  
       ax2.plot(iterations, losses['value_loss'])
       ax2.set_title('Value Loss')
       
       # Total loss
       ax3.plot(iterations, losses['total_loss'])
       ax3.set_title('Total Loss')
   ```
   - Plots training loss curves over time
   - Separate plots for policy, value, and total loss

2. **plot_win_rates()**:
   ```python
   def plot_win_rates(self, win_rate_file="win_rates.json"):
       with open(win_rate_path, 'r') as f:
           win_rates = json.load(f)
       
       plt.plot(iterations, rates, marker='o')
       plt.axhline(y=0.55, color='r', linestyle='--', label='Acceptance Threshold')
       plt.title('Model Win Rate vs Previous Best')
   ```
   - Shows model improvement over training
   - Horizontal line at 55% acceptance threshold

3. **visualize_board_state()**:
   ```python
   def visualize_board_state(self, board, title="Go Board"):
       # Draw grid
       for i in range(19):
           ax.axhline(i, color='black', linewidth=0.5)
           ax.axvline(i, color='black', linewidth=0.5)
       
       # Draw stones
       for i in range(board.shape[0]):
           for j in range(board.shape[1]):
               if board[i, j] == 1:  # Black stone
                   circle = plt.Circle((j, 18-i), 0.4, color='black')
                   ax.add_patch(circle)
               elif board[i, j] == 2:  # White stone
                   circle = plt.Circle((j, 18-i), 0.4, color='white', edgecolor='black')
                   ax.add_patch(circle)
   ```
   - Visualizes Go board positions
   - Shows black stones, white stones, and grid

---

This detailed documentation covers every major component in the AlphaGo Zero implementation. Each function is explained with its purpose, parameters, return values, and implementation details. Use this guide to understand the codebase and make informed modifications to suit your specific needs.