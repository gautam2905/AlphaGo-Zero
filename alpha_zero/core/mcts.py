import numpy as np
from typing import List, Optional, Tuple
import math
import torch
from torch.amp import autocast
import copy as cp


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0, to_play=None):
        self.game = game
        self.args = args
        self.state = state  # This will be the observation
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.to_play = to_play  # Track which player's turn it is
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy, game_env):
        for action, prob in enumerate(policy):
            if prob > 0:
                # No need to create child states here, we'll handle that during selection
                child = Node(self.game, self.args, None, self, action, prob, to_play=game_env.opponent_player)
                self.children.append(child)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        # Flip value for opponent
        value = -value
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args, model):
        self.game = game  # This should be a GoEnv instance
        self.args = args
        self.model = model
        self.device = next(model.parameters()).device
        
    @torch.no_grad()
    def search(self, state=None):
        # Use current game state
        if state is None:
            state = self.game.observation()
        
        root = Node(self.game, self.args, state, visit_count=1, to_play=self.game.to_play)
        
        # Get neural network prediction for root with mixed precision (BFloat16 for H100)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        use_bfloat16 = self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        with autocast('cuda', enabled=self.device.type == 'cuda', dtype=dtype):
            policy, _ = self.model(state_tensor)
            # Convert to float32 for softmax to avoid BFloat16 issues
            policy = policy.float()
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_dim)
        
        # Mask invalid moves
        valid_moves = self.game.legal_actions
        policy *= valid_moves
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum
        else:
            # If all moves are masked, use uniform distribution over valid moves
            policy = valid_moves / np.sum(valid_moves)
        
        root.expand(policy, self.game)
        
        # Batch neural network evaluations for H100 efficiency
        # Larger batch sizes for better H100 GPU utilization with 800+ simulations
        batch_size = min(64, max(16, self.args['num_searches'] // 10))  # Adaptive batch size
        
        for batch_start in range(0, self.args['num_searches'], batch_size):
            batch_end = min(batch_start + batch_size, self.args['num_searches'])
            batch_nodes = []
            batch_paths = []
            
            # Collect leaf nodes for batch evaluation
            for search in range(batch_start, batch_end):
                node = root
                simulation_moves = []
                
                # Traverse the tree to find leaf node
                path = [node]
                while node.is_fully_expanded():
                    node = node.select()
                    path.append(node)
                    if node.action_taken is not None:
                        simulation_moves.append(node.action_taken)
                
                batch_nodes.append(node)
                batch_paths.append(path)
            
            # Batch evaluate all leaf nodes
            if batch_nodes:
                states = []
                for node in batch_nodes:
                    # Use current game state (simplified - in practice would apply simulation moves)
                    current_state = self.game.observation()
                    states.append(current_state)
                
                # Stack states into batch tensor
                if states:
                    states_tensor = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
                    use_bfloat16 = self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
                    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
                    with autocast('cuda', enabled=self.device.type == 'cuda', dtype=dtype):
                        policies_batch, values_batch = self.model(states_tensor)
                        # Convert to float32 for softmax to avoid BFloat16 issues
                        policies_batch = policies_batch.float()
                        values_batch = values_batch.float()
                    policies_batch = torch.softmax(policies_batch, axis=1).cpu().numpy()
                    values_batch = values_batch.cpu().numpy()
                    
                    # Process batch results
                    for i, (node, path) in enumerate(zip(batch_nodes, batch_paths)):
                        policy = policies_batch[i]
                        value = values_batch[i].item()
                        
                        # Mask invalid moves
                        valid_moves = self.game.legal_actions
                        policy *= valid_moves
                        policy_sum = np.sum(policy)
                        if policy_sum > 0:
                            policy /= policy_sum
                        else:
                            policy = valid_moves / np.sum(valid_moves)
                        
                        # Expand node if not terminal
                        if not node.is_fully_expanded():
                            node.expand(policy, self.game)
                        
                        # Backpropagate value through the path
                        current_value = value
                        for j, path_node in enumerate(reversed(path)):
                            # For the leaf node, use the value directly
                            # For other nodes, flip the value based on turn alternation
                            if j == 0:
                                path_node.value_sum += current_value
                                path_node.visit_count += 1
                            else:
                                current_value = -current_value
                                path_node.value_sum += current_value
                                path_node.visit_count += 1    
            
            
        action_probs = np.zeros(self.game.action_dim)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        