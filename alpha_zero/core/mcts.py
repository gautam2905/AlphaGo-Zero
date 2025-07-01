import numpy as np
from typing import List, Optional, Tuple
import math
import torch
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
        
        # Get neural network prediction
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        policy, _ = self.model(state_tensor)
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
        
        for search in range(self.args['num_searches']):
            node = root
            
            # Create a lightweight copy of the game for this simulation
            # Instead of deep copying the entire game, we'll track moves and apply them
            simulation_moves = []
            
            # Traverse the tree
            path = [node]
            while node.is_fully_expanded():
                node = node.select()
                path.append(node)
                # Track moves for this simulation
                if node.action_taken is not None:
                    simulation_moves.append(node.action_taken)
            
            # For now, assume game is not terminal at leaf node
            # In a full implementation, we'd apply simulation_moves to check
            is_terminal = False
            value = 0
            
            if not is_terminal:
                # Use current game state for leaf evaluation (simplified)
                current_state = self.game.observation()
                state_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                policy, value = self.model(state_tensor)
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                
                # Mask invalid moves
                valid_moves = self.game.legal_actions
                policy *= valid_moves
                policy_sum = np.sum(policy)
                if policy_sum > 0:
                    policy /= policy_sum
                else:
                    policy = valid_moves / np.sum(valid_moves)
                
                value = value.item()
                
                # Update node state for expansion
                node.state = current_state
                node.expand(policy, self.game)
            
            # Backpropagate value through the path
            # The value is from the perspective of the player at the leaf node
            # We need to be careful about which player's perspective the value is from
            for i, node in enumerate(reversed(path)):
                # For the leaf node, use the value directly
                # For other nodes, flip the value based on turn alternation
                if i == 0:
                    node.value_sum += value
                    node.visit_count += 1
                else:
                    value = -value
                    node.value_sum += value
                    node.visit_count += 1    
            
            
        action_probs = np.zeros(self.game.action_dim)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        