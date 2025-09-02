"""
Actor Manager for AlphaGo Zero Self-Play
Handles distributed self-play across multiple GPUs with proper process management.
"""

import os
import time
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import numpy as np
from typing import List, Dict, Optional
import threading
import traceback

from alpha_zero.envs import GoEnv
from alpha_zero.core import MCTS
from alpha_zero.core.replay import Transition


class SelfPlayActor:
    """Individual self-play actor for generating games."""
    
    def __init__(self, actor_id: int, gpu_id: int, config, model_queue: Queue, 
                 game_queue: Queue, stop_event: Event):
        self.actor_id = actor_id
        self.gpu_id = gpu_id
        self.config = config
        self.model_queue = model_queue
        self.game_queue = game_queue
        self.stop_event = stop_event
        
        # Stats tracking
        self.games_played = 0
        self.total_moves = 0
        self.start_time = time.time()
    
    def run(self):
        """Main actor loop."""
        try:
            # Set GPU for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
                device = torch.device(f'cuda:{self.gpu_id}')
            else:
                device = torch.device('cpu')
            
            # Initialize environment and model
            env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
            model = None
            mcts_args = self.config.get_mcts_args()
            
            # Actor initialized silently
            
            while not self.stop_event.is_set():
                try:
                    # Check for model updates
                    if not self.model_queue.empty():
                        model_state = self.model_queue.get_nowait()
                        if model is None:
                            # Import here to avoid issues with multiprocessing
                            from alpha_zero.core.network import AlphaGoZeroResNet
                            model = AlphaGoZeroResNet(env, self.config.num_res_blocks, 
                                                    self.config.num_hidden, device)
                        model.load_state_dict(model_state)
                        model.eval()
                    
                    if model is None:
                        time.sleep(0.1)
                        continue
                    
                    # Play a game
                    game_data = self._play_game(env, model, mcts_args)
                    if game_data:
                        self.game_queue.put(game_data)
                        self.games_played += 1
                        
                        # Track progress silently
                
                except Exception as e:
                    pass  # Silently handle errors
                    time.sleep(1)
                    
        except Exception as e:
            pass  # Silently handle fatal errors
    
    def _play_game(self, env, model, mcts_args) -> Optional[List[Transition]]:
        """Play a single self-play game."""
        env.reset()
        transitions = []
        move_count = 0
        
        while not env.is_game_over() and move_count < self.config.max_game_length:
            # Get current state
            state = env.observation().copy()
            
            # MCTS search
            mcts = MCTS(env, mcts_args, model)
            pi_probs = mcts.search()
            
            # Temperature-based action selection
            if move_count < self.config.temperature_drop:
                # Sample with temperature
                temperature_probs = np.power(pi_probs, 1.0 / self.config.temperature)
                temperature_probs = temperature_probs / np.sum(temperature_probs)
                action = np.random.choice(len(temperature_probs), p=temperature_probs)
            else:
                # Greedy selection
                action = np.argmax(pi_probs)
            
            # Store transition (value will be filled later)
            transitions.append(Transition(
                state=state,
                pi_prob=pi_probs.copy(),
                value=None
            ))
            
            # Make move
            env.step(action)
            move_count += 1
        
        # Fill in values based on game outcome
        if env.is_game_over():
            winner = env.winner
            for i, transition in enumerate(transitions):
                # Value from perspective of player who made the move
                player_to_move = env.get_player_from_state(transition.state)
                if winner == 0:  # Draw
                    value = 0.0
                elif winner == player_to_move:
                    value = 1.0
                else:
                    value = -1.0
                
                transitions[i] = Transition(
                    state=transition.state,
                    pi_prob=transition.pi_prob,
                    value=value
                )
            
            self.total_moves += len(transitions)
            return transitions
        
        return None


class ActorManager:
    """Manages multiple self-play actors across GPUs."""
    
    def __init__(self, config):
        self.config = config
        self.actors = []
        self.processes = []
        
        # Communication queues
        self.model_queues = []  # One per GPU
        self.game_queue = Queue(maxsize=1000)
        self.stop_event = Event()
        
        # Initialize queues for each GPU
        num_gpus = config.get_device_count()
        for gpu_id in range(num_gpus):
            self.model_queues.append(Queue(maxsize=10))
        
        # Create actors
        actors_per_gpu = config.num_actors // num_gpus
        actor_id = 0
        
        for gpu_id in range(num_gpus):
            for _ in range(actors_per_gpu):
                actor = SelfPlayActor(
                    actor_id=actor_id,
                    gpu_id=gpu_id,
                    config=config,
                    model_queue=self.model_queues[gpu_id],
                    game_queue=self.game_queue,
                    stop_event=self.stop_event
                )
                self.actors.append(actor)
                actor_id += 1
        
        # ActorManager initialized silently
    
    def start_actors(self):
        """Start all actor processes."""
        for actor in self.actors:
            process = Process(target=actor.run, daemon=True)
            process.start()
            self.processes.append(process)
        
        # Actors started silently
    
    def stop_actors(self):
        """Stop all actor processes."""
        self.stop_event.set()
        
        # Wait for processes to finish
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        
        # All actors stopped
    
    def update_model(self, model_state_dict):
        """Update model weights for all actors."""
        # Send model to all GPU queues
        for queue in self.model_queues:
            try:
                # Clear old models
                while not queue.empty():
                    queue.get_nowait()
                # Add new model
                queue.put(model_state_dict)
            except:
                pass
    
    def get_games(self, timeout=1.0) -> List[List[Transition]]:
        """Collect completed games from actors."""
        games = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                game = self.game_queue.get_nowait()
                games.append(game)
            except:
                break
        
        return games
    
    def get_stats(self) -> Dict:
        """Get actor statistics."""
        # This is simplified - in practice you'd need inter-process communication
        # for real-time stats
        return {
            'num_actors': len(self.actors),
            'num_processes': len([p for p in self.processes if p.is_alive()]),
            'queue_size': self.game_queue.qsize()
        }


# Example usage
if __name__ == "__main__":
    from config import Config
    
    config = Config()
    manager = ActorManager(config)
    
    try:
        manager.start_actors()
        
        # Simulate training loop
        for iteration in range(5):
            print(f"Iteration {iteration}")
            
            # In real training, you'd update model weights here
            # manager.update_model(model.state_dict())
            
            # Collect games
            games = manager.get_games(timeout=5.0)
            print(f"Collected {len(games)} games")
            
            time.sleep(10)
        
    except KeyboardInterrupt:
        print("Interrupted")
    
    finally:
        manager.stop_actors()