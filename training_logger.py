"""Comprehensive training logger for AlphaGo Zero."""

import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import torch
from typing import Dict, List, Any
from datetime import datetime


class TrainingLogger:
    """Comprehensive logger for all training metrics and visualizations."""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Training metrics
        self.metrics = {
            'iteration': [],
            'timestamp': [],
            
            # Loss metrics
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'learning_rate': [],
            
            # Self-play metrics
            'games_generated': [],
            'avg_game_length': [],
            'avg_moves_per_second': [],
            'self_play_time': [],
            
            # MCTS metrics
            'avg_mcts_time': [],
            'avg_tree_size': [],
            'avg_simulations': [],
            
            # Model evaluation
            'win_rate_vs_previous': [],
            'win_rate_vs_random': [],
            'win_rate_vs_expert': [],
            'elo_rating': [],
            
            # Training performance
            'training_time': [],
            'samples_per_second': [],
            'gpu_utilization': [],
            'memory_usage': [],
            
            # Game statistics
            'black_wins': [],
            'white_wins': [],
            'draws': [],
            'resignations': [],
            'avg_score_margin': [],
            
            # Neural network metrics
            'policy_entropy': [],
            'value_accuracy': [],
            'gradient_norm': [],
            'weight_norm': [],
        }
        
        # Current iteration data
        self.current_iteration = 0
        self.iteration_start_time = None
        
        # Game history for detailed analysis
        self.game_histories = []
        self.position_evaluations = []
        
        # Epoch-level tracking
        self.epoch_metrics = defaultdict(lambda: defaultdict(list))
        self.current_epoch_data = {}
        
        # Auto-save settings
        self.auto_save = True
        self.figures_dir = os.path.join(log_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def start_iteration(self, iteration: int):
        """Start logging for a new iteration."""
        self.current_iteration = iteration
        self.iteration_start_time = time.time()
        self.metrics['iteration'].append(iteration)
        self.metrics['timestamp'].append(time.time())
        
        # Reset epoch data for new iteration
        self.current_epoch_data = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'samples_per_second': [],
            'policy_entropy': [],
            'value_accuracy': []
        }
        
    def log_self_play_metrics(self, games_generated: int, avg_game_length: float, 
                             total_time: float, game_stats: Dict):
        """Log self-play metrics."""
        self.metrics['games_generated'].append(games_generated)
        self.metrics['avg_game_length'].append(avg_game_length)
        self.metrics['self_play_time'].append(total_time)
        self.metrics['avg_moves_per_second'].append(
            (games_generated * avg_game_length) / total_time if total_time > 0 else 0
        )
        
        # Game outcome statistics
        self.metrics['black_wins'].append(game_stats.get('black_wins', 0))
        self.metrics['white_wins'].append(game_stats.get('white_wins', 0))
        self.metrics['draws'].append(game_stats.get('draws', 0))
        self.metrics['resignations'].append(game_stats.get('resignations', 0))
        self.metrics['avg_score_margin'].append(game_stats.get('avg_score_margin', 0))
        
    def log_mcts_metrics(self, avg_time: float, avg_tree_size: float, avg_simulations: float):
        """Log MCTS performance metrics."""
        self.metrics['avg_mcts_time'].append(avg_time)
        self.metrics['avg_tree_size'].append(avg_tree_size)
        self.metrics['avg_simulations'].append(avg_simulations)
        
    def log_training_metrics(self, policy_loss: float, value_loss: float, 
                           total_loss: float, learning_rate: float, training_time: float,
                           samples_per_second: float, model=None):
        """Log neural network training metrics."""
        self.metrics['policy_loss'].append(policy_loss)
        self.metrics['value_loss'].append(value_loss)
        self.metrics['total_loss'].append(total_loss)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['training_time'].append(training_time)
        self.metrics['samples_per_second'].append(samples_per_second)
        
        # Neural network specific metrics
        if model is not None:
            total_norm = 0
            param_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                param_count += p.numel()
            
            self.metrics['gradient_norm'].append(total_norm ** 0.5)
            
            # Weight norm
            weight_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            self.metrics['weight_norm'].append(weight_norm)
        else:
            self.metrics['gradient_norm'].append(0)
            self.metrics['weight_norm'].append(0)
            
    def log_evaluation_metrics(self, win_rate_vs_previous: float, win_rate_vs_random: float,
                              win_rate_vs_expert: float, elo_rating: float):
        """Log model evaluation results."""
        self.metrics['win_rate_vs_previous'].append(win_rate_vs_previous)
        self.metrics['win_rate_vs_random'].append(win_rate_vs_random)
        self.metrics['win_rate_vs_expert'].append(win_rate_vs_expert)
        self.metrics['elo_rating'].append(elo_rating)
        
    def log_system_metrics(self):
        """Log system performance metrics."""
        try:
            import psutil
            import GPUtil
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].append(memory.percent)
            
            # GPU utilization
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_util = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
                self.metrics['gpu_utilization'].append(gpu_util)
            else:
                self.metrics['gpu_utilization'].append(0)
                
        except ImportError:
            self.metrics['memory_usage'].append(0)
            self.metrics['gpu_utilization'].append(0)
            
    def save_metrics(self):
        """Save all metrics to JSON files."""
        # Save main metrics
        with open(os.path.join(self.log_dir, 'training_metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        # Save detailed game histories
        with open(os.path.join(self.log_dir, 'game_histories.json'), 'w') as f:
            json.dump(self.game_histories, f, indent=2)
            
    def log_epoch_metrics(self, epoch: int, policy_loss: float, value_loss: float, 
                          total_loss: float, learning_rate: float, 
                          samples_per_second: float = None, gradient_norm: float = None):
        """Log metrics for a specific epoch within current iteration."""
        iter_key = f'iteration_{self.current_iteration}_epochs'
        
        self.epoch_metrics[iter_key]['epoch'].append(epoch)
        self.epoch_metrics[iter_key]['policy_loss'].append(policy_loss)
        self.epoch_metrics[iter_key]['value_loss'].append(value_loss)
        self.epoch_metrics[iter_key]['total_loss'].append(total_loss)
        self.epoch_metrics[iter_key]['learning_rate'].append(learning_rate)
        
        if samples_per_second is not None:
            self.epoch_metrics[iter_key]['samples_per_second'].append(samples_per_second)
        if gradient_norm is not None:
            self.epoch_metrics[iter_key]['gradient_norm'].append(gradient_norm)
        
        # Store in current epoch data
        self.current_epoch_data['policy_loss'].append(policy_loss)
        self.current_epoch_data['value_loss'].append(value_loss)
        self.current_epoch_data['total_loss'].append(total_loss)
        self.current_epoch_data['learning_rate'].append(learning_rate)
        
    def save_figure(self, fig, filename: str, dpi: int = 300):
        """Save figure with auto-save functionality."""
        if self.auto_save:
            base_path = os.path.join(self.figures_dir, filename)
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            
            # Save in multiple formats
            fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
            fig.savefig(f"{base_path}.pdf", bbox_inches='tight', facecolor='white')
            print(f"Auto-saved: {filename}.png and .pdf")
    
    def plot_all_metrics(self):
        """Generate 3 ESSENTIAL PLOTS for professor demo."""
        iterations = self.metrics['iteration']
        
        if not iterations:
            print("No data to plot yet.")
            return
        
        print(f"Creating plots for {len(iterations)} iterations...")
        
        # 1. LOSS PLOT - Policy, Value, Total
        plt.figure(figsize=(12, 8))
        if self.metrics['policy_loss']:
            plt.plot(iterations[:len(self.metrics['policy_loss'])], self.metrics['policy_loss'], 
                    'b-o', linewidth=2, label='Policy Loss')
        if self.metrics['value_loss']:
            plt.plot(iterations[:len(self.metrics['value_loss'])], self.metrics['value_loss'], 
                    'r-s', linewidth=2, label='Value Loss')
        if self.metrics['total_loss']:
            plt.plot(iterations[:len(self.metrics['total_loss'])], self.metrics['total_loss'], 
                    'g-^', linewidth=2, label='Total Loss')
        
        plt.xlabel('Training Iteration')
        plt.ylabel('Loss Value')
        plt.title('Neural Network Training Loss Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        loss_path = os.path.join(self.figures_dir, 'loss_convergence.png')
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        print(f"✓ Loss plot saved: {loss_path}")
        plt.close()
        
        # 2. WIN RATE PLOT - Score vs Human-level performance
        plt.figure(figsize=(12, 8))
        if self.metrics['win_rate_vs_previous']:
            plt.plot(iterations[:len(self.metrics['win_rate_vs_previous'])], 
                    self.metrics['win_rate_vs_previous'], 'b-o', linewidth=3, markersize=8, 
                    label='Win Rate vs Previous Model')
        
        # Add human skill level reference lines
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% Random')
        plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Beginner Level')
        plt.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='Intermediate Level')
        plt.axhline(y=0.95, color='purple', linestyle='--', alpha=0.7, label='Advanced Level')
        
        plt.xlabel('Training Iteration')
        plt.ylabel('Win Rate')
        plt.title('Model Strength vs Human Performance Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        score_path = os.path.join(self.figures_dir, 'human_performance.png')
        plt.savefig(score_path, dpi=150, bbox_inches='tight')
        print(f"✓ Human performance plot saved: {score_path}")
        plt.close()
        
        # 3. MCTS PERFORMANCE PLOT
        plt.figure(figsize=(12, 8))
        if self.metrics['avg_mcts_time']:
            plt.subplot(2, 1, 1)
            plt.plot(iterations[:len(self.metrics['avg_mcts_time'])], 
                    self.metrics['avg_mcts_time'], 'g-o', linewidth=2)
            plt.ylabel('MCTS Time (seconds)')
            plt.title('MCTS Search Performance')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            # Calculate move strength (inverse of time = faster decisions)
            move_strength = [1/max(t, 0.01) for t in self.metrics['avg_mcts_time']]
            plt.plot(iterations[:len(move_strength)], move_strength, 'r-s', linewidth=2)
            plt.xlabel('Training Iteration')
            plt.ylabel('Search Efficiency (1/time)')
            plt.title('MCTS Decision Speed')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        mcts_path = os.path.join(self.figures_dir, 'mcts_performance.png')
        plt.savefig(mcts_path, dpi=150, bbox_inches='tight')
        print(f"✓ MCTS plot saved: {mcts_path}")
        plt.close()
        
        print("\n" + "="*50)
        print("PROFESSOR DEMO PLOTS READY!")
        print("="*50)
        print(f"📊 1. Loss Convergence: {loss_path}")
        print(f"📈 2. Human Performance: {score_path}")
        print(f"🎯 3. MCTS Performance: {mcts_path}")
        print("="*50)
    
    def _plot_loss_correlation(self):
        plt.plot(iterations, self.metrics['win_rate_vs_expert'], label='vs Expert', marker='^')
        plt.axhline(y=0.55, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        plt.title('Win Rates')
        plt.xlabel('Iteration')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)
        
        # 3. Elo rating
        plt.subplot(6, 3, 3)
        if self.metrics['elo_rating']:
            plt.plot(iterations, self.metrics['elo_rating'], color='purple', marker='o')
            plt.title('Elo Rating Over Time')
            plt.xlabel('Iteration')
            plt.ylabel('Elo Rating')
            plt.grid(True)
        
        # 4. Game length distribution
        plt.subplot(6, 3, 4)
        if self.metrics['avg_game_length']:
            plt.plot(iterations, self.metrics['avg_game_length'], color='green')
            plt.title('Average Game Length')
            plt.xlabel('Iteration')
            plt.ylabel('Moves per Game')
            plt.grid(True)
        
        # 5. MCTS performance
        plt.subplot(6, 3, 5)
        if self.metrics['avg_mcts_time']:
            plt.plot(iterations, self.metrics['avg_mcts_time'], color='orange')
            plt.title('MCTS Search Time')
            plt.xlabel('Iteration')
            plt.ylabel('Seconds per Search')
            plt.grid(True)
        
        # 6. Training speed
        plt.subplot(6, 3, 6)
        if self.metrics['samples_per_second']:
            plt.plot(iterations, self.metrics['samples_per_second'], color='brown')
            plt.title('Training Speed')
            plt.xlabel('Iteration')
            plt.ylabel('Samples per Second')
            plt.grid(True)
        
        # 7. Game outcomes
        plt.subplot(6, 3, 7)
        if self.metrics['black_wins'] and self.metrics['white_wins']:
            plt.plot(iterations, self.metrics['black_wins'], label='Black Wins', color='black')
            plt.plot(iterations, self.metrics['white_wins'], label='White Wins', color='gray')
            plt.plot(iterations, self.metrics['draws'], label='Draws', color='blue')
            plt.title('Game Outcomes')
            plt.xlabel('Iteration')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True)
        
        # 8. System performance
        plt.subplot(6, 3, 8)
        if self.metrics['gpu_utilization']:
            plt.plot(iterations, self.metrics['gpu_utilization'], label='GPU %', color='red')
        if self.metrics['memory_usage']:
            plt.plot(iterations, self.metrics['memory_usage'], label='Memory %', color='blue')
        plt.title('System Utilization')
        plt.xlabel('Iteration')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        
        # 9. Neural network metrics
        plt.subplot(6, 3, 9)
        if self.metrics['gradient_norm']:
            plt.plot(iterations, self.metrics['gradient_norm'], label='Gradient Norm', color='purple')
        if self.metrics['weight_norm']:
            plt.plot(iterations, self.metrics['weight_norm'], label='Weight Norm', color='orange')
        plt.title('Neural Network Metrics')
        plt.xlabel('Iteration')
        plt.ylabel('Norm')
        plt.legend()
        plt.grid(True)
        
        # 10. Learning rate schedule
        plt.subplot(6, 3, 10)
        if self.metrics['learning_rate']:
            plt.plot(iterations, self.metrics['learning_rate'], color='green')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Iteration')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
        
        # 11. Self-play efficiency
        plt.subplot(6, 3, 11)
        if self.metrics['avg_moves_per_second']:
            plt.plot(iterations, self.metrics['avg_moves_per_second'], color='cyan')
            plt.title('Self-play Efficiency')
            plt.xlabel('Iteration')
            plt.ylabel('Moves per Second')
            plt.grid(True)
        
        # 12. Score margins
        plt.subplot(6, 3, 12)
        if self.metrics['avg_score_margin']:
            plt.plot(iterations, self.metrics['avg_score_margin'], color='magenta')
            plt.title('Average Score Margin')
            plt.xlabel('Iteration')
            plt.ylabel('Points')
            plt.grid(True)
        
        # 13. Training time breakdown
        plt.subplot(6, 3, 13)
        if self.metrics['self_play_time'] and self.metrics['training_time']:
            plt.plot(iterations, self.metrics['self_play_time'], label='Self-play', color='blue')
            plt.plot(iterations, self.metrics['training_time'], label='Training', color='red')
            plt.title('Time Breakdown')
            plt.xlabel('Iteration')
            plt.ylabel('Seconds')
            plt.legend()
            plt.grid(True)
        
        # 14. Tree size evolution
        plt.subplot(6, 3, 14)
        if self.metrics['avg_tree_size']:
            plt.plot(iterations, self.metrics['avg_tree_size'], color='brown')
            plt.title('MCTS Tree Size')
            plt.xlabel('Iteration')
            plt.ylabel('Nodes')
            plt.grid(True)
        
        # 15. Model comparison
        plt.subplot(6, 3, 15)
        win_rates = []
        labels = []
        if self.metrics['win_rate_vs_random'] and self.metrics['win_rate_vs_random'][-1] > 0:
            win_rates.append(self.metrics['win_rate_vs_random'][-1])
            labels.append('vs Random')
        if self.metrics['win_rate_vs_previous'] and self.metrics['win_rate_vs_previous'][-1] > 0:
            win_rates.append(self.metrics['win_rate_vs_previous'][-1])
            labels.append('vs Previous')
        if self.metrics['win_rate_vs_expert'] and self.metrics['win_rate_vs_expert'][-1] > 0:
            win_rates.append(self.metrics['win_rate_vs_expert'][-1])
            labels.append('vs Expert')
        
        if win_rates:
            plt.bar(labels, win_rates, color=['green', 'blue', 'red'])
            plt.title('Current Win Rates')
            plt.ylabel('Win Rate')
            plt.ylim(0, 1)
            for i, v in enumerate(win_rates):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        # Auto-save the comprehensive dashboard
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        filename = f'comprehensive_training_dashboard_{timestamp}'
        fig = plt.gcf()
        self.save_figure(fig, filename)
        
        # Also save with standard name for latest version
        plt.savefig(os.path.join(self.log_dir, 'comprehensive_training_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate additional specialized plots
        self._plot_loss_correlation()
        self._plot_performance_vs_time()
        self._plot_win_rate_progression()
        
        # Generate epoch-specific plots for current iteration
        if hasattr(self, 'current_iteration'):
            self._plot_current_epoch_progress()
        
    def _plot_loss_correlation(self):
        """Plot correlation between different loss types."""
        if not self.metrics['policy_loss'] or not self.metrics['value_loss']:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss correlation scatter plot
        ax1.scatter(self.metrics['policy_loss'], self.metrics['value_loss'], alpha=0.6)
        ax1.set_xlabel('Policy Loss')
        ax1.set_ylabel('Value Loss')
        ax1.set_title('Policy vs Value Loss Correlation')
        ax1.grid(True)
        
        # Loss progression over time
        iterations = self.metrics['iteration']
        ax2.plot(iterations, np.array(self.metrics['policy_loss']) / np.array(self.metrics['value_loss']), 
                color='purple')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Policy/Value Loss Ratio')
        ax2.set_title('Loss Ratio Evolution')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Auto-save loss analysis
        fig = plt.gcf()
        self.save_figure(fig, 'loss_analysis')
        plt.close()
        
    def _plot_performance_vs_time(self):
        """Plot performance metrics vs time."""
        if not self.metrics['timestamp']:
            return
            
        timestamps = np.array(self.metrics['timestamp'])
        timestamps = (timestamps - timestamps[0]) / 3600  # Convert to hours
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Win rate vs time
        if self.metrics['win_rate_vs_random']:
            ax1.plot(timestamps, self.metrics['win_rate_vs_random'], label='vs Random')
        if self.metrics['win_rate_vs_expert']:
            ax1.plot(timestamps, self.metrics['win_rate_vs_expert'], label='vs Expert')
        ax1.set_xlabel('Training Time (hours)')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Win Rate vs Training Time')
        ax1.legend()
        ax1.grid(True)
        
        # Loss vs time
        ax2.plot(timestamps, self.metrics['total_loss'], color='red')
        ax2.set_xlabel('Training Time (hours)')
        ax2.set_ylabel('Total Loss')
        ax2.set_title('Loss vs Training Time')
        ax2.grid(True)
        
        # Efficiency vs time
        if self.metrics['samples_per_second']:
            ax3.plot(timestamps, self.metrics['samples_per_second'], color='green')
            ax3.set_xlabel('Training Time (hours)')
            ax3.set_ylabel('Samples per Second')
            ax3.set_title('Training Efficiency vs Time')
            ax3.grid(True)
        
        # System utilization vs time
        if self.metrics['gpu_utilization']:
            ax4.plot(timestamps, self.metrics['gpu_utilization'], label='GPU %', color='red')
        if self.metrics['memory_usage']:
            ax4.plot(timestamps, self.metrics['memory_usage'], label='Memory %', color='blue')
        ax4.set_xlabel('Training Time (hours)')
        ax4.set_ylabel('Utilization %')
        ax4.set_title('System Utilization vs Time')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Auto-save performance vs time
        fig = plt.gcf()
        self.save_figure(fig, 'performance_vs_time')
        plt.close()
        
    def _plot_win_rate_progression(self):
        """Plot detailed win rate progression."""
        if not any([self.metrics['win_rate_vs_random'], self.metrics['win_rate_vs_expert']]):
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        iterations = self.metrics['iteration']
        
        # Plot win rates with confidence intervals (if available)
        if self.metrics['win_rate_vs_random']:
            ax.plot(iterations, self.metrics['win_rate_vs_random'], 
                   label='vs Random Player', linewidth=2, marker='o')
        
        if self.metrics['win_rate_vs_expert']:
            ax.plot(iterations, self.metrics['win_rate_vs_expert'], 
                   label='vs Expert Data', linewidth=2, marker='^')
        
        if self.metrics['win_rate_vs_previous']:
            ax.plot(iterations, self.metrics['win_rate_vs_previous'], 
                   label='vs Previous Best', linewidth=2, marker='s')
        
        # Add milestone lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Level')
        ax.axhline(y=0.55, color='orange', linestyle='--', alpha=0.7, label='Update Threshold')
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Strong Level')
        
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Win Rate')
        ax.set_title('Model Strength Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Auto-save win rate progression
        fig = plt.gcf()
        self.save_figure(fig, 'win_rate_progression')
        plt.close()
        
    def _plot_current_epoch_progress(self):
        """Plot epoch-level progress for current iteration."""
        iter_key = f'iteration_{self.current_iteration}_epochs'
        
        if iter_key not in self.epoch_metrics or not self.epoch_metrics[iter_key]['epoch']:
            return
            
        epochs = self.epoch_metrics[iter_key]['epoch']
        policy_losses = self.epoch_metrics[iter_key]['policy_loss']
        value_losses = self.epoch_metrics[iter_key]['value_loss']
        total_losses = self.epoch_metrics[iter_key]['total_loss']
        learning_rates = self.epoch_metrics[iter_key]['learning_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss progression
        axes[0, 0].plot(epochs, policy_losses, 'o-', label='Policy Loss', color='blue')
        axes[0, 0].plot(epochs, value_losses, 's-', label='Value Loss', color='red')
        axes[0, 0].plot(epochs, total_losses, '^-', label='Total Loss', color='green')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'Loss Progression - Iteration {self.current_iteration}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Learning rate
        axes[0, 1].plot(epochs, learning_rates, 'o-', color='purple')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Loss reduction rate
        if len(total_losses) > 1:
            reduction_rates = [-((total_losses[i+1] - total_losses[i]) / total_losses[i] * 100) 
                             for i in range(len(total_losses)-1)]
            axes[1, 0].plot(epochs[1:], reduction_rates, 'o-', color='orange')
            axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Reduction Rate (%)')
            axes[1, 0].set_title('Training Efficiency per Epoch')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient norm (if available)
        if 'gradient_norm' in self.epoch_metrics[iter_key] and self.epoch_metrics[iter_key]['gradient_norm']:
            grad_norms = self.epoch_metrics[iter_key]['gradient_norm']
            axes[1, 1].plot(epochs[:len(grad_norms)], grad_norms, 'o-', color='brown')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Gradient Magnitude')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        else:
            # Show training speed if gradient norm not available
            if 'samples_per_second' in self.epoch_metrics[iter_key] and self.epoch_metrics[iter_key]['samples_per_second']:
                speeds = self.epoch_metrics[iter_key]['samples_per_second']
                axes[1, 1].bar(epochs[:len(speeds)], speeds, color='teal', alpha=0.7)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Samples/Second')
                axes[1, 1].set_title('Training Speed')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Epoch-Level Training Progress - Iteration {self.current_iteration}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Auto-save epoch progress
        filename = f'iteration_{self.current_iteration:03d}/epoch_progress'
        self.save_figure(fig, filename)
        plt.close()
        
    def generate_training_report(self):
        """Generate a comprehensive training report."""
        if not self.metrics['iteration']:
            return
            
        total_iterations = len(self.metrics['iteration'])
        latest_win_rate_random = self.metrics['win_rate_vs_random'][-1] if self.metrics['win_rate_vs_random'] else 0
        latest_win_rate_expert = self.metrics['win_rate_vs_expert'][-1] if self.metrics['win_rate_vs_expert'] else 0
        latest_elo = self.metrics['elo_rating'][-1] if self.metrics['elo_rating'] else 1500
        
        report = f"""
# AlphaGo Zero Training Report

## Training Overview
- **Total Iterations**: {total_iterations}
- **Current Win Rate vs Random**: {latest_win_rate_random:.3f}
- **Current Win Rate vs Expert**: {latest_win_rate_expert:.3f}
- **Current Elo Rating**: {latest_elo:.1f}

## Performance Metrics
- **Average Game Length**: {np.mean(self.metrics['avg_game_length']):.1f} moves
- **Training Efficiency**: {np.mean(self.metrics['samples_per_second']):.1f} samples/sec
- **MCTS Speed**: {np.mean(self.metrics['avg_mcts_time']):.3f} sec/search

## Model Quality
- **Final Policy Loss**: {self.metrics['policy_loss'][-1]:.4f}
- **Final Value Loss**: {self.metrics['value_loss'][-1]:.4f}
- **Loss Improvement**: {(self.metrics['total_loss'][0] - self.metrics['total_loss'][-1]):.4f}

## System Performance
- **Average GPU Utilization**: {np.mean(self.metrics['gpu_utilization']):.1f}%
- **Average Memory Usage**: {np.mean(self.metrics['memory_usage']):.1f}%

## Recommendations
{"✓ Model is performing well!" if latest_win_rate_random > 0.8 else "→ Consider training longer or adjusting hyperparameters"}
{"✓ Beating expert level!" if latest_win_rate_expert > 0.6 else "→ Need more training to reach expert level"}
{"✓ Efficient resource usage" if np.mean(self.metrics['gpu_utilization']) > 80 else "→ Could utilize GPU more efficiently"}
"""
        
        with open(os.path.join(self.log_dir, 'training_report.md'), 'w') as f:
            f.write(report)
            
        return report