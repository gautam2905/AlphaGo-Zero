"""Plotting utilities for AlphaGo Zero training."""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from collections import defaultdict


class TrainingPlotter:
    """Plot training metrics and game analysis."""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
    
    def plot_training_loss(self, loss_file="training_losses.json"):
        """Plot training loss curves."""
        loss_path = os.path.join(self.log_dir, loss_file)
        
        if not os.path.exists(loss_path):
            print(f"Loss file not found: {loss_path}")
            return
        
        with open(loss_path, 'r') as f:
            losses = json.load(f)
        
        iterations = list(range(len(losses['policy_loss'])))
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Policy loss
        ax1.plot(iterations, losses['policy_loss'])
        ax1.set_title('Policy Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Value loss
        ax2.plot(iterations, losses['value_loss'])
        ax2.set_title('Value Loss')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # Total loss
        ax3.plot(iterations, losses['total_loss'])
        ax3.set_title('Total Loss')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_losses.png'), dpi=300)
        plt.show()
    
    def plot_win_rates(self, win_rate_file="win_rates.json"):
        """Plot evaluation win rates over time."""
        win_rate_path = os.path.join(self.log_dir, win_rate_file)
        
        if not os.path.exists(win_rate_path):
            print(f"Win rate file not found: {win_rate_path}")
            return
        
        with open(win_rate_path, 'r') as f:
            win_rates = json.load(f)
        
        iterations = list(win_rates.keys())
        rates = list(win_rates.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, rates, marker='o')
        plt.axhline(y=0.55, color='r', linestyle='--', label='Acceptance Threshold')
        plt.title('Model Win Rate vs Previous Best')
        plt.xlabel('Iteration')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'win_rates.png'), dpi=300)
        plt.show()
    
    def plot_game_length_distribution(self, game_data_file="game_stats.json"):
        """Plot distribution of game lengths."""
        game_data_path = os.path.join(self.log_dir, game_data_file)
        
        if not os.path.exists(game_data_path):
            print(f"Game data file not found: {game_data_path}")
            return
        
        with open(game_data_path, 'r') as f:
            game_data = json.load(f)
        
        game_lengths = game_data['game_lengths']
        
        plt.figure(figsize=(10, 6))
        plt.hist(game_lengths, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Game Lengths')
        plt.xlabel('Number of Moves')
        plt.ylabel('Frequency')
        plt.axvline(x=np.mean(game_lengths), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(game_lengths):.1f}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'game_lengths.png'), dpi=300)
        plt.show()
    
    def plot_mcts_tree_size(self, mcts_file="mcts_stats.json"):
        """Plot MCTS tree size statistics."""
        mcts_path = os.path.join(self.log_dir, mcts_file)
        
        if not os.path.exists(mcts_path):
            print(f"MCTS file not found: {mcts_path}")
            return
        
        with open(mcts_path, 'r') as f:
            mcts_data = json.load(f)
        
        iterations = list(range(len(mcts_data['avg_tree_size'])))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Average tree size
        ax1.plot(iterations, mcts_data['avg_tree_size'])
        ax1.set_title('Average MCTS Tree Size')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Number of Nodes')
        ax1.grid(True)
        
        # Search time
        ax2.plot(iterations, mcts_data['avg_search_time'])
        ax2.set_title('Average Search Time')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'mcts_stats.png'), dpi=300)
        plt.show()
    
    def visualize_board_state(self, board, title="Go Board"):
        """Visualize a Go board state."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
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
        
        # Mark star points
        star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for x, y in star_points:
            ax.plot(x, 18-y, 'ko', markersize=3)
        
        ax.set_xlim(-0.5, 18.5)
        ax.set_ylim(-0.5, 18.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xticks(range(19))
        ax.set_yticks(range(19))
        ax.set_xticklabels([chr(ord('A') + i) if i < 8 else chr(ord('A') + i + 1) for i in range(19)])
        ax.set_yticklabels(range(19, 0, -1))
        
        plt.tight_layout()
        return fig, ax
    
    def plot_training_summary(self):
        """Create a comprehensive training summary plot."""
        fig = plt.figure(figsize=(16, 12))
        
        # Plot training losses
        try:
            plt.subplot(2, 3, 1)
            self.plot_training_loss()
        except:
            pass
        
        # Plot win rates
        try:
            plt.subplot(2, 3, 2)
            self.plot_win_rates()
        except:
            pass
        
        # Plot game lengths
        try:
            plt.subplot(2, 3, 3)
            self.plot_game_length_distribution()
        except:
            pass
        
        # Plot MCTS stats
        try:
            plt.subplot(2, 3, (4, 5))
            self.plot_mcts_tree_size()
        except:
            pass
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_summary.png'), dpi=300)
        plt.show()


def analyze_game_sgf(sgf_content):
    """Analyze a game from SGF content."""
    # This would parse SGF and extract game statistics
    # For now, just a placeholder
    print("SGF analysis functionality would go here")
    pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot AlphaGo Zero training metrics")
    parser.add_argument("--log_dir", default="logs", help="Directory containing log files")
    parser.add_argument("--plot_type", 
                       choices=["losses", "win_rates", "game_lengths", "mcts", "summary"],
                       default="summary", help="Type of plot to generate")
    
    args = parser.parse_args()
    
    plotter = TrainingPlotter(args.log_dir)
    
    if args.plot_type == "losses":
        plotter.plot_training_loss()
    elif args.plot_type == "win_rates":
        plotter.plot_win_rates()
    elif args.plot_type == "game_lengths":
        plotter.plot_game_length_distribution()
    elif args.plot_type == "mcts":
        plotter.plot_mcts_tree_size()
    elif args.plot_type == "summary":
        plotter.plot_training_summary()