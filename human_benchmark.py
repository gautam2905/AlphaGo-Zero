"""Human performance benchmarking for AlphaGo Zero.

This module provides tools to compare AI performance against human players
of various skill levels, from beginner to professional.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

from alpha_zero.core import MCTS
from alpha_zero.envs import GoEnv


class HumanSkillLevel:
    """Standard human skill levels with corresponding Elo ratings."""
    
    # Based on standard Go rating systems (KGS, EGF, AGA)
    SKILL_LEVELS = {
        'Beginner': {
            'elo': 100,
            'kyu': '30k',
            'description': 'Just learned the rules',
            'win_rate_vs_random': 0.6,
            'typical_mistakes': ['random moves', 'no pattern recognition']
        },
        'Novice': {
            'elo': 500,
            'kyu': '20k',
            'description': 'Understands basic tactics',
            'win_rate_vs_random': 0.8,
            'typical_mistakes': ['weak shape', 'no strategy']
        },
        'Intermediate': {
            'elo': 1000,
            'kyu': '10k',
            'description': 'Solid tactical understanding',
            'win_rate_vs_random': 0.95,
            'typical_mistakes': ['poor timing', 'weak positional judgment']
        },
        'Advanced': {
            'elo': 1500,
            'kyu': '5k',
            'description': 'Strong amateur player',
            'win_rate_vs_random': 0.99,
            'typical_mistakes': ['subtle strategic errors']
        },
        'Expert': {
            'elo': 1800,
            'kyu': '1k',
            'description': 'Very strong amateur',
            'win_rate_vs_random': 0.999,
            'typical_mistakes': ['minor inefficiencies']
        },
        'Master': {
            'elo': 2100,
            'dan': '1d',
            'description': 'Dan level player',
            'win_rate_vs_random': 0.9999,
            'typical_mistakes': ['rare tactical oversights']
        },
        'Strong_Amateur': {
            'elo': 2400,
            'dan': '4d',
            'description': 'Very strong dan player',
            'win_rate_vs_random': 1.0,
            'typical_mistakes': ['very subtle positional misjudgments']
        },
        'Professional': {
            'elo': 2700,
            'dan': '7d+',
            'description': 'Professional level',
            'win_rate_vs_random': 1.0,
            'typical_mistakes': ['extremely rare']
        },
        'Top_Professional': {
            'elo': 3000,
            'dan': '9d',
            'description': 'World champion level',
            'win_rate_vs_random': 1.0,
            'typical_mistakes': ['virtually none']
        }
    }
    
    # Historical AI milestones for comparison
    AI_MILESTONES = {
        'Deep_Blue_Era': {'elo': 1800, 'year': 1997, 'description': 'Chess AI level projected to Go'},
        'Monte_Carlo_Go': {'elo': 2200, 'year': 2006, 'description': 'Early MCTS programs'},
        'Crazy_Stone_6d': {'elo': 2500, 'year': 2013, 'description': 'First strong MCTS+NN'},
        'AlphaGo_Fan': {'elo': 3000, 'year': 2015, 'description': 'Defeated Fan Hui 2p'},
        'AlphaGo_Lee': {'elo': 3500, 'year': 2016, 'description': 'Defeated Lee Sedol 9p'},
        'AlphaGo_Master': {'elo': 4000, 'year': 2017, 'description': '60-0 online record'},
        'AlphaGo_Zero': {'elo': 5000, 'year': 2017, 'description': 'Surpassed all versions'}
    }


class EloRatingSystem:
    """Elo rating calculation for Go."""
    
    def __init__(self, k_factor: int = 32):
        self.k_factor = k_factor
        
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, rating_a: float, rating_b: float, 
                      score_a: float) -> Tuple[float, float]:
        """Update ratings based on game result.
        
        Args:
            rating_a: Current rating of player A
            rating_b: Current rating of player B
            score_a: Actual score (1.0 for win, 0.5 for draw, 0.0 for loss)
            
        Returns:
            Updated ratings for both players
        """
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - score_a) - expected_b)
        
        return new_rating_a, new_rating_b
    
    def win_probability(self, rating_a: float, rating_b: float) -> float:
        """Calculate win probability for player A against player B."""
        return self.expected_score(rating_a, rating_b)


class HumanBenchmark:
    """Benchmark AI performance against human skill levels."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.elo_system = EloRatingSystem()
        self.benchmark_results = defaultdict(list)
        
        # Initialize simulated human players
        self.human_simulators = self._create_human_simulators()
        
        # Current AI Elo (starts at beginner level)
        self.current_ai_elo = 100
        
    def _create_human_simulators(self) -> Dict:
        """Create simulated players for each human skill level."""
        simulators = {}
        
        for level, info in HumanSkillLevel.SKILL_LEVELS.items():
            simulators[level] = {
                'elo': info['elo'],
                'play_strength': self._get_play_strength(info['elo']),
                'info': info
            }
            
        return simulators
    
    def _get_play_strength(self, elo: float) -> Dict:
        """Convert Elo rating to play characteristics."""
        # Map Elo to MCTS simulations and temperature
        # Higher Elo = more simulations, lower temperature
        
        if elo < 500:  # Beginner
            return {'simulations': 10, 'temperature': 2.0, 'resign_threshold': -0.99}
        elif elo < 1000:  # Novice
            return {'simulations': 50, 'temperature': 1.5, 'resign_threshold': -0.95}
        elif elo < 1500:  # Intermediate
            return {'simulations': 100, 'temperature': 1.0, 'resign_threshold': -0.9}
        elif elo < 2000:  # Advanced
            return {'simulations': 200, 'temperature': 0.7, 'resign_threshold': -0.85}
        elif elo < 2500:  # Expert
            return {'simulations': 400, 'temperature': 0.5, 'resign_threshold': -0.8}
        else:  # Master+
            return {'simulations': 800, 'temperature': 0.3, 'resign_threshold': -0.75}
    
    def benchmark_against_humans(self, model, num_games: int = 10) -> Dict:
        """Benchmark model against all human skill levels."""
        results = {}
        model.eval()
        
        print("\n" + "="*60)
        print("HUMAN PERFORMANCE BENCHMARK")
        print("="*60)
        
        for level, simulator in self.human_simulators.items():
            wins = 0
            draws = 0
            game_records = []
            
            print(f"\nTesting against {level} ({simulator['info']['description']})...")
            print(f"Elo: {simulator['elo']}, Level: {simulator['info'].get('kyu', simulator['info'].get('dan', 'N/A'))}")
            
            for game_idx in range(num_games):
                # Alternate colors
                if game_idx % 2 == 0:
                    ai_color = 'black'
                    result, game_record = self._play_game(model, simulator, ai_plays_black=True)
                else:
                    ai_color = 'white'
                    result, game_record = self._play_game(model, simulator, ai_plays_black=False)
                
                if result == 1:  # AI wins
                    wins += 1
                elif result == 0:  # Draw
                    draws += 1
                
                game_records.append(game_record)
                
                # Update Elo ratings
                if result == 1:
                    self.current_ai_elo, _ = self.elo_system.update_ratings(
                        self.current_ai_elo, simulator['elo'], 1.0
                    )
                elif result == -1:
                    self.current_ai_elo, _ = self.elo_system.update_ratings(
                        self.current_ai_elo, simulator['elo'], 0.0
                    )
                else:  # Draw
                    self.current_ai_elo, _ = self.elo_system.update_ratings(
                        self.current_ai_elo, simulator['elo'], 0.5
                    )
            
            win_rate = wins / num_games
            expected_win_rate = self.elo_system.win_probability(self.current_ai_elo, simulator['elo'])
            
            results[level] = {
                'wins': wins,
                'losses': num_games - wins - draws,
                'draws': draws,
                'win_rate': win_rate,
                'expected_win_rate': expected_win_rate,
                'game_records': game_records,
                'opponent_elo': simulator['elo']
            }
            
            print(f"Results: {wins}W-{num_games-wins-draws}L-{draws}D (Win rate: {win_rate:.1%})")
            print(f"Expected win rate based on Elo: {expected_win_rate:.1%}")
        
        # Determine AI's skill level
        skill_level = self._determine_skill_level(results)
        
        print("\n" + "="*60)
        print(f"Current AI Elo Rating: {self.current_ai_elo:.0f}")
        print(f"Estimated Skill Level: {skill_level}")
        print("="*60)
        
        # Store benchmark results
        self.benchmark_results['results'].append({
            'timestamp': datetime.now().isoformat(),
            'ai_elo': self.current_ai_elo,
            'skill_level': skill_level,
            'detailed_results': results
        })
        
        return {
            'ai_elo': self.current_ai_elo,
            'skill_level': skill_level,
            'results': results
        }
    
    def _play_game(self, model, simulator, ai_plays_black: bool) -> Tuple[int, Dict]:
        """Play a single game between AI and simulated human."""
        env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
        
        # Create MCTS for both players
        ai_mcts = MCTS(env, self.config.get_mcts_args(), model)
        
        # Simulated human uses weaker MCTS settings
        human_args = self.config.get_mcts_args().copy()
        human_args['num_searches'] = simulator['play_strength']['simulations']
        human_mcts = MCTS(env, human_args, model)  # Uses same model but different search
        
        game_record = {
            'moves': [],
            'positions': [],
            'ai_plays_black': ai_plays_black
        }
        
        while not env.is_game_over() and env.steps < 200:  # Limit game length
            current_player_is_ai = (env.to_play == env.black_player) == ai_plays_black
            
            if current_player_is_ai:
                # AI move
                pi_probs = ai_mcts.search()
                if self.config.eval_temperature == 0:
                    action = np.argmax(pi_probs)
                else:
                    pi_temp = np.power(pi_probs, 1/self.config.eval_temperature)
                    pi_temp = pi_temp / np.sum(pi_temp)
                    action = np.random.choice(len(pi_probs), p=pi_temp)
            else:
                # Simulated human move
                pi_probs = human_mcts.search()
                temperature = simulator['play_strength']['temperature']
                
                # Add some randomness for lower-level players
                if simulator['elo'] < 1000:
                    # Occasionally make random moves
                    if np.random.random() < 0.1:
                        valid_moves = env.get_valid_moves()
                        valid_indices = np.where(valid_moves)[0]
                        action = np.random.choice(valid_indices)
                    else:
                        pi_temp = np.power(pi_probs, 1/temperature)
                        pi_temp = pi_temp / np.sum(pi_temp)
                        action = np.random.choice(len(pi_probs), p=pi_temp)
                else:
                    pi_temp = np.power(pi_probs, 1/temperature)
                    pi_temp = pi_temp / np.sum(pi_temp)
                    action = np.random.choice(len(pi_probs), p=pi_temp)
            
            game_record['moves'].append(action)
            game_record['positions'].append(env.observation().copy())
            env.step(action)
        
        # Determine winner
        if env.winner == env.black_player:
            result = 1 if ai_plays_black else -1
        elif env.winner == env.white_player:
            result = -1 if ai_plays_black else 1
        else:
            result = 0  # Draw
        
        game_record['result'] = result
        game_record['final_position'] = env.observation().copy()
        
        return result, game_record
    
    def _determine_skill_level(self, results: Dict) -> str:
        """Determine AI's skill level based on benchmark results."""
        # Find the highest level where AI has >50% win rate
        skill_estimate = 'Below Beginner'
        
        for level in ['Beginner', 'Novice', 'Intermediate', 'Advanced', 
                     'Expert', 'Master', 'Strong_Amateur', 'Professional', 'Top_Professional']:
            if level in results and results[level]['win_rate'] >= 0.5:
                skill_estimate = level
            else:
                break
        
        return skill_estimate
    
    def plot_benchmark_results(self, save_path: str = None):
        """Plot benchmark results comparing AI to human skill levels."""
        if not self.benchmark_results['results']:
            print("No benchmark results to plot")
            return
        
        latest_results = self.benchmark_results['results'][-1]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Win rates against different skill levels
        levels = []
        win_rates = []
        elos = []
        
        for level, info in HumanSkillLevel.SKILL_LEVELS.items():
            if level in latest_results['detailed_results']:
                levels.append(level.replace('_', ' '))
                win_rates.append(latest_results['detailed_results'][level]['win_rate'])
                elos.append(info['elo'])
        
        ax1.bar(levels, win_rates, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
        ax1.set_xlabel('Human Skill Level')
        ax1.set_ylabel('AI Win Rate')
        ax1.set_title('AI Performance vs Human Skill Levels')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Elo progression
        iterations = list(range(len(self.benchmark_results['results'])))
        ai_elos = [r['ai_elo'] for r in self.benchmark_results['results']]
        
        ax2.plot(iterations, ai_elos, 'b-', linewidth=2, marker='o')
        
        # Add human skill level lines
        for level, info in HumanSkillLevel.SKILL_LEVELS.items():
            ax2.axhline(y=info['elo'], color='gray', linestyle='--', alpha=0.5)
            ax2.text(len(iterations)-1, info['elo'], level.replace('_', ' '), 
                    fontsize=8, ha='right', va='bottom')
        
        ax2.set_xlabel('Benchmark Iteration')
        ax2.set_ylabel('Elo Rating')
        ax2.set_title('AI Elo Rating Progression')
        ax2.grid(True, alpha=0.3)
        
        # 3. Comparison with AI milestones
        ai_years = [info['year'] for info in HumanSkillLevel.AI_MILESTONES.values()]
        ai_elos = [info['elo'] for info in HumanSkillLevel.AI_MILESTONES.values()]
        ai_names = list(HumanSkillLevel.AI_MILESTONES.keys())
        
        ax3.scatter(ai_years, ai_elos, s=100, c='red', marker='*', label='Historical AI')
        ax3.axhline(y=latest_results['ai_elo'], color='blue', linewidth=2, 
                   label=f'Current AI ({latest_results["ai_elo"]:.0f})')
        
        for i, name in enumerate(ai_names):
            ax3.annotate(name.replace('_', ' '), (ai_years[i], ai_elos[i]), 
                        fontsize=8, ha='center', va='bottom')
        
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Elo Rating')
        ax3.set_title('AI Progress in Go: Historical Context')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Skill distribution pie chart
        skill_distribution = {
            'Defeats': 0,
            'Competitive': 0,
            'Dominated by': 0
        }
        
        for level, result in latest_results['detailed_results'].items():
            if result['win_rate'] >= 0.7:
                skill_distribution['Defeats'] += 1
            elif result['win_rate'] >= 0.3:
                skill_distribution['Competitive'] += 1
            else:
                skill_distribution['Dominated by'] += 1
        
        ax4.pie(skill_distribution.values(), labels=skill_distribution.keys(), 
               autopct='%1.0f%%', startangle=90, colors=['lightgreen', 'yellow', 'lightcoral'])
        ax4.set_title(f'Performance Distribution\n(Current Level: {latest_results["skill_level"]})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.config.log_dir, 'human_benchmark.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate detailed comparison plot
        self._plot_detailed_comparison(latest_results)
    
    def _plot_detailed_comparison(self, results):
        """Create a detailed comparison visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a visual Elo scale
        elo_range = np.arange(0, 5500, 100)
        
        # Plot human skill levels as horizontal bars
        for i, (level, info) in enumerate(HumanSkillLevel.SKILL_LEVELS.items()):
            elo = info['elo']
            ax.barh(i, 200, left=elo-100, height=0.6, alpha=0.3, color='lightblue',
                   label='Human Range' if i == 0 else '')
            ax.text(elo, i, level.replace('_', ' '), ha='center', va='center', fontweight='bold')
        
        # Plot AI milestones
        for j, (ai_name, info) in enumerate(HumanSkillLevel.AI_MILESTONES.items()):
            y_pos = len(HumanSkillLevel.SKILL_LEVELS) + j * 0.7
            ax.scatter(info['elo'], y_pos, s=200, marker='*', color='red', zorder=5)
            ax.text(info['elo'], y_pos + 0.3, f"{ai_name.replace('_', ' ')} ({info['year']})", 
                   ha='center', fontsize=8)
        
        # Plot current AI position
        ax.axvline(x=results['ai_elo'], color='blue', linewidth=3, linestyle='-', 
                  label=f'Current AI ({results["ai_elo"]:.0f})')
        
        # Formatting
        ax.set_xlim(0, 5500)
        ax.set_xlabel('Elo Rating', fontsize=12)
        ax.set_title('Go AI Performance: Human and AI Comparison', fontsize=16)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_yticks([])
        ax.legend(loc='upper right')
        
        # Add Elo scale reference
        ax.text(0.02, 0.98, 'Elo Difference → Expected Score:\n'
                           '0 → 50%\n'
                           '200 → 75%\n'
                           '400 → 90%\n'
                           '800 → 99%',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'detailed_ai_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_benchmark_results(self):
        """Save benchmark results to file."""
        results_path = os.path.join(self.config.log_dir, 'human_benchmark_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2, default=str)
        print(f"Benchmark results saved to {results_path}")
    
    def get_skill_description(self, elo: float) -> str:
        """Get a human-readable description of skill level based on Elo."""
        for level, info in HumanSkillLevel.SKILL_LEVELS.items():
            if elo <= info['elo'] + 100:
                return f"{level.replace('_', ' ')} level ({info['description']})"
        return "Beyond Top Professional level"