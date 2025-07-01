"""Expert evaluation system using professional game data."""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from alpha_zero.envs import GoEnv
from alpha_zero.core import MCTS
from alpha_zero.utils import SGFWrapper
import requests
from io import StringIO


class ExpertGameDatabase:
    """Database of professional Go games for evaluation."""
    
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Use D drive directory by default
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, "expert_games")
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Download some professional games if not available
        self.ensure_expert_games()
        
    def ensure_expert_games(self):
        """Download or create a database of expert games."""
        
        # Sample expert game positions (simplified for demo)
        # In a real implementation, you would download from sources like:
        # - GoGoD database
        # - Online Go servers (OGS, KGS)
        # - Professional tournament databases
        
        expert_positions_file = os.path.join(self.data_dir, "expert_positions.json")
        
        if not os.path.exists(expert_positions_file):
            # Create sample expert positions for testing
            expert_positions = self._create_sample_expert_positions()
            
            with open(expert_positions_file, 'w') as f:
                json.dump(expert_positions, f, indent=2)
                
        # Download famous games SGF files (URLs would be real in production)
        self._download_famous_games()
        
    def _create_sample_expert_positions(self):
        """Create sample expert positions for testing."""
        
        # These are simplified positions - in reality you'd have real professional games
        positions = []
        
        # Sample opening positions
        positions.append({
            'position_id': 'fuseki_1',
            'moves': [60, 159, 300, 18],  # Sample moves on 19x19 board
            'expert_move': 174,
            'strength': 'professional',
            'source': 'sample_opening',
            'difficulty': 'medium'
        })
        
        positions.append({
            'position_id': 'joseki_1', 
            'moves': [60, 159, 300, 18, 174, 158],
            'expert_move': 175,
            'strength': 'professional',
            'source': 'sample_joseki',
            'difficulty': 'hard'
        })
        
        # Add more sample positions for different game phases
        for i in range(20):
            positions.append({
                'position_id': f'sample_{i}',
                'moves': list(np.random.choice(361, size=np.random.randint(5, 30), replace=False)),
                'expert_move': int(np.random.choice(361)),
                'strength': 'professional',
                'source': 'random_sample',
                'difficulty': np.random.choice(['easy', 'medium', 'hard'])
            })
            
        return positions
    
    def _download_famous_games(self):
        """Download famous Go games (placeholder implementation)."""
        
        # In a real implementation, you would download from sources like:
        # - https://www.go4go.net/
        # - https://gogameguru.com/
        # - Professional tournament websites
        
        famous_games_file = os.path.join(self.data_dir, "famous_games.json")
        
        if not os.path.exists(famous_games_file):
            # Sample famous games metadata
            famous_games = [
                {
                    'game_id': 'lee_sedol_vs_alphago_game1',
                    'black_player': 'Lee Sedol',
                    'white_player': 'AlphaGo',
                    'result': 'W+R',
                    'date': '2016-03-09',
                    'moves': list(range(0, 200, 3)),  # Sample moves
                    'notable_positions': [37, 78, 102, 145]
                },
                {
                    'game_id': 'lee_changho_vs_cho_hunhyun',
                    'black_player': 'Lee Changho',
                    'white_player': 'Cho Hunhyun', 
                    'result': 'B+2.5',
                    'date': '1995-07-15',
                    'moves': list(range(1, 180, 2)),
                    'notable_positions': [45, 89, 123, 156]
                }
            ]
            
            with open(famous_games_file, 'w') as f:
                json.dump(famous_games, f, indent=2)
    
    def load_expert_positions(self) -> List[Dict]:
        """Load expert positions for evaluation."""
        positions_file = os.path.join(self.data_dir, "expert_positions.json")
        
        with open(positions_file, 'r') as f:
            return json.load(f)
    
    def load_famous_games(self) -> List[Dict]:
        """Load famous games for analysis."""
        games_file = os.path.join(self.data_dir, "famous_games.json")
        
        with open(games_file, 'r') as f:
            return json.load(f)


class ExpertEvaluator:
    """Evaluates AlphaGo models against expert play."""
    
    def __init__(self, model, config, expert_db: ExpertGameDatabase):
        self.model = model
        self.config = config
        self.expert_db = expert_db
        self.device = next(model.parameters()).device
        
    def evaluate_position_accuracy(self, num_positions: int = 50) -> Dict:
        """Evaluate how often the model chooses the same move as experts."""
        
        expert_positions = self.expert_db.load_expert_positions()
        if len(expert_positions) > num_positions:
            expert_positions = np.random.choice(expert_positions, num_positions, replace=False)
        
        results = {
            'total_positions': len(expert_positions),
            'exact_matches': 0,
            'top3_matches': 0,
            'top5_matches': 0,
            'position_details': [],
            'difficulty_breakdown': {'easy': {'total': 0, 'correct': 0},
                                   'medium': {'total': 0, 'correct': 0},
                                   'hard': {'total': 0, 'correct': 0}}
        }
        
        for position in expert_positions:
            # Set up position
            env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
            
            # Play moves to reach position
            for move in position['moves']:
                if move < env.action_dim:  # Valid move
                    try:
                        env.step(move)
                    except:
                        continue  # Skip invalid moves
            
            # Get model's move prediction
            mcts = MCTS(env, self.config.get_mcts_args(), self.model)
            pi_probs = mcts.search()
            
            # Get top moves
            top_moves = np.argsort(pi_probs)[::-1]
            expert_move = position['expert_move']
            
            # Check accuracy
            exact_match = top_moves[0] == expert_move
            top3_match = expert_move in top_moves[:3]
            top5_match = expert_move in top_moves[:5]
            
            if exact_match:
                results['exact_matches'] += 1
            if top3_match:
                results['top3_matches'] += 1
            if top5_match:
                results['top5_matches'] += 1
            
            # Track by difficulty
            difficulty = position['difficulty']
            results['difficulty_breakdown'][difficulty]['total'] += 1
            if exact_match:
                results['difficulty_breakdown'][difficulty]['correct'] += 1
            
            # Store detailed results
            results['position_details'].append({
                'position_id': position['position_id'],
                'expert_move': expert_move,
                'model_top_move': int(top_moves[0]),
                'model_top3': [int(m) for m in top_moves[:3]],
                'exact_match': exact_match,
                'top3_match': top3_match,
                'move_probability': float(pi_probs[expert_move]),
                'top_move_probability': float(pi_probs[top_moves[0]]),
                'difficulty': difficulty
            })
        
        # Calculate final accuracies
        total = results['total_positions']
        results['exact_accuracy'] = results['exact_matches'] / total
        results['top3_accuracy'] = results['top3_matches'] / total
        results['top5_accuracy'] = results['top5_matches'] / total
        
        return results
    
    def evaluate_vs_expert_games(self, num_games: int = 10) -> Dict:
        """Play model against expert game continuations."""
        
        famous_games = self.expert_db.load_famous_games()
        selected_games = np.random.choice(famous_games, 
                                        min(num_games, len(famous_games)), 
                                        replace=False)
        
        results = {
            'total_games': len(selected_games),
            'games_won': 0,
            'games_lost': 0,
            'games_drawn': 0,
            'average_score_difference': 0,
            'game_details': []
        }
        
        for game in selected_games:
            game_result = self._play_vs_expert_game(game)
            results['game_details'].append(game_result)
            
            if game_result['result'] == 'win':
                results['games_won'] += 1
            elif game_result['result'] == 'loss':
                results['games_lost'] += 1
            else:
                results['games_drawn'] += 1
        
        # Calculate win rate
        results['win_rate'] = results['games_won'] / results['total_games']
        results['average_score_difference'] = np.mean([g['score_difference'] 
                                                      for g in results['game_details']])
        
        return results
    
    def _play_vs_expert_game(self, expert_game: Dict) -> Dict:
        """Play model against an expert game continuation."""
        
        env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
        
        # Play first part of expert game
        expert_moves = expert_game['moves']
        split_point = len(expert_moves) // 2  # Take over at middle game
        
        # Play expert moves up to split point
        for i, move in enumerate(expert_moves[:split_point]):
            if move < env.action_dim and not env.is_game_over():
                try:
                    env.step(move)
                except:
                    break
        
        # Model takes over for one side, expert continuation for other
        model_is_black = (split_point % 2 == 0)
        model_moves = []
        expert_continuation = expert_moves[split_point:]
        
        move_index = 0
        while not env.is_game_over() and move_index < len(expert_continuation):
            current_player_is_model = (env.to_play == env.black_player) == model_is_black
            
            if current_player_is_model:
                # Model's turn
                mcts = MCTS(env, self.config.get_mcts_args(), self.model)
                pi_probs = mcts.search()
                action = np.argmax(pi_probs)
                model_moves.append(action)
            else:
                # Expert's turn
                if move_index < len(expert_continuation):
                    action = expert_continuation[move_index]
                    move_index += 1
                else:
                    break
            
            try:
                env.step(action)
            except:
                break
        
        # Determine result
        if env.winner is None:
            result = 'draw'
            score_diff = 0
        elif (env.winner == env.black_player) == model_is_black:
            result = 'win'
            score_diff = 1
        else:
            result = 'loss'
            score_diff = -1
        
        return {
            'game_id': expert_game['game_id'],
            'model_color': 'black' if model_is_black else 'white',
            'result': result,
            'score_difference': score_diff,
            'model_moves': model_moves,
            'game_length': env.steps
        }
    
    def analyze_playing_style(self) -> Dict:
        """Analyze the model's playing style compared to experts."""
        
        style_metrics = {
            'aggressiveness': 0,  # How often model plays fighting moves
            'territory_focus': 0,  # Focus on territory vs influence
            'opening_similarity': 0,  # Similarity to common openings
            'endgame_precision': 0,  # Accuracy in endgame
            'move_variety': 0  # Diversity of move choices
        }
        
        # This would involve complex analysis of move patterns
        # For now, return placeholder metrics
        return style_metrics
    
    def generate_strength_estimate(self, position_accuracy: float, game_win_rate: float) -> Dict:
        """Estimate model strength in human terms."""
        
        # Rough mapping based on accuracy and win rate
        if position_accuracy > 0.7 and game_win_rate > 0.6:
            estimated_strength = "Professional level"
            estimated_rank = "6-9 dan"
        elif position_accuracy > 0.5 and game_win_rate > 0.5:
            estimated_strength = "Strong amateur"
            estimated_rank = "1-5 dan"
        elif position_accuracy > 0.3 and game_win_rate > 0.4:
            estimated_strength = "Amateur"
            estimated_rank = "1-5 kyu"
        else:
            estimated_strength = "Beginner"
            estimated_rank = "6+ kyu"
        
        return {
            'estimated_strength': estimated_strength,
            'estimated_rank': estimated_rank,
            'position_accuracy': position_accuracy,
            'game_win_rate': game_win_rate,
            'confidence': 'medium'  # Would calculate confidence based on sample size
        }


def evaluate_model_vs_experts(model, config, num_positions: int = 50, 
                             num_games: int = 10) -> Dict:
    """Complete expert evaluation of a model."""
    
    expert_db = ExpertGameDatabase()
    evaluator = ExpertEvaluator(model, config, expert_db)
    
    print("Evaluating position accuracy...")
    position_results = evaluator.evaluate_position_accuracy(num_positions)
    
    print("Evaluating game continuation...")
    game_results = evaluator.evaluate_vs_expert_games(num_games)
    
    print("Analyzing playing style...")
    style_analysis = evaluator.analyze_playing_style()
    
    print("Generating strength estimate...")
    strength_estimate = evaluator.generate_strength_estimate(
        position_results['exact_accuracy'],
        game_results['win_rate']
    )
    
    return {
        'position_accuracy': position_results,
        'game_performance': game_results,
        'style_analysis': style_analysis,
        'strength_estimate': strength_estimate,
        'overall_score': (position_results['exact_accuracy'] + game_results['win_rate']) / 2
    }


if __name__ == "__main__":
    # Test the expert evaluation system
    from alpha_zero.core import ResNet
    from alpha_zero.envs import GoEnv
    from config import Config
    
    config = Config()
    env = GoEnv(komi=config.komi, num_stack=config.num_stack)
    
    # Create a dummy model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(env, config.num_res_blocks, config.num_hidden, device)
    
    # Run evaluation
    results = evaluate_model_vs_experts(model, config, num_positions=10, num_games=3)
    
    print("\n=== Expert Evaluation Results ===")
    print(f"Position Accuracy: {results['position_accuracy']['exact_accuracy']:.3f}")
    print(f"Game Win Rate: {results['game_performance']['win_rate']:.3f}")
    print(f"Estimated Strength: {results['strength_estimate']['estimated_strength']}")
    print(f"Overall Score: {results['overall_score']:.3f}")