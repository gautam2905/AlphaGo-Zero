"""SGF Human Game Data Integration for AlphaGo Zero Benchmarking.

This module handles downloading, parsing, and utilizing professional Go game
records from various online sources for benchmarking AI performance.
"""

import os
import requests
import zipfile
import tarfile
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sgf
from tqdm import tqdm
import sqlite3
from datetime import datetime
import re

from alpha_zero.utils.sgf_wrapper import SgfWrapper
from alpha_zero.envs import GoEnv


class SGFDataSource:
    """Available SGF data sources with professional games."""
    
    SOURCES = {
        'KGS': {
            'url': 'https://www.u-go.net/gamerecords/',
            'description': 'KGS Go Server high-dan games',
            'min_rank': '6d',
            'game_count': '~100k'
        },
        'GoGoD': {
            'url': 'https://gogodonline.co.uk/',
            'description': 'Games of Go on Disk - Professional games database',
            'min_rank': 'Pro',
            'game_count': '~100k'
        },
        'Kifu': {
            'url': 'https://github.com/featurecat/go-dataset',
            'description': 'Collection of professional games',
            'min_rank': 'Pro',
            'game_count': '~170k'
        },
        'AlphaGo_Games': {
            'url': 'https://github.com/yenw/AlphaGo-Games',
            'description': 'AlphaGo match games collection',
            'min_rank': 'Pro',
            'game_count': '~100'
        },
        'Pro_Games_Collection': {
            'url': 'https://github.com/pwmarcz/sgflib',
            'description': 'Curated professional games',
            'min_rank': 'Pro',
            'game_count': '~50k'
        },
        'GoBase': {
            'url': 'http://gobase.org/',
            'description': 'Historical professional games',
            'min_rank': 'Pro',
            'game_count': '~80k'
        }
    }
    
    # Direct downloadable datasets
    DOWNLOADABLE_SOURCES = {
        'minigo_dataset': {
            'url': 'https://github.com/tensorflow/minigo/tree/master/data',
            'description': 'Minigo training games'
        },
        'leela_zero_dataset': {
            'url': 'https://github.com/leela-zero/leela-zero/tree/master/data',
            'description': 'Leela Zero training games'
        },
        'go4go_dataset': {
            'url': 'http://www.go4go.net/go/games/sgffiles',
            'description': 'Professional game collection'
        }
    }


class HumanGameDatabase:
    """Database for storing and querying human game records."""
    
    def __init__(self, db_path: str = "human_games.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        
    def _create_tables(self):
        """Create database tables for storing game information."""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sgf_content TEXT NOT NULL,
                black_player TEXT,
                white_player TEXT,
                black_rank TEXT,
                white_rank TEXT,
                result TEXT,
                date_played TEXT,
                komi REAL,
                board_size INTEGER,
                game_name TEXT,
                source TEXT,
                elo_estimate INTEGER,
                moves_count INTEGER,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                move_number INTEGER,
                board_state TEXT,
                next_move INTEGER,
                player_to_move INTEGER,
                FOREIGN KEY (game_id) REFERENCES games (id)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_rank ON games (black_rank, white_rank)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_elo ON games (elo_estimate)
        ''')
        
        self.conn.commit()
    
    def add_game(self, sgf_content: str, source: str) -> int:
        """Add a game to the database."""
        try:
            game = sgf.parse(sgf_content)[0]
            root = game.root
            
            # Extract game information
            black_player = root.properties.get('PB', ['Unknown'])[0]
            white_player = root.properties.get('PW', ['Unknown'])[0]
            black_rank = root.properties.get('BR', ['Unknown'])[0]
            white_rank = root.properties.get('WR', ['Unknown'])[0]
            result = root.properties.get('RE', ['Unknown'])[0]
            date_played = root.properties.get('DT', ['Unknown'])[0]
            komi = float(root.properties.get('KM', ['0'])[0])
            board_size = int(root.properties.get('SZ', ['19'])[0])
            game_name = root.properties.get('GN', [''])[0]
            
            # Estimate Elo based on rank
            elo_estimate = self._estimate_elo_from_rank(black_rank, white_rank)
            
            # Count moves
            moves_count = len(list(game.main_sequence())) - 1
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO games (
                    sgf_content, black_player, white_player, black_rank, white_rank,
                    result, date_played, komi, board_size, game_name, source,
                    elo_estimate, moves_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sgf_content, black_player, white_player, black_rank, white_rank,
                result, date_played, komi, board_size, game_name, source,
                elo_estimate, moves_count
            ))
            
            game_id = cursor.lastrowid
            self.conn.commit()
            
            return game_id
            
        except Exception as e:
            print(f"Error adding game: {e}")
            return -1
    
    def _estimate_elo_from_rank(self, black_rank: str, white_rank: str) -> int:
        """Estimate Elo rating from rank information."""
        def rank_to_elo(rank: str) -> int:
            rank = rank.lower()
            
            # Professional ranks
            if 'p' in rank or 'pro' in rank:
                if '9p' in rank or '9d' in rank:
                    return 3000
                elif '8p' in rank or '8d' in rank:
                    return 2900
                elif '7p' in rank or '7d' in rank:
                    return 2800
                elif '6p' in rank or '6d' in rank:
                    return 2700
                elif '5p' in rank or '5d' in rank:
                    return 2600
                elif '4p' in rank or '4d' in rank:
                    return 2500
                elif '3p' in rank or '3d' in rank:
                    return 2400
                elif '2p' in rank or '2d' in rank:
                    return 2300
                elif '1p' in rank or '1d' in rank:
                    return 2200
                else:
                    return 2500  # Default pro
            
            # Amateur dan ranks
            elif 'd' in rank and 'k' not in rank:
                match = re.search(r'(\d+)d', rank)
                if match:
                    dan = int(match.group(1))
                    return 2100 - (dan - 1) * 100
                return 2000
            
            # Kyu ranks
            elif 'k' in rank:
                match = re.search(r'(\d+)k', rank)
                if match:
                    kyu = int(match.group(1))
                    return 2000 - kyu * 100
                return 1000
            
            return 1500  # Default
        
        black_elo = rank_to_elo(black_rank)
        white_elo = rank_to_elo(white_rank)
        
        return (black_elo + white_elo) // 2
    
    def get_professional_games(self, min_elo: int = 2200, limit: int = 1000) -> List[Dict]:
        """Retrieve professional games above a certain Elo threshold."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, sgf_content, black_player, white_player, black_rank, 
                   white_rank, result, elo_estimate
            FROM games
            WHERE elo_estimate >= ?
            ORDER BY elo_estimate DESC
            LIMIT ?
        ''', (min_elo, limit))
        
        games = []
        for row in cursor.fetchall():
            games.append({
                'id': row[0],
                'sgf_content': row[1],
                'black_player': row[2],
                'white_player': row[3],
                'black_rank': row[4],
                'white_rank': row[5],
                'result': row[6],
                'elo_estimate': row[7]
            })
        
        return games
    
    def get_positions_by_skill(self, min_elo: int, max_elo: int, 
                              limit: int = 1000) -> List[Dict]:
        """Get positions from games within a skill range."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT p.board_state, p.next_move, p.player_to_move, g.elo_estimate
            FROM positions p
            JOIN games g ON p.game_id = g.id
            WHERE g.elo_estimate BETWEEN ? AND ?
            AND p.move_number BETWEEN 20 AND 200
            ORDER BY RANDOM()
            LIMIT ?
        ''', (min_elo, max_elo, limit))
        
        positions = []
        for row in cursor.fetchall():
            positions.append({
                'board_state': json.loads(row[0]),
                'next_move': row[1],
                'player_to_move': row[2],
                'elo_estimate': row[3]
            })
        
        return positions
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class SGFHumanBenchmark:
    """Benchmark AI against real human games from SGF files."""
    
    def __init__(self, config, data_dir: str = "sgf_data"):
        self.config = config
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.db = HumanGameDatabase(os.path.join(data_dir, "human_games.db"))
        self.sgf_wrapper = SgfWrapper()
        
    def download_sgf_datasets(self):
        """Download SGF datasets from various sources."""
        print("Downloading SGF datasets...")
        
        # Download from GitHub repositories
        datasets = {
            'alphago_games': {
                'url': 'https://raw.githubusercontent.com/yenw/AlphaGo-Games/master/',
                'files': [
                    'AlphaGo%20vs%20Lee%20Sedol%201.sgf',
                    'AlphaGo%20vs%20Lee%20Sedol%202.sgf',
                    'AlphaGo%20vs%20Lee%20Sedol%203.sgf',
                    'AlphaGo%20vs%20Lee%20Sedol%204.sgf',
                    'AlphaGo%20vs%20Lee%20Sedol%205.sgf'
                ]
            },
            'pro_collection': {
                'url': 'https://raw.githubusercontent.com/featurecat/go-dataset/master/AI/Other/',
                'files': ['alphago-vs-alphago-1.sgf', 'alphago-vs-alphago-2.sgf']
            }
        }
        
        for dataset_name, dataset_info in datasets.items():
            dataset_dir = os.path.join(self.data_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            for file_name in dataset_info['files']:
                try:
                    url = dataset_info['url'] + file_name
                    response = requests.get(url)
                    if response.status_code == 200:
                        file_path = os.path.join(dataset_dir, file_name.replace('%20', '_'))
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded: {file_name}")
                except Exception as e:
                    print(f"Failed to download {file_name}: {e}")
        
        # Download sample professional games dataset
        self._download_sample_pro_games()
        
    def _download_sample_pro_games(self):
        """Download a sample collection of professional games."""
        # This would typically download from a larger dataset
        # For now, we'll create a sample dataset structure
        
        sample_games = [
            {
                'url': 'https://raw.githubusercontent.com/featurecat/go-dataset/master/AI/AlphaGo/AG-vs-AG/AlphaGo%20vs%20AlphaGo%20-%201.sgf',
                'name': 'alphago_self_play_1.sgf'
            },
            {
                'url': 'https://raw.githubusercontent.com/featurecat/go-dataset/master/KGS/2017/01/18/Ke%20Jie-livego2.sgf',
                'name': 'ke_jie_sample.sgf'
            }
        ]
        
        pro_dir = os.path.join(self.data_dir, 'professional_games')
        os.makedirs(pro_dir, exist_ok=True)
        
        for game_info in sample_games:
            try:
                response = requests.get(game_info['url'])
                if response.status_code == 200:
                    file_path = os.path.join(pro_dir, game_info['name'])
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    print(f"Downloaded professional game: {game_info['name']}")
            except Exception as e:
                print(f"Failed to download {game_info['name']}: {e}")
    
    def load_sgf_files_to_db(self):
        """Load all downloaded SGF files into the database."""
        print("Loading SGF files into database...")
        
        loaded_count = 0
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.sgf'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            sgf_content = f.read()
                        
                        source = os.path.basename(root)
                        game_id = self.db.add_game(sgf_content, source)
                        
                        if game_id > 0:
                            loaded_count += 1
                            print(f"Loaded: {file}")
                            
                            # Extract positions from the game
                            self._extract_positions_from_game(game_id, sgf_content)
                            
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
        
        print(f"Total games loaded: {loaded_count}")
    
    def _extract_positions_from_game(self, game_id: int, sgf_content: str):
        """Extract board positions from a game for position-based evaluation."""
        try:
            game = sgf.parse(sgf_content)[0]
            env = GoEnv(board_size=19, komi=7.5)
            
            # Play through the game and record positions
            move_number = 0
            for node in game.main_sequence():
                if node.move:
                    color, (row, col) = node.move
                    
                    # Convert SGF coordinates to board coordinates
                    move = row * 19 + col
                    
                    # Store position before move
                    board_state = env.observation().tolist()
                    player_to_move = 0 if color == 'B' else 1
                    
                    cursor = self.db.conn.cursor()
                    cursor.execute('''
                        INSERT INTO positions (
                            game_id, move_number, board_state, next_move, player_to_move
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', (game_id, move_number, json.dumps(board_state), move, player_to_move))
                    
                    # Make the move
                    env.step(move)
                    move_number += 1
            
            self.db.conn.commit()
            
        except Exception as e:
            print(f"Error extracting positions: {e}")
    
    def benchmark_against_human_games(self, model, num_positions: int = 100):
        """Benchmark model against professional human moves."""
        print("\n" + "="*60)
        print("BENCHMARKING AGAINST REAL HUMAN GAMES")
        print("="*60)
        
        model.eval()
        results_by_level = {}
        
        # Define skill levels to test
        skill_levels = [
            ('Amateur Dan', 1800, 2200),
            ('Low Professional', 2200, 2500),
            ('Mid Professional', 2500, 2800),
            ('High Professional', 2800, 3200)
        ]
        
        for level_name, min_elo, max_elo in skill_levels:
            print(f"\nTesting against {level_name} players (Elo {min_elo}-{max_elo})...")
            
            # Get positions from games in this skill range
            positions = self.db.get_positions_by_skill(min_elo, max_elo, num_positions)
            
            if not positions:
                print(f"No positions found for {level_name}")
                continue
            
            correct_predictions = 0
            top3_accuracy = 0
            top5_accuracy = 0
            
            for pos_data in tqdm(positions, desc=f"Evaluating {level_name}"):
                # Create game state
                env = GoEnv(board_size=19, komi=7.5)
                # Reconstruct board state (simplified - would need proper reconstruction)
                
                # Get model prediction
                board_tensor = torch.tensor(pos_data['board_state'], dtype=torch.float32)
                board_tensor = board_tensor.unsqueeze(0).to(model.device)
                
                with torch.no_grad():
                    policy, value = model(board_tensor)
                    policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
                
                # Get top predictions
                top_moves = np.argsort(policy_probs)[-5:][::-1]
                human_move = pos_data['next_move']
                
                # Check accuracy
                if top_moves[0] == human_move:
                    correct_predictions += 1
                if human_move in top_moves[:3]:
                    top3_accuracy += 1
                if human_move in top_moves[:5]:
                    top5_accuracy += 1
            
            # Calculate accuracies
            total = len(positions)
            results_by_level[level_name] = {
                'top1_accuracy': correct_predictions / total,
                'top3_accuracy': top3_accuracy / total,
                'top5_accuracy': top5_accuracy / total,
                'positions_evaluated': total,
                'elo_range': (min_elo, max_elo)
            }
            
            print(f"Results for {level_name}:")
            print(f"  Top-1 Accuracy: {results_by_level[level_name]['top1_accuracy']:.1%}")
            print(f"  Top-3 Accuracy: {results_by_level[level_name]['top3_accuracy']:.1%}")
            print(f"  Top-5 Accuracy: {results_by_level[level_name]['top5_accuracy']:.1%}")
        
        # Generate visualization
        self._plot_human_game_benchmark(results_by_level)
        
        return results_by_level
    
    def _plot_human_game_benchmark(self, results: Dict):
        """Plot benchmark results against human games."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        levels = list(results.keys())
        top1_acc = [results[l]['top1_accuracy'] for l in levels]
        top3_acc = [results[l]['top3_accuracy'] for l in levels]
        top5_acc = [results[l]['top5_accuracy'] for l in levels]
        
        # Plot 1: Accuracy by level
        x = np.arange(len(levels))
        width = 0.25
        
        ax1.bar(x - width, top1_acc, width, label='Top-1', color='blue', alpha=0.7)
        ax1.bar(x, top3_acc, width, label='Top-3', color='green', alpha=0.7)
        ax1.bar(x + width, top5_acc, width, label='Top-5', color='red', alpha=0.7)
        
        ax1.set_xlabel('Player Level')
        ax1.set_ylabel('Move Prediction Accuracy')
        ax1.set_title('AI Move Prediction vs Human Professional Games')
        ax1.set_xticks(x)
        ax1.set_xticklabels(levels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs Elo
        elos = [(results[l]['elo_range'][0] + results[l]['elo_range'][1]) / 2 for l in levels]
        
        ax2.plot(elos, top1_acc, 'bo-', label='Top-1', linewidth=2, markersize=8)
        ax2.plot(elos, top3_acc, 'go-', label='Top-3', linewidth=2, markersize=8)
        ax2.plot(elos, top5_acc, 'ro-', label='Top-5', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Average Elo Rating')
        ax2.set_ylabel('Move Prediction Accuracy')
        ax2.set_title('Prediction Accuracy vs Player Strength')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'human_game_benchmark.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed analysis plot
        self._create_detailed_analysis(results)
    
    def _create_detailed_analysis(self, results: Dict):
        """Create a detailed analysis visualization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Professional player reference lines
        pro_levels = {
            'Top Amateur (1d)': 0.15,
            'Low Professional (1p)': 0.25,
            'Mid Professional (5p)': 0.35,
            'High Professional (9p)': 0.45,
            'Top AI (AlphaGo)': 0.57
        }
        
        for level, accuracy in pro_levels.items():
            ax.axhline(y=accuracy, color='gray', linestyle='--', alpha=0.5)
            ax.text(0.02, accuracy + 0.01, level, fontsize=8, 
                   transform=ax.get_yaxis_transform())
        
        # Plot actual results
        levels = list(results.keys())
        top1_acc = [results[l]['top1_accuracy'] for l in levels]
        
        bars = ax.bar(levels, top1_acc, color='skyblue', edgecolor='navy', linewidth=2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, top1_acc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.1%}', ha='center', va='bottom')
        
        ax.set_ylabel('Top-1 Move Prediction Accuracy')
        ax.set_title('AI Performance on Professional Game Positions\n' + 
                    'Comparison with Typical Professional Accuracy Levels')
        ax.set_ylim(0, 0.7)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add interpretation text
        ax.text(0.98, 0.02, 
               'Note: Top professionals typically achieve 45-50% top-1 accuracy\n' +
               'on other professional games. AlphaGo achieved ~57%.',
               transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'professional_game_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


# Add this import at the top of the file
import matplotlib.pyplot as plt
import torch