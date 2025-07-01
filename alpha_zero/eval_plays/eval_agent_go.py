"""Evaluation script for AlphaGo Zero agent."""

import torch
import numpy as np
from alpha_zero.envs import GoEnv
from alpha_zero.core import ResNet, MCTS
from config import Config


class AlphaGoEvaluator:
    """Evaluate AlphaGo Zero agent."""
    
    def __init__(self, model_path, config=None):
        if config is None:
            config = Config()
        self.config = config
        
        # Initialize environment
        self.env = GoEnv(
            komi=config.komi,
            num_stack=config.num_stack
        )
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet(
            self.env,
            config.num_res_blocks,
            config.num_hidden,
            self.device
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if hasattr(checkpoint, 'state_dict'):
            self.model.load_state_dict(checkpoint.state_dict())
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Initialize MCTS
        self.mcts = MCTS(self.env, config.get_mcts_args(), self.model)
    
    def play_game_vs_random(self, agent_plays_black=True):
        """Play a game against random player."""
        env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
        
        while not env.is_game_over():
            if (env.to_play == env.black_player and agent_plays_black) or \
               (env.to_play == env.white_player and not agent_plays_black):
                # Agent's turn
                mcts = MCTS(env, self.config.get_mcts_args(), self.model)
                pi_probs = mcts.search()
                action = np.argmax(pi_probs)
            else:
                # Random player's turn
                valid_actions = np.where(env.legal_actions == 1)[0]
                action = np.random.choice(valid_actions)
            
            env.step(action)
        
        # Determine winner from agent's perspective
        if env.winner is None:
            return 0  # Draw
        elif (env.winner == env.black_player and agent_plays_black) or \
             (env.winner == env.white_player and not agent_plays_black):
            return 1  # Agent wins
        else:
            return -1  # Agent loses
    
    def evaluate_vs_random(self, num_games=100):
        """Evaluate agent against random player."""
        print(f"Evaluating against random player for {num_games} games...")
        
        wins = 0
        draws = 0
        losses = 0
        
        for i in range(num_games):
            # Alternate who plays black
            agent_plays_black = (i % 2 == 0)
            result = self.play_game_vs_random(agent_plays_black)
            
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
            
            if (i + 1) % 10 == 0:
                print(f"Games {i+1}/{num_games}: Wins={wins}, Draws={draws}, Losses={losses}")
        
        win_rate = wins / num_games
        print(f"\nFinal results: {wins}W-{draws}D-{losses}L (Win rate: {win_rate:.2%})")
        return win_rate
    
    def play_interactive_game(self):
        """Play an interactive game against the agent."""
        print("Starting interactive game. You are Black, Agent is White.")
        print("Enter moves in GTP format (e.g., 'D4') or 'pass' to pass, 'quit' to exit.")
        
        env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
        
        while not env.is_game_over():
            env.render()
            
            if env.to_play == env.black_player:
                # Human's turn
                while True:
                    move_str = input("Your move: ").strip().upper()
                    
                    if move_str == 'QUIT':
                        return
                    elif move_str == 'PASS':
                        action = env.pass_move
                        break
                    else:
                        action = env.gtp_to_action(move_str)
                        if action is not None:
                            break
                        else:
                            print("Invalid move. Try again.")
            else:
                # Agent's turn
                print("Agent is thinking...")
                mcts = MCTS(env, self.config.get_mcts_args(), self.model)
                pi_probs = mcts.search()
                action = np.argmax(pi_probs)
                
                if action == env.pass_move:
                    print("Agent passes.")
                else:
                    move_gtp = env.action_to_gtp(action)
                    print(f"Agent plays: {move_gtp}")
            
            env.step(action)
        
        env.render()
        result = env.get_result_string()
        print(f"Game over! Result: {result}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate AlphaGo Zero agent")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument("--mode", choices=["random", "interactive"], default="random",
                       help="Evaluation mode")
    parser.add_argument("--num_games", type=int, default=100,
                       help="Number of games for random evaluation")
    
    args = parser.parse_args()
    
    evaluator = AlphaGoEvaluator(args.model_path)
    
    if args.mode == "random":
        evaluator.evaluate_vs_random(args.num_games)
    elif args.mode == "interactive":
        evaluator.play_interactive_game()