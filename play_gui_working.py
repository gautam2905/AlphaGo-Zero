#!/usr/bin/env python3
"""
Play against your trained AlphaGo Zero model with GUI!
Working version with proper human vs AI support.
"""

import os
os.environ['BOARD_SIZE'] = '9'  # Match training board size

import torch
import numpy as np
from alpha_zero.envs import GoEnv
from alpha_zero.envs.gui import BoardGameGui
from alpha_zero.core import ResNet, MCTS
from config import Config

class PlayAgainstAI:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model()
        
    def load_model(self):
        """Load the trained model with proper DataParallel handling."""
        env = GoEnv(komi=self.config.komi, num_stack=self.config.num_stack)
        model = ResNet(env, self.config.num_res_blocks, self.config.num_hidden, self.device)
        
        # Try to load model
        model_paths = [
            'checkpoints/best_model.pth',
            'checkpoints/final_model.pth',
            'checkpoints/latest_checkpoint.pth'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"Loading model from: {path}")
                try:
                    checkpoint = torch.load(path, map_location=self.device)
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # Fix DataParallel state dict (remove 'module.' prefix)
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('module.'):
                            # Remove 'module.' prefix
                            new_key = key[7:]
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    
                    # Load the fixed state dict
                    model.load_state_dict(new_state_dict)
                    print("✓ Model loaded successfully!")
                    model.eval()
                    return model
                    
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    continue
        
        print("⚠️ No trained model found! Using random model.")
        model.eval()
        return model
    
    def get_ai_move_function(self):
        """Returns a function that the GUI can call to get AI moves."""
        def ai_move(env):
            """AI makes a move using MCTS."""
            with torch.no_grad():  # Ensure no gradients during inference
                mcts = MCTS(env, self.config.get_mcts_args(), self.model)
                
                # Get move probabilities from MCTS
                pi_probs = mcts.search()
                
                # Choose best move (deterministic for strong play)
                action = np.argmax(pi_probs)
                
                # Show top 3 moves in console
                top_moves = np.argsort(pi_probs)[::-1][:3]
                print("\nAI thinking...")
                for i, move in enumerate(top_moves):
                    if move == env.action_dim - 1:
                        print(f"  {i+1}. PASS: {pi_probs[move]:.1%}")
                    else:
                        row, col = move // env.board_size, move % env.board_size
                        print(f"  {i+1}. ({row},{col}): {pi_probs[move]:.1%}")
                
                return action
        
        return ai_move

def play_as_black():
    """Play as Black against AI."""
    print("\n" + "="*50)
    print("YOU PLAY AS BLACK (First move)")
    print("="*50)
    print("\nInstructions:")
    print("- Click on empty intersections to place your black stones")
    print("- Click PASS button to pass your turn")
    print("- Close window to exit game")
    print("\nStarting GUI...")
    
    game = PlayAgainstAI()
    env = GoEnv(komi=game.config.komi, num_stack=game.config.num_stack)
    
    gui = BoardGameGui(
        env=env,
        black_player='human',  # Human plays black
        white_player=game.get_ai_move_function(),  # AI plays white
        show_steps=True,
        delay=1000  # Minimum required delay is 1000ms
    )
    
    print("GUI window opened! Make your move by clicking on the board.")
    gui.start()

def play_as_white():
    """Play as White against AI."""
    print("\n" + "="*50)
    print("YOU PLAY AS WHITE (Second move)")
    print("="*50)
    print("\nInstructions:")
    print("- AI (Black) will move first")
    print("- Click on empty intersections to place your white stones")
    print("- Click PASS button to pass your turn")
    print("- Close window to exit game")
    print("\nStarting GUI...")
    
    game = PlayAgainstAI()
    env = GoEnv(komi=game.config.komi, num_stack=game.config.num_stack)
    
    gui = BoardGameGui(
        env=env,
        black_player=game.get_ai_move_function(),  # AI plays black
        white_player='human',  # Human plays white
        show_steps=True,
        delay=1000  # Minimum required delay is 1000ms
    )
    
    print("GUI window opened! AI will make the first move.")
    gui.start()

def watch_ai_vs_ai():
    """Watch AI play against itself."""
    print("\n" + "="*50)
    print("AI vs AI DEMONSTRATION")
    print("="*50)
    print("\nWatch the AI play against itself!")
    print("The game will auto-play with 1 second delay between moves.")
    print("Close the window to exit.")
    print("\nStarting GUI...")
    
    game = PlayAgainstAI()
    env = GoEnv(komi=game.config.komi, num_stack=game.config.num_stack)
    
    # For AI vs AI, both players use the same AI function
    ai_move_func = game.get_ai_move_function()
    
    gui = BoardGameGui(
        env=env,
        black_player=ai_move_func,
        white_player=ai_move_func,
        show_steps=True,
        delay=1000  # 1 second between moves for viewing
    )
    
    print("GUI window opened! AI vs AI game starting...")
    gui.start()

def test_ai_only():
    """Test that the AI can make moves without GUI."""
    print("\n" + "="*50)
    print("TESTING AI MOVE GENERATION")
    print("="*50)
    
    game = PlayAgainstAI()
    env = GoEnv(komi=game.config.komi, num_stack=game.config.num_stack)
    env.reset()
    
    print("\nGenerating AI move for initial position...")
    ai_func = game.get_ai_move_function()
    move = ai_func(env)
    
    if move == env.action_dim - 1:
        print(f"AI chooses: PASS")
    else:
        row, col = move // env.board_size, move % env.board_size
        print(f"AI chooses: ({row},{col})")
    
    print("✓ AI move generation working!")

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════╗
    ║   PLAY AGAINST ALPHAGO ZERO MODEL      ║
    ║           9x9 Go Board                 ║
    ╚════════════════════════════════════════╝
    
    Choose an option:
    1. Play as BLACK (you go first)
    2. Play as WHITE (AI goes first)  
    3. Watch AI vs AI
    4. Test AI (no GUI)
    0. Exit
    """)
    
    try:
        choice = input("Enter choice (0-4): ").strip()
        
        if choice == '1':
            play_as_black()
        elif choice == '2':
            play_as_white()
        elif choice == '3':
            watch_ai_vs_ai()
        elif choice == '4':
            test_ai_only()
        elif choice == '0':
            print("Goodbye!")
        else:
            print("Invalid choice! Please run again.")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*50)
        print("TROUBLESHOOTING TIPS:")
        print("="*50)
        print("1. Make sure you have tkinter installed:")
        print("   - Windows: Usually included with Python")
        print("   - Linux: sudo apt-get install python3-tk")
        print("   - Mac: Usually included with Python")
        print("\n2. If using WSL/Remote connection:")
        print("   - Install X server (VcXsrv for Windows)")
        print("   - Set DISPLAY environment variable")
        print("\n3. Try option 4 to test AI without GUI")
        print("\n4. Check that model file exists in checkpoints/")
        print("="*50)