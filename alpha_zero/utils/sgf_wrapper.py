"""SGF (Smart Game Format) wrapper for Go games.

This module provides functionality to save and load Go games in SGF format,
which is the standard format for storing Go game records.
"""

import re
from typing import List, Tuple, Optional, Dict
from datetime import datetime


class SGFWrapper:
    """A wrapper class for creating and parsing SGF files."""
    
    def __init__(self, board_size: int = 19, komi: float = 7.5):
        self.board_size = board_size
        self.komi = komi
        self.moves = []
        self.metadata = {
            'GM': '1',  # Game type (1 = Go)
            'FF': '4',  # File format version
            'CA': 'UTF-8',  # Character encoding
            'SZ': str(board_size),  # Board size
            'KM': str(komi),  # Komi
            'DT': datetime.now().strftime('%Y-%m-%d'),  # Date
            'AP': 'AlphaGoZero:1.0',  # Application name
        }
    
    def add_move(self, color: str, coords: Tuple[int, int]):
        """Add a move to the game record.
        
        Args:
            color: 'B' for black or 'W' for white
            coords: (row, col) coordinates, or None for pass
        """
        if coords is None:
            move_str = f"{color}[]"  # Pass move
        else:
            row, col = coords
            # Convert to SGF coordinates (a-s)
            sgf_col = chr(ord('a') + col)
            sgf_row = chr(ord('a') + row)
            move_str = f"{color}[{sgf_col}{sgf_row}]"
        
        self.moves.append(move_str)
    
    def set_metadata(self, key: str, value: str):
        """Set metadata property."""
        self.metadata[key] = value
    
    def to_sgf(self) -> str:
        """Convert the game to SGF format string."""
        sgf = "(;"
        
        # Add metadata
        for key, value in self.metadata.items():
            sgf += f"{key}[{value}]"
        
        # Add moves
        for move in self.moves:
            sgf += f";{move}"
        
        sgf += ")"
        return sgf
    
    def save(self, filepath: str):
        """Save the game to an SGF file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_sgf())
    
    @classmethod
    def from_sgf(cls, sgf_content: str) -> 'SGFWrapper':
        """Parse SGF content and create SGFWrapper instance."""
        wrapper = cls()
        
        # Extract metadata
        metadata_pattern = r'(\w+)\[([^\]]*)\]'
        metadata_matches = re.findall(metadata_pattern, sgf_content)
        
        for key, value in metadata_matches:
            if key in ['B', 'W']:
                continue  # Skip moves
            wrapper.metadata[key] = value
        
        # Extract board size and komi
        if 'SZ' in wrapper.metadata:
            wrapper.board_size = int(wrapper.metadata['SZ'])
        if 'KM' in wrapper.metadata:
            wrapper.komi = float(wrapper.metadata['KM'])
        
        # Extract moves
        move_pattern = r';([BW])\[([^\]]*)\]'
        move_matches = re.findall(move_pattern, sgf_content)
        
        for color, coords_str in move_matches:
            if coords_str == '':  # Pass move
                wrapper.add_move(color, None)
            else:
                col = ord(coords_str[0]) - ord('a')
                row = ord(coords_str[1]) - ord('a')
                wrapper.add_move(color, (row, col))
        
        return wrapper
    
    @classmethod
    def load(cls, filepath: str) -> 'SGFWrapper':
        """Load SGF from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            sgf_content = f.read()
        return cls.from_sgf(sgf_content)
    
    def get_moves(self) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
        """Get list of moves as (color, coords) tuples."""
        moves = []
        for move_str in self.moves:
            match = re.match(r'([BW])\[([^\]]*)\]', move_str)
            if match:
                color, coords_str = match.groups()
                if coords_str == '':
                    moves.append((color, None))
                else:
                    col = ord(coords_str[0]) - ord('a')
                    row = ord(coords_str[1]) - ord('a')
                    moves.append((color, (row, col)))
        return moves


def moves_to_sgf(moves: List[Tuple[int, int]], board_size: int = 19, 
                 komi: float = 7.5, winner: Optional[str] = None,
                 player_black: str = "AlphaGoZero", 
                 player_white: str = "AlphaGoZero") -> str:
    """Convert a list of moves to SGF format.
    
    Args:
        moves: List of (row, col) tuples, None for pass
        board_size: Board size
        komi: Komi value
        winner: 'B' or 'W' for winner
        player_black: Name of black player
        player_white: Name of white player
    
    Returns:
        SGF format string
    """
    wrapper = SGFWrapper(board_size, komi)
    wrapper.set_metadata('PB', player_black)
    wrapper.set_metadata('PW', player_white)
    
    if winner:
        wrapper.set_metadata('RE', f"{winner}+")
    
    # Alternate between black and white
    for i, move in enumerate(moves):
        color = 'B' if i % 2 == 0 else 'W'
        wrapper.add_move(color, move)
    
    return wrapper.to_sgf()


def make_sgf(board_size: int, move_history, result_string: str, 
             ruleset: str = "Chinese", komi: float = 7.5, date: str = None,
             player_black: str = "AlphaGoZero", player_white: str = "AlphaGoZero") -> str:
    """Create SGF from game history (compatible with existing code).
    
    Args:
        board_size: Size of the board
        move_history: List of PlayerMove objects with color and move attributes
        result_string: Game result (e.g., "B+R", "W+7.5")
        ruleset: Go ruleset used
        komi: Komi value
        date: Date string
        player_black: Black player name
        player_white: White player name
    
    Returns:
        SGF format string
    """
    wrapper = SGFWrapper(board_size, komi)
    wrapper.set_metadata('PB', player_black)
    wrapper.set_metadata('PW', player_white)
    wrapper.set_metadata('RU', ruleset)
    wrapper.set_metadata('RE', result_string)
    
    if date:
        wrapper.set_metadata('DT', date)
    
    # Add moves from history
    for move_record in move_history:
        color = move_record.color  # 'B' or 'W'
        move = move_record.move    # Action index
        
        # Convert action index to coordinates
        if move is None or move < 0:
            coords = None  # Pass move
        else:
            # Convert flat action to (row, col)
            row = move // board_size
            col = move % board_size
            coords = (row, col)
        
        wrapper.add_move(color, coords)
    
    return wrapper.to_sgf()