"""Game environments for AlphaGo Zero."""

from .base import BoardGameEnv
from .go import GoEnv

__all__ = ['BoardGameEnv', 'GoEnv']