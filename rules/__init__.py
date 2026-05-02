"""
温州麻将核心规则引擎
"""

from .tiles import (
    TileType, ALL_TILE_IDS, TILE_COUNT, generate_full_deck,
    tile_id_to_str, get_next_tile, get_suit, get_number, get_tile_group,
    ACTION_DISCARD, ACTION_PASS, ACTION_CHI, ACTION_PENG,
    ACTION_MING_GANG, ACTION_JIA_GANG, ACTION_AN_GANG, ACTION_HU, NUM_ACTIONS
)
from .caishen import CaishenEngine
from .hand import Hand
from .win import WinChecker, WinPattern
from .score import ScoreCalculator

__all__ = [
    'TileType', 'ALL_TILE_IDS', 'TILE_COUNT', 'generate_full_deck',
    'tile_id_to_str', 'get_next_tile', 'get_suit', 'get_number', 'get_tile_group',
    'ACTION_DISCARD', 'ACTION_PASS', 'ACTION_CHI', 'ACTION_PENG',
    'ACTION_MING_GANG', 'ACTION_JIA_GANG', 'ACTION_AN_GANG', 'ACTION_HU', 'NUM_ACTIONS',
    'CaishenEngine', 'Hand', 'WinChecker', 'WinPattern', 'ScoreCalculator'
]
