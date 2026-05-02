"""
训练模块
"""

from .self_play import FastTrainer
from .alpha_train import AlphaZeroTrainer

__all__ = ['FastTrainer', 'AlphaZeroTrainer']
