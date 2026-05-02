"""
手牌管理 - 温州麻将17张手牌
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter


class Hand:
    """
    手牌管理类

    支持：
    - 摸牌、出牌
    - 副露（吃、碰、杠）
    - 手牌分析
    """

    def __init__(self, tiles: Optional[List[int]] = None):
        self.hand: List[int] = tiles if tiles else []
        self.fulu: List[Tuple[str, List[int]]] = []  # 副露列表
        self.discus: List[int] = []  # 弃牌列表

    def add(self, tile: int) -> None:
        """摸牌"""
        self.hand.append(tile)
        self.hand.sort()

    def discard(self, tile: int) -> bool:
        """出牌"""
        if tile in self.hand:
            self.hand.remove(tile)
            self.discus.append(tile)
            return True
        return False

    def count(self, tile_id: int) -> int:
        """某张牌的数量"""
        return self.hand.count(tile_id)

    def get_counts(self) -> Dict[int, int]:
        """获取所有牌的数量统计"""
        return dict(Counter(self.hand))

    @property
    def size(self) -> int:
        """手牌数量"""
        return len(self.hand)

    def can_pong(self, tile: int) -> bool:
        """能否碰"""
        return self.hand.count(tile) >= 2

    def can_kong(self, tile: int) -> bool:
        """能否杠"""
        return self.hand.count(tile) >= 3

    def can_hidden_kong(self, tile: int) -> bool:
        """能否暗杠"""
        return self.hand.count(tile) >= 4

    def add_fulu(self, fulu_type: str, tiles: List[int]) -> None:
        """
        添加副露

        Args:
            fulu_type: 'pong'(碰), 'kong'(杠), 'chow'(吃)
            tiles: 相关的牌
        """
        self.fulu.append((fulu_type, tiles))
        for t in tiles:
            if t in self.hand:
                self.hand.remove(t)

    def get_full_hand(self) -> List[int]:
        """获取完整手牌（含副露）"""
        full = self.hand.copy()
        for _, tiles in self.fulu:
            full.extend(tiles)
        return full

    def get_ting_tiles(self) -> List[int]:
        """获取听牌列表"""
        # 这个需要调用win.py的逻辑
        from .win import WinChecker
        # 简化版本，返回空
        return []

    def __repr__(self):
        from .tiles import tile_id_to_str
        hand_str = ''.join([tile_id_to_str(t) for t in sorted(self.hand)])
        fulu_str = str(self.fulu) if self.fulu else ''
        return f"Hand({hand_str}, Fulu:{fulu_str})"
