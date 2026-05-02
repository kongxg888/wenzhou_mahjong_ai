"""
财神引擎 - 温州麻将核心特色
"""

from typing import List, Set
from .tiles import get_next_tile, WHITE_TILE_ID


class CaishenEngine:
    """
    财神管理器

    财神可代替任意牌，白板只能代替财神本身
    财神和白板都不能打出
    """

    def __init__(self, caishen_id: int):
        """
        Args:
            caishen_id: 财神牌ID (0-33)
        """
        self.caishen_id = caishen_id

    def is_caishen(self, tile_id: int) -> bool:
        """是否是财神牌本身"""
        return tile_id == self.caishen_id

    def is_white_tile(self, tile_id: int) -> bool:
        """是否是白板"""
        return tile_id == WHITE_TILE_ID

    def is_wild(self, tile_id: int) -> bool:
        """是否是万能牌（财神或白板）"""
        return self.is_caishen(tile_id) or self.is_white_tile(tile_id)

    def can_replace(self, tile_id: int) -> bool:
        """
        这个牌能否作为万能牌使用
        财神可以代替任何牌，白板只能代替财神
        """
        if self.is_caishen(tile_id):
            return True
        if self.is_white_tile(tile_id):
            return True
        return False

    def count_caishen(self, hand: List[int]) -> int:
        """统计手牌中财神数量（财神+白板都算财神功能）"""
        return sum(1 for t in hand if self.is_wild(t))

    def get_wild_tiles(self) -> Set[int]:
        """获取所有万能牌"""
        wild = {self.caishen_id}
        if self.caishen_id != WHITE_TILE_ID:
            wild.add(WHITE_TILE_ID)
        return wild

    @staticmethod
    def get_caishen_from_flip(flip_card_id: int) -> int:
        """
        从翻出的牌计算财神ID
        翻出牌的下一张是财神
        """
        return get_next_tile(flip_card_id)

    def __repr__(self):
        return f"CaishenEngine(caishen={self.caishen_id})"
