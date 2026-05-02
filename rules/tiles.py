"""
牌定义 - 温州麻将34种牌

牌ID: 0-8=万, 9-17=筒, 18-26=索, 27-33=字牌
"""

from enum import IntEnum
from typing import List
import random


class TileType(IntEnum):
    """牌类型"""
    WAN = 0      # 万子 (0-8)
    TONG = 1     # 筒子 (9-17)
    SUO = 2      # 索子 (18-26)
    ZI = 3       # 字牌 (27-33)


# 常量
ALL_TILE_IDS = list(range(34))  # 0-33
TILE_COUNT = 4  # 每种牌4张
TOTAL_TILES = 34 * 4  # 136张

# 牌名字（显示用）
TILE_NAMES = [
    '一万', '二万', '三万', '四万', '五万', '六万', '七万', '八万', '九万',  # 0-8
    '一筒', '二筒', '三筒', '四筒', '五筒', '六筒', '七筒', '八筒', '九筒',  # 9-17
    '一索', '二索', '三索', '四索', '五索', '六索', '七索', '八索', '九索',  # 18-26
    '东', '南', '西', '北', '中', '发', '白'  # 27-33
]

# 白板ID
WHITE_TILE_ID = 33

# ===== 动作常量 =====
ACTION_DISCARD = 0      # 出牌动作索引范围: 0-33
ACTION_PASS = 34         # 过
ACTION_CHI = 35         # 吃
ACTION_PENG = 36        # 碰
ACTION_MING_GANG = 37   # 明杠（别人打出第4张）
ACTION_JIA_GANG = 38    # 加杠/补杠（碰后摸到第4张）
ACTION_AN_GANG = 39     # 暗杠（自己摸到4张）
ACTION_HU = 40          # 胡
NUM_ACTIONS = 41         # 总动作数


def get_suit(tile_id: int) -> TileType:
    """获取牌的花色"""
    if tile_id < 9:
        return TileType.WAN
    elif tile_id < 18:
        return TileType.TONG
    elif tile_id < 27:
        return TileType.SUO
    else:
        return TileType.ZI


def get_number(tile_id: int) -> int:
    """获取牌的数字（1-9），字牌返回0"""
    return tile_id % 9 + 1


def get_tile_group(tile_id: int) -> int:
    """获取花色组（0=万，1=筒，2=索，3=字）"""
    if tile_id < 9:
        return 0
    elif tile_id < 18:
        return 1
    elif tile_id < 27:
        return 2
    return 3


def tile_id_to_str(tile_id: int) -> str:
    """牌ID转字符串"""
    if 0 <= tile_id < 34:
        return TILE_NAMES[tile_id]
    return '?'


def get_next_tile(tile_id: int) -> int:
    """获取下一张牌（用于财神计算）"""
    if tile_id == 32:  # 发之后是白
        return 33
    elif tile_id == 33:  # 白之后回头
        return 27
    return tile_id + 1


def is_word_tile(tile_id: int) -> bool:
    """是否是字牌"""
    return tile_id >= 27


def is_same_suit(t1: int, t2: int) -> bool:
    """是否同花色"""
    return get_tile_group(t1) == get_tile_group(t2)


def is_adjacent(t1: int, t2: int) -> bool:
    """是否相邻（可用于顺子判断）"""
    if get_tile_group(t1) != get_tile_group(t2):
        return False
    return abs(get_number(t1) - get_number(t2)) == 1


def generate_full_deck() -> List[int]:
    """生成完整牌墙"""
    deck = []
    for tile_id in ALL_TILE_IDS:
        deck.extend([tile_id] * TILE_COUNT)
    random.shuffle(deck)
    return deck


def shuffle_deck(deck: List[int]) -> List[int]:
    """洗牌"""
    random.shuffle(deck)
    return deck
