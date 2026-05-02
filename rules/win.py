"""
胡牌判定器 - 温州麻将17张牌型检测

基本牌型: AA + ABC×5 (5顺子 或 4顺子+1刻子 等)

胡牌等级:
  软牌 ×1 - 有财神参与
  硬牌 ×2 - 无财神/自摸/抢杠胡/3财神强制胡
  双翻 ×4 - 天胡/地胡/硬8对/3财神+基本牌型/单吊/全球神/碰碰胡/杠上开花/双财神归位/半清/清一色
  特殊4翻 ×8 - 7对+3财神归位

注意：3财神强制胡强烈不推荐，做大牌才是最优策略
"""

from typing import List, Dict, Tuple
from enum import IntEnum
from collections import Counter

from .tiles import get_tile_group, ALL_TILE_IDS, get_suit
from .caishen import CaishenEngine


class WinPattern(IntEnum):
    """胡牌类型"""
    NONE = 0
    SOFT = 1       # 软牌 ×1
    HARD = 2       # 硬牌 ×2
    DOUBLE_FAN = 4  # 双翻 ×4
    SPECIAL_8FAN = 8  # 特殊4翻 ×8


class WinChecker:
    """胡牌检测器"""

    def __init__(self, caishen_id: int):
        self.caishen = CaishenEngine(caishen_id)

    def check_win(self, hand: List[int]) -> Dict:
        """
        检测是否胡牌

        Returns:
            {
                'can_win': bool,
                'fan': int,          # 番数
                'win_type': str,      # soft/hard/double_fan/special_8fan
                'details': List[str]  # 详情
            }
        """
        if len(hand) != 17:
            return {'can_win': False}

        caishen_count = self.caishen.count_caishen(hand)

        # 1. 特殊4翻(×8): 7对普通 + 3财神归位 (硬8对7对+3财神)
        if self._check_special_8fan(hand, caishen_count):
            return {
                'can_win': True,
                'fan': 8,
                'win_type': 'special_8fan',
                'details': ['7对+3财神', '特殊4翻']
            }

        # 2. 硬8对: AA×8 + 1张，无财神 = 双翻
        if self._check_hard_8_pairs(hand, caishen_count):
            return {
                'can_win': True,
                'fan': 4,
                'win_type': 'double_fan',
                'details': ['硬八对', '双翻']
            }

        # 3. 基本牌型检测 (AA + ABC×5的各种变体)
        basic = self._check_basic_pattern(hand, caishen_count)
        if basic['can_win']:
            return basic

        # 4. 软8对: AA×7 + BB + 1张，有财神参与
        if self._check_soft_8_pairs(hand, caishen_count):
            return {
                'can_win': True,
                'fan': 1,
                'win_type': 'soft',
                'details': ['软八对']
            }

        # 5. 3财神强制胡 (fan=2，但强烈不推荐)
        if caishen_count >= 3:
            return {
                'can_win': True,
                'fan': 2,
                'win_type': 'hard',
                'details': ['三财神', '强烈不推荐胡']
            }

        return {'can_win': False}

    def _check_special_8fan(self, hand: List[int], caishen_count: int) -> bool:
        """
        特殊4翻(×8): 7对普通 + 3财神归位
        即 硬8对7对 + 3财神牌型
        """
        if caishen_count != 3:
            return False
        non_wild = [t for t in hand if not self.caishen.is_wild(t)]
        counts = Counter(non_wild)
        pairs = sum(c // 2 for c in counts.values())
        return pairs == 7

    def _check_hard_8_pairs(self, hand: List[int], caishen_count: int) -> bool:
        """
        硬8对: AA×8 + 1张，无财神 = 双翻
        条件：8对 + 1单张，且无财神参与
        """
        if caishen_count != 0:
            return False
        counts = Counter(hand)
        pairs = sum(c // 2 for c in counts.values())
        singles = sum(c % 2 for c in counts.values())
        return pairs == 8 and singles == 1

    def _check_soft_8_pairs(self, hand: List[int], caishen_count: int) -> bool:
        """
        软8对: AA×7 + BB + 1张，有财神参与
        7对普通牌 + 1财神当对 + 1单张(任意)
        """
        if caishen_count == 0:
            return False
        non_wild = [t for t in hand if not self.caishen.is_wild(t)]
        counts = Counter(non_wild)
        pairs = sum(c // 2 for c in counts.values())
        singles = sum(c % 2 for c in counts.values())
        # 需要 7对 + 1财神对 + 1单张
        return pairs >= 7 and singles <= 1 and caishen_count >= 2

    def _check_basic_pattern(self, hand: List[int], caishen_count: int) -> Dict:
        """
        检查基本牌型: AA + ABC×5 (或变体 AA + ABC×4 + AAA×1 等)

        基本牌型有5种:
        1. AA + ABC×5           (1对将 + 5顺子)
        2. AA + ABC×4 + AAA×1   (1对将 + 4顺子 + 1刻子)
        3. AA + ABC×3 + AAA×2   (1对将 + 3顺子 + 2刻子)
        4. AA + ABC×2 + AAA×3   (1对将 + 2顺子 + 3刻子)
        5. AA + ABC×1 + AAA×4   (1对将 + 1顺子 + 4刻子)
        """
        non_wild = [t for t in hand if not self.caishen.is_wild(t)]
        counts = Counter(non_wild)

        # 尝试每个可能的对子
        pair_tiles = [(t, counts[t]) for t in counts.keys() if counts[t] >= 2]
        pair_tiles.sort(key=lambda x: x[1])

        for pair_tile, _ in pair_tiles:
            test_counts = counts.copy()
            test_counts[pair_tile] -= 2
            if test_counts[pair_tile] == 0:
                del test_counts[pair_tile]

            success, meld_counts = self._form_melds_5(test_counts, caishen_count)
            if success:
                # 判断是软牌还是硬牌
                if caishen_count == 0:
                    return {
                        'can_win': True,
                        'win_type': 'hard',
                        'fan': 2,
                        'details': ['硬胡']
                    }
                else:
                    return {
                        'can_win': True,
                        'win_type': 'soft',
                        'fan': 1,
                        'details': ['软胡']
                    }

        # 用财神做对子
        if caishen_count >= 2:
            success, meld_counts = self._form_melds_5(counts.copy(), caishen_count - 2)
            if success:
                return {
                    'can_win': True,
                    'win_type': 'soft',
                    'fan': 1,
                    'details': ['软胡', '财神做对']
                }

        return {'can_win': False}

    def _form_melds_5(self, counts: Counter, caishen: int) -> Tuple[bool, List]:
        """尝试用counts中的牌组成5个面子"""
        return self._try_meld_5(dict(counts), caishen, 5, [])

    def _try_meld_5(self, remaining: Dict, caishen: int,
                     melds_left: int, details: List) -> Tuple[bool, List]:
        """递归尝试组成5个面子"""
        if melds_left == 0:
            return sum(remaining.values()) == 0, details
        if not remaining:
            return caishen >= melds_left * 3, details

        tile = max(remaining.keys(), key=lambda t: remaining[t])
        cnt = remaining[tile]
        g = get_tile_group(tile)

        # 刻子 (AAA)
        if cnt >= 3:
            nr = remaining.copy()
            nr[tile] -= 3
            if nr[tile] == 0:
                del nr[tile]
            ok, d = self._try_meld_5(nr, caishen, melds_left - 1, details + ['pong'])
            if ok:
                return True, d

        # 财神补刻子
        if caishen > 0 and cnt >= 1:
            need = 3 - cnt
            cs = min(caishen, need)
            if cs == need:
                nr = remaining.copy()
                nr[tile] -= cnt
                if nr[tile] == 0:
                    del nr[tile]
                ok, d = self._try_meld_5(nr, caishen - cs, melds_left - 1, details + ['pong'])
                if ok:
                    return True, d

        # 顺子 (ABC) - 字牌不能成顺
        if g < 3:  # 万/筒/索
            for off in [0, -1, -2]:
                t1, t2, t3 = tile + off, tile + off + 1, tile + off + 2
                if t1 < 0 or t3 > 26:
                    continue
                if get_tile_group(t1) != g or get_tile_group(t2) != g or get_tile_group(t3) != g:
                    continue
                c1, c2, c3 = remaining.get(t1, 0), remaining.get(t2, 0), remaining.get(t3, 0)
                need = (1 if c1 < 1 else 0) + (1 if c2 < 1 else 0) + (1 if c3 < 1 else 0)
                if need <= caishen:
                    nr = remaining.copy()
                    for t in [t1, t2, t3]:
                        if nr.get(t, 0) > 0:
                            nr[t] -= 1
                            if nr[t] == 0:
                                del nr[t]
                    ok, d = self._try_meld_5(nr, caishen - need, melds_left - 1, details + ['chow'])
                    if ok:
                        return True, d

        return False, details

    def get_ting_tiles(self, hand: List[int]) -> List[int]:
        """获取所有能让人听牌的牌"""
        ting = []
        for tile in ALL_TILE_IDS:
            if not self.caishen.is_wild(tile):
                if self.check_win(hand + [tile])['can_win']:
                    ting.append(tile)
        return ting

    def is_ting(self, hand: List[int]) -> bool:
        """是否听牌"""
        return len(self.get_ting_tiles(hand)) > 0

    def check_pengpeng(self, hand: List[int]) -> bool:
        """检查是否是碰碰胡 (全部刻子+将对)"""
        non_wild = [t for t in hand if not self.caishen.is_wild(t)]
        counts = Counter(non_wild)

        # 需要有1对 + 5刻子(15张)
        pairs = sum(c // 2 for c in counts.values())
        triples = sum(c // 3 for c in counts.values())
        singles = sum(c % 2 for c in counts.values())

        # 碰碰胡: AAA×5 + AA (5刻子 + 1对 = 17张)
        return pairs == 1 and triples == 5 and singles == 0

    def check_qing_yi_se(self, hand: List[int]) -> bool:
        """检查是否清一色 (全为一种花色)"""
        non_wild = [t for t in hand if not self.caishen.is_wild(t)]
        if not non_wild:
            return False
        suits = set(get_suit(t) for t in non_wild)
        return len(suits) == 1

    def check_ban_qing(self, hand: List[int]) -> bool:
        """检查是否半清 (一种花色 + 风字牌)"""
        non_wild = [t for t in hand if not self.caishen.is_wild(t)]
        if not non_wild:
            return False

        suit_tiles = [t for t in non_wild if t < 27]  # 万筒索
        zi_tiles = [t for t in non_wild if t >= 27]   # 字牌

        return (len(suit_tiles) > 0 and len(zi_tiles) > 0 and
                len(set(get_suit(t) for t in suit_tiles)) == 1)