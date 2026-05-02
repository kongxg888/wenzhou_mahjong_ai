"""
共享游戏状态类 - 消除代码重复
"""
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
from rules.tiles import ACTION_CHI, ACTION_PENG, ACTION_MING_GANG, ACTION_JIA_GANG, ACTION_AN_GANG, ACTION_HU

# 花色组查找表 (比if-elif快)
_TILE_GROUP_LOOKUP = tuple(
    0 if i < 9 else 1 if i < 18 else 2 if i < 27 else 3
    for i in range(34)
)


class GameState:
    """简化游戏状态（用于MCTS）"""

    def __init__(self, hand, caishen_id, wall_remaining, current_player,
                 last_discarded=None, fulus=None, opponent_discards=None, discarded=None):
        self.hand = list(hand) if not isinstance(hand, list) else hand.copy()
        self.caishen_id = caishen_id
        self.wall_remaining = wall_remaining
        self.current_player = current_player
        self.last_discarded = last_discarded
        self.fulus = fulus if fulus else []
        self.discarded = discarded if discarded else []
        self.opponent_discards = opponent_discards if opponent_discards else []

    def get_legal_actions(self) -> List[int]:
        """获取合法出牌动作"""
        legal = []
        for tile in set(self.hand):
            if tile != self.caishen_id and tile != 33:
                legal.append(tile)
        return legal if legal else list(set(self.hand))

    def get_legal_actions_with_special(self, last_discarded=None) -> List[int]:
        """获取完整合法动作包括吃碰杠胡"""
        legal = self.get_legal_actions()

        if last_discarded is not None and last_discarded < 34:
            # 碰（手中有一对 + 对手打出）
            if self.hand.count(last_discarded) >= 2:
                legal.append(ACTION_PENG)

            # 明杠
            if self.hand.count(last_discarded) >= 3:
                legal.append(ACTION_MING_GANG)

        # 吃（只有闲家能吃）
        if last_discarded is not None and last_discarded < 34 and self.current_player == 0:
            if self._can_chi(last_discarded):
                legal.append(ACTION_CHI)

        # 胡
        if len(self.hand) == 17:
            legal.append(ACTION_HU)

        return legal

    def _can_chi(self, discarded_tile: int) -> bool:
        """检查是否能吃"""
        tile_group = _TILE_GROUP_LOOKUP[discarded_tile]
        if tile_group >= 3:
            return False
        for offset in [-1, 0, 1]:
            t1 = discarded_tile + offset
            t2 = t1 + 1
            t3 = t1 + 2
            if t1 < 0 or t3 > 26:
                continue
            if _TILE_GROUP_LOOKUP[t1] != tile_group:
                continue
            if (self.hand.count(t1) >= 1 and
                self.hand.count(t2) >= 1 and
                self.hand.count(t3) >= 1):
                return True
        return False

    def do_action(self, action: int):
        """执行动作（包括吃/碰/杠/胡）"""
        if action == ACTION_CHI:  # 吃
            if self.last_discarded is not None:
                for offset in [-1, 0, 1]:
                    t1 = self.last_discarded + offset
                    t2 = t1 + 1
                    t3 = t1 + 2
                    if t1 < 0 or t3 > 26:
                        continue
                    if _TILE_GROUP_LOOKUP[t1] != _TILE_GROUP_LOOKUP[self.last_discarded]:
                        continue
                    if (self.hand.count(t1) >= 1 and
                        self.hand.count(t2) >= 1 and
                        self.hand.count(t3) >= 1):
                        self._do_chi((t1, t2, t3))
                        self.last_discarded = None
                        self.current_player = 1 - self.current_player
                        return
            return
        elif action == ACTION_PENG:  # 碰
            if self.last_discarded is not None and self.hand.count(self.last_discarded) >= 2:
                self._do_peng(self.last_discarded)
                self.last_discarded = None
                self.current_player = 1 - self.current_player
                return
        elif action == ACTION_MING_GANG:  # 明杠
            if self.last_discarded is not None and self.hand.count(self.last_discarded) >= 3:
                self._do_gang(self.last_discarded, 'ming')
                self.last_discarded = None
                self.current_player = 1 - self.current_player
                return
        elif action == ACTION_JIA_GANG:  # 加杠
            if self.last_discarded is not None and self._has_jia_gang(self.last_discarded):
                self._do_gang(self.last_discarded, 'jia')
                self.last_discarded = None
                self.current_player = 1 - self.current_player
                return
        elif action == ACTION_AN_GANG:  # 暗杠
            if self._can_an_gang():
                tile = self._get_an_gang_tile()
                self._do_gang(tile, 'an')
                self.last_discarded = None
                # 暗杠后需要继续（不切换玩家，因为要补牌）
                return
        elif action == ACTION_HU:  # 胡 - 终局
            self.wall_remaining = 0
            return

        # 出牌动作
        if action in self.hand:
            self.hand.remove(action)
            self.discarded.append(action)
        self.wall_remaining -= 1
        self.current_player = 1 - self.current_player

    def _do_chi(self, chi_tiles: Tuple[int, int, int]):
        for t in chi_tiles:
            if t in self.hand:
                self.hand.remove(t)
        self.fulus.append(('chow', list(chi_tiles)))

    def _do_peng(self, tile: int):
        for _ in range(2):
            if tile in self.hand:
                self.hand.remove(tile)
        self.fulus.append(('pong', [tile, tile, tile]))

    def _do_gang(self, tile: int, gang_type: str = 'ming'):
        """执行杠牌

        Args:
            tile: 杠的牌
            gang_type: 'ming'(明杠), 'jia'(加杠), 'an'(暗杠)
        """
        if gang_type == 'ming':
            for _ in range(3):
                if tile in self.hand:
                    self.hand.remove(tile)
        elif gang_type == 'jia':
            if tile in self.hand:
                self.hand.remove(tile)
        elif gang_type == 'an':
            for _ in range(4):
                if tile in self.hand:
                    self.hand.remove(tile)
        self.fulus.append(('kong', [tile, tile, tile, tile]))

    def _has_jia_gang(self, tile: int) -> bool:
        """检查是否能加杠（碰后摸到第4张）"""
        # 检查是否已经碰过这张牌
        for fulu in self.fulus:
            if fulu[0] == 'pong' and fulu[1][0] == tile:
                # 已经碰过，检查手中有无第4张
                return self.hand.count(tile) >= 1
        return False

    def _can_an_gang(self) -> bool:
        """检查能否暗杠"""
        for tile in set(self.hand):
            if self.hand.count(tile) >= 4:
                return True
        return False

    def _get_an_gang_tile(self) -> int:
        """获取第一组可暗杠的牌"""
        for tile in set(self.hand):
            if self.hand.count(tile) >= 4:
                return tile
        return -1

    def is_terminal(self) -> bool:
        """是否终局"""
        return self.wall_remaining <= 0

    def get_observation(self) -> np.ndarray:
        """获取状态编码"""
        obs = np.zeros(136, dtype=np.float32)

        # 使用Counter加速牌计数
        hand_counts = Counter(self.hand)

        # 手牌
        for tile, count in hand_counts.items():
            if 0 <= tile < 34:
                obs[tile] = count

        # 弃牌
        disc_counts = Counter(self.discarded[-20:])
        for tile, count in disc_counts.items():
            if 0 <= tile < 34:
                obs[34 + tile] = count

        # 对手弃牌
        opp_disc_counts = Counter(self.opponent_discards[-20:])
        for tile, count in opp_disc_counts.items():
            if 0 <= tile < 34:
                obs[68 + tile] = count

        # 牌墙
        if self.wall_remaining > 0:
            idx = min(33, max(0, 102 + (self.wall_remaining // 3)))
            obs[idx] = 1

        # 财神
        if 0 <= self.caishen_id < 34:
            obs[104 + min(self.caishen_id, 31)] = 1

        return obs
