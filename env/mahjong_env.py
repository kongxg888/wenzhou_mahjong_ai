"""
温州麻将 RLCard环境

支持标准RLCard接口
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

# 尝试导入RLCard
try:
    from rlcard.envs.env import Env
    HAS_RLCARD = True
except ImportError:
    HAS_RLCARD = False
    Env = object  # 基础类


from rules import (
    generate_full_deck, CaishenEngine, WinChecker, ScoreCalculator,
    get_next_tile, tile_id_to_str, ALL_TILE_IDS, get_suit, get_tile_group
)


class WenzhouMahjongEnv:
    """
    温州麻将环境

    支持RLCard标准接口：
    - reset()
    - step(action)
    - get_legal_actions()
    - extract_state()
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 配置字典 {
                'seed': int,
                'allow_step_back': bool,
                'base_score': int (default=2)
            }
        """
        config = config or {}

        self.seed = config.get('seed', None)
        self.allow_step_back = config.get('allow_step_back', False)
        self.base_score = config.get('base_score', 2)

        # 游戏状态
        self.num_players = 2
        self.winner = None
        self.chaos = False  # 荒庄

        # 牌相关
        self.deck: List[int] = []
        self.wall_idx: int = 0

        # 玩家状态
        self.hands: List[List[int]] = [[], []]
        self.discards: List[List[int]] = [[], []]
        self.fulus: List[List[Tuple]] = [[], []]  # 副露

        # 财神
        self.caishen_id: int = 0
        self.caishen: Optional[CaishenEngine] = None

        # 计分
        self.score_calc: Optional[ScoreCalculator] = None
        self.lianzhuang: int = 0  # 连庄次数

        # 杠分记录
        self.gang_scores: List[int] = []  # 每次杠的得分

        # 当前玩家
        self.current_player: int = 0

        # 最后打出的牌（用于吃/碰/杠判断）
        self.last_discarded: int = None

        # 过手记录（放弃胡的牌，一轮内不能再胡）
        self.passed_hu_tiles: List[int] = []

        # 历史（用于step_back）
        self.history: List[Dict] = []

        # RLCard兼容
        self.state_shape = [34 * 4 + 10]  # 手牌编码 + 游戏状态
        self.action_shape = [34]  # 34种牌

    def reset(self) -> Dict:
        """重置环境，返回初始状态"""
        import random
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # 生成牌墙
        self.deck = generate_full_deck()
        random.shuffle(self.deck)

        # 翻财神
        flip_card = self.deck[33]
        self.caishen_id = get_next_tile(flip_card)
        self.caishen = CaishenEngine(self.caishen_id)
        self.score_calc = ScoreCalculator(self.caishen_id, self.base_score)

        # 发牌：闲16张，庄17张
        self.hands = [
            sorted(self.deck[:16]),  # 闲家: 16张
            sorted(self.deck[16:33])  # 庄家: 17张
        ]

        # 初始化弃牌和副露
        self.discards = [[], []]
        self.fulus = [[], []]

        # 牌墙从第35张开始(index 34)
        self.wall_idx = 34

        # 当前玩家：庄家先出牌
        self.current_player = 1

        # 最后打出的牌
        self.last_discarded = None

        # 过手记录（一轮内放弃胡的牌不能再胡）
        self.passed_hu_tiles = []

        # 杠分记录
        self.gang_scores = []

        # 重置状态
        self.winner = None
        self.chaos = False
        self.history = []

        return self.get_state(0)

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作

        Args:
            action: 0-33 出牌, 34=过, 35=吃, 36=碰,
                   37=明杠, 38=加杠, 39=暗杠, 40=胡

        Returns:
            (next_state, reward, done, info)
        """
        # 保存历史
        if self.allow_step_back:
            self.history.append(self._get_state_snapshot())

        player = self.current_player
        opponent = 1 - player

        # 处理特殊动作
        if action == 40:  # 胡
            # 检查是否能胡（考虑过手规则）
            can_win = False
            if self.last_discarded is not None:
                # 检查是否已经过手过这张牌
                if self.last_discarded not in self.passed_hu_tiles:
                    if self._check_win(player, just_discarded=self.last_discarded):
                        can_win = True
            elif self._check_win(player):
                can_win = True

            if can_win:
                self.winner = player
                is_self_draw = (self.last_discarded is None)
                reward = self._calc_reward(player, is_self_draw=is_self_draw)
                return self._get_observation(player), reward, True, {'winner': player}
            return self._get_observation(player), 0.0, False, {}

        elif action == 36:  # 碰
            return self._handle_peng(player)
        elif action == 37:  # 明杠
            return self._handle_ming_gang(player)
        elif action == 38:  # 加杠/补杠
            return self._handle_jia_gang(player)
        elif action == 39:  # 暗杠
            return self._handle_an_gang(player)
        elif action == 35:  # 吃
            return self._handle_chi(player)
        elif action == 34:  # 过（跳过吃/碰/杠）
            # 过手：记录放弃胡的牌（一轮内不能再胡这张牌）
            if self.last_discarded is not None:
                self.passed_hu_tiles.append(self.last_discarded)

        # 摸牌阶段
        if self._is_draw_phase():
            tile = self._draw_tile(player)
            if tile is None:
                self.chaos = True
                return self._get_observation(player), 0.0, True, {'winner': -1}

            # 检查自摸
            if self._check_win(player):
                self.winner = player
                reward = self._calc_reward(player, is_self_draw=True)
                return self._get_observation(player), reward, True, {'winner': player}

        # 出牌阶段
        if action < 34 and action in self._get_legal_actions(player):
            self._discard(player, action)
            # 更新last_discarded前，清除过手记录（新的一轮开始）
            if player == 1:  # 庄家出牌后，闲家能看到
                self.passed_hu_tiles = []  # 新一轮开始，清除过手记录
            self.last_discarded = action

        # 检查对手能否和牌（点炮）
        if self.last_discarded is not None:
            # 检查是否已经过手过这张牌
            if self.last_discarded not in self.passed_hu_tiles:
                if self._check_win(opponent, just_discarded=self.last_discarded):
                    self.winner = opponent
                    reward = self._calc_reward(opponent, is_self_draw=False)
                    return self._get_observation(player), -1.0, True, {
                        'winner': opponent,
                        'is_fang_pao': True,
                        'is_self_draw': False
                    }
            else:
                # 已经过手了，不能再胡（过手规则）
                pass

        # 切换玩家
        self.current_player = opponent

        # 检查荒庄
        if self.wall_idx >= len(self.deck) - 8:
            self.chaos = True
            return self._get_observation(player), 0.0, True, {'winner': -1}

        # 检查双方手牌都为空（异常结束）
        if len(self.hands[0]) == 0 and len(self.hands[1]) == 0:
            # 双方都没手牌了，算荒庄
            self.chaos = True
            return self._get_observation(player), 0.0, True, {'winner': -1}

        return self._get_observation(player), 0.0, False, {}

    def _handle_chi(self, player: int) -> Tuple[Dict, float, bool, Dict]:
        """处理吃牌"""
        if self.last_discarded is None:
            return self._get_observation(player), 0.0, False, {}

        chi_list = self.get_legal_chi(player, self.last_discarded)
        if not chi_list:
            return self._get_observation(player), 0.0, False, {}

        # 使用第一个合法吃牌组合
        chi_tiles = chi_list[0]
        self.do_chi(player, chi_tiles)
        self.last_discarded = None

        # 吃后直接切换到对手出牌（不摸牌）
        self.current_player = 1 - player

        # 检查荒庄
        if self.wall_idx >= len(self.deck) - 8:
            self.chaos = True
            return self._get_observation(player), 0.0, True, {'winner': -1}

        return self._get_observation(player), 0.0, False, {}

    def _handle_peng(self, player: int) -> Tuple[Dict, float, bool, Dict]:
        """处理碰牌"""
        if self.last_discarded is None:
            return self._get_observation(player), 0.0, False, {}

        if not self.get_legal_peng(player, self.last_discarded):
            return self._get_observation(player), 0.0, False, {}

        self.do_peng(player, self.last_discarded)
        self.last_discarded = None

        # 碰后直接切换到对手出牌（不摸牌）
        self.current_player = 1 - player

        # 检查荒庄
        if self.wall_idx >= len(self.deck) - 8:
            self.chaos = True
            return self._get_observation(player), 0.0, True, {'winner': -1}

        return self._get_observation(player), 0.0, False, {}

    def _handle_ming_gang(self, player: int) -> Tuple[Dict, float, bool, Dict]:
        """处理明杠（别人打出第4张）"""
        if self.last_discarded is None:
            return self._get_observation(player), 0.0, False, {}

        gang_tiles = self.get_legal_gang(player, self.last_discarded)
        if not gang_tiles:
            return self._get_observation(player), 0.0, False, {}

        tile = gang_tiles[0]
        self.do_gang(player, tile, 'ming_gang')

        # 记录杠分（明杠得1分）
        self.gang_scores.append(1)

        # 杠后补牌
        new_tile = self._draw_tile(player)
        if new_tile is None:
            self.chaos = True
            return self._get_observation(player), 0.0, True, {'winner': -1}

        self.last_discarded = None

        # 检查杠上开花
        if self._check_win(player):
            self.winner = player
            reward = self._calc_reward(player, is_self_draw=True, is_gang_kai=True)
            return self._get_observation(player), reward, True, {'winner': player, 'is_gang_kai': True}

        # 切换到对手出牌
        self.current_player = 1 - player

        # 检查荒庄
        if self.wall_idx >= len(self.deck) - 8:
            self.chaos = True
            return self._get_observation(player), 0.0, True, {'winner': -1}

        return self._get_observation(player), 0.0, False, {}

    def _handle_jia_gang(self, player: int) -> Tuple[Dict, float, bool, Dict]:
        """处理加杠/补杠（碰后摸到第4张）"""
        # 加杠需要玩家已经碰了这张牌
        fulu_tiles = [tile for fulu in self.fulus[player] for tile in fulu[1]]
        if self.last_discarded not in fulu_tiles:
            return self._get_observation(player), 0.0, False, {}

        if self.last_discarded not in self.hands[player]:
            return self._get_observation(player), 0.0, False, {}

        # 执行加杠
        self.do_gang(player, self.last_discarded, 'jia_gang')

        # 记录杠分（加杠得1分）
        self.gang_scores.append(1)

        # 杠后补牌
        new_tile = self._draw_tile(player)
        if new_tile is None:
            self.chaos = True
            return self._get_observation(player), 0.0, True, {'winner': -1}

        self.last_discarded = None

        # 检查杠上开花
        if self._check_win(player):
            self.winner = player
            reward = self._calc_reward(player, is_self_draw=True, is_gang_kai=True)
            return self._get_observation(player), reward, True, {'winner': player, 'is_gang_kai': True}

        # 切换到对手出牌
        self.current_player = 1 - player

        # 检查荒庄
        if self.wall_idx >= len(self.deck) - 8:
            self.chaos = True
            return self._get_observation(player), 0.0, True, {'winner': -1}

        return self._get_observation(player), 0.0, False, {}

    def _handle_an_gang(self, player: int) -> Tuple[Dict, float, bool, Dict]:
        """处理暗杠（自己摸到4张）"""
        # 暗杠需要手中有4张相同的牌
        hand = self.hands[player]
        an_gang_tiles = [t for t in set(hand) if hand.count(t) >= 4]

        if not an_gang_tiles:
            return self._get_observation(player), 0.0, False, {}

        # 使用第一组暗杠
        tile = an_gang_tiles[0]
        self.do_gang(player, tile, 'an_gang')

        # 记录杠分（暗杠得2分）
        self.gang_scores.append(2)

        # 杠后补牌（暗杠补两张）
        for _ in range(2):
            new_tile = self._draw_tile(player)
            if new_tile is None:
                self.chaos = True
                return self._get_observation(player), 0.0, True, {'winner': -1}

        self.last_discarded = None

        # 检查暗杠后能否胡牌（一般暗杠后要继续打牌）
        # 切换到对手出牌
        self.current_player = 1 - player

        # 检查荒庄
        if self.wall_idx >= len(self.deck) - 8:
            self.chaos = True
            return self._get_observation(player), 0.0, True, {'winner': -1}

        return self._get_observation(player), 0.0, False, {}

    def _handle_win(self, player: int) -> Tuple[Dict, float, bool, Dict]:
        """处理胡牌"""
        # 检查是否真的能胡
        if self.last_discarded is not None:
            if self._check_win(player, just_discarded=self.last_discarded):
                self.winner = player
                is_self_draw = False
                reward = self._calc_reward(player, is_self_draw=is_self_draw)
                return self._get_observation(player), reward, True, {'winner': player}
        elif self._check_win(player):
            self.winner = player
            reward = self._calc_reward(player, is_self_draw=True)
            return self._get_observation(player), reward, True, {'winner': player}

        return self._get_observation(player), 0.0, False, {}

    def step_back(self) -> bool:
        """回退到上一个状态"""
        if not self.allow_step_back or not self.history:
            return False

        snapshot = self.history.pop()
        self.hands = snapshot['hands']
        self.discards = snapshot['discards']
        self.fulus = snapshot['fulus']
        self.wall_idx = snapshot['wall_idx']
        self.current_player = snapshot['current_player']
        self.winner = snapshot['winner']
        self.chaos = snapshot['chaos']

        return True

    def get_legal_actions(self) -> List[int]:
        """获取当前玩家的合法动作"""
        return self._get_legal_actions(self.current_player)

    def get_state(self, player: int = None) -> Dict:
        """获取玩家状态，默认当前玩家"""
        if player is None:
            player = self.current_player
        return self._get_observation(player)

    def _get_observation(self, player: int) -> Dict:
        """获取当前观察"""
        opponent = 1 - player

        # 编码手牌
        hand_encoding = self._encode_hand(self.hands[player])

        # 编码对手弃牌
        opp_disc_encoding = self._encode_discards(self.discards[opponent])

        # 编码己方弃牌
        own_disc_encoding = self._encode_discards(self.discards[player])

        # 游戏状态
        game_state = np.array([
            self.wall_idx / 100.0,  # 牌墙剩余
            self.current_player / 1.0,  # 当前玩家
            self.lianzhuang / 4.0,  # 连庄次数
            1.0 if self.caishen_id == 33 else 0.0,  # 财神是否白板
        ] + [0.0] * 6)  # padding

        return {
            'obs': np.concatenate([hand_encoding, opp_disc_encoding, own_disc_encoding, game_state]),
            'legal_actions': self._get_legal_actions(player),
            'raw_legal_actions': self._get_legal_actions(player),
            'player': player,
            'hand': self.hands[player].copy(),
            'caishen': self.caishen_id,
            'wall_remaining': len(self.deck) - self.wall_idx,
        }

    def _encode_hand(self, hand: List[int]) -> np.ndarray:
        """编码手牌 (34*4 one-hot + count)"""
        encoding = np.zeros(34 * 5)
        counts = Counter(hand)

        wild_count = sum(1 for t in hand if self.caishen and self.caishen.is_wild(t))

        for tile_id in range(34):
            idx = tile_id * 5
            count = counts.get(tile_id, 0)
            encoding[idx] = 1 if count > 0 else 0  # 有没有
            encoding[idx + 1] = 1 if count >= 2 else 0  # 是否有对
            encoding[idx + 2] = 1 if count >= 3 else 0  # 是否有刻
            encoding[idx + 3] = 1 if count >= 4 else 0  # 是否暗杠
            encoding[idx + 4] = min(count, 5) / 5.0  # 数量归一化

        return encoding

    def _encode_discards(self, discards: List[int]) -> np.ndarray:
        """编码弃牌 (34 one-hot + 最近3张)"""
        encoding = np.zeros(34 * 2)
        counts = Counter(discards)

        for tile_id, count in counts.items():
            encoding[tile_id] = min(count, 3) / 3.0

        # 最近弃牌标记
        for i, tile in enumerate(discards[-3:]):
            encoding[34 + tile] = (3 - i) / 3.0

        return encoding

    def _get_legal_actions(self, player: int) -> List[int]:
        """获取合法出牌动作"""
        hand = self.hands[player]
        if not hand:
            return []

        legal = []

        for tile in set(hand):
            # 财神和白板不能出
            if self.caishen and self.caishen.is_wild(tile):
                continue
            legal.append(tile)

        return legal if legal else [hand[0]]

    def get_legal_chi(self, player: int, discarded_tile: int) -> List[Tuple[int, int, int]]:
        """
        获取能吃的牌组合

        吃牌规则：
        - 只能吃上家（对手是 current_player，自己是 next player）
        - 只能吃顺子，分吃头(左)、吃中、吃尾(右)

        Returns:
            List of (chi_head, chi_middle, chi_tail) tuples
        """
        # 只有闲家(玩家0)能吃上家(庄家)打出的牌
        if player != 0:
            return []

        hand = self.hands[player]
        chi_list = []

        # 检查是否有足够的牌组成顺子
        for offset in [-1, 0, 1]:  # 吃头、中、尾
            t1 = discarded_tile + offset
            t2 = discarded_tile + offset + 1
            t3 = discarded_tile + offset + 2

            # 检查范围
            if t1 < 0 or t3 > 26:
                continue

            # 检查是否同花色（字牌不能吃）
            if get_tile_group(t1) != get_tile_group(discarded_tile):
                continue

            # 检查手牌是否有这些牌
            count1 = hand.count(t1)
            count2 = hand.count(t2)
            count3 = hand.count(t3)

            # 需要的牌数（排除打出的那张）
            need1 = 1 if t1 != discarded_tile else 0
            need2 = 1 if t2 != discarded_tile else 0
            need3 = 1 if t3 != discarded_tile else 0

            if count1 >= need1 and count2 >= need2 and count3 >= need3:
                chi_list.append((t1, t2, t3))

        return chi_list

    def get_legal_peng(self, player: int, discarded_tile: int) -> bool:
        """
        获取能否碰牌

        碰牌规则：
        - 手中有一对
        - 他人打出同张可碰
        """
        hand = self.hands[player]
        return hand.count(discarded_tile) >= 2

    def get_legal_gang(self, player: int, discarded_tile: int = None) -> List[int]:
        """
        获取能否杠牌

        杠牌规则：
        - 明杠：三张同牌 + 他人打出同张可杠
        - 暗杠：四张相同（自己摸到）
        - 加杠：碰后补杠（自己碰了一对，再摸到第四张）
        """
        hand = self.hands[player]
        gang_tiles = []

        # 明杠：手中有三张， opponent打出第四张
        if discarded_tile is not None:
            if hand.count(discarded_tile) >= 3:
                gang_tiles.append(discarded_tile)

        # 暗杠：手中有四张（需要单独操作）
        for tile in set(hand):
            if hand.count(tile) >= 4:
                if tile not in gang_tiles:
                    gang_tiles.append(tile)

        return gang_tiles

    def can_chi(self, player: int) -> bool:
        """检查是否能吃"""
        # 需要知道上一张打出的牌
        # 简化：吃需要上家出牌
        return self.current_player == 0  # 只有闲家能吃（庄家先出，闲家是上家）

    def do_chi(self, player: int, chi_tiles: Tuple[int, int, int]):
        """
        执行吃牌

        Args:
            player: 玩家ID
            chi_tiles: (t1, t2, t3) 要吃的顺子组合
        """
        # 吃牌后需要从手牌中移除相应牌
        # 注意：chi_tiles中包含打出的牌(最后打出的那张)，不需要从手牌移除
        discarded = self.last_discarded
        for t in chi_tiles:
            # 跳过大米（打出的牌不在手牌中）
            if t == discarded:
                continue
            if t in self.hands[player]:
                self.hands[player].remove(t)

        # 记录副露
        self.fulus[player].append(('chow', list(chi_tiles)))

    def do_peng(self, player: int, tile: int):
        """
        执行碰牌

        Args:
            player: 玩家ID
            tile: 碰的牌（三张相同）
        """
        # 从手牌移除两张
        for _ in range(2):
            if tile in self.hands[player]:
                self.hands[player].remove(tile)

        # 记录副露
        self.fulus[player].append(('pong', [tile, tile, tile]))

    def do_gang(self, player: int, tile: int, gang_type: str = 'ming_gang'):
        """
        执行杠牌

        Args:
            player: 玩家ID
            tile: 杠的牌
            gang_type: 'ming_gang'(明杠), 'an_gang'(暗杠), 'jia_gang'(加杠)
        """
        if gang_type == 'ming_gang':
            # 明杠：从手牌移除三张
            for _ in range(3):
                if tile in self.hands[player]:
                    self.hands[player].remove(tile)
        elif gang_type == 'an_gang':
            # 暗杠：四张都留下，但需要特殊标记
            pass
        elif gang_type == 'jia_gang':
            # 加杠：碰后再摸一张杠
            if tile in self.hands[player]:
                self.hands[player].remove(tile)

        # 记录副露
        self.fulus[player].append(('kong', [tile, tile, tile, tile]))

    def _is_draw_phase(self) -> bool:
        """是否是摸牌阶段（玩家手牌为16张时）"""
        return len(self.hands[self.current_player]) == 16

    def _draw_tile(self, player: int) -> Optional[int]:
        """摸牌"""
        if self.wall_idx >= len(self.deck):
            return None
        tile = self.deck[self.wall_idx]
        self.wall_idx += 1
        self.hands[player].append(tile)
        self.hands[player].sort()
        return tile

    def _discard(self, player: int, tile: int):
        """出牌"""
        if tile in self.hands[player]:
            self.hands[player].remove(tile)
            self.discards[player].append(tile)

    def _check_win(self, player: int, just_discarded: int = None) -> bool:
        """检查玩家能否和牌"""
        hand = self.hands[player].copy()
        if just_discarded is not None:
            hand.append(just_discarded)

        if len(hand) != 17:
            return False

        checker = WinChecker(self.caishen_id)
        return checker.check_win(hand)['can_win']

    def _calc_reward(self, winner: int, is_self_draw: bool, is_gang_kai: bool = False) -> float:
        """计算奖励"""
        hand = self.hands[winner]
        result = self.score_calc.calc_score(hand, is_self_draw=is_self_draw)
        return float(result['total_score'])

    def _get_state_snapshot(self) -> Dict:
        """获取状态快照"""
        return {
            'hands': [h.copy() for h in self.hands],
            'discards': [d.copy() for d in self.discards],
            'fulus': [f.copy() for f in self.fulus],
            'wall_idx': self.wall_idx,
            'current_player': self.current_player,
            'winner': self.winner,
            'chaos': self.chaos,
        }

    @property
    def num_actions(self) -> int:
        """动作数量: 34种牌 + 过/吃/碰/明杠/加杠/暗杠/胡"""
        return 41  # 0-33=出牌, 34=过, 35=吃, 36=碰, 37=明杠, 38=加杠, 39=暗杠, 40=胡

    def get_legal_actions_full(self, player: int) -> List[int]:
        """
        获取完整合法动作列表（包括吃/碰/杠/胡）

        注意：
        - 过手规则 - 如果玩家放弃了胡某张牌，一轮内不能再胡
        - 最后四张禁吃碰杠
        - 明杠+1分，加杠+1分，暗杠+2分
        """
        actions = []

        # 基本出牌
        actions.extend(self._get_legal_actions(player))

        # 检查是否最后四张（最后四张禁吃碰杠）
        wall_remaining = len(self.deck) - self.wall_idx
        is_last_four = wall_remaining <= 4

        # 检查能否碰（最后四张除外）
        if not is_last_four and self.last_discarded is not None and self.get_legal_peng(player, self.last_discarded):
            actions.append(36)

        # 检查能否明杠（最后四张除外）
        if not is_last_four and self.last_discarded is not None:
            hand = self.hands[player]
            if self.last_discarded in hand and hand.count(self.last_discarded) >= 3:
                actions.append(37)  # 明杠

        # 检查能否加杠（碰后摸到第4张）- 最后四张除外
        if not is_last_four and self.last_discarded is not None:
            fulu_tiles = [tile for fulu in self.fulus[player] for tile in fulu[1]]
            pair_count = fulu_tiles.count(self.last_discarded) // 3
            if pair_count > 0 and self.last_discarded in hand and hand.count(self.last_discarded) >= 1:
                actions.append(38)  # 加杠

        # 检查能否暗杠（自己摸到4张）- 最后四张除外
        if not is_last_four:
            hand = self.hands[player]
            for tile in set(hand):
                if hand.count(tile) >= 4:
                    actions.append(39)  # 暗杠
                    break

        # 检查能否吃（只有闲家能吃上家，最后四张除外）
        if not is_last_four and self.last_discarded is not None and self.can_chi(player):
            if self.get_legal_chi(player, self.last_discarded):
                actions.append(35)

        # 检查能否胡（考虑过手规则）
        if self.last_discarded is not None:
            if self.last_discarded not in self.passed_hu_tiles:
                if self._check_win(player, just_discarded=self.last_discarded):
                    actions.append(40)  # 胡
        elif self._check_win(player):
            actions.append(40)  # 自摸胡

        return sorted(set(actions))

    def get_perfect_information(self) -> Dict:
        """获取完整信息（用于debug）"""
        return {
            'hands': self.hands,
            'discards': self.discards,
            'fulus': self.fulus,
            'caishen': self.caishen_id,
            'caishen_name': tile_id_to_str(self.caishen_id),
            'wall_remaining': len(self.deck) - self.wall_idx,
            'current_player': self.current_player,
            'lianzhuang': self.lianzhuang,
            'winner': self.winner,
            'chaos': self.chaos,
        }
