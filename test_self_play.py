"""
AlphaZero AI 自对弈测试
"""

import numpy as np
import time
from collections import deque
import random

from env import WenzhouMahjongEnv
from ai import AlphaZeroAgent, AlphaZeroMCTS


class SelfPlayTester:
    """自对弈测试"""

    def __init__(self):
        self.device = 'mps' if __import__('torch').backends.mps.is_available() else 'cpu'
        self.agent = AlphaZeroAgent(state_dim=136, action_dim=34, device=self.device)
        self.mcts = AlphaZeroMCTS(self.agent.net, caishen_id=0, num_simulations=30)
        self.env = WenzhouMahjongEnv()
        print(f"使用设备: {self.device}")

    def play_one_game(self, verbose=False):
        """玩一局"""
        state = self.env.reset()

        trajectories = []
        current = 1  # 庄家先出
        game_history = {0: [], 1: []}
        last_discarded = None

        while True:
            state = self.env.get_state()
            hand = state['hand']
            legal_actions = state['legal_actions']

            # MCTS搜索 - 使用完整合法动作
            game_state = GameState(hand, state['caishen'],
                                  wall_remaining=state['wall_remaining'],
                                  current_player=current,
                                  last_discarded=last_discarded)

            try:
                self.mcts.caishen_id = state['caishen']
                policy_dict = self.mcts.search(game_state, last_discarded)
            except:
                policy_dict = {a: 1.0 / len(legal_actions) for a in legal_actions if a < 34}

            # 转换为39维向量
            policy = np.zeros(39)
            for a, p in policy_dict.items():
                if a < 34:
                    policy[a] = p

            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                policy = np.ones(39) / 39

            # 采样动作
            mask = np.zeros(39)
            for a in legal_actions:
                mask[a] = policy[a]

            if mask.sum() == 0:
                action = random.choice(legal_actions) if legal_actions else 0
            else:
                probs = mask / mask.sum()
                action = np.random.choice(39, p=probs)

            # 更新last_discarded
            if action < 34:
                last_discarded = action

            next_state, reward, done, info = self.env.step(action)

            game_history[current].append({
                'state': game_state.get_observation(),
                'policy': policy,
                'value': 0.0,
            })

            if done:
                winner = info.get('winner', -1)

                for p in [0, 1]:
                    for traj in game_history[p]:
                        if winner == p:
                            traj['value'] = 1.0
                        elif winner == 1 - p:
                            traj['value'] = -1.0
                        else:
                            traj['value'] = 0.0

                if verbose:
                    if winner == 0:
                        print("闲家胡!")
                    elif winner == 1:
                        print("庄家胡!")
                    else:
                        print("流局!")

                return winner, game_history

            current = 1 - current

    def test(self, num_games=200, print_freq=50):
        """测试"""
        print(f"\n开始自对弈测试: {num_games}局")
        print("=" * 50)

        stats = {'p0': 0, 'p1': 0, 'draw': 0}
        start_time = time.time()

        for game in range(num_games):
            winner, _ = self.play_one_game()

            if winner == 0:
                stats['p0'] += 1
            elif winner == 1:
                stats['p1'] += 1
            else:
                stats['draw'] += 1

            if (game + 1) % print_freq == 0:
                elapsed = time.time() - start_time
                p0_rate = stats['p0'] / (game + 1) * 100
                p1_rate = stats['p1'] / (game + 1) * 100
                draw_rate = stats['draw'] / (game + 1) * 100
                speed = (game + 1) / elapsed

                print(f"\n已测试 {game+1} 局:")
                print(f"  闲家(先手)胜率: {p0_rate:.1f}%")
                print(f"  庄家(后手)胜率: {p1_rate:.1f}%")
                print(f"  流局率: {draw_rate:.1f}%")
                print(f"  速度: {speed:.1f}局/秒")

        print("\n" + "=" * 50)
        print("最终结果:")
        print(f"  闲家(先手): {stats['p0']} 胜 ({stats['p0']/num_games*100:.1f}%)")
        print(f"  庄家(后手): {stats['p1']} 胜 ({stats['p1']/num_games*100:.1f}%)")
        print(f"  流局: {stats['draw']} 次 ({stats['draw']/num_games*100:.1f}%)")
        print(f"  总用时: {time.time() - start_time:.1f}秒")

        return stats


class GameState:
    """简化游戏状态（用于MCTS）"""

    ACTION_CHI = 35
    ACTION_PENG = 36
    ACTION_MING_GANG = 37
    ACTION_JIA_GANG = 38
    ACTION_AN_GANG = 39
    ACTION_HU = 40

    def __init__(self, hand, caishen_id, wall_remaining, current_player, last_discarded=None):
        self.hand = list(hand) if not isinstance(hand, list) else hand
        self.caishen_id = caishen_id
        self.wall_remaining = wall_remaining
        self.current_player = current_player
        self.last_discarded = last_discarded
        self.discarded = []
        self.fulus = []

    def get_legal_actions(self):
        legal = []
        for tile in set(self.hand):
            if tile != self.caishen_id and tile != 33:
                legal.append(tile)
        return legal if legal else list(set(self.hand))

    def get_legal_actions_with_special(self, last_discarded=None) -> list:
        """获取完整合法动作包括吃碰杠胡"""
        legal = self.get_legal_actions()

        # 明杠（对手打出第4张）
        if last_discarded is not None and self.hand.count(last_discarded) >= 3:
            legal.append(self.ACTION_MING_GANG)

        # 吃（只有闲家能吃上家）
        if last_discarded is not None and self.current_player == 0:
            if self._can_chi(last_discarded):
                legal.append(self.ACTION_CHI)

        # 胡（手牌17张时可胡）
        if len(self.hand) == 17:
            legal.append(self.ACTION_HU)

        return legal

    def _can_chi(self, discarded_tile: int) -> bool:
        """检查是否能吃"""
        tile_group = 0 if discarded_tile < 9 else 1 if discarded_tile < 18 else 2 if discarded_tile < 27 else 3
        if tile_group >= 3:
            return False
        for offset in [-1, 0, 1]:
            t1 = discarded_tile + offset
            t2 = t1 + 1
            t3 = t1 + 2
            if t1 < 0 or t3 > 26:
                continue
            t1_group = 0 if t1 < 9 else 1 if t1 < 18 else 2 if t1 < 27 else 3
            if t1_group != tile_group:
                continue
            if (self.hand.count(t1) >= 1 and
                self.hand.count(t2) >= 1 and
                self.hand.count(t3) >= 1):
                return True
        return False

    def do_action(self, action: int):
        """执行动作（包括吃/碰/杠/胡）"""
        if action == self.ACTION_CHI:  # 吃
            if self.last_discarded is not None:
                for offset in [-1, 0, 1]:
                    t1 = self.last_discarded + offset
                    t2 = t1 + 1
                    t3 = t1 + 2
                    if t1 < 0 or t3 > 26:
                        continue
                    t1_group = 0 if t1 < 9 else 1 if t1 < 18 else 2 if t1 < 27 else 3
                    last_group = 0 if self.last_discarded < 9 else 1 if self.last_discarded < 18 else 2 if self.last_discarded < 27 else 3
                    if t1_group != last_group:
                        continue
                    if (self.hand.count(t1) >= 1 and
                        self.hand.count(t2) >= 1 and
                        self.hand.count(t3) >= 1):
                        self._do_chi((t1, t2, t3))
                        self.last_discarded = None
                        self.current_player = 1 - self.current_player
                        return
            return
        elif action == self.ACTION_PENG:  # 碰
            if self.last_discarded is not None and self.hand.count(self.last_discarded) >= 2:
                self._do_peng(self.last_discarded)
                self.last_discarded = None
                self.current_player = 1 - self.current_player
                return
        elif action == self.ACTION_MING_GANG:  # 明杠
            if self.last_discarded is not None and self.hand.count(self.last_discarded) >= 3:
                self._do_gang(self.last_discarded, 'ming')
                self.last_discarded = None
                self.current_player = 1 - self.current_player
                return
        elif action == self.ACTION_JIA_GANG:  # 加杠
            if self.last_discarded is not None and self._has_jia_gang(self.last_discarded):
                self._do_gang(self.last_discarded, 'jia')
                self.last_discarded = None
                self.current_player = 1 - self.current_player
                return
        elif action == self.ACTION_AN_GANG:  # 暗杠
            if self._can_an_gang():
                tile = self._get_an_gang_tile()
                self._do_gang(tile, 'an')
                self.last_discarded = None
                return
        elif action == self.ACTION_HU:  # 胡 - 终局
            self.wall_remaining = 0
            return

        # 出牌动作
        if action in self.hand:
            self.hand.remove(action)
            self.discarded.append(action)
            self.last_discarded = action
        self.wall_remaining -= 1
        self.current_player = 1 - self.current_player

    def _do_chi(self, chi_tiles):
        for t in chi_tiles:
            if t in self.hand:
                self.hand.remove(t)
        self.fulus.append(('chow', list(chi_tiles))) if hasattr(self, 'fulus') else None

    def _do_peng(self, tile):
        for _ in range(2):
            if tile in self.hand:
                self.hand.remove(tile)
        if hasattr(self, 'fulus'):
            self.fulus.append(('pong', [tile, tile, tile]))

    def _do_gang(self, tile, gang_type='ming'):
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
        if hasattr(self, 'fulus'):
            self.fulus.append(('kong', [tile, tile, tile, tile]))

    def _has_jia_gang(self, tile) -> bool:
        """检查能否加杠"""
        for fulu in getattr(self, 'fulus', []):
            if fulu[0] == 'pong' and fulu[1][0] == tile:
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

    def is_terminal(self):
        return self.wall_remaining <= 0

    def get_observation(self):
        obs = np.zeros(136, dtype=np.float32)

        for tile in self.hand:
            if 0 <= tile < 34:
                obs[tile] += 1

        for tile in self.discarded[-20:]:
            if 0 <= tile < 34:
                obs[34 + tile] += 1

        if self.wall_remaining > 0:
            idx = min(33, max(0, 68 + (self.wall_remaining // 3)))
            obs[idx] = 1

        if 0 <= self.caishen_id < 34:
            obs[104 + min(self.caishen_id, 31)] = 1

        return obs


if __name__ == '__main__':
    tester = SelfPlayTester()
    stats = tester.test(num_games=200, print_freq=50)