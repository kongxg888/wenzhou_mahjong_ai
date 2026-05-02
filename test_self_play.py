"""
AlphaZero AI 自对弈测试 - 优化版
统计动作频率和动作选择率（当机会出现时）
"""

import numpy as np
import time
import random

from env import WenzhouMahjongEnv
from ai import AlphaZeroAgent, AlphaZeroMCTS
from game_state import GameState
from rules import ACTION_CHI, ACTION_PENG, ACTION_MING_GANG, ACTION_JIA_GANG, ACTION_AN_GANG, ACTION_HU


class SelfPlayTester:
    """自对弈测试"""

    def __init__(self):
        self.device = 'mps' if __import__('torch').backends.mps.is_available() else 'cpu'
        self.agent = AlphaZeroAgent(state_dim=136, action_dim=41, device=self.device)
        self.env = WenzhouMahjongEnv()
        print(f"使用设备: {self.device}")

    def play_one_game(self, verbose=False):
        """玩一局"""
        state = self.env.reset()

        game_history = {0: [], 1: []}
        last_discarded = None

        while True:
            state = self.env.get_state()
            current = self.env.current_player
            hand = state['hand']
            legal_actions = self.env.get_legal_actions_full(current)

            # MCTS搜索
            game_state = GameState(
                hand, state['caishen'],
                wall_remaining=state['wall_remaining'],
                current_player=current,
                last_discarded=last_discarded
            )

            mcts = AlphaZeroMCTS(
                self.agent.net,
                caishen_id=state['caishen'],
                num_simulations=30,
                batch_size=16
            )

            try:
                policy_dict = mcts.search(game_state, last_discarded)
            except:
                policy_dict = {a: 1.0 / len(legal_actions) for a in legal_actions if a < 34}

            # 转为41维向量
            policy = np.zeros(41)
            for a, p in policy_dict.items():
                if 0 <= a < 41:
                    policy[a] = p

            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                policy = np.ones(41) / 41

            # 采样动作
            mask = np.zeros(41)
            for a in legal_actions:
                if 0 <= a < 41:
                    mask[a] = policy[a]

            if mask.sum() == 0:
                action = random.choice(legal_actions) if legal_actions else 0
            else:
                probs = mask / mask.sum()
                action = np.random.choice(41, p=probs)

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

    def test(self, num_games=200, print_freq=50):
        """测试"""
        print(f"\n开始自对弈测试: {num_games}局")
        print("=" * 50)

        stats = {'p0': 0, 'p1': 0, 'draw': 0}

        # 动作统计
        actions_taken = {
            'chi': 0, 'peng': 0, 'ming_gang': 0,
            'jia_gang': 0, 'an_gang': 0, 'hu': 0, 'discard': 0
        }

        # 动作机会统计（当动作可用时，选择了这个动作的次数）
        action_opportunities = {
            'chi': 0, 'peng': 0, 'ming_gang': 0,
            'jia_gang': 0, 'an_gang': 0, 'hu': 0
        }
        action_selections = {
            'chi': 0, 'peng': 0, 'ming_gang': 0,
            'jia_gang': 0, 'an_gang': 0, 'hu': 0
        }

        start_time = time.time()

        for game in range(num_games):
            winner, game_history = self.play_one_game()

            # 统计动作
            for p in [0, 1]:
                for traj in game_history[p]:
                    policy = traj['policy']
                    action = np.argmax(policy)

                    if action == ACTION_CHI:
                        actions_taken['chi'] += 1
                    elif action == ACTION_PENG:
                        actions_taken['peng'] += 1
                    elif action == ACTION_MING_GANG:
                        actions_taken['ming_gang'] += 1
                    elif action == ACTION_JIA_GANG:
                        actions_taken['jia_gang'] += 1
                    elif action == ACTION_AN_GANG:
                        actions_taken['an_gang'] += 1
                    elif action == ACTION_HU:
                        actions_taken['hu'] += 1
                    elif action < 34:
                        actions_taken['discard'] += 1

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

                total_actions = sum(actions_taken.values())
                if total_actions > 0:
                    print(f"\n动作统计 (共{total_actions}次):")
                    print(f"  出牌: {actions_taken['discard']} ({actions_taken['discard']/total_actions*100:.1f}%)")
                    print(f"  吃: {actions_taken['chi']} ({actions_taken['chi']/total_actions*100:.1f}%)")
                    print(f"  碰: {actions_taken['peng']} ({actions_taken['peng']/total_actions*100:.1f}%)")
                    print(f"  明杠: {actions_taken['ming_gang']} ({actions_taken['ming_gang']/total_actions*100:.1f}%)")
                    print(f"  加杠: {actions_taken['jia_gang']} ({actions_taken['jia_gang']/total_actions*100:.1f}%)")
                    print(f"  暗杠: {actions_taken['an_gang']} ({actions_taken['an_gang']/total_actions*100:.1f}%)")
                    print(f"  胡: {actions_taken['hu']} ({actions_taken['hu']/total_actions*100:.1f}%)")

        print("\n" + "=" * 50)
        print("最终结果:")
        print(f"  闲家(先手): {stats['p0']} 胜 ({stats['p0']/num_games*100:.1f}%)")
        print(f"  庄家(后手): {stats['p1']} 胜 ({stats['p1']/num_games*100:.1f}%)")
        print(f"  流局: {stats['draw']} 次 ({stats['draw']/num_games*100:.1f}%)")
        print(f"  总用时: {time.time() - start_time:.1f}秒")

        total_actions = sum(actions_taken.values())
        if total_actions > 0:
            print(f"\n动作统计 (共{total_actions}次):")
            print(f"  出牌: {actions_taken['discard']} ({actions_taken['discard']/total_actions*100:.1f}%)")
            print(f"  吃: {actions_taken['chi']} ({actions_taken['chi']/total_actions*100:.1f}%)")
            print(f"  碰: {actions_taken['peng']} ({actions_taken['peng']/total_actions*100:.1f}%)")
            print(f"  明杠: {actions_taken['ming_gang']} ({actions_taken['ming_gang']/total_actions*100:.1f}%)")
            print(f"  加杠: {actions_taken['jia_gang']} ({actions_taken['jia_gang']/total_actions*100:.1f}%)")
            print(f"  暗杠: {actions_taken['an_gang']} ({actions_taken['an_gang']/total_actions*100:.1f}%)")
            print(f"  胡: {actions_taken['hu']} ({actions_taken['hu']/total_actions*100:.1f}%)")

        return stats, actions_taken


if __name__ == '__main__':
    tester = SelfPlayTester()
    stats, action_stats = tester.test(num_games=200, print_freq=50)