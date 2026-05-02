"""
对战测试 - 使用训练好的模型进行AI对战
"""

import argparse
import numpy as np
from env import WenzhouMahjongEnv
from ai import PPOAgent, DQNAgent, MCTSPlayer
from rules import tile_id_to_str


class BattleGame:
    """AI对战"""

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.env = WenzhouMahjongEnv()
        self.device = device

        state_dim = 340  # 34*5 + 34*2 + 10
        action_dim = 34

        # 加载模型
        if model_path and model_path.endswith('.pt'):
            try:
                self.agent_0 = PPOAgent(state_dim, action_dim, device=device)
                self.agent_0.load(model_path)
                self.agent_1 = PPOAgent(state_dim, action_dim, device=device)
                self.agent_1.load(model_path)
                print(f"加载模型: {model_path}")
            except:
                print("模型加载失败，使用随机AI")
                self.agent_0 = MCTSPlayer(0, use_mcts=False)
                self.agent_1 = MCTSPlayer(0, use_mcts=False)
        else:
            self.agent_0 = MCTSPlayer(0, use_mcts=False)
            self.agent_1 = MCTSPlayer(0, use_mcts=False)

    def play(self, verbose: bool = True) -> dict:
        """玩一局"""
        state = self.env.reset()

        if verbose:
            info = self.env.get_perfect_information()
            print(f"财神: {info['caishen_name']}")
            print(f"庄家: Player1")

        rounds = 0

        while True:
            # 获取当前玩家的状态
            current = self.env.current_player
            state = self.env.get_state()  # 获取当前玩家状态
            agent = self.agent_0 if current == 0 else self.agent_1

            # 选择动作
            legal_actions = state['legal_actions']
            hand = state['hand']

            if hasattr(agent, 'select_action'):
                action = agent.select_action(hand, legal_actions, state['wall_remaining'])
            else:
                action = agent.select_action(hand, legal_actions)

            # 执行
            next_state, reward, done, info = self.env.step(action)

            if verbose:
                player_name = "闲家" if current == 0 else "庄家"
                print(f"{player_name}打出: {tile_id_to_str(action)}")

            if done:
                winner = info.get('winner', -1)
                if winner == 0:
                    print("闲家胡牌!")
                elif winner == 1:
                    print("庄家胡牌!")
                else:
                    print("流局!")

                return {
                    'winner': winner,
                    'rounds': rounds,
                    'reward': reward
                }

            current = 1 - current
            state = next_state
            rounds += 1

    def battle(self, num_games: int = 100, verbose: bool = False) -> dict:
        """多局对战"""
        stats = {'p0': 0, 'p1': 0, 'draw': 0, 'avg_rounds': 0}

        for i in range(num_games):
            result = self.play(verbose=False)
            if result['winner'] == 0:
                stats['p0'] += 1
            elif result['winner'] == 1:
                stats['p1'] += 1
            else:
                stats['draw'] += 1
            stats['avg_rounds'] += result['rounds']

            if verbose and (i + 1) % 10 == 0:
                print(f"已完成: {i+1}/{num_games}")

        stats['avg_rounds'] /= num_games

        print(f"\n=== 对战结果 ({num_games}局) ===")
        print(f"闲家胜: {stats['p0']} ({100*stats['p0']/num_games:.1f}%)")
        print(f"庄家胜: {stats['p1']} ({100*stats['p1']/num_games:.1f}%)")
        print(f"流局: {stats['draw']} ({100*stats['draw']/num_games:.1f}%)")
        print(f"平均回合: {stats['avg_rounds']:.1f}")

        return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--games', type=int, default=10, help='对战局数')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    game = BattleGame(args.model, args.device)

    if args.games == 1:
        game.play(verbose=True)
    else:
        game.battle(args.games, verbose=args.verbose)


if __name__ == '__main__':
    main()
