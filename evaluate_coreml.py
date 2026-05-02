"""
温州麻将AI评估工具 - Core ML Neural Engine 加速版 (简化版)
"""

import numpy as np
import time
import coremltools as ct
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from env import WenzhouMahjongEnv


@dataclass
class EvaluationReport:
    total_games: int = 0
    p0_wins: int = 0
    p1_wins: int = 0
    draws: int = 0
    p0_win_rate: float = 0.0
    p1_win_rate: float = 0.0
    draw_rate: float = 0.0
    fan_distribution: field(default_factory=Counter) = field(default_factory=Counter)
    avg_win_fan: float = 0.0
    total_chi: int = 0
    total_peng: int = 0
    total_gang: int = 0
    fang_pao_rate: float = 0.0
    self_draw_rate: float = 0.0


class CoreMLEvaluator:
    """Core ML 加速的 AI 评估器"""

    def __init__(self, model_path: Optional[str] = None):
        self.env = WenzhouMahjongEnv()

        # 加载 Core ML 模型
        if model_path:
            self.model = ct.models.MLModel(model_path)
            print(f"Core ML 模型加载: {model_path}")
        else:
            self.model = None

        self.records: List[dict] = []

    def evaluate(self, num_games: int = 1000, print_freq: int = 100) -> EvaluationReport:
        """运行评估"""
        print(f"\n{'='*50}")
        print(f"Core ML 加速评估 - {num_games}局")
        print(f"{'='*50}")

        start_time = time.time()

        for game_idx in range(num_games):
            record = self._play_one_game()
            self.records.append(record)

            if (game_idx + 1) % print_freq == 0:
                elapsed = time.time() - start_time
                speed = (game_idx + 1) / elapsed
                p0_wr = sum(1 for r in self.records if r['winner'] == 0) / (game_idx + 1)
                p1_wr = sum(1 for r in self.records if r['winner'] == 1) / (game_idx + 1)
                print(f"[{game_idx+1}局 | {speed:.1f}局/秒 | 闲{p0_wr:.1%} 庄{p1_wr:.1%}]")

        return self._calc_stats(num_games)

    def _play_one_game(self) -> dict:
        """玩一局"""
        state = self.env.reset()

        record = {
            'winner': -1,
            'chi': 0,
            'peng': 0,
            'gang': 0,
            'fang_pao': 0,
            'is_self_draw': False,
            'fan': 0,
        }

        current = 1
        last_discarded = None

        while True:
            state = self.env.get_state()
            hand = state['hand']
            legal_actions = self.env.get_legal_actions_full(current)

            # 获取策略
            if self.model is not None:
                obs = self._get_obs(state)
                policy = self._predict(obs)
            else:
                policy = np.ones(39) / 39

            # 选择动作
            mask = np.zeros(39)
            for a in legal_actions:
                if a < len(policy):
                    mask[a] = max(0, policy[a])

            if mask.sum() > 0:
                mask = mask / mask.sum()
                action = np.random.choice(39, p=mask)
            else:
                action = legal_actions[0] if legal_actions else 0

            # 记录
            if action == 35:
                record['chi'] += 1
            elif action == 36:
                record['peng'] += 1
            elif action == 37:
                record['gang'] += 1

            # 更新
            if action < 34:
                last_discarded = action

            # 执行
            next_state, reward, done, info = self.env.step(action)

            if done:
                record['winner'] = info.get('winner', -1)
                if record['winner'] >= 0:
                    record['is_self_draw'] = info.get('is_self_draw', False)
                    record['fang_pao'] = 1 if info.get('is_fang_pao') else 0

                    win_hand = self.env.hands[record['winner']].copy()
                    if info.get('is_fang_pao') and self.env.last_discarded:
                        win_hand.append(self.env.last_discarded)

                    from rules import WinChecker
                    checker = WinChecker(self.env.caishen_id)
                    result = checker.check_win(win_hand)
                    record['fan'] = result.get('fan', 1)

                return record

            current = 1 - current

    def _get_obs(self, state: dict) -> np.ndarray:
        """获取观察"""
        obs = np.zeros(136, dtype=np.float32)
        for tile in state['hand']:
            if 0 <= tile < 34:
                obs[tile] += 1

        opp = 1 - self.env.current_player
        for tile in self.env.discards[opp][-20:]:
            if 0 <= tile < 34:
                obs[34 + tile] += 1

        wall_rem = len(self.env.deck) - self.env.wall_idx
        if wall_rem > 0:
            idx = min(33, max(0, 102 + (wall_rem // 3)))
            obs[idx] = 1

        if 0 <= self.env.caishen_id < 34:
            obs[104 + min(self.env.caishen_id, 31)] = 1

        return obs

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        """Core ML 推理"""
        input_data = obs.astype(np.float32).reshape(1, -1)
        result = self.model.predict({"input": input_data})

        policy = None
        for key, val in result.items():
            if isinstance(val, np.ndarray) and val.shape in [(39,), (1, 39)]:
                policy = val.flatten()
                break

        return policy if policy is not None else np.ones(39) / 39

    def _calc_stats(self, num_games: int) -> EvaluationReport:
        """计算统计"""
        records = self.records[:num_games]

        report = EvaluationReport()
        report.total_games = num_games

        report.p0_wins = sum(1 for r in records if r['winner'] == 0)
        report.p1_wins = sum(1 for r in records if r['winner'] == 1)
        report.draws = sum(1 for r in records if r['winner'] == -1)

        report.p0_win_rate = report.p0_wins / num_games
        report.p1_win_rate = report.p1_wins / num_games
        report.draw_rate = report.draws / num_games

        hu = [r for r in records if r['winner'] >= 0]
        if hu:
            for r in hu:
                report.fan_distribution[r['fan']] += 1
            report.avg_win_fan = sum(r['fan'] for r in hu) / len(hu)

        report.total_chi = sum(r['chi'] for r in records)
        report.total_peng = sum(r['peng'] for r in records)
        report.total_gang = sum(r['gang'] for r in records)
        report.fang_pao_rate = sum(r['fang_pao'] for r in records) / num_games
        report.self_draw_rate = sum(1 for r in hu if r['is_self_draw']) / len(hu) if hu else 0

        return report

    def print_report(self, report: EvaluationReport):
        """打印报告"""
        print(f"\n{'='*50}")
        print("评估报告")
        print(f"{'='*50}")

        print(f"\n📊 基础统计 ({report.total_games}局)")
        print(f"   闲家: {report.p0_win_rate:.1%} ({report.p0_wins}胜)")
        print(f"   庄家: {report.p1_win_rate:.1%} ({report.p1_wins}胜)")
        print(f"   流局: {report.draw_rate:.1%} ({report.draws}局)")

        print(f"\n🎯 番数: 平均{report.avg_win_fan:.1f}")

        print(f"\n🔄 吃碰杠: {report.total_chi}/{report.total_peng}/{report.total_gang}")
        print(f"💨 点炮率: {report.fang_pao_rate:.1%}")
        print(f"🎲 自摸率: {report.self_draw_rate:.1%}")
        print(f"{'='*50}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--model', type=str,
                       default='checkpoints/alpha_zero_coreml.mlpackage')
    parser.add_argument('--print_freq', type=int, default=100)
    args = parser.parse_args()

    evaluator = CoreMLEvaluator(model_path=args.model)
    report = evaluator.evaluate(num_games=args.games, print_freq=args.print_freq)
    evaluator.print_report(report)


if __name__ == '__main__':
    main()
