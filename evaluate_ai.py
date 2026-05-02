"""
温州麻将AI详细评估工具 (修复版)

评估指标：
1. 胜率统计（闲家/庄家/流局）
2. 胡牌番数分布
3. 特殊牌型出现率
4. 吃碰杠使用分析
5. 每局得分统计
6. 点炮率统计
7. AI决策质量分析
"""

import numpy as np
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from env import WenzhouMahjongEnv
from ai import AlphaZeroAgent, AlphaZeroMCTS


@dataclass
class GameRecord:
    """单局记录"""
    winner: int  # -1=流局, 0=闲家, 1=庄家
    is_self_draw: bool = False
    is_fang_pao: bool = False  # 是否点炮
    win_fan: int = 0
    win_type: str = 'unknown'
    caishen_count: int = 0
    total_score: int = 0
    hand_tiles: List[int] = field(default_factory=list)
    chi_count: int = 0
    peng_count: int = 0
    gang_count: int = 0
    fang_pao_count: int = 0  # 点炮次数
    decisions: List[Dict] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """评估报告"""
    total_games: int = 0
    p0_wins: int = 0
    p1_wins: int = 0
    draws: int = 0

    # 胜率
    p0_win_rate: float = 0.0
    p1_win_rate: float = 0.0
    draw_rate: float = 0.0

    # 番数分布
    fan_distribution: Counter = field(default_factory=Counter)
    avg_win_fan: float = 0.0

    # 牌型统计
    special_types: Counter = field(default_factory=Counter)
    hu_types: Counter = field(default_factory=Counter)

    # 财神统计
    avg_caishen_in_winning_hand: float = 0.0
    caishen_distribution: Counter = field(default_factory=Counter)

    # 吃碰杠
    total_chi: int = 0
    total_peng: int = 0
    total_gang: int = 0
    chi_as_p0: int = 0
    chi_as_p1: int = 0

    # 点炮统计
    total_fang_pao: int = 0
    fang_pao_rate: float = 0.0

    # 得分
    avg_score: float = 0.0
    max_score: int = 0
    score_distribution: Counter = field(default_factory=Counter)

    # 流局相关
    liuju_rate: float = 0.0
    liuju_by_exhaustion: int = 0

    # 自摸率
    self_draw_rate: float = 0.0
    self_draw_as_winner: int = 0


class AIEvaluator:
    """AI评估器"""

    def __init__(self, model_path: Optional[str] = None, device: str = 'mps'):
        self.device = device
        self.env = WenzhouMahjongEnv()

        # 创建AI
        self.agent = AlphaZeroAgent(state_dim=316, action_dim=39, device=device)
        if model_path:
            self.agent.load(model_path)
            print(f"加载模型: {model_path}")

        self.mcts = AlphaZeroMCTS(self.agent.net, caishen_id=0, num_simulations=50)

        self.records: List[GameRecord] = []

    def evaluate(self, num_games: int = 1000, verbose: bool = True,
                 print_freq: int = 100) -> EvaluationReport:
        """运行评估"""
        print(f"\n{'='*60}")
        print(f"温州麻将AI评估 - {num_games}局")
        print(f"{'='*60}")

        start_time = time.time()

        for game_idx in range(num_games):
            record = self._play_one_game()
            self.records.append(record)

            if verbose and (game_idx + 1) % print_freq == 0:
                elapsed = time.time() - start_time
                speed = (game_idx + 1) / elapsed
                interim = self._calc_stats(game_idx + 1)
                print(f"\n[已评估 {game_idx+1} 局 | 速度: {speed:.1f}局/秒]")
                print(f"  闲家胜率: {interim.p0_win_rate:.1%}")
                print(f"  庄家胜率: {interim.p1_win_rate:.1%}")
                print(f"  流局率: {interim.draw_rate:.1%}")
                print(f"  平均番数: {interim.avg_win_fan:.2f}")

        report = self._calc_stats(num_games)
        report.total_games = num_games

        return report

    def _play_one_game(self) -> GameRecord:
        """玩一局并记录"""
        state = self.env.reset()

        record = GameRecord(
            winner=-1,
            is_self_draw=False,
            is_fang_pao=False,
            win_fan=0,
            win_type='unknown',
            caishen_count=0,
            total_score=0
        )

        current = 1  # 庄家先出
        last_player = None  # 上一个出牌者
        last_discarded = None  # 对手上一步打出的牌

        while True:
            state = self.env.get_state()
            hand = state['hand']
            legal_actions = state['legal_actions']
            legal_actions_full = self.env.get_legal_actions_full(current)

            # MCTS决策 - 传递last_discarded以便MCTS考虑吃碰杠
            game_state = GameState(hand, state['caishen'],
                                   wall_remaining=state['wall_remaining'],
                                   current_player=current,
                                   last_discarded=last_discarded)

            try:
                self.mcts.caishen_id = state['caishen']
                policy_dict = self.mcts.search(game_state, last_discarded)
            except:
                policy_dict = {a: 1.0 / len(legal_actions) for a in legal_actions}

            # 转换为39维策略 - 包括MCTS返回的特殊动作
            policy = np.zeros(39)
            for a, p in policy_dict.items():
                if a < 39:
                    policy[a] = p

            if policy[:34].sum() > 0:
                policy[:34] = policy[:34] / policy[:34].sum()
            else:
                policy[:34] = np.ones(34) / 34

            # 过滤合法动作 - 包括35-38的特殊动作
            mask = np.zeros(39)
            for a in legal_actions_full:
                if a < 39:
                    mask[a] = policy[a]

            if mask.sum() == 0:
                action = legal_actions[0] if legal_actions else 0
            else:
                probs = mask / mask.sum()
                action = np.random.choice(39, p=probs)

            # 记录吃碰杠
            if action == 35:  # 吃
                record.chi_count += 1
                if current == 0:
                    record.chi_as_p0 += 1
                else:
                    record.chi_as_p1 += 1
            elif action == 36:  # 碰
                record.peng_count += 1
            elif action == 37:  # 杠
                record.gang_count += 1

            # 更新last_discarded（只有出牌才更新）
            if action < 34:
                last_player = current
                last_discarded = action
            else:
                # 特殊动作后不产生新的last_discarded
                pass

            next_state, reward, done, info = self.env.step(action)

            if done:
                winner = info.get('winner', -1)
                record.winner = winner

                if winner >= 0:
                    # 正确获取赢家的完整手牌
                    win_hand = self.env.hands[winner].copy()

                    # 如果是点炮胡，赢家手牌要加上打出的牌
                    is_fang_pao = info.get('is_fang_pao', False)
                    if is_fang_pao and self.env.last_discarded is not None:
                        win_hand.append(self.env.last_discarded)

                    # 如果是自摸，赢家手牌已经完整（已经包含了摸的牌）
                    is_self_draw = info.get('is_self_draw', False) or (not is_fang_pao and winner == current)

                    from rules import WinChecker
                    checker = WinChecker(self.env.caishen_id)
                    result = checker.check_win(win_hand)

                    record.win_fan = result.get('fan', 1)
                    record.win_type = result.get('win_type', 'unknown')
                    record.caishen_count = self.env.caishen.count_caishen(win_hand)
                    record.total_score = int(abs(reward)) if reward != 0 else 0
                    record.is_self_draw = is_self_draw
                    record.is_fang_pao = is_fang_pao

                    # 点炮者的统计
                    if is_fang_pao and last_player is not None:
                        # 找到点炮的玩家
                        loser = last_player
                        # 在records里没法直接更新对方，改为记录在当前局
                        record.fang_pao_count = 1

                # 流局
                if winner == -1:
                    record.win_type = 'liuju'

                return record

            current = 1 - current

    def _calc_stats(self, num_games: int) -> EvaluationReport:
        """计算统计数据"""
        records = self.records[:num_games]

        report = EvaluationReport()

        # 基础统计
        report.p0_wins = sum(1 for r in records if r.winner == 0)
        report.p1_wins = sum(1 for r in records if r.winner == 1)
        report.draws = sum(1 for r in records if r.winner == -1)

        report.p0_win_rate = report.p0_wins / num_games
        report.p1_win_rate = report.p1_wins / num_games
        report.draw_rate = report.draws / num_games

        # 番数分布
        win_records = [r for r in records if r.winner >= 0]
        for r in win_records:
            report.fan_distribution[r.win_fan] += 1

        if win_records:
            report.avg_win_fan = sum(r.win_fan for r in win_records) / len(win_records)

        # 胡牌类型
        for r in win_records:
            report.hu_types[r.win_type] += 1

        # 特殊牌型
        for r in win_records:
            report.special_types[r.win_type] += 1

        # 财神统计 - 使用正确的赢手牌
        if win_records:
            total_caishen = sum(r.caishen_count for r in win_records)
            report.avg_caishen_in_winning_hand = total_caishen / len(win_records)

        for r in win_records:
            report.caishen_distribution[r.caishen_count] += 1

        # 吃碰杠
        report.total_chi = sum(r.chi_count for r in records)
        report.total_peng = sum(r.peng_count for r in records)
        report.total_gang = sum(r.gang_count for r in records)

        # 点炮统计
        report.total_fang_pao = sum(r.fang_pao_count for r in records)
        report.fang_pao_rate = report.total_fang_pao / num_games

        # 得分统计
        if win_records:
            scores = [r.total_score for r in win_records]
            report.avg_score = np.mean(scores)
            report.max_score = max(scores)
            for s in scores:
                bucket = (s // 10) * 10  # 按10分桶
                report.score_distribution[bucket] += 1

        # 流局
        report.liuju_rate = report.draws / num_games

        # 自摸率
        report.self_draw_as_winner = sum(1 for r in win_records if r.is_self_draw)
        if win_records:
            report.self_draw_rate = report.self_draw_as_winner / len(win_records)

        return report

    def print_report(self, report: EvaluationReport):
        """打印评估报告"""
        print(f"\n{'='*60}")
        print("评估报告")
        print(f"{'='*60}")

        # 基础统计
        print(f"\n📊 基础统计 (共{report.total_games}局)")
        print(f"   闲家胜率: {report.p0_win_rate:.1%} ({report.p0_wins}胜)")
        print(f"   庄家胜率: {report.p1_win_rate:.1%} ({report.p1_wins}胜)")
        print(f"   流局率:   {report.draw_rate:.1%} ({report.draws}局)")

        # 番数分布
        print(f"\n🎯 胡牌番数分布")
        total_hu = sum(report.fan_distribution.values())
        if total_hu > 0:
            for fan in sorted(report.fan_distribution.keys()):
                count = report.fan_distribution[fan]
                pct = count / total_hu * 100
                bar = '█' * int(pct / 2)
                print(f"   ×{fan:2d}番: {count:4d}局 ({pct:5.1f}%) {bar}")
        else:
            print("   无胡牌记录")
        print(f"   平均番数: {report.avg_win_fan:.2f}")

        # 牌型统计
        print(f"\n🀄 胡牌类型")
        total_types = sum(report.hu_types.values())
        if total_types > 0:
            for hu_type, count in report.hu_types.most_common(10):
                pct = count / total_types * 100
                print(f"   {hu_type:12s}: {count:4d}局 ({pct:5.1f}%)")
        else:
            print("   无胡牌记录")

        # 财神统计
        print(f"\n✨ 财神统计")
        print(f"   平均财神数: {report.avg_caishen_in_winning_hand:.2f}")
        total_caishen_records = sum(report.caishen_distribution.values())
        if total_caishen_records > 0:
            print(f"   财神分布:")
            for cs_count in sorted(report.caishen_distribution.keys()):
                count = report.caishen_distribution[cs_count]
                pct = count / total_caishen_records * 100
                print(f"      {cs_count}个财神: {count:4d}局 ({pct:5.1f}%)")
        else:
            print("   无胡牌记录")

        # 吃碰杠
        print(f"\n🔄 吃碰杠统计")
        total_ops = report.total_chi + report.total_peng + report.total_gang
        print(f"   总操作数: {total_ops}")
        if total_ops > 0:
            print(f"   吃牌: {report.total_chi}次 ({report.total_chi/total_ops*100:.1f}%)")
            print(f"   碰牌: {report.total_peng}次 ({report.total_peng/total_ops*100:.1f}%)")
            print(f"   杠牌: {report.total_gang}次 ({report.total_gang/total_ops*100:.1f}%)")
        else:
            print("   (AI未使用吃碰杠)")
        print(f"   闲家吃牌: {report.chi_as_p0}次 | 庄家吃牌: {report.chi_as_p1}次")

        # 点炮统计
        print(f"\n💨 点炮统计")
        print(f"   点炮率: {report.fang_pao_rate:.1%} ({report.total_fang_pao}次)")

        # 得分统计
        print(f"\n💰 得分统计")
        print(f"   平均得分: {report.avg_score:.1f}")
        print(f"   最高得分: {report.max_score}")
        total_scores = sum(report.score_distribution.values())
        if total_scores > 0:
            print(f"   得分分布:")
            for bucket in sorted(report.score_distribution.keys()):
                count = report.score_distribution[bucket]
                pct = count / total_scores * 100
                bar = '█' * int(pct / 2)
                print(f"      {bucket:4d}-{bucket+9:4d}: {count:4d}局 ({pct:5.1f}%) {bar}")

        # 自摸率
        print(f"\n🎲 自摸统计")
        print(f"   自摸率: {report.self_draw_rate:.1%}")
        print(f"   自摸胡牌: {report.self_draw_as_winner}局")

        print(f"\n{'='*60}")


class GameState:
    """简化游戏状态（用于MCTS）"""

    def __init__(self, hand, caishen_id, wall_remaining, current_player,
                 last_discarded=None, fulus=None, discards=None):
        self.hand = list(hand) if not isinstance(hand, list) else hand
        self.caishen_id = caishen_id
        self.wall_remaining = wall_remaining
        self.current_player = current_player
        self.last_discarded = last_discarded
        self.fulus = fulus if fulus else []
        self.discards = discards if discards else []
        self._opponent_discards = []  # 对手弃牌

    def get_legal_actions(self):
        legal = []
        for tile in set(self.hand):
            if tile != self.caishen_id and tile != 33:
                legal.append(tile)
        return legal if legal else list(set(self.hand))

    def get_legal_actions_with_special(self, opponent_discarded=None):
        """获取完整合法动作包括吃碰杠胡"""
        legal = self.get_legal_actions()

        # 碰 (36): 对手打出的牌，手里有对子
        if opponent_discarded is not None:
            if self.hand.count(opponent_discarded) >= 2:
                legal.append(36)

        # 杠 (37): 手里有三张 + 对手打出
        if opponent_discarded is not None:
            if self.hand.count(opponent_discarded) >= 3:
                legal.append(37)

        # 吃 (35): 只有闲家能吃上家，且只吃顺子
        # 简化：这里暂不展开吃的具体组合
        if opponent_discarded is not None and self.current_player == 0:
            # 检查是否能组成顺子
            if self._can_chi(opponent_discarded):
                legal.append(35)

        # 胡 (38): 检查是否能和牌 - 简化处理
        # 在实际MCTS中，这里应该检测是否听牌

        return legal

    def _can_chi(self, discarded_tile):
        """检查是否能吃（简化版）"""
        if self.current_player != 0:  # 只有闲家能吃
            return False

        # 吃需要是顺子
        tile_group = self._get_tile_group(discarded_tile)
        if tile_group >= 3:  # 字牌不能吃
            return False

        # 检查是否有相邻的牌可以组成顺子
        for offset in [-1, 0, 1]:
            t1 = discarded_tile + offset
            t2 = discarded_tile + offset + 1
            t3 = discarded_tile + offset + 2
            if t1 < 0 or t3 > 26:
                continue
            if self._get_tile_group(t1) != tile_group:
                continue
            if self.hand.count(t1) >= 1 and self.hand.count(t2) >= 1 and self.hand.count(t3) >= 1:
                return True
        return False

    def _get_tile_group(self, tile_id):
        """获取牌的花色组: 0=万, 1=筒, 2=索, 3=字"""
        if tile_id < 9:
            return 0
        elif tile_id < 18:
            return 1
        elif tile_id < 27:
            return 2
        else:
            return 3

    def do_action(self, action):
        """执行动作"""
        if action < 34 and action in self.hand:
            self.hand.remove(action)
            self.discards.append(action)
        self.wall_remaining -= 1
        self.current_player = 1 - self.current_player

    def do_peng(self, tile):
        """执行碰牌"""
        for _ in range(2):
            if tile in self.hand:
                self.hand.remove(tile)
        self.fulus.append(('pong', [tile, tile, tile]))

    def do_gang(self, tile):
        """执行杠牌"""
        for _ in range(3):
            if tile in self.hand:
                self.hand.remove(tile)
        self.fulus.append(('kong', [tile, tile, tile, tile]))

    def do_chi(self, chi_tiles):
        """执行吃牌"""
        for t in chi_tiles:
            if t in self.hand:
                self.hand.remove(t)
        self.fulus.append(('chow', list(chi_tiles)))

    def is_terminal(self):
        return self.wall_remaining <= 0

    def get_observation(self):
        obs = np.zeros(136, dtype=np.float32)
        for tile in self.hand:
            if 0 <= tile < 34:
                obs[tile] += 1
        for tile in self.discards[-20:]:
            if 0 <= tile < 34:
                obs[34 + tile] += 1
        if self.wall_remaining > 0:
            idx = min(33, max(0, 68 + (self.wall_remaining // 3)))
            obs[idx] = 1
        if 0 <= self.caishen_id < 34:
            obs[104 + min(self.caishen_id, 31)] = 1
        return obs


def main():
    import argparse
    parser = argparse.ArgumentParser(description='温州麻将AI评估工具')
    parser.add_argument('--games', type=int, default=1000, help='评估局数')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--mcts', type=int, default=50, help='MCTS模拟次数')
    parser.add_argument('--print_freq', type=int, default=100, help='打印频率')
    args = parser.parse_args()

    evaluator = AIEvaluator(model_path=args.model)
    evaluator.mcts.num_simulations = args.mcts

    report = evaluator.evaluate(
        num_games=args.games,
        verbose=True,
        print_freq=args.print_freq
    )

    evaluator.print_report(report)


if __name__ == '__main__':
    main()
