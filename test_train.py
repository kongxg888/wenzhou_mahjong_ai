#!/usr/bin/env python3
import time
import sys

print("开始训练...", flush=True)

from train.alpha_train import AlphaZeroTrainer
from ai.models.alpha_zero import AlphaZeroMCTS

print("创建 trainer...", flush=True)
trainer = AlphaZeroTrainer()

print("创建 MCTS...", flush=True)
mcts = AlphaZeroMCTS(trainer.agent.net, caishen_id=0, num_simulations=20)

print("开始训练 50 局...", flush=True)
start = time.time()

win_stats = {'p0': 0, 'p1': 0, 'draw': 0}

for game in range(50):
    print(f"第{game+1}局开始...", flush=True)
    winner, trajectories = trainer._play_one_game(mcts)
    print(f"第{game+1}局完成: winner={winner}, 轨迹={len(trajectories)}", flush=True)

    trainer.memory.extend(trajectories)

    if winner == 0:
        win_stats['p0'] += 1
    elif winner == 1:
        win_stats['p1'] += 1
    else:
        win_stats['draw'] += 1

    if (game + 1) % 10 == 0:
        elapsed = time.time() - start
        speed = (game + 1) / elapsed if elapsed > 0 else 0
        wr = win_stats['p0'] / (game + 1) * 100
        print(f"进度: {game+1}/50 | 速度:{speed:.0f}局/秒 | 胜率:{wr:.1f}%", flush=True)

    # 训练
    if (game + 1) % 1 == 0 and len(trainer.memory) >= 32:
        batch = list(trainer.memory)[-32:]
        loss = trainer.agent.train_step(batch, 32)

elapsed = time.time() - start
print(f"\n训练完成!", flush=True)
print(f"50局总耗时: {elapsed:.1f}秒", flush=True)
print(f"速度: {50/elapsed:.0f}局/秒", flush=True)
print(f"胜率: 闲{win_stats['p0']/50*100:.0f}% 庄{win_stats['p1']/50*100:.0f}% 流{win_stats['draw']/50*100:.0f}%", flush=True)
