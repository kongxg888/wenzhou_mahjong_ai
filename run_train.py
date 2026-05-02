#!/usr/bin/env python3
import time
import sys
from train.alpha_train import AlphaZeroTrainer
from ai.models.alpha_zero import AlphaZeroMCTS

trainer = AlphaZeroTrainer()
mcts = AlphaZeroMCTS(trainer.agent.net, caishen_id=0, num_simulations=20)

start = time.time()
for i in range(50):
    winner, trajectories = trainer._play_one_game(mcts)
    trainer.memory.extend(trajectories)

    if (i+1) % 10 == 0:
        elapsed = time.time() - start
        speed = (i+1) / elapsed
        print(f'{i+1}局 | {speed:.1f}局/秒')

    if len(trainer.memory) >= 32:
        batch = list(trainer.memory)[-32:]
        loss = trainer.agent.train_step(batch, 32)

print(f'完成! 50局/{time.time()-start:.1f}秒')
