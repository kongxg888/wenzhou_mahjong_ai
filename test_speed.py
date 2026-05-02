#!/usr/bin/env python3
import time
from train.alpha_train import AlphaZeroTrainer
from ai.models.alpha_zero import AlphaZeroMCTS

trainer = AlphaZeroTrainer()
mcts = AlphaZeroMCTS(trainer.agent.net, caishen_id=0, num_simulations=20)

start = time.time()
times = []

for i in range(20):
    game_start = time.time()
    winner, trajectories = trainer._play_one_game(mcts)
    game_time = time.time() - game_start
    times.append(game_time)

    elapsed = time.time() - start
    print(f'Game {i+1}: {game_time:.3f}s (total: {elapsed:.1f}s)')

avg = sum(times) / len(times)
print(f'\nAverage: {avg:.3f}s/game = {1/avg:.1f} games/s')
print(f'Total time: {time.time()-start:.1f}s for 20 games')
