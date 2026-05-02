# 温州麻将 2人 AI

基于AlphaZero强化学习的温州麻将AI，使用MCTS + 神经网络自对弈训练。

## 项目结构

```
wenzhou_mahjong_ai/
├── SPEC.md                # 完整规则文档
├── rules/                 # 游戏规则引擎
│   ├── __init__.py       # 动作常量定义
│   ├── tiles.py          # 牌定义
│   ├── caishen.py        # 财神引擎
│   ├── hand.py           # 手牌管理
│   ├── win.py            # 胡牌判定
│   └── score.py          # 计分
├── env/                   # RLCard环境
│   └── mahjong_env.py    # 自定义环境
├── ai/models/
│   ├── alpha_zero.py     # AlphaZero网络 + MCTS
│   └── ppo.py            # PPO算法（备用）
├── train/
│   └── alpha_train.py    # AlphaZero自对弈训练
├── game_state.py         # MCTS用简化游戏状态
├── test_self_play.py     # 自对弈测试
└── play.py               # 人机对战
```

## 安装依赖

```bash
pip install torch numpy
```

## 快速开始

### 训练AI

```bash
python3 train/alpha_train.py --games 10000 --mcts 50
```

### 自对弈测试

```bash
python3 test_self_play.py
```

## 核心规则闭环

### 动作流程

| 玩家A出牌后 | 玩家B动作 | 后续处理 |
|------------|----------|----------|
| 吃 | 直接出牌 | 不摸牌 |
| 碰 | 直接出牌 | 不摸牌 |
| 明杠 | 补牌→检查杠上开花→出牌 | 补1张 |
| 加杠 | 出牌 | 摸牌后直接出 |
| 暗杠 | 补牌→检查杠上开花→出牌 | 补1张 |
| 过 | 摸牌→出牌 | - |
| 胡 | 终局 | - |

### 杠分
- 明杠: +1分
- 加杠: +1分  
- 暗杠: +2分

### 荒庄
牌墙剩余≤8张时禁止吃/碰/杠，只能出牌/过/胡。

## 算法说明

### AlphaZero + MCTS
- 神经网络评估状态
- MCTS搜索最优动作
- 自对弈收集经验训练
- Mac M4 MPS GPU加速
