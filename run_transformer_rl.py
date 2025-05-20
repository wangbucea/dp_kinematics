import torch
import numpy as np
import matplotlib.pyplot as plt
from robotEnv import RoboticArmEnv
from RL_with_transformer import train_rl_with_transformer

# 创建环境
env = RoboticArmEnv()

# 设置参数
state_dim = 31  # 7(关节角度) + 21(关节位置) + 3(目标位置)
action_dim = 7  # 7个关节的动作

# 预训练模型路径
pretrained_model_path = "c:/DiskD/trae_doc/robot_gym/transformer_model.pth"

# 使用预训练的Transformer模型进行强化学习训练
actor, critic, rewards = train_rl_with_transformer(
    env=env,
    transformer_path=pretrained_model_path,
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=128,  # 与预训练模型保持一致
    buffer_size=1000000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    lr=1e-4,         # 较小的学习率
    start_steps=5000, # 减少随机探索步数
    update_after=1000,
    update_every=50,
    num_updates=1,
    max_ep_len=1000,
    num_episodes=500,
    save_freq=50,
    finetune_transformer=True,  # 设置为True可以微调Transformer模型
    render_freq=1
)

# 绘制训练曲线
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.xlabel('回合')
plt.ylabel('奖励')
plt.title('使用Transformer作为Actor的强化学习训练进度')
plt.savefig('transformer_rl_training_progress.png')
plt.show()
