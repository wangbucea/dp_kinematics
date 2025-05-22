import os
import gym
import numpy as np
from robotEnv import RoboticArmEnv
from RL_with_mlp import train_rl_with_mlp

# 创建环境
env = RoboticArmEnv()

# 设置参数
state_dim = 31
action_dim = 7
mlp_path = r"C:\DiskD\trae_doc\robot_gym\model_results\best_sequence_mlp_model.pth"  # 预训练MLP模型路径

# 训练
actor, critic, rewards = train_rl_with_mlp(
    env, 
    mlp_path, 
    state_dim, action_dim, 
    hidden_dim=256, 
    buffer_size=1000000, 
    batch_size=256, 
    gamma=0.99, 
    tau=0.005,
    alpha=0.2, 
    lr=3e-4, 
    start_steps=0, 
    update_after=50,
    update_every=50, 
    num_updates=1, 
    max_ep_len=5000,
    num_episodes=1000, 
    save_freq=100, finetune_actor=True,
    render_freq=1000
)

# 保存训练结果
np.save("mlp_rl_rewards.npy", rewards)
print("训练完成！")
