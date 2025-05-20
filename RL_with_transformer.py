import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformerModel2 import TrajectoryTransformer2
import math
from torch.nn import TransformerEncoderLayer

class TransformerActor:
    """
    使用预训练的TrajectoryTransformer2作为Actor网络的包装类
    """
    def __init__(self, transformer_model):
        self.transformer_model = transformer_model
        self.device = next(transformer_model.parameters()).device
        
    def act(self, state, deterministic=False):
        """
        根据当前状态生成动作
        
        Args:
            state: 环境状态
            deterministic: 是否使用确定性策略
            
        Returns:
            选择的动作
        """
        # 解析状态并确保它们在正确的设备上
        joint_angles = torch.FloatTensor(state[:7]).unsqueeze(0).to(self.device)
        joint_positions = torch.FloatTensor(state[7:-3]).reshape(1, 7, 3).to(self.device)
        target_position = torch.FloatTensor(state[-3:]).unsqueeze(0).to(self.device)
        
        # 使用Transformer模型生成动作序列
        with torch.no_grad():
            predicted_trajectory = self.transformer_model(
                joint_angles,
                joint_positions,
                target_position,
                target_trajectory=None  # 推理模式
            )
            
            # 只取第一个动作
            action = predicted_trajectory[:, 0, :].cpu().numpy().flatten()
            
        return action
    
    def save(self, path):
        """保存模型"""
        torch.save(self.transformer_model.state_dict(), path)
    
    def load(self, path):
        """加载模型"""
        self.transformer_model.load_state_dict(torch.load(path))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=30):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class TransformerCritic(nn.Module):
    """
    基于Transformer架构的Critic网络，用于评估状态-动作对的价值
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerCritic, self).__init__()
        
        input_dim = state_dim + action_dim
        
        # Q1网络
        self.q1_embedding = nn.Linear(input_dim, hidden_dim)
        self.q1_pos_encoder = PositionalEncoding(hidden_dim)
        self.q1_transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
            for _ in range(num_layers)
        ])
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        # Q2网络
        self.q2_embedding = nn.Linear(input_dim, hidden_dim)
        self.q2_pos_encoder = PositionalEncoding(hidden_dim)
        self.q2_transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
            for _ in range(num_layers)
        ])
        self.q2_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        """
        计算状态-动作对的Q值
        
        Args:
            state: 环境状态
            action: 执行的动作
            
        Returns:
            两个Q网络的输出值
        """
        # 合并状态和动作
        x = torch.cat([state, action], dim=1)
        
        # 添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, state_dim+action_dim]
        
        # Q1处理
        q1 = self.q1_embedding(x)
        q1 = self.q1_pos_encoder(q1)
        for layer in self.q1_transformer_layers:
            q1 = layer(q1)
        q1 = q1[:, -1]  # 取最后一个位置的输出
        q1 = self.q1_out(q1)
        
        # Q2处理
        q2 = self.q2_embedding(x)
        q2 = self.q2_pos_encoder(q2)
        for layer in self.q2_transformer_layers:
            q2 = layer(q2)
        q2 = q2[:, -1]  # 取最后一个位置的输出
        q2 = self.q2_out(q2)
        
        return q1, q2


class ReplayBuffer:
    """
    经验回放缓冲区
    """
    def __init__(self, state_dim, action_dim, capacity):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.reward = np.zeros((capacity, 1))
        self.next_state = np.zeros((capacity, state_dim))
        self.done = np.zeros((capacity, 1))
        
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """采样经验批次"""
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind]
        )


def train_rl_with_transformer(env, transformer_path, state_dim, action_dim, hidden_dim=256, 
                             buffer_size=1000000, batch_size=256, gamma=0.99, tau=0.005, 
                             alpha=0.2, lr=3e-4, start_steps=10000, update_after=1000, 
                             update_every=50, num_updates=1, max_ep_len=1000, 
                             num_episodes=1000, save_freq=100, finetune_transformer=False,
                             render_freq=50):  # 添加渲染频率参数
    """
    使用预训练的Transformer模型作为Actor进行强化学习训练
    
    Args:
        env: 环境
        transformer_path: 预训练Transformer模型路径
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        buffer_size: 经验回放缓冲区大小
        batch_size: 批次大小
        gamma: 折扣因子
        tau: 目标网络软更新系数
        alpha: 熵正则化系数
        lr: 学习率
        start_steps: 随机动作的步数
        update_after: 开始更新的步数
        update_every: 更新频率
        num_updates: 每次更新的次数
        max_ep_len: 最大回合长度
        num_episodes: 训练回合数
        save_freq: 保存模型的频率
        finetune_transformer: 是否微调Transformer模型
        render_freq: 渲染频率，每隔多少个episode渲染一次
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载预训练的Transformer模型
    transformer_model = TrajectoryTransformer2()
    transformer_model.load_state_dict(torch.load(transformer_path))
    transformer_model.to(device)  # 确保模型在正确的设备上
    
    # 创建Actor（使用预训练的Transformer）
    actor = TransformerActor(transformer_model)
    
    # 创建Critic网络（从头开始训练）
    critic = TransformerCritic(state_dim, action_dim, hidden_dim)
    
    # 创建目标Critic网络
    target_critic = TransformerCritic(state_dim, action_dim, hidden_dim)
    target_critic.load_state_dict(critic.state_dict())
    
    # 冻结目标网络的参数
    for p in target_critic.parameters():
        p.requires_grad = False
    
    # 设置优化器
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    
    # 如果需要微调Transformer模型，则为其创建优化器
    if finetune_transformer:
        actor_optimizer = optim.Adam(transformer_model.parameters(), lr=lr*0.1)  # 使用较小的学习率
    
    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
    
    # 将网络移动到设备上
    critic.to(device)
    target_critic.to(device)
    
    # 导入PyBullet用于渲染
    import pybullet as p
    import time
    import cv2
    import os
    
    # 设置PyBullet渲染参数
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # 启用GUI，便于调试
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # 默认开启渲染
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # 关闭阴影以提高性能
    
    # 添加目标位置可视化 - 使用固定颜色，避免闪烁
    target_visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.8])
    
    # 创建轨迹标记列表
    if not hasattr(env, 'trajectory_markers'):
        env.trajectory_markers = []
    
    # 确保目标对象ID属性存在
    if not hasattr(env, 'target_object_id'):
        env.target_object_id = None
    
    # 设置相机位置 - 在循环外设置一次固定相机位置
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=60,
        cameraPitch=-30,
        cameraTargetPosition=[0.2, 0.0, 0.3]
    )
    
    # 训练循环
    total_steps = 0
    rewards = []
    
    for episode in tqdm(range(num_episodes)):
        episode_reward = 0
        state = env.reset()
        done = False
        episode_steps = 0
        
        # 判断是否需要渲染当前episode
        render_episode = (episode + 1) % render_freq == 0
        
        if render_episode:
            # 更新目标位置可视化
            target_position = env.target_position
            
            # 安全删除之前的目标对象
            if env.target_object_id is not None:
                try:
                    p.removeBody(env.target_object_id)
                except:
                    pass  # 忽略错误
            
            # 创建新的目标对象
            env.target_object_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=target_visual_id,
                basePosition=target_position
            )
            
            # 清除之前的轨迹标记
            if hasattr(env, 'trajectory_markers') and env.trajectory_markers:
                # 使用集合去重，避免重复删除
                unique_ids = set()
                for marker_id in env.trajectory_markers:
                    if isinstance(marker_id, int) and marker_id not in unique_ids:
                        unique_ids.add(marker_id)
                        try:
                            # 尝试作为调试项删除
                            p.removeUserDebugItem(marker_id)
                        except:
                            try:
                                # 尝试作为物体删除
                                p.removeBody(marker_id)
                            except:
                                pass  # 忽略错误
                
                # 清空标记列表
                env.trajectory_markers = []
            
            # 记录初始末端执行器位置
            env.last_end_effector_pos = env._get_end_effector_position()
            
            print(f"Episode {episode+1}: 启用特殊渲染...")
            
            # 添加轨迹起点标记
            visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 1.0])
            marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_id,
                basePosition=env.last_end_effector_pos
            )
            env.trajectory_markers.append(marker_id)
            
            # 添加目标点连接线
            try:
                line_id = p.addUserDebugLine(
                    env.last_end_effector_pos,
                    target_position,
                    lineColorRGB=[1, 0.5, 0],
                    lineWidth=1.0,
                    lifeTime=0
                )
                env.trajectory_markers.append(line_id)
            except:
                pass
            
            # 记录当前episode的轨迹点
            trajectory_points = []
            trajectory_points.append(env._get_end_effector_position())
            
            # 确保机械臂可见
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            
            print(f"初始状态 - 目标位置: {target_position}")
            print(f"初始状态 - 末端执行器位置: {env.last_end_effector_pos}")
            print("按Enter键开始执行动作...")
            input()
        
        while not done and episode_steps < max_ep_len:
            # 在开始阶段使用随机动作进行探索
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                # 使用Actor（Transformer）选择动作
                action = actor.act(state)
            
            # 确保动作在合理范围内
            action = np.clip(action, -0.5, 0.5)
            
            # 执行动作前打印信息
            if render_episode:
                print(f"Step {episode_steps+1} - 执行动作: {action}")
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # 渲染处理
            if render_episode:
                # 获取当前末端执行器位置
                current_end_effector_pos = env._get_end_effector_position()
                trajectory_points.append(current_end_effector_pos)
                
                # 计算移动距离
                move_distance = np.linalg.norm(np.array(current_end_effector_pos) - np.array(env.last_end_effector_pos))
                
                # 只有当移动距离超过阈值时才添加轨迹线条
                if move_distance > 0.001:  # 1mm的阈值
                    try:
                        line_id = p.addUserDebugLine(
                            env.last_end_effector_pos,
                            current_end_effector_pos,
                            lineColorRGB=[0, 1, 0],
                            lineWidth=2.0,
                            lifeTime=0  # 永久显示
                        )
                        env.trajectory_markers.append(line_id)
                    except Exception as e:
                        print(f"添加轨迹线条失败: {e}")
                    
                    # 在末端执行器位置添加小球标记
                    if episode_steps % 3 == 0:  # 每3步添加一个标记点，避免过多标记
                        try:
                            visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=[0, 1, 0, 0.7])
                            marker_id = p.createMultiBody(
                                baseMass=0,
                                baseVisualShapeIndex=visual_id,
                                basePosition=current_end_effector_pos
                            )
                            env.trajectory_markers.append(marker_id)
                        except Exception as e:
                            print(f"添加轨迹点标记失败: {e}")
                
                # 显示当前距离信息
                distance = np.linalg.norm(np.array(current_end_effector_pos) - np.array(env.target_position))
                try:
                    # 使用固定位置的文本，避免闪烁
                    text_id = p.addUserDebugText(
                        f"距离目标: {distance:.4f} | 步数: {episode_steps+1}/{max_ep_len}",
                        [0.0, 0.0, 0.8],  # 在机械臂上方显示
                        textColorRGB=[1, 1, 1],
                        textSize=1.5,
                        lifeTime=0.1
                    )
                except:
                    pass
                
                # 打印调试信息
                print(f"Step {episode_steps+1} - 末端位置: {current_end_effector_pos}, 移动距离: {move_distance:.6f}, 到目标距离: {distance:.4f}")
                
                # 添加短暂延迟以便观察
                time.sleep(0.2)
                
                # 更新上一个位置
                env.last_end_effector_pos = current_end_effector_pos
                
                # 如果渲染步数达到1000，提前结束渲染
                if episode_steps >= 1000:
                    print("渲染步数达到1000步，提前结束渲染...")
                    break
            
            # 存储经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            total_steps += 1
            episode_steps += 1
            
            # 更新网络
            if total_steps > update_after and total_steps % update_every == 0:
                for _ in range(num_updates):
                    batch = replay_buffer.sample(batch_size)
                    update_networks(batch, actor, critic, target_critic, 
                                   critic_optimizer, actor_optimizer if finetune_transformer else None, 
                                   alpha, gamma, tau, device, finetune_transformer)
            
            # 如果episode结束且正在渲染，添加终点标记
            if done and render_episode:
                try:
                    visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1.0])
                    marker_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=visual_id,
                        basePosition=current_end_effector_pos
                    )
                    env.trajectory_markers.append(marker_id)
                    print(f"Episode结束！最终距离: {distance:.4f}")
                except Exception as e:
                    print(f"添加终点标记失败: {e}")
        
        rewards.append(episode_reward)
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards[-10:]) / 10
            print(f"回合 {episode+1}, 平均奖励: {avg_reward:.2f}")
        
        # 如果是渲染episode，暂停一下让用户观察最终状态
        if render_episode:
            # 保存轨迹截图
            try:
                screenshot_dir = os.path.join(os.path.dirname(transformer_path), "trajectory_screenshots")
                os.makedirs(screenshot_dir, exist_ok=True)
                screenshot_path = os.path.join(screenshot_dir, f"trajectory_episode_{episode+1}.png")
                p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
                img_arr = p.getCameraImage(1280, 720)[2]
                cv2.imwrite(screenshot_path, img_arr)
                print(f"轨迹截图已保存至 {screenshot_path}")
            except Exception as e:
                print(f"保存截图失败: {e}")
            
            print("轨迹渲染完成，按Enter键继续下一个episode...")
            input()
        
        # 保存模型
        if (episode + 1) % save_freq == 0:
            actor.save(f"transformer_actor_ep{episode+1}.pth")
            torch.save(critic.state_dict(), f"critic_ep{episode+1}.pth")
    
    # 保存最终模型
    actor.save("transformer_actor_final.pth")
    torch.save(critic.state_dict(), "critic_final.pth")
    
    return actor, critic, rewards



def update_networks(batch, actor, critic, target_critic, critic_optimizer, actor_optimizer=None, 
                   alpha=0.2, gamma=0.99, tau=0.005, device=None, finetune_transformer=False):
    """
    更新网络
    
    Args:
        batch: 经验批次
        actor: Actor网络
        critic: Critic网络
        target_critic: 目标Critic网络
        critic_optimizer: Critic优化器
        actor_optimizer: Actor优化器（如果需要微调Transformer）
        alpha: 熵正则化系数
        gamma: 折扣因子
        tau: 目标网络软更新系数
        device: 计算设备
        finetune_transformer: 是否微调Transformer模型
    """
    # 解包批次数据
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    
    # 将数据转移到设备
    state_batch = torch.FloatTensor(state_batch).to(device)
    action_batch = torch.FloatTensor(action_batch).to(device)
    reward_batch = torch.FloatTensor(reward_batch).to(device)
    next_state_batch = torch.FloatTensor(next_state_batch).to(device)
    done_batch = torch.FloatTensor(done_batch).to(device)
    
    # 计算目标Q值
    with torch.no_grad():
        # 使用Actor获取下一个动作
        next_action = torch.FloatTensor(np.array([actor.act(next_state) for next_state in next_state_batch.cpu().numpy()])).to(device)
        
        # 计算目标Q值
        target_q1, target_q2 = target_critic(next_state_batch, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward_batch + gamma * (1 - done_batch) * target_q
    
    # 当前Q值
    current_q1, current_q2 = critic(state_batch, action_batch)
    
    # 计算Critic损失
    critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
    
    # 更新Critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    # 如果需要微调Transformer模型
    if finetune_transformer and actor_optimizer is not None:
        # 获取当前策略的动作
        current_actions = []
        
        # 批量处理状态，避免逐个处理
        batch_size = state_batch.size(0)
        joint_angles = state_batch[:, :7].to(device)
        joint_positions = state_batch[:, 7:-3].reshape(batch_size, 7, 3).to(device)
        target_position = state_batch[:, -3:].to(device)
        
        # 使用Transformer生成动作
        predicted_trajectory = actor.transformer_model(
            joint_angles,
            joint_positions,
            target_position,
            target_trajectory=None
        )
        
        # 只取第一个动作
        current_actions = predicted_trajectory[:, 0, :]
        
        # 计算Q值
        q1, q2 = critic(state_batch, current_actions)
        q = torch.min(q1, q2)
        
        # 计算Actor损失（最大化Q值）
        actor_loss = -q.mean()
        
        # 更新Actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
    
    # 软更新目标网络
    for param, target_param in zip(critic.parameters(), target_critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
