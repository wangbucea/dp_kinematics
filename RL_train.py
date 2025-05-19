import torch.optim as optim
from torch.distributions import Normal
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
from transformerModel import *
# 添加位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 多头自注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # 重塑并连接
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(output)

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# 基于Transformer的Actor网络
class TransformerActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1, log_std_min=-20, log_std_max=2):
        super(TransformerActor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 输入嵌入
        self.embedding = nn.Linear(state_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, joint_angles, joint_positions, target_position, max_len=1):
        # 合并输入特征
        batch_size = joint_angles.size(0)
        joint_positions_flat = joint_positions.reshape(batch_size, -1)
        state = torch.cat([joint_angles, joint_positions_flat, target_position], dim=1)
        
        # 添加序列维度
        x = state.unsqueeze(1)  # [batch_size, 1, state_dim]
            
        # 嵌入和位置编码
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
        x = self.pos_encoder(x)
        
        # Transformer编码器层
        for layer in self.transformer_layers:
            x = layer(x)
            
        # 取最后一个位置的输出
        x = x[:, -1]  # [batch_size, hidden_dim]
        
        # 输出动作分布参数
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        # 检查输入类型
        if isinstance(state, tuple) and len(state) == 3:
            # 如果输入是分开的特征，直接使用
            joint_angles, joint_positions, target_position = state
            mean, log_std = self.forward(joint_angles, joint_positions, target_position)
        else:
            # 如果输入是批量状态数据，需要分解
            batch_size = state.shape[0]
            
            # 假设状态向量的格式是：[joint_angles(7), joint_positions(21), target_position(3)]
            joint_angles = state[:, :7]
            joint_positions = state[:, 7:-3].reshape(batch_size, 7, 3)
            target_position = state[:, -3:]
            
            mean, log_std = self.forward(joint_angles, joint_positions, target_position)
        
        std = log_std.exp()
        normal = Normal(mean, std)
        
        x_t = normal.rsample()  # 使用重参数化技巧
        action = torch.tanh(x_t)
        
        # 计算对数概率
        log_prob = normal.log_prob(x_t)
        
        # 修正因tanh变换导致的对数概率变化
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)

# 基于Transformer的Critic网络
class TransformerCritic(nn.Module):
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

# 修改 SAC 类以使用预训练的 TrajectoryTransformer
class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = TransformerActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_target = TransformerActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
            
        # Critic 网络保持不变
        self.critic = TransformerCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TransformerCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state, evaluate=False):
        # 将状态分解为 TrajectoryTransformer 所需的输入格式
        joint_angles = torch.FloatTensor(state[:7]).unsqueeze(0).to(self.device)
        joint_positions = torch.FloatTensor(state[7:-3]).unsqueeze(0).reshape(1, 7, 3).to(self.device)
        target_position = torch.FloatTensor(state[-3:]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 使用 TrajectoryTransformer 预测轨迹，取第一个动作
            mean, _ = self.actor(joint_angles, joint_positions, target_position, max_len=1)
            action = torch.tanh(mean).cpu().numpy().flatten()
            
        return action
    
    def update_parameters(self, memory, batch_size=256):
        # 采样一批数据
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)
        
        # 转换为张量
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)  # 形状应为 [batch_size, 1]
        done_batch = torch.FloatTensor(done_batch).to(self.device)  # 形状应为 [batch_size, 1]
        
        # 确保形状正确
        if reward_batch.dim() == 1:
            reward_batch = reward_batch.unsqueeze(1)
        if done_batch.dim() == 1:
            done_batch = done_batch.unsqueeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target
        
        # 当前Q值
        qf1, qf2 = self.critic(state_batch, action_batch)
        
        # 确保形状匹配
        if qf1.shape != next_q_value.shape:
            print(f"形状不匹配: qf1.shape={qf1.shape}, next_q_value.shape={next_q_value.shape}")
            # 尝试调整形状
            if qf1.dim() > next_q_value.dim():
                next_q_value = next_q_value.expand_as(qf1)
            elif qf1.dim() < next_q_value.dim():
                qf1 = qf1.unsqueeze(-1)
                qf2 = qf2.unsqueeze(-1)
        
        # 计算critic损失
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        # 更新critic
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()
        
        # 更新actor
        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        # Actor损失 = α * log(π(a|s)) - Q(s,a)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # 更新alpha（温度参数）
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # 软更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        return qf_loss.item(), policy_loss.item()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []
        
        for experience in batch:
            state, action, reward, next_state, done = experience
            
            # 确保状态和下一个状态不为None
            if state is not None and next_state is not None:
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)
                next_state_list.append(next_state)
                done_list.append(done)
        
        # 检查采样的经验数量是否足够
        if len(state_list) < batch_size // 2:
            # 如果有效样本太少，递归调用直到获取足够样本
            return self.sample(batch_size)
            
        # 确保所有状态和下一个状态具有相同的形状
        # 找出最常见的状态形状
        if state_list and next_state_list:
            # 获取第一个状态的形状作为参考
            ref_shape = np.array(state_list[0]).shape
            
            # 过滤掉形状不一致的样本
            valid_indices = []
            for i in range(len(state_list)):
                if (np.array(state_list[i]).shape == ref_shape and 
                    np.array(next_state_list[i]).shape == ref_shape):
                    valid_indices.append(i)
            
            # 只保留形状一致的样本
            state_list = [state_list[i] for i in valid_indices]
            action_list = [action_list[i] for i in valid_indices]
            reward_list = [reward_list[i] for i in valid_indices]
            next_state_list = [next_state_list[i] for i in valid_indices]
            done_list = [done_list[i] for i in valid_indices]
            
            # 如果过滤后样本太少，递归调用
            if len(state_list) < batch_size // 2:
                return self.sample(batch_size)
        
        # 转换为numpy数组
        try:
            state_batch = np.array(state_list)
            action_batch = np.array(action_list)
            reward_batch = np.array(reward_list).reshape(-1, 1)
            next_state_batch = np.array(next_state_list)
            done_batch = np.array(done_list).reshape(-1, 1)
            
            return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        except ValueError as e:
            # 如果仍然出现值错误，打印详细信息并重新采样
            print(f"采样错误: {e}")
            print(f"状态形状: {[np.array(s).shape for s in state_list]}")
            print(f"下一状态形状: {[np.array(s).shape for s in next_state_list]}")
            return self.sample(batch_size)
    
    def __len__(self):
        return len(self.buffer)


def train_sac(env, state_dim, action_dim, hidden_dim=256, num_episodes=1000, 
              max_steps=1000, batch_size=256, gamma=0.99, tau=0.005, 
              alpha=0.2, lr=3e-4, updates_per_step=1, start_steps=10000,
              replay_size=1000000, eval_interval=10):
    """
    训练SAC算法
    
    Args:
        env: 环境
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        num_episodes: 训练的总回合数
        max_steps: 每个回合的最大步数
        batch_size: 批次大小
        gamma: 折扣因子
        tau: 目标网络软更新系数
        alpha: 初始温度参数
        lr: 学习率
        updates_per_step: 每步更新次数
        start_steps: 开始使用策略前的随机步数
        replay_size: 经验回放缓冲区大小
        eval_interval: 评估间隔
    
    Returns:
        训练好的SAC智能体
    """
    # 初始化SAC智能体
    agent = SAC(state_dim, action_dim, hidden_dim, gamma, tau, alpha, lr)
    
    # 初始化经验回放缓冲区
    memory = ReplayBuffer(replay_size)
    
    # 训练统计
    total_steps = 0
    episode_rewards = []
    
    for episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 在开始阶段使用随机动作进行探索
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            memory.push(state, action, reward, next_state, done)
            
            # 更新状态和统计信息
            state = next_state
            episode_reward += reward
            step += 1
            total_steps += 1
            
            # 当收集了足够的样本后开始更新
            if len(memory) > batch_size:
                for _ in range(updates_per_step):
                    critic_loss, actor_loss = agent.update_parameters(memory, batch_size)
            
        # 记录回合奖励
        episode_rewards.append(episode_reward)
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
        
        # 定期评估和保存模型
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_policy(agent, env)
            print(f"Evaluation at episode {episode+1}: {eval_reward:.2f}")
            
            # 保存模型
            torch.save(agent.actor.state_dict(), 'sac_actor.pth')
            torch.save(agent.critic.state_dict(), 'sac_critic.pth')
    
    return agent

def evaluate_policy(agent, env, eval_episodes=10):
    """
    评估策略
    
    Args:
        agent: SAC智能体
        env: 环境
        eval_episodes: 评估回合数
    
    Returns:
        平均奖励
    """
    avg_reward = 0.
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            avg_reward += reward
            state = next_state
    
    avg_reward /= eval_episodes
    return avg_reward
