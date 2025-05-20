import torch
from torch.cuda import device
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm

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

class TrajectoryTransformer2(nn.Module):
    def __init__(self, input_dim=31, hidden_dim=128, num_layers=2, nhead=4, max_seq_len=30, action_dim=7):
        super(TrajectoryTransformer2, self).__init__()
        
        self.action_dim = action_dim
        # 输入处理
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)  # 专门用于嵌入动作的层
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Transformer解码器
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, action_dim)  # 7个关节的动作
        
        # 用于自回归生成的起始token
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.max_seq_len = max_seq_len
    
    def _prepare_inputs(self, joint_angles, joint_positions, target_position):
        """准备编码器的输入"""
        batch_size = joint_angles.size(0)
        
        # 合并输入特征
        joint_positions_flat = joint_positions.reshape(batch_size, -1)
        inputs = torch.cat([joint_angles, joint_positions_flat, target_position], dim=1)
        
        # 编码输入
        encoded_input = self.input_embed(inputs).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 使用Transformer编码器处理输入
        memory = self.transformer_encoder(encoded_input)  # [batch_size, 1, hidden_dim]
        
        return memory
    
    def forward(self, joint_angles, joint_positions, target_position, target_trajectory=None):
        """
        自回归前向传播
        Args:
            joint_angles: 初始关节角度 [batch_size, 7]
            joint_positions: 初始关节位置 [batch_size, 7, 3]
            target_position: 目标位置 [batch_size, 3]
            target_trajectory: 用于训练时的目标轨迹 [batch_size, seq_len, 7]，如果为None则为推理模式
        """
        batch_size = joint_angles.size(0)
        device = joint_angles.device
        
        # 准备编码器记忆
        memory = self._prepare_inputs(joint_angles, joint_positions, target_position)
        
        # 初始化解码器输入（起始token）
        start_tokens = self.start_token.repeat(batch_size, 1, 1)  # [batch_size, 1, hidden_dim]
        decoder_input = start_tokens
        
        outputs = []
        
        # 训练模式（使用教师强制）
        if target_trajectory is not None and self.training:
            # 将目标轨迹嵌入
            target_embedded = self.action_embed(target_trajectory)  # [batch_size, seq_len, hidden_dim]
            
            # 添加位置编码
            decoder_input_full = torch.cat([start_tokens, target_embedded[:, :-1]], dim=1)  # 不包括最后一个时间步
            decoder_input_full = self.pos_encoder(decoder_input_full)
            
            # 创建注意力掩码（确保每个时间步只能看到之前的时间步）
            seq_len = decoder_input_full.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            # 解码整个序列
            decoder_output = self.transformer_decoder(decoder_input_full, memory, tgt_mask=tgt_mask)
            
            # 预测动作
            predicted_trajectory = self.output_layer(decoder_output)
            
            return predicted_trajectory
        
        # 推理模式（自回归生成）
        else:
            for i in range(self.max_seq_len):
                # 添加位置编码
                pos_decoder_input = self.pos_encoder(decoder_input)
                
                # 创建注意力掩码
                seq_len = decoder_input.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                
                # 解码当前序列
                decoder_output = self.transformer_decoder(pos_decoder_input, memory, tgt_mask=tgt_mask)
                
                # 预测下一个动作
                next_action = self.output_layer(decoder_output[:, -1:])  # [batch_size, 1, 7]
                outputs.append(next_action)
                
                # 为下一步准备输入 - 嵌入预测的动作
                next_embedding = self.action_embed(next_action.reshape(batch_size, -1)).unsqueeze(1)
                decoder_input = torch.cat([decoder_input, next_embedding], dim=1)
            
            # 合并所有预测的动作
            trajectory = torch.cat(outputs, dim=1)  # [batch_size, max_seq_len, 7]
            
            return trajectory

def train_transformer2(model, dataset, num_epochs=100, batch_size=64, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = nn.MSELoss()
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    
    # 获取模型的最大序列长度
    max_seq_len = model.max_seq_len
    processed_trajectories = []
    
    for traj in dataset['trajectories']:
        traj_len = traj.size(0)
        
        if traj_len <= max_seq_len:
            # 如果轨迹长度小于等于max_seq_len，进行填充
            padding = torch.zeros(max_seq_len - traj_len, model.action_dim)
            processed_traj = torch.cat([traj, padding], dim=0)
        else:
            # 如果轨迹长度大于max_seq_len，从随机位置开始截取max_seq_len长度的序列
            start_idx = torch.randint(0, traj_len - max_seq_len + 1, (1,)).item()
            processed_traj = traj[start_idx:start_idx + max_seq_len]
            
        processed_trajectories.append(processed_traj)
    
    processed_trajectories = torch.stack(processed_trajectories)
    
    # 创建数据集
    tensor_dataset = TensorDataset(
        dataset['joint_angles'],
        dataset['joint_positions'],
        dataset['target_positions'],
        processed_trajectories
    )
    
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for joint_angles, joint_positions, target_positions, target_trajectories in tqdm(dataloader):
            joint_angles = joint_angles.to(device)
            joint_positions = joint_positions.to(device)
            target_positions = target_positions.to(device)
            target_trajectories = target_trajectories.to(device)
            
            # 前向传播（使用教师强制）
            pred_trajectories = model(
                joint_angles, 
                joint_positions, 
                target_positions, 
                target_trajectory=target_trajectories,
            )
            
            # 计算损失
            loss = criterion(pred_trajectories, target_trajectories)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
            
            # 添加验证步骤
            model.eval()
            with torch.no_grad():
                # 随机选择一个样本进行可视化
                sample_idx = np.random.randint(0, len(tensor_dataset))
                sample_joint_angles, sample_joint_positions, sample_target_positions, sample_target_trajectories = tensor_dataset[sample_idx]
                
                # 将样本移动到设备上
                sample_joint_angles = sample_joint_angles.unsqueeze(0).to(device)
                sample_joint_positions = sample_joint_positions.unsqueeze(0).to(device)
                sample_target_positions = sample_target_positions.unsqueeze(0).to(device)
                
                # 生成轨迹（推理模式）
                generated_trajectory = model(
                    sample_joint_angles,
                    sample_joint_positions,
                    sample_target_positions,
                    target_trajectory=None,  # 推理模式
                )
                
                # 打印一些统计信息
                print(f"生成轨迹形状: {generated_trajectory.shape}")
                print(f"目标轨迹形状: {sample_target_trajectories.shape}")
                
                # 计算生成轨迹与目标轨迹的MSE
                sample_target_trajectories = sample_target_trajectories.unsqueeze(0).to(device)
                val_mse = F.mse_loss(generated_trajectory, sample_target_trajectories[:, :model.max_seq_len]).item()
                print(f"验证MSE: {val_mse:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), r'C:\DiskD\trae_doc\robot_gym\transformer_model.pth')
    return model

def collect_data_with_transformer(env, transformer_model, num_trajectories=1000, max_steps=200, use_rolling_prediction=True, decay_factor=0.9):
    """
    使用Transformer模型收集轨迹数据
    
    参数:
        env: 机械臂环境
        transformer_model: 训练好的Transformer模型
        num_trajectories: 要收集的轨迹数量
        max_steps: 每条轨迹的最大步数
        use_rolling_prediction: 是否使用滚动时域预测
        decay_factor: 时间衰减因子，用于加权不同时刻的预测（越接近当前时刻权重越大）
    """
    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'dones': []
    }
    
    # 获取模型所在的设备
    device = next(transformer_model.parameters()).device
    transformer_model.eval()  # 设置为评估模式
    
    for _ in tqdm(range(num_trajectories)):
        obs = env.reset()
        
        # 执行预测的轨迹并收集RL数据
        current_obs = obs
        
        if not use_rolling_prediction:
            # 原始方法：只在初始状态预测一次
            # 解析初始观测并移动到正确的设备
            joint_angles = torch.tensor(obs[:transformer_model.action_dim], dtype=torch.float32).unsqueeze(0).to(device)
            joint_positions = torch.tensor(obs[transformer_model.action_dim:-3], dtype=torch.float32).unsqueeze(0).reshape(1, transformer_model.action_dim, 3).to(device)
            target_position = torch.tensor(obs[-3:], dtype=torch.float32).unsqueeze(0).to(device)
            
            # 使用Transformer预测轨迹（推理模式）
            with torch.no_grad():
                predicted_trajectory = transformer_model(
                    joint_angles, 
                    joint_positions, 
                    target_position, 
                    target_trajectory=None,  # 推理模式
                ).squeeze(0).cpu().numpy()  # 将结果转回CPU并转为numpy数组
            
            for step in range(max_steps):
                if step >= len(predicted_trajectory):
                    break
                    
                action = predicted_trajectory[step]
                
                # 执行动作
                next_obs, reward, done, _ = env.step(action)
                
                # 存储经验
                dataset['observations'].append(current_obs)
                dataset['actions'].append(action)
                dataset['rewards'].append(reward)
                dataset['next_observations'].append(next_obs)
                dataset['dones'].append(done)
                
                current_obs = next_obs
                
                if done:
                    break
        else:
            # 改进方法：滚动时域预测
            # 在每个时间步重新预测未来轨迹
            action_buffer = None  # 用于存储加权融合后的动作序列
            
            for step in range(max_steps):
                # 解析当前观测
                joint_angles = torch.tensor(current_obs[:transformer_model.action_dim], dtype=torch.float32).unsqueeze(0).to(device)
                joint_positions = torch.tensor(current_obs[transformer_model.action_dim:-3], dtype=torch.float32).unsqueeze(0).reshape(1, transformer_model.action_dim, 3).to(device)
                target_position = torch.tensor(current_obs[-3:], dtype=torch.float32).unsqueeze(0).to(device)
                
                # 使用Transformer预测未来轨迹
                with torch.no_grad():
                    new_predicted_trajectory = transformer_model(
                        joint_angles, 
                        joint_positions, 
                        target_position, 
                        target_trajectory=None,
                    ).squeeze(0).cpu().numpy()
                
                # 如果是第一步或者action_buffer为空，直接使用新预测的轨迹
                if action_buffer is None:
                    action_buffer = new_predicted_trajectory
                else:
                    # 加权融合新旧预测
                    # 新预测的权重随时间衰减
                    weights = np.array([decay_factor**i for i in range(len(new_predicted_trajectory))])
                    weights = weights / weights.sum()  # 归一化权重
                    
                    # 更新action_buffer，保留其长度
                    buffer_length = min(len(action_buffer) - 1, len(new_predicted_trajectory))
                    
                    # 旧预测向前移动一步（丢弃已执行的第一个动作）
                    old_actions = action_buffer[1:buffer_length+1] if len(action_buffer) > 1 else []
                    
                    # 新预测的前buffer_length个动作
                    new_actions = new_predicted_trajectory[:buffer_length]
                    
                    if len(old_actions) > 0:
                        # 加权融合
                        fused_actions = (1 - weights[:buffer_length])[:, np.newaxis] * old_actions + \
                                        weights[:buffer_length][:, np.newaxis] * new_actions
                        
                        # 如果新预测比旧预测长，添加剩余部分
                        if len(new_predicted_trajectory) > buffer_length:
                            fused_actions = np.vstack([fused_actions, new_predicted_trajectory[buffer_length:]])
                        
                        action_buffer = fused_actions
                    else:
                        action_buffer = new_predicted_trajectory
                
                # 执行当前动作（始终使用融合后序列的第一个动作）
                action = action_buffer[0]
                
                # 执行动作
                next_obs, reward, done, _ = env.step(action)
                
                # 存储经验
                dataset['observations'].append(current_obs)
                dataset['actions'].append(action)
                dataset['rewards'].append(reward)
                dataset['next_observations'].append(next_obs)
                dataset['dones'].append(done)
                
                current_obs = next_obs
                
                if done:
                    break
    
    # 转换为numpy数组
    for key in dataset:
        dataset[key] = np.array(dataset[key])
        
    return dataset
