import torch
from torch.cuda import device
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
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

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=31, hidden_dim=128, num_layers=2, nhead=4, max_seq_len=200):
        super(TrajectoryTransformer, self).__init__()
        
        # 输入处理
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        self.action_embed = nn.Linear(7, hidden_dim)  # 专门用于嵌入动作的层
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Transformer解码器
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 7)  # 7个关节的动作
        
        # 用于自回归生成的起始token
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, joint_angles, joint_positions, target_position, max_len=200):
        batch_size = joint_angles.size(0)
        
        # 合并输入特征
        joint_positions_flat = joint_positions.reshape(batch_size, -1)
        inputs = torch.cat([joint_angles, joint_positions_flat, target_position], dim=1)
        
        # 编码输入
        encoded_input = self.input_embed(inputs).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 使用Transformer编码器处理输入
        memory = self.transformer_encoder(encoded_input)  # [batch_size, 1, hidden_dim]
        
        # 自回归解码
        start_tokens = self.start_token.repeat(batch_size, 1, 1)  # [batch_size, 1, hidden_dim]
        decoder_output = start_tokens
        
        outputs = []
        for i in range(max_len):
            # 位置编码
            pos_decoder_output = self.pos_encoder(decoder_output)
            
            # 解码当前序列
            seq_len = decoder_output.size(1)
            # 确保掩码尺寸正确，并且与当前批次大小匹配
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(inputs.device)
            
            output = self.transformer_decoder(pos_decoder_output, memory, tgt_mask=tgt_mask)
            
            # 预测下一个动作
            next_action = self.output_layer(output[:, -1:])  # [batch_size, 1, 7]
            outputs.append(next_action)
            
            # 为下一步准备输入 - 使用专门的动作嵌入层
            next_embedding = self.action_embed(next_action.reshape(batch_size, -1)).unsqueeze(1)
            decoder_output = torch.cat([decoder_output, next_embedding], dim=1)
        
        # 合并所有预测的动作
        trajectory = torch.cat(outputs, dim=1)  # [batch_size, max_len, 7]
        
        return trajectory

def train_transformer(model, dataset, num_epochs=100, batch_size=64, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    criterion = nn.MSELoss()
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    
    # 填充轨迹到相同长度，但最大不超过30
    max_traj_len = min(200, max([traj.size(0) for traj in dataset['trajectories']]))
    padded_trajectories = []
    
    for traj in dataset['trajectories']:
        # 如果轨迹长度超过30，截断
        if traj.size(0) > max_traj_len:
            traj = traj[:max_traj_len]
        
        # 如果轨迹长度小于max_traj_len，填充
        if traj.size(0) < max_traj_len:
            padding = torch.zeros(max_traj_len - traj.size(0), 7)
            padded_traj = torch.cat([traj, padding], dim=0)
        else:
            padded_traj = traj
            
        padded_trajectories.append(padded_traj)
    
    padded_trajectories = torch.stack(padded_trajectories)
    
    # 创建数据集
    tensor_dataset = TensorDataset(
        dataset['joint_angles'],
        dataset['joint_positions'],
        dataset['target_positions'],
        padded_trajectories
    )
    
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        
        for joint_angles, joint_positions, target_positions, target_trajectories in tqdm(dataloader):
            # 获取当前批次的实际大小
            current_batch_size = joint_angles.size(0)
            joint_angles = joint_angles.to(device)
            joint_positions = joint_positions.to(device)
            target_positions = target_positions.to(device)
            target_trajectories = target_trajectories.to(device)
            # 前向传播
            pred_trajectories = model(joint_angles, joint_positions, target_positions, max_len=200)
            
            # 计算损失（只考虑非填充部分）
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
    torch.save(model.state_dict(), r'C:\DiskD\trae_doc\robot_gym\transformer_model.pth')
    return model

def collect_data_with_transformer(env, transformer_model, num_trajectories=1000, max_steps=200):
    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'dones': []
    }
    
    # 获取模型所在的设备
    device = next(transformer_model.parameters()).device
    
    for _ in tqdm(range(num_trajectories)):
        obs = env.reset()
        
        # 解析初始观测并移动到正确的设备
        joint_angles = torch.tensor(obs[:7], dtype=torch.float32).unsqueeze(0).to(device)
        joint_positions = torch.tensor(obs[7:-3], dtype=torch.float32).unsqueeze(0).reshape(1, 7, 3).to(device)
        target_position = torch.tensor(obs[-3:], dtype=torch.float32).unsqueeze(0).to(device)
        
        # 使用Transformer预测轨迹
        with torch.no_grad():
            predicted_trajectory = transformer_model(
                joint_angles, joint_positions, target_position, max_len=max_steps
            ).squeeze(0).cpu().numpy()  # 将结果转回CPU并转为numpy数组
        
        # 执行预测的轨迹并收集RL数据
        current_obs = obs
        
        for step in range(min(max_steps, len(predicted_trajectory))):
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
    
    # 转换为numpy数组
    for key in dataset:
        dataset[key] = np.array(dataset[key])
        
    return dataset
