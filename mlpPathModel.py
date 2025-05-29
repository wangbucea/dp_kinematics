import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import pybullet as p
import h5py
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class HDF5Dataset(Dataset):
    """
    支持大型HDF5文件的流式数据集类
    """
    def __init__(self, h5_file_path, max_seq_len=30, cache_size=1000):
        self.h5_file_path = h5_file_path
        self.max_seq_len = max_seq_len
        self.cache_size = cache_size
        self.cache = {}
        self.cache_keys = []
        
        # 打开文件获取数据长度
        with h5py.File(h5_file_path, 'r') as f:
            self.length = len(f['joint_angles'])
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]
            
        # 从文件读取数据
        with h5py.File(self.h5_file_path, 'r') as f:
            idx_str = str(idx)
            
            # 读取数据
            initial_joint_angles = torch.tensor(f['joint_angles'][idx_str][:], dtype=torch.float32)
            initial_joint_positions = torch.tensor(f['joint_positions'][idx_str][:], dtype=torch.float32)
            target_position = torch.tensor(f['target_positions'][idx_str][:], dtype=torch.float32)
            sequence_joint_angles = torch.tensor(f['sequence_joint_angles'][idx_str][:], dtype=torch.float32)
            
            # 处理序列长度
            if sequence_joint_angles.size(0) > self.max_seq_len:
                sequence_joint_angles = sequence_joint_angles[:self.max_seq_len]
            elif sequence_joint_angles.size(0) < self.max_seq_len:
                padding = torch.zeros(self.max_seq_len - sequence_joint_angles.size(0), 
                                    sequence_joint_angles.size(1), dtype=torch.float32)
                sequence_joint_angles = torch.cat([sequence_joint_angles, padding], dim=0)
            
            # 获取目标关节角度
            target_joint_angles = sequence_joint_angles[-1]
            for j in range(sequence_joint_angles.size(0)-1, -1, -1):
                if torch.sum(torch.abs(sequence_joint_angles[j])) > 1e-6:
                    target_joint_angles = sequence_joint_angles[j]
                    break
            
            data = (initial_joint_angles, initial_joint_positions, target_position, 
                   target_joint_angles, sequence_joint_angles)
            
            # 更新缓存
            if len(self.cache) >= self.cache_size:
                # 移除最旧的缓存项
                oldest_key = self.cache_keys.pop(0)
                del self.cache[oldest_key]
            
            self.cache[idx] = data
            self.cache_keys.append(idx)
            
            return data

class MLPEncoder(nn.Module):
    """
    编码器：将机械臂状态和目标位置编码为隐藏表示
    参考MPNet中的Encoder结构
    """
    
    def __init__(self, input_size, output_size):
        super(MLPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(0.1),
            nn.Linear(512, 768), nn.GELU(), nn.LayerNorm(768),nn.Dropout(0.1),
            nn.Linear(768, 896), nn.GELU(), nn.LayerNorm(896),nn.Dropout(0.1),
            nn.Linear(896, output_size)
        )
    def forward(self, x):
        return self.encoder(x)

class MLPPathGenerator(nn.Module):
    """
    路径生成器：生成机械臂的运动轨迹
    """
    def __init__(self, input_size, output_size, hidden_sizes=[1280, 1024, 896, 768, 640,  512, 384,  256]):
        # 1280, 896, 512, 384, 256, 128, 64, 32                            1280 1024 896 768 640  512 384  256
        super(MLPPathGenerator, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Dropout(0.1))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            prev_size = hidden_size
        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class EnhancedMLPPathGenerator(nn.Module):
    """
    增强版路径生成器：添加残差连接和注意力机制
    """
    def __init__(self, input_size, output_size, hidden_sizes=[1280, 1024, 896, 768, 640, 512, 384, 256]):
        super(EnhancedMLPPathGenerator, self).__init__()
        
        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        prev_size = input_size
        
        # 构建隐藏层和残差连接
        for i, hidden_size in enumerate(hidden_sizes):
            # 主层
            self.layers.append(nn.Sequential(
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
            
            # 残差连接（如果维度匹配）
            if i > 0 and hidden_sizes[i-1] == hidden_size:
                self.skip_connections.append(True)
            else:
                self.skip_connections.append(False)
                
            prev_size = hidden_size
        
        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(hidden_sizes[-1], num_heads=4, batch_first=True)
    
    def forward(self, x):
        # 保存中间结果用于残差连接
        residuals = []
        
        # 前向传播通过隐藏层
        for i, layer in enumerate(self.layers):
            if i > 0 and self.skip_connections[i]:
                x = layer(x) + residuals[-1]  # 残差连接
            else:
                x = layer(x)
            residuals.append(x)
        
        # 重塑以适应注意力层 [batch_size, 1, hidden_dim]
        x_reshaped = x.unsqueeze(1)
        
        # 应用自注意力
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x = x + attn_output.squeeze(1)  # 残差连接
        
        # 输出层
        return self.output_layer(x)
   
class BidirectionalPathGenerator(nn.Module):
    """
    双向轨迹生成器：从起点和终点同时生成轨迹，然后融合
    借鉴RRT*的双向搜索思想
    """
    def __init__(self, input_size, output_size, hidden_sizes=[1280, 1024, 896, 768, 640,  512, 384,  256], seq_len=30):
        super(BidirectionalPathGenerator, self).__init__()
        
        self.seq_len = seq_len
        # 修改half_seq_len的计算方式，确保2*half_seq_len-1 == seq_len
        self.half_seq_len = (seq_len + 1) // 2
        self.output_size = output_size  # 保存输出维度大小
        
        # 前向生成器（从起点到中间点）
        self.forward_generator = MLPPathGenerator(
            input_size, 
            output_size * self.half_seq_len,
            hidden_sizes
        )
        
        # 后向生成器（从终点到中间点）
        self.backward_generator = MLPPathGenerator(
            input_size, 
            output_size * self.half_seq_len,
            hidden_sizes
        )
        
        # 轨迹融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(output_size * 2, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size)
        )
    def forward(self, start_encoding, end_encoding):
        """
        双向生成轨迹
        Args:
            start_encoding: 起点编码 [batch_size, input_size]
            end_encoding: 终点编码 [batch_size, input_size]
        Returns:
            full_trajectory: 完整轨迹 [batch_size, seq_len, output_size/joint_dim]
        """
        batch_size = start_encoding.size(0)
        
        # 从起点生成前半段轨迹
        forward_traj_flat = self.forward_generator(start_encoding)
        forward_traj = forward_traj_flat.view(batch_size, self.half_seq_len, self.output_size)
        
        # 从终点生成后半段轨迹（需要反转）
        backward_traj_flat = self.backward_generator(end_encoding)
        backward_traj = backward_traj_flat.view(batch_size, self.half_seq_len, self.output_size)
        backward_traj = torch.flip(backward_traj, dims=[1])  # 反转时间维度
        
        # 获取连接点（前向轨迹的最后一点和后向轨迹的第一点）
        forward_end = forward_traj[:, -1, :]  # [batch_size, output_size]
        backward_start = backward_traj[:, 0, :]  # [batch_size, output_size]
        
        # 融合连接点
        fusion_input = torch.cat([forward_end, backward_start], dim=1)
        fusion_point = self.fusion_network(fusion_input).unsqueeze(1)  # [batch_size, 1, output_size]
        
        # 构建完整轨迹：前向轨迹（除最后一点）+ 融合点 + 后向轨迹（除第一点）
        full_trajectory = torch.cat([
            forward_traj[:, :-1, :],
            fusion_point,
            backward_traj[:, 1:, :]
        ], dim=1)
        
        # 确保轨迹长度正确
        actual_len = full_trajectory.size(1)
        if actual_len != self.seq_len:
            # 如果长度不匹配，进行调整
            if actual_len < self.seq_len:
                # 如果太短，在末尾添加一个额外的点（使用最后一个点的复制）
                extra_point = full_trajectory[:, -1:, :].clone()
                full_trajectory = torch.cat([full_trajectory, extra_point], dim=1)
            else:
                # 如果太长，截断
                full_trajectory = full_trajectory[:, :self.seq_len, :]
        
        return full_trajectory

class BidirectionalSequenceMLPPathModel(nn.Module):
    """
    双向序列编码的路径规划模型：能够从起点和终点同时生成轨迹并融合
    """
    def __init__(self, 
                 joint_dim=7,           # 关节角度维度
                 position_dim=3,        # 位置维度
                 hidden_dim=1024,        # 隐藏层维度
                 seq_len=30,            # 生成轨迹的长度
                 dropout=0.1,
                 num_attention_layers=2): # 注意力层数量
        super(BidirectionalSequenceMLPPathModel, self).__init__()
        
        # 计算输入维度：关节角度 + 关节位置 + 目标位置
        # self.input_dim = joint_dim + joint_dim * position_dim + position_dim
        self.input_dim = joint_dim + position_dim
        self.joint_dim = joint_dim
        self.position_dim = position_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.training_mode = True  # 默认为训练模式
        
        # 起点编码器
        self.start_encoder = MLPEncoder(self.input_dim, hidden_dim).to(device)
        
        # 终点编码器
        self.end_encoder = MLPEncoder(self.input_dim, hidden_dim)
        
        # 双向轨迹生成器
        self.bidirectional_generator = BidirectionalPathGenerator(
            hidden_dim, 
            joint_dim,
            seq_len=seq_len
        )
        
        # 双向LSTM层，用于捕捉时序依赖
        self.lstm = nn.LSTM(
            input_size=joint_dim,
            hidden_size=joint_dim*2,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True
        )
        
        self.layer_norm1 = nn.LayerNorm(joint_dim*4)
        self.layer_norm2 = nn.LayerNorm(joint_dim)
        
        # LSTM输出投影
        self.lstm_projection = nn.Linear(joint_dim*4, joint_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def _prepare_inputs(self, joint_angles, target_position):
        """准备编码器的输入"""
        # 确保所有输入至少是二维的
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)
        # if joint_positions.dim() == 1:
        #     joint_positions = joint_positions.unsqueeze(0)
        if target_position.dim() == 1:
            target_position = target_position.unsqueeze(0)
        
        batch_size = joint_angles.size(0)
        
        # 合并输入特征
        # joint_positions_flat = joint_positions.reshape(batch_size, -1)
        inputs = torch.cat([joint_angles, target_position], dim=1)
        
        return inputs
        
    def forward(self, joint_angles,joint_positions, target_position, target_joint_angles=None):
        """
        前向传播 - 双向生成轨迹
        Args:
            joint_angles: 初始关节角度 [batch_size, joint_dim]
            joint_positions: 初始关节位置 [batch_size, joint_dim, position_dim]
            target_position: 目标位置 [batch_size, position_dim]
            target_joint_angles: 目标关节角度 [batch_size, joint_dim]，如果为None则使用零向量
            teacher_forcing_ratio: 教师强制比例，用于训练时
        Returns:
            predicted_trajectory: 预测的轨迹 [batch_size, seq_len, joint_dim]
        """
        batch_size = joint_angles.size(0)
        
        # 如果没有提供目标关节角度，使用零向量
        if target_joint_angles is None:
            target_joint_angles = torch.zeros_like(joint_angles)
        
        # 准备起点输入并编码
        # start_inputs = self._prepare_inputs(joint_angles, joint_positions, target_position)
        start_inputs = self._prepare_inputs(joint_angles, target_position)
        
        start_encoded = self.start_encoder(start_inputs)
        start_encoded = self.dropout(start_encoded)
        # 准备终点输入并编码
        # end_inputs = self._prepare_inputs(target_joint_angles, joint_positions, target_position)
        end_inputs = self._prepare_inputs(target_joint_angles, target_position)
        
        end_encoded = self.end_encoder(end_inputs)
        end_encoded = self.dropout(end_encoded)
        # 生成双向轨迹
        predicted_trajectory = self.bidirectional_generator(start_encoded, end_encoded)
        
        # 应用LSTM层捕捉时序依赖
        lstm_out, _ = self.lstm(predicted_trajectory)
        lstm_out = self.layer_norm1(lstm_out)
        
        # 投影LSTM输出
        trajectory = self.lstm_projection(lstm_out)
        
        return trajectory  # 返回预测轨迹 每个关节的角度

def trajectory_similarity_loss(pred, target, alpha=1, beta=0.5, gamma=0.5, kl_weight=0.2):
    """
    高级轨迹损失函数：结合多种损失来减少误差累积，并添加KL散度正则化
    
    Args:
        pred: 预测轨迹 [batch_size, seq_len, joint_dim]
        target: 目标轨迹 [batch_size, seq_len, joint_dim]
        alpha: 基础MSE损失权重
        beta: 方向一致性损失权重
        gamma: 累积误差损失权重
        kl_weight: KL散度损失权重
    
    Returns:
        combined_loss: 组合损失
    """
    batch_size, seq_len, joint_dim = pred.shape
    
    # 1. 基础MSE损失
    mse_loss = F.mse_loss(pred, target)
    
    # 2. 方向一致性损失
    pred_direction = pred[:, 1:] - pred[:, :-1]
    target_direction = target[:, 1:] - target[:, :-1]
    
    direction_sim = 1.0 - F.cosine_similarity(
        pred_direction.reshape(-1, pred_direction.size(-1)),
        target_direction.reshape(-1, target_direction.size(-1)),
        dim=1
    ).mean()
    
    # 3. 累积误差损失 - 对后期预测给予更高权重
    weights = torch.linspace(1.0, 2.0, seq_len, device=pred.device)
    weighted_errors = weights.view(1, -1, 1) * (pred - target).pow(2)
    cumulative_loss = weighted_errors.mean()
    
    # 4. 关键点损失 - 特别关注序列中的关键点
    key_indices = [0, seq_len//4, seq_len//2, 3*seq_len//4, -1]
    key_points_loss = F.mse_loss(
        pred[:, key_indices], 
        target[:, key_indices]
    )

    key_indices = [0, seq_len//16, seq_len//8, 3*seq_len//16,
    seq_len//4, 5*seq_len//16, 3*seq_len//8,
    7*seq_len//16, 1*seq_len//2, 9*seq_len//16,
    5*seq_len//8, 11*seq_len//16, 3*seq_len//4,
    13*seq_len//16, 7*seq_len//8, 15*seq_len//16, -1]
    key_points_loss = F.mse_loss(
        pred[:, key_indices], 
        target[:, key_indices]
    )

    
    # 5. 速度一致性损失
    pred_velocity = pred_direction[:, 1:] - pred_direction[:, :-1]
    target_velocity = target_direction[:, 1:] - target_direction[:, :-1]
    velocity_loss = F.mse_loss(pred_velocity, target_velocity)
    # 计算加速度（二阶导数）
    pred_accel = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    target_accel = target_velocity[:, 1:] - target_velocity[:, :-1]
    
    # 加速度损失（对转弯点更敏感）
    accel_loss = F.mse_loss(pred_accel, target_accel)
    
    # 6. KL散度损失 - 使生成的动作轨迹分布接近专家轨迹分布
    # 计算预测和目标轨迹的均值和方差
    pred_mean = pred.mean(dim=1)  # [batch_size, joint_dim]
    target_mean = target.mean(dim=1)  # [batch_size, joint_dim]
    
    pred_var = ((pred - pred_mean.unsqueeze(1)) ** 2).mean(dim=1)  # [batch_size, joint_dim]
    target_var = ((target - target_mean.unsqueeze(1)) ** 2).mean(dim=1)  # [batch_size, joint_dim]
    
    # 确保方差为正数，避免数值问题
    pred_var = torch.clamp(pred_var, min=1e-6)
    target_var = torch.clamp(target_var, min=1e-6)
    
    # 计算KL散度: 0.5 * (log(σ2²/σ1²) + (σ1² + (μ1-μ2)²)/σ2² - 1)
    kl_div = 0.5 * (
        torch.log(target_var / pred_var) + 
        (pred_var + (pred_mean - target_mean) ** 2) / target_var - 1
    ).sum(dim=1).mean()
    
    # 组合损失
    combined_loss = (
        mse_loss + 
        200*direction_sim + 
        50 * cumulative_loss + 
        200 * key_points_loss + 
        200 * velocity_loss + 
        200 * accel_loss
        )
    
    return combined_loss

def train_sequence_mlp_path_model(model, dataset, num_epochs=100, batch_size=64, lr=1e-4, 
                                 weight_decay=1e-5, test_ratio=0.1, save_dir='model_results', early_stop_patience=10,
                                 save_best_only=True, temporal_reg_weight=0.3):
    """
    训练序列MLP路径模型
    
    Args:
        model: SequenceMLPPathModel或BidirectionalSequenceMLPPathModel实例
        dataset: 包含训练数据的字典，格式为：
                {
                    'joint_angles': list [num_samples, joint_dim],
                    'joint_positions': list [num_samples, joint_dim, position_dim],
                    'target_positions': list [num_samples, position_dim],
                    'sequence_joint_angles': list [num_samples, seq_len, joint_dim]
                }
        num_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        weight_decay: 权重衰减系数
        test_ratio: 测试集比例
        save_dir: 保存结果的目录
    
    Returns:
        训练好的模型和训练/测试损失
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import matplotlib.pyplot as plt
    
    # 获取模型的最大序列长度
    max_seq_len = model.seq_len
    
    # 处理数据集
    processed_data = {
        'initial_joint_angles': [],
        'initial_joint_positions': [],
        'target_positions': [],
        'target_joint_angles': [],
        'sequence_joint_angles': []
    }
    
    # 首先检查数据集中是否有必要的键
    required_keys = ['joint_angles', 'joint_positions', 'target_positions', 'sequence_joint_angles']
    for key in required_keys:
        if key not in dataset:
            raise KeyError(f"数据集中缺少必要的键: '{key}'")
    
    # 首先确保所有数据都是张量
    for key in dataset:
        if key != 'trajectories' and key != 'sequence_joint_angles' and key != 'sequence_joint_positions' and not isinstance(dataset[key][0], torch.Tensor):
            # 使用numpy.array()先转换为单个numpy数组，再转换为张量，提高效率
            dataset[key] = [torch.tensor(np.array(item), dtype=torch.float32) for item in dataset[key]]
    
    # 处理序列数据
    for i in range(len(dataset['sequence_joint_angles'])):
        seq_joint_angles = dataset['sequence_joint_angles'][i]
        if not isinstance(seq_joint_angles, torch.Tensor):
            seq_joint_angles = torch.tensor(np.array(seq_joint_angles), dtype=torch.float32)
        
        # 确保序列长度不超过模型的最大序列长度
        if seq_joint_angles.size(0) > max_seq_len:
            seq_joint_angles = seq_joint_angles[:max_seq_len]
        
        # 如果序列长度小于max_seq_len，填充
        if seq_joint_angles.size(0) < max_seq_len:
            padding = torch.zeros(max_seq_len - seq_joint_angles.size(0), seq_joint_angles.size(1), dtype=torch.float32)
            seq_joint_angles = torch.cat([seq_joint_angles, padding], dim=0)
        
        # 获取初始关节角度和目标关节角度
        initial_joint_angles = dataset['joint_angles'][i]
        if not isinstance(initial_joint_angles, torch.Tensor):
            initial_joint_angles = torch.tensor(np.array(initial_joint_angles), dtype=torch.float32)
        
        # 目标关节角度是序列的最后一个非零元素
        target_joint_angles = seq_joint_angles[-1]
        for j in range(seq_joint_angles.size(0)-1, -1, -1):
            if torch.sum(torch.abs(seq_joint_angles[j])) > 1e-6:
                target_joint_angles = seq_joint_angles[j]
                break
        
        # 获取初始关节位置和目标位置
        initial_joint_positions = dataset['joint_positions'][i]
        if not isinstance(initial_joint_positions, torch.Tensor):
            initial_joint_positions = torch.tensor(np.array(initial_joint_positions), dtype=torch.float32)
        
        target_position = dataset['target_positions'][i]
        if not isinstance(target_position, torch.Tensor):
            target_position = torch.tensor(np.array(target_position), dtype=torch.float32)
        
        # 添加到处理后的数据中
        processed_data['initial_joint_angles'].append(initial_joint_angles)
        processed_data['initial_joint_positions'].append(initial_joint_positions)
        processed_data['target_positions'].append(target_position)
        processed_data['target_joint_angles'].append(target_joint_angles)
        processed_data['sequence_joint_angles'].append(seq_joint_angles)
    
    # 转换为张量
    for key in processed_data:
        processed_data[key] = torch.stack(processed_data[key])
    
    # 划分训练集和测试集
    dataset_size = len(processed_data['initial_joint_angles'])
    test_size = int(dataset_size * test_ratio)
    train_size = dataset_size - test_size
    
    train_dataset, test_dataset = random_split(
        TensorDataset(
            processed_data['initial_joint_angles'],
            processed_data['initial_joint_positions'],
            processed_data['target_positions'],
            processed_data['target_joint_angles'],
            processed_data['sequence_joint_angles']
        ),
        [train_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr*10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 前10%的步骤用于预热
        div_factor=25,  # 初始学习率 = max_lr/div_factor
        final_div_factor=10000  # 最终学习率 = max_lr/(div_factor*final_div_factor)
    )
    criterion = nn.MSELoss()
    

    # 训练循环
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 早停相关变量
    best_test_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        if hasattr(model, 'set_training_mode'):
            model.set_training_mode(True)
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            initial_joint_angles, initial_joint_positions, target_positions, target_joint_angles, sequence_joint_angles = batch
            initial_joint_angles, initial_joint_positions, target_positions, target_joint_angles, sequence_joint_angles = initial_joint_angles.to(device), initial_joint_positions.to(device), target_positions.to(device), target_joint_angles.to(device), sequence_joint_angles.to(device)
            
            # 前向传播
            outputs = model(initial_joint_angles, initial_joint_positions, target_positions, target_joint_angles)
            
            # 检查是否返回了时间维度轨迹
            # 计算主要损失
            loss = trajectory_similarity_loss(outputs, sequence_joint_angles)
    
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 在测试集上评估
        model.eval()
        if hasattr(model, 'set_training_mode'):
            model.set_training_mode(False)  # 在评估时禁用时间维度正则化
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                initial_joint_angles, initial_joint_positions, target_positions, target_joint_angles, sequence_joint_angles = batch
                initial_joint_angles, initial_joint_positions, target_positions, target_joint_angles, sequence_joint_angles = initial_joint_angles.to(device), initial_joint_positions.to(device), target_positions.to(device), target_joint_angles.to(device), sequence_joint_angles.to(device)
                
                predicted_sequence = model(initial_joint_angles, initial_joint_positions, target_positions, target_joint_angles)
                # 计算损失
                loss = trajectory_similarity_loss(predicted_sequence, sequence_joint_angles)
            
                test_loss += loss.item()
        
        # 计算平均测试损失
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        # 早停检查
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # 如果设置了只保存最佳模型，则立即保存
            if save_best_only:
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(best_model_state, best_model_path)
                print(f"保存最佳模型，测试损失: {best_test_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"早停触发！连续{early_stop_patience}个epoch没有改善。")
                break
        # 更新学习率
        # scheduler.step(avg_test_loss)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        # 每10个epoch保存一次损失曲线
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Testing Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'loss_curve_epoch_{epoch+1}.png'))
            plt.close()
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    # 绘制最终的损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'final_loss_curve.png'))
    plt.close()
    
    return model, train_losses, test_losses

def generate_sequence_trajectory(model, initial_joint_angles, initial_joint_positions, target_position):
    """
    使用训练好的序列MLP模型生成轨迹
    
    Args:
        model: 训练好的SequenceMLPPathModel
        initial_joint_angles: 初始关节角度 [1, joint_dim]
        initial_joint_positions: 初始关节位置 [1, joint_dim, position_dim]
        target_position: 目标位置 [1, position_dim]
    
    Returns:
        生成的轨迹 [seq_len, joint_dim]
    """
    device = next(model.parameters()).device
    
    # 确保输入是张量并且有批次维度
    if not isinstance(initial_joint_angles, torch.Tensor):
        initial_joint_angles = torch.tensor(initial_joint_angles, dtype=torch.float32)
    if initial_joint_angles.dim() == 1:
        initial_joint_angles = initial_joint_angles.unsqueeze(0)
    
    if not isinstance(initial_joint_positions, torch.Tensor):
        initial_joint_positions = torch.tensor(initial_joint_positions, dtype=torch.float32)
    if initial_joint_positions.dim() == 2:
        initial_joint_positions = initial_joint_positions.unsqueeze(0)
    
    if not isinstance(target_position, torch.Tensor):
        target_position = torch.tensor(target_position, dtype=torch.float32)
    if target_position.dim() == 1:
        target_position = target_position.unsqueeze(0)
    
    # 移动到正确的设备
    initial_joint_angles = initial_joint_angles.to(device)
    initial_joint_positions = initial_joint_positions.to(device)
    target_position = target_position.to(device)
    
    # 设置为评估模式
    model.eval()
    
    # 生成轨迹
    with torch.no_grad():
        trajectory = model(initial_joint_angles, initial_joint_positions, target_position)
    
    # 返回轨迹，去掉批次维度
    return trajectory.squeeze(0).cpu().numpy()

def visualize_model_trajectory(model, env, joint_angles=None, joint_positions=None, target_position=None, 
                              output_file="model_trajectory.gif", fps=30, max_steps=10000):
    """
    将模型生成的动作序列在gym环境中进行可视化渲染
    
    参数:
        model: 训练好的模型（SequenceMLPPathModel实例）
        env: 机械臂环境
        joint_angles: 初始关节角度，如果为None则使用环境重置后的状态
        joint_positions: 初始关节位置，如果为None则使用环境重置后的状态
        target_position: 目标位置，如果为None则使用环境重置后的状态
        output_file: 输出文件路径
        fps: 每秒帧数
        max_steps: 最大步数
    
    返回:
        生成的轨迹数据
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np
    import torch
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    model.eval()  # 设置为评估模式
    
    # 检查是否需要重置环境
    need_reset = (joint_angles is None or joint_positions is None or target_position is None)
    
    # 如果没有提供初始状态，则重置环境获取
    if need_reset:
        obs = env.reset(random_init=True)
        joint_angles = torch.tensor(obs[:env.num_joints], dtype=torch.float32).unsqueeze(0).to(device)
        joint_positions_flat = obs[env.num_joints:-3]
        joint_positions = torch.tensor(joint_positions_flat, dtype=torch.float32).unsqueeze(0).reshape(1, env.num_joints, 3).to(device)
        target_position = torch.tensor(obs[-3:], dtype=torch.float32).unsqueeze(0).to(device)
    else:
        # 确保输入是张量并且维度正确
        if not isinstance(joint_angles, torch.Tensor):
            joint_angles = torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0).to(device)
        if not isinstance(joint_positions, torch.Tensor):
            joint_positions = torch.tensor(joint_positions, dtype=torch.float32).unsqueeze(0).to(device)
        if not isinstance(target_position, torch.Tensor):
            target_position = torch.tensor(target_position, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 执行生成的轨迹并收集数据
    traj_data = []
    
    # 记录初始状态
    current_joint_angles = np.array([p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
    current_joint_positions = np.array([p.getLinkState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)]).reshape(env.num_joints, 3)
    current_target_position = env.target_position.copy() if hasattr(env, 'target_position') else target_position.squeeze(0).cpu().numpy()
    
    traj_data.append({
        'joint_angles': current_joint_angles.copy(),
        'joint_positions': current_joint_positions.copy(),
        'target_position': current_target_position.copy()
    })
    
    # 创建权重数组 - 前面的动作权重更大
    # 使用指数衰减权重，前面的动作权重更大
    def create_weights(length, decay_factor=0.8):
        weights = np.array([decay_factor ** i for i in range(length)])
        return weights / weights.sum()  # 归一化权重
    
    # 执行预测的轨迹
    current_step = 0
    done = False
    
    while current_step < max_steps and not done:
        # 将当前状态转换为模型输入
        current_joint_angles_tensor = torch.tensor(current_joint_angles, dtype=torch.float32).unsqueeze(0).to(device)
        current_joint_positions_tensor = torch.tensor(current_joint_positions.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        current_joint_positions_tensor = current_joint_positions_tensor.reshape(1, env.num_joints, 3)
        current_target_position_tensor = torch.tensor(current_target_position, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 使用模型预测未来30步动作
        with torch.no_grad():
            predicted_trajectory = model(
                current_joint_angles_tensor, 
                current_joint_positions_tensor, 
                current_target_position_tensor
            )
            predicted_trajectory = predicted_trajectory.squeeze(0).cpu().numpy()
        
        # 计算要执行的步数 - 最多执行预测的前5步
        steps_to_execute = min(3, len(predicted_trajectory))
        
        # 创建权重 - 前面的动作权重更大
        weights = create_weights(7)
        # 执行加权后的动作序列
        for i in range(steps_to_execute):
            if current_step >= max_steps:
                break
                
            # 如果是第一步，直接使用预测的动作
            if i == 0:
                action = predicted_trajectory[0]
            else:
                # 对未来几步的动作进行加权平均
                action = np.zeros_like(predicted_trajectory[0])
                for j in range(min(3, steps_to_execute - i)):
                    action += weights*predicted_trajectory[i + j]
        # action = predicted_trajectory[0]
        # 确保动作在合理范围内
        action = np.clip(action, -0.001, 0.001)
        
        # 执行动作
        next_obs, reward, done, _ = env.step(action)
        current_step += 1
        
        # 提取当前状态
        current_joint_angles = next_obs[:env.num_joints]
        current_joint_positions = next_obs[env.num_joints:-3].reshape(env.num_joints, 3)
        
        # 记录当前状态
        traj_data.append({
            'joint_angles': current_joint_angles.copy(),
            'joint_positions': current_joint_positions.copy(),
            'target_position': current_target_position.copy()
        })
        
        if done:
            print(f"目标达成！步数: {current_step}")
            break
    
        if done:
            break
    
    if not done and current_step >= max_steps:
        print(f"达到最大步数 {max_steps}，但未完成目标")
    
    # 创建轨迹动画
    create_trajectory_animation(traj_data, output_file, fps)
    
    return traj_data

def create_trajectory_animation(trajectory, output_file="model_trajectory.gif", fps=10):
    """
    创建机械臂轨迹的动画并保存为GIF文件
    
    参数:
        trajectory: 轨迹数据列表，每个元素是包含关节位置的字典
        output_file: 输出文件路径
        fps: 每秒帧数
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取关节位置和目标位置
    joint_positions_list = []
    target_position = None
    
    for step in trajectory:
        joint_positions_list.append(step['joint_positions'])
        if target_position is None and 'target_position' in step:
            target_position = step['target_position']
    
    # 初始化机械臂线条
    line, = ax.plot([], [], [], 'bo-', linewidth=2, markersize=6)
    
    # 初始化末端轨迹线条
    end_trajectory, = ax.plot([], [], [], 'g--', alpha=0.5)
    
    # 绘制目标位置
    if target_position is not None:
        ax.scatter(target_position[0], target_position[1], target_position[2], 
                  color='red', s=150, marker='*', label='target location')
    
    # 设置坐标轴范围
    all_positions = np.vstack(joint_positions_list)
    max_range = np.array([
        np.max(all_positions[:, 0]) - np.min(all_positions[:, 0]),
        np.max(all_positions[:, 1]) - np.min(all_positions[:, 1]),
        np.max(all_positions[:, 2]) - np.min(all_positions[:, 2])
    ]).max() / 2.0
    
    mid_x = (np.max(all_positions[:, 0]) + np.min(all_positions[:, 0])) / 2
    mid_y = (np.max(all_positions[:, 1]) + np.min(all_positions[:, 1])) / 2
    mid_z = (np.max(all_positions[:, 2]) + np.min(all_positions[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('MLP Generated Trajectory')
    
    # 添加时间标签
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    
    # 存储末端位置历史
    end_positions_history = []
    
    # 初始化函数
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        end_trajectory.set_data([], [])
        end_trajectory.set_3d_properties([])
        time_text.set_text('')
        return line, end_trajectory, time_text
    
    # 更新函数 - 每一帧调用
    def update(frame):
        # 计算当前帧的机械臂位置
        positions = joint_positions_list[frame]
        
        # 更新机械臂线条
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        
        # 更新末端位置历史
        if frame == 0:
            # 重置历史记录，确保每次动画开始时都是从头开始
            end_positions_history.clear()
        end_positions_history.append(positions[-1])
        
        # 更新末端轨迹线条 - 只显示到当前帧的轨迹
        if len(end_positions_history) > 1:
            end_x = [p[0] for p in end_positions_history]
            end_y = [p[1] for p in end_positions_history]
            end_z = [p[2] for p in end_positions_history]
            end_trajectory.set_data(end_x, end_y)
            end_trajectory.set_3d_properties(end_z)
        
        # 更新时间标签
        progress = frame / (len(joint_positions_list) - 1) * 100
        time_text.set_text(f'进度: {progress:.1f}%')
        
        return line, end_trajectory, time_text
    
    # 创建动画
    frames = min(len(joint_positions_list), 100)  # 限制帧数，避免过多计算
    step = max(1, len(joint_positions_list) // frames)
    
    # 使用实际的帧索引，确保按照时间顺序显示
    frame_indices = list(range(0, len(joint_positions_list), step))
    if len(joint_positions_list) - 1 not in frame_indices:
        frame_indices.append(len(joint_positions_list) - 1)  # 确保包含最后一帧
    
    ani = FuncAnimation(fig, update, frames=frame_indices,
                       init_func=init, interval=1000/fps, blit=True)
    
    # 添加图例
    ax.legend()
    
    # 保存动画
    ani.save(output_file, writer='pillow', fps=fps)
    
    print(f"动画已保存至 {output_file}")
    
    plt.close(fig)
    
    return ani

def compare_model_trajectories(model1, model2, env, output_file="model_comparison.gif", fps=10, max_steps=30, 
                              title1="模型1轨迹", title2="模型2轨迹"):
    """
    比较两个模型生成的轨迹
    
    参数:
        model1, model2: 两个训练好的模型
        env: 机械臂环境
        output_file: 输出文件路径
        fps: 每秒帧数
        max_steps: 最大步数
        title1, title2: 两个模型的标题
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 重置环境，确保两个模型使用相同的初始状态
    obs = env.reset(random_init=False)
    
    # 生成两个模型的轨迹
    traj1 = visualize_model_trajectory(model1, env, output_file=f"model1_{output_file}", fps=fps, max_steps=max_steps)
    
    # 重置环境到相同的初始状态
    env.reset()
    for i, joint_angle in enumerate(traj1[0]['joint_angles']):
        if i < env.num_joints:
            p.resetJointState(env.robot_id, i, joint_angle)
    
    traj2 = visualize_model_trajectory(model2, env, output_file=f"model2_{output_file}", fps=fps, max_steps=max_steps)
    
    # 提取末端执行器位置
    end_positions1 = np.array([step['joint_positions'][-1] for step in traj1])
    end_positions2 = np.array([step['joint_positions'][-1] for step in traj2])
    
    # 创建比较图
    fig = plt.figure(figsize=(15, 7))
    
    # 第一个轨迹
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(end_positions1[:, 0], end_positions1[:, 1], end_positions1[:, 2], 'b-', linewidth=2)
    ax1.scatter(end_positions1[0, 0], end_positions1[0, 1], end_positions1[0, 2], color='green', s=100, label='起点')
    ax1.scatter(end_positions1[-1, 0], end_positions1[-1, 1], end_positions1[-1, 2], color='red', s=100, label='终点')
    
    # 如果有目标位置，绘制目标
    if 'target_position' in traj1[0]:
        target = traj1[0]['target_position']
        ax1.scatter(target[0], target[1], target[2], color='purple', s=150, marker='*', label='目标')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(title1)
    ax1.legend()
    
    # 第二个轨迹
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(end_positions2[:, 0], end_positions2[:, 1], end_positions2[:, 2], 'r-', linewidth=2)
    ax2.scatter(end_positions2[0, 0], end_positions2[0, 1], end_positions2[0, 2], color='green', s=100, label='起点')
    ax2.scatter(end_positions2[-1, 0], end_positions2[-1, 1], end_positions2[-1, 2], color='red', s=100, label='终点')
    
    # 如果有目标位置，绘制目标
    if 'target_position' in traj2[0]:
        target = traj2[0]['target_position']
        ax2.scatter(target[0], target[1], target[2], color='purple', s=150, marker='*', label='目标')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(title2)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"comparison_{output_file.replace('.gif', '.png')}")
    plt.show()
    
    return traj1, traj2

def train_sequence_mlp_path_model_hdf5(model, h5_file_path, num_epochs=100, batch_size=64, lr=1e-4,
                                      weight_decay=1e-5, test_ratio=0.1, save_dir='model_results', 
                                      early_stop_patience=10, save_best_only=True, temporal_reg_weight=0.3):
    """
    使用HDF5文件训练序列MLP路径模型（内存优化版本）
    
    Args:
        model: SequenceMLPPathModel或BidirectionalSequenceMLPPathModel实例
        h5_file_path: HDF5数据文件路径
        其他参数同原函数
    
    Returns:
        训练好的模型和训练/测试损失
    """
    import gc
    from torch.utils.data import DataLoader, random_split
    import matplotlib.pyplot as plt
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建HDF5数据集
    full_dataset = HDF5Dataset(h5_file_path, max_seq_len=model.seq_len, cache_size=batch_size*2)
    
    # 划分训练集和测试集
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_ratio)
    train_size = dataset_size - test_size
    
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 创建数据加载器，使用较小的batch_size和多进程
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    
    # 设置优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr*10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=10000
    )
    
    # 训练循环
    train_losses = []
    test_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 早停相关变量
    best_test_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        if hasattr(model, 'set_training_mode'):
            model.set_training_mode(True)
        epoch_loss = 0.0
        
        # 训练阶段
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            try:
                initial_joint_angles, initial_joint_positions, target_positions, target_joint_angles, sequence_joint_angles = batch
                
                # 移动到设备
                initial_joint_angles = initial_joint_angles.to(device, non_blocking=True)
                initial_joint_positions = initial_joint_positions.to(device, non_blocking=True)
                target_positions = target_positions.to(device, non_blocking=True)
                target_joint_angles = target_joint_angles.to(device, non_blocking=True)
                sequence_joint_angles = sequence_joint_angles.to(device, non_blocking=True)
                
                # 前向传播
                outputs = model(initial_joint_angles, initial_joint_positions, target_positions, target_joint_angles)
                
                # 计算损失
                loss = trajectory_similarity_loss(outputs, sequence_joint_angles)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                # 定期清理GPU内存
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU内存不足，跳过批次 {batch_idx}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        if hasattr(model, 'set_training_mode'):
            model.set_training_mode(False)
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    initial_joint_angles, initial_joint_positions, target_positions, target_joint_angles, sequence_joint_angles = batch
                    
                    initial_joint_angles = initial_joint_angles.to(device, non_blocking=True)
                    initial_joint_positions = initial_joint_positions.to(device, non_blocking=True)
                    target_positions = target_positions.to(device, non_blocking=True)
                    target_joint_angles = target_joint_angles.to(device, non_blocking=True)
                    sequence_joint_angles = sequence_joint_angles.to(device, non_blocking=True)
                    
                    predicted_sequence = model(initial_joint_angles, initial_joint_positions, target_positions, target_joint_angles)
                    loss = trajectory_similarity_loss(predicted_sequence, sequence_joint_angles)
                    test_loss += loss.item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("测试时GPU内存不足，跳过批次")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
        
        # 计算平均测试损失
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # 早停检查
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            if save_best_only:
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(best_model_state, best_model_path)
                print(f"Epoch {epoch+1}: 保存最佳模型，测试损失: {best_test_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
        # 定期清理内存
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 保存最终模型和损失曲线
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练过程损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'final_loss_curve.png'))
    plt.close()
    
    return model, train_losses, test_losses

def test_model_visualization():
    """
    测试模型可视化函数 - 可视化3组运动学数据和3组MLP模型生成的数据
    """
    from robotEnv import RoboticArmEnv
    import torch
    import os
    import pybullet as p
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import time
    
    # 检查是否已有连接
    try:
        # 尝试断开现有连接
        p.disconnect()
    except:
        pass
    
    # 创建环境 - 使用GUI模式
    env = RoboticArmEnv(use_gui=True)
    
    # 加载模型
    model_path = os.path.join('model_results', 'best_sequence_mlp_model.pth')
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请先训练模型")
        env.close()  # 确保关闭环境
        return
    
    # 创建模型实例
    model = BidirectionalSequenceMLPPathModel(joint_dim=7, position_dim=3, hidden_dim=512, seq_len=30)
    
    # 加载模型参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    
    print("模型加载成功，开始生成轨迹...")
    
    try:
        # 创建图形窗口
        plt.figure(figsize=(18, 12))
        
        # 生成3组运动学数据和3组MLP模型数据
        for group_idx in range(3):
            # 重置环境，获取随机目标
            obs = env.reset()
            joint_angles = torch.tensor(obs[:7], dtype=torch.float32).unsqueeze(0).to(device)
            joint_positions = torch.tensor(obs[7:-3], dtype=torch.float32).reshape(1, 7, 3).to(device)
            target_position = torch.tensor(obs[-3:], dtype=torch.float32).unsqueeze(0).to(device)
            
            # 使用MLP模型生成轨迹
            with torch.no_grad():
                predicted_trajectory = model(joint_angles, joint_positions, target_position).squeeze(0).cpu().numpy()
            
            # 收集运动学轨迹数据
            kinematics_trajectory = []
            current_angles = obs[:7].copy()
            
            # 重置环境到初始状态
            for joint_id in range(7):
                p.resetJointState(env.robot_id, joint_id, current_angles[joint_id])
            
            # 使用运动学方法生成轨迹
            target_pos = obs[-3:].copy()
            steps = 30  # 与模型生成的轨迹长度相同
            
            # 使用逆运动学计算目标关节角度
            target_orn = p.getQuaternionFromEuler([0, 0, 0])
            ik_solution = p.calculateInverseKinematics(
                env.robot_id, 
                6,  # 末端执行器关节ID
                target_pos,
                targetOrientation=target_orn,
                maxNumIterations=100,
                residualThreshold=0.01
            )
            target_angles = np.array(ik_solution[:7])
            
            # 线性插值生成轨迹
            for step in range(steps):
                alpha = step / (steps - 1)
                interpolated_angles = (1 - alpha) * current_angles + alpha * target_angles
                
                # 收集关节角度
                kinematics_trajectory.append(interpolated_angles)
            
            kinematics_trajectory = np.array(kinematics_trajectory)
            
            # 可视化轨迹 - 运动学方法
            ax1 = plt.subplot(3, 2, group_idx*2+1, projection='3d')
            ax1.set_title(f'组 {group_idx+1}: 运动学轨迹')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 计算并绘制运动学轨迹中的末端执行器位置
            kinematics_positions = []
            for angles in kinematics_trajectory:
                # 设置关节角度
                for i, angle in enumerate(angles):
                    p.resetJointState(env.robot_id, i, angle)
                
                # 获取末端执行器位置
                link_state = p.getLinkState(env.robot_id, 6)
                pos = link_state[0]  # 位置
                kinematics_positions.append(pos)
            
            kinematics_positions = np.array(kinematics_positions)
            ax1.plot(kinematics_positions[:, 0], kinematics_positions[:, 1], kinematics_positions[:, 2], 'b-', linewidth=2)
            ax1.scatter(target_pos[0], target_pos[1], target_pos[2], c='r', marker='*', s=100, label='目标')
            ax1.legend()
            
            # 可视化轨迹 - MLP模型
            ax2 = plt.subplot(3, 2, group_idx*2+2, projection='3d')
            ax2.set_title(f'组 {group_idx+1}: MLP模型轨迹')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # 计算并绘制MLP模型轨迹中的末端执行器位置
            mlp_positions = []
            for angles in predicted_trajectory:
                # 设置关节角度
                for i, angle in enumerate(angles):
                    p.resetJointState(env.robot_id, i, float(angle))
                
                # 获取末端执行器位置
                link_state = p.getLinkState(env.robot_id, 6)
                pos = link_state[0]  # 位置
                mlp_positions.append(pos)
            
            mlp_positions = np.array(mlp_positions)
            ax2.plot(mlp_positions[:, 0], mlp_positions[:, 1], mlp_positions[:, 2], 'g-', linewidth=2)
            ax2.scatter(target_pos[0], target_pos[1], target_pos[2], c='r', marker='*', s=100, label='目标')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig('trajectory_comparison.png')
        plt.show()
        
        print("轨迹可视化完成！结果已保存为 trajectory_comparison.png")
    
    except Exception as e:
        print(f"轨迹生成过程中出错: {e}")

    finally:
        # 确保环境被关闭
        env.close()
        try:
            p.disconnect()
        except:
            pass
        print("环境已关闭")

def visualize_sequence_data2_trajectories(env, num_trajectories=5, save_path=None):
    """
    可视化collect_sequence_data2函数采集的动作轨迹动画
    
    Args:
        env: 机械臂环境
        num_trajectories: 要可视化的轨迹数量
        save_path: 保存动画的路径，如果为None则使用当前目录
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from tqdm import tqdm
    import os
    import pybullet as p
    from robotEnv import collect_sequence_data2, RoboticArmEnv
    
    # 确保环境已正确初始化
    if not p.isConnected():
        print("重新连接到物理服务器...")
        p.connect(p.DIRECT)  # 或者使用 p.GUI 进行可视化
    
    # 如果传入的环境已关闭，创建一个新的环境
    try:
        env.reset()
    except:
        print("环境已关闭，创建新环境...")
        env = RoboticArmEnv()
    
    # 收集数据
    print("收集轨迹数据...")
    dataset = collect_sequence_data2(env, num_trajectories=num_trajectories)
    
    # 确定保存路径
    if save_path is None:
        save_path = os.getcwd()
    
    # 为每条轨迹创建动画
    for i in range(min(num_trajectories, len(dataset['trajectories']))):
        print(f"创建轨迹 {i+1} 的动画...")
        
        # 构建完整轨迹
        traj = []
        
        # 初始关节角度和位置
        joint_angles = dataset['joint_angles'][i]
        joint_positions = dataset['joint_positions'][i]
        target_position = dataset['target_positions'][i]
        
        # 初始状态
        traj.append({
            'joint_angles': joint_angles,
            'joint_positions': joint_positions,
            'target_position': target_position
        })
        
        # 执行轨迹中的每一步
        current_angles = joint_angles.copy()
        for action in dataset['trajectories'][i]:
            action = action
            current_angles = current_angles + action
            
            # 使用环境计算新的关节位置
            env.reset()
            for j, angle in enumerate(current_angles):
                if j < env.num_joints:
                    p.resetJointState(env.robot_id, j, angle)
            
            # 获取关节位置
            joint_positions = []
            for joint_id in range(env.num_joints):
                joint_info = p.getLinkState(env.robot_id, joint_id)
                joint_positions.append(joint_info[0])
            joint_positions = np.array(joint_positions)
            
            traj.append({
                'joint_angles': current_angles,
                'joint_positions': joint_positions,
                'target_position': target_position
            })
        
        # 创建动画
        create_trajectory_animation(traj, os.path.join(save_path, f"sequence_data2_trajectory_{i+1}.gif"))
    
    print("动画创建完成！")
    
    # 确保在函数结束时不关闭环境，让调用者决定何时关闭
    return env


def visualize_model_predictions(model, env, dataset, num_samples=5, save_dir='model_results'):
    """
    可视化模型预测结果
    
    Args:
        model: 训练好的模型
        env: 机械臂环境
        dataset: 测试数据集
        num_samples: 要可视化的样本数量
        save_dir: 保存结果的目录
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 将模型设置为评估模式
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 随机选择样本
    indices = np.random.choice(len(dataset['joint_angles']), min(num_samples, len(dataset['joint_angles'])), replace=False)
    
    for i, idx in enumerate(indices):
        # 获取样本数据
        joint_angles = dataset['joint_angles'][idx]
        joint_positions = dataset['joint_positions'][idx]
        target_position = dataset['target_positions'][idx]
        sequence_joint_angles = dataset['sequence_joint_angles'][idx]
        
        # 确保数据是张量
        if not isinstance(joint_angles, torch.Tensor):
            joint_angles = torch.tensor(np.array(joint_angles), dtype=torch.float32)
        if not isinstance(joint_positions, torch.Tensor):
            joint_positions = torch.tensor(np.array(joint_positions), dtype=torch.float32)
        if not isinstance(target_position, torch.Tensor):
            target_position = torch.tensor(np.array(target_position), dtype=torch.float32)
        if not isinstance(sequence_joint_angles, torch.Tensor):
            sequence_joint_angles = torch.tensor(np.array(sequence_joint_angles), dtype=torch.float32)
        
        # 添加批次维度
        joint_angles = joint_angles.unsqueeze(0)
        joint_positions = joint_positions.unsqueeze(0)
        target_position = target_position.unsqueeze(0)
        
        # 获取目标关节角度（序列的最后一个非零元素）
        target_joint_angles = None
        for j in range(sequence_joint_angles.size(0)-1, -1, -1):
            if torch.sum(torch.abs(sequence_joint_angles[j])) > 1e-6:
                target_joint_angles = sequence_joint_angles[j].unsqueeze(0)
                break
        
        # 模型预测 - 在推理时不使用目标关节角度
        with torch.no_grad():
            joint_angles = joint_angles.to(device)
            joint_positions = joint_positions.to(device)
            target_position = target_position.to(device)
            outputs = model(joint_angles, joint_positions, target_position, None)
            if isinstance(outputs, tuple):
                predicted_sequence = outputs[0]
            else:
                predicted_sequence = outputs
            # predicted_sequence, _ = model(joint_angles, joint_positions, target_position, None)
            predicted_sequence = predicted_sequence.squeeze(0)  # 移除批次维度
        
        # 获取真实轨迹和预测轨迹的末端执行器位置
        true_end_effector_positions = []
        pred_end_effector_positions = []
        
        # 重置环境
        env.reset()
        
        # 设置初始关节角度
        for j, joint_id in enumerate(range(env.num_joints)):
            p.resetJointState(env.robot_id, joint_id, joint_angles[0, j].item())
        
        # 计算真实轨迹的末端执行器位置
        for j in range(sequence_joint_angles.size(0)):
            if torch.sum(torch.abs(sequence_joint_angles[j])) > 1e-6:  # 跳过填充的零
                # 设置关节角度
                for k, joint_id in enumerate(range(env.num_joints)):
                    p.resetJointState(env.robot_id, joint_id, sequence_joint_angles[j, k].item())
                
                # 获取末端执行器位置
                end_effector_pos = env._get_end_effector_position()
                true_end_effector_positions.append(end_effector_pos)
        
        # 计算预测轨迹的末端执行器位置
        for j in range(predicted_sequence.size(0)):
            if torch.sum(torch.abs(predicted_sequence[j])) > 1e-6:  # 跳过填充的零
                # 设置关节角度
                for k, joint_id in enumerate(range(env.num_joints)):
                    p.resetJointState(env.robot_id, joint_id, predicted_sequence[j, k].item())
                
                # 获取末端执行器位置
                end_effector_pos = env._get_end_effector_position()
                pred_end_effector_positions.append(end_effector_pos)
        
        # 转换为numpy数组
        true_end_effector_positions = np.array(true_end_effector_positions)
        pred_end_effector_positions = np.array(pred_end_effector_positions)
        
        # 创建3D图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制真实轨迹
        if len(true_end_effector_positions) > 0:
            ax.plot(true_end_effector_positions[:, 0], true_end_effector_positions[:, 1], true_end_effector_positions[:, 2], 
                    'b-', linewidth=2, label='exper traj')
            ax.scatter(true_end_effector_positions[0, 0], true_end_effector_positions[0, 1], true_end_effector_positions[0, 2], 
                      c='g', marker='o', s=100, label='start')
            ax.scatter(true_end_effector_positions[-1, 0], true_end_effector_positions[-1, 1], true_end_effector_positions[-1, 2], 
                      c='r', marker='x', s=100, label='end')
        
        # 绘制预测轨迹
        if len(pred_end_effector_positions) > 0:
            ax.plot(pred_end_effector_positions[:, 0], pred_end_effector_positions[:, 1], pred_end_effector_positions[:, 2], 
                    'r--', linewidth=2, label='pred traj')
        
        # 绘制目标位置
        ax.scatter(target_position[0, 0].item(), target_position[0, 1].item(), target_position[0, 2].item(), 
                  c='m', marker='*', s=200, label='target location')
        
        # 设置图表属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'sample {i+1}: expert traj vs pred traj')
        ax.legend()
        
        # 保存图表
        plt.savefig(os.path.join(save_dir, f'trajectory_comparison_{i+1}.png'))
        plt.close()
    
    print(f"已保存 {num_samples} 个轨迹比较图到 {save_dir}")

def create_trajectory_animation(trajectory, filename, fps=10):
    """
    创建轨迹动画并保存为GIF
    
    Args:
        trajectory: 轨迹数据列表，每个元素包含关节角度、关节位置和目标位置
        filename: 保存的文件名
        fps: 帧率
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    
    # 设置标题和标签
    ax.set_title('arm traj')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 初始化绘图元素
    arm_line, = ax.plot([], [], [], 'bo-', linewidth=2, markersize=6)  # 机械臂
    target_point, = ax.plot([], [], [], 'ro', markersize=10)  # 目标点
    end_effector_trace, = ax.plot([], [], [], 'g-', linewidth=1)  # 末端执行器轨迹
    
    # 存储末端执行器位置历史
    end_effector_history = []
    
    def init():
        arm_line.set_data([], [])
        arm_line.set_3d_properties([])
        target_point.set_data([], [])
        target_point.set_3d_properties([])
        end_effector_trace.set_data([], [])
        end_effector_trace.set_3d_properties([])
        return arm_line, target_point, end_effector_trace
    
    def update(frame):
        # 获取当前帧的数据
        data = trajectory[frame]
        joint_positions = data['joint_positions']
        target_position = data['target_position']
        
        # 提取关节位置的坐标
        x = [pos[0] for pos in joint_positions]
        y = [pos[1] for pos in joint_positions]
        z = [pos[2] for pos in joint_positions]
        
        # 添加基座位置
        x.insert(0, 0)
        y.insert(0, 0)
        z.insert(0, 0)
        
        # 更新机械臂线条
        arm_line.set_data(x, y)
        arm_line.set_3d_properties(z)
        
        # 更新目标点
        target_point.set_data([target_position[0]], [target_position[1]])
        target_point.set_3d_properties([target_position[2]])
        
        # 更新末端执行器轨迹
        end_effector_history.append((x[-1], y[-1], z[-1]))
        trace_x = [p[0] for p in end_effector_history]
        trace_y = [p[1] for p in end_effector_history]
        trace_z = [p[2] for p in end_effector_history]
        end_effector_trace.set_data(trace_x, trace_y)
        end_effector_trace.set_3d_properties(trace_z)
        
        return arm_line, target_point, end_effector_trace
    
    # 创建动画
    ani = animation.FuncAnimation(
        fig, update, frames=len(trajectory),
        init_func=init, blit=True, interval=1000/fps
    )
    
    # 保存为GIF
    ani.save(filename, writer='pillow', fps=fps)
    plt.close(fig)

def generate_trajectory_with_model(model, env, joint_angles, joint_positions, target_position):
    """
    使用训练好的模型生成轨迹
    
    Args:
        model: 训练好的模型
        env: 机械臂环境
        joint_angles: 初始关节角度 [joint_dim]
        joint_positions: 初始关节位置 [joint_dim, position_dim]
        target_position: 目标位置 [position_dim]
    
    Returns:
        predicted_sequence: 预测的关节角度序列 [seq_len, joint_dim]
    """
    # 确保数据是张量
    if not isinstance(joint_angles, torch.Tensor):
        joint_angles = torch.tensor(np.array(joint_angles), dtype=torch.float32)
    if not isinstance(joint_positions, torch.Tensor):
        joint_positions = torch.tensor(np.array(joint_positions), dtype=torch.float32)
    if not isinstance(target_position, torch.Tensor):
        target_position = torch.tensor(np.array(target_position), dtype=torch.float32)
    
    # 添加批次维度
    joint_angles = joint_angles.unsqueeze(0)
    joint_positions = joint_positions.unsqueeze(0)
    target_position = target_position.unsqueeze(0)
    
    # 将模型设置为评估模式
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型预测 - 在推理时不使用目标关节角度
    with torch.no_grad():
        joint_angles = joint_angles.to(device)
        joint_positions = joint_positions.to(device)
        target_position = target_position.to(device)
        predicted_sequence = model(joint_angles, joint_positions, target_position, None)
        predicted_sequence = predicted_sequence.squeeze(0)  # 移除批次维度
    
    return predicted_sequence

def test_hdf5_training():
    """
    测试HDF5训练函数
    """
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BidirectionalSequenceMLPPathModel(
        joint_dim=7, position_dim=3, 
        hidden_dim=1024, seq_len=30
    )
    model.to(device)
    
    # HDF5文件路径
    h5_file_path = 'robot_trajectory_data.h5'
    
    if not os.path.exists(h5_file_path):
        print(f"HDF5文件 {h5_file_path} 不存在，请先创建数据文件")
        return
    
    # 使用HDF5训练
    trained_model, train_losses, test_losses = train_sequence_mlp_path_model_hdf5(
        model,
        h5_file_path,
        num_epochs=100,
        batch_size=256,  # 可以使用更大的batch_size
        lr=1e-4,
        weight_decay=1e-5,
        test_ratio=0.2,
        save_dir='model_results_hdf5'
    )
    
    print("HDF5训练完成！")
    return trained_model, train_losses, test_losses

if __name__ == "__main__":
    # 创建序列MLP模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BidirectionalSequenceMLPPathModel(
        joint_dim=7, position_dim=3, 
        hidden_dim=1024, seq_len=30)
    print(model)
    model.to(device)
    
    # 使用HDF5文件路径而不是加载整个数据集
    h5_data_path = r'C:/DiskD/trae_doc/robot_gym/result/robot_trajectory_data.h5'
    from robotEnv import collect_sequence_data3, RoboticArmEnv
    env = RoboticArmEnv(use_gui=False)
    dataset = collect_sequence_data3(env, 
                                    output_file=h5_data_path,
                                    num_trajectories=50000, 
                                    max_steps=30)
    # 训练模型（使用内存优化版本）
    trained_model, train_losses, test_losses = train_sequence_mlp_path_model_hdf5(
        model, 
        h5_data_path,
        num_epochs=1000, 
        batch_size=1024,  # 减小批次大小
        lr=5e-5, 
        weight_decay=5e-4,
        test_ratio=0.1,
        save_dir='model_results'
    )
    # 创建环境用于测试
    from robotEnv import RoboticArmEnv
    env = RoboticArmEnv(use_gui=False)
    
    # 加载训练好的模型
    model_path = r'C:\DiskD\trae_doc\robot_gym\model_results\best_model.pth'
    if os.path.exists(model_path):
        trained_model.load_state_dict(torch.load(model_path))
    else:
        print(f"模型文件 {model_path} 不存在，使用当前训练的模型")
    obs = env.reset()
    joint_angles = obs[:env.num_joints]
    joint_positions = obs[env.num_joints:-3].reshape(env.num_joints, 3)
    target_position = obs[-3:]
    # 生成轨迹
    predicted_sequence = generate_trajectory_with_model(
        trained_model, 
        env, 
        joint_angles, 
        joint_positions, 
        target_position
    )
    # 可视化模型预测（使用HDF5数据集的一个小样本）
    import time
    # 从HDF5文件中读取少量数据用于可视化
    sample_dataset = {'joint_angles': [], 'joint_positions': [], 'target_positions': [], 'sequence_joint_angles': []}
    with h5py.File(h5_data_path, 'r') as f:
        num_samples = min(5, len(f['joint_angles']))
        for i in range(num_samples):
            idx_str = str(i)
            sample_dataset['joint_angles'].append(torch.tensor(f['joint_angles'][idx_str][:], dtype=torch.float32))
            sample_dataset['joint_positions'].append(torch.tensor(f['joint_positions'][idx_str][:], dtype=torch.float32))
            sample_dataset['target_positions'].append(torch.tensor(f['target_positions'][idx_str][:], dtype=torch.float32))
            sample_dataset['sequence_joint_angles'].append(torch.tensor(f['sequence_joint_angles'][idx_str][:], dtype=torch.float32))
    
    visualize_model_predictions(trained_model, env, sample_dataset, num_samples=num_samples)
    # 可视化预测的轨迹
    for i in range(predicted_sequence.size(0)):
        if torch.sum(torch.abs(predicted_sequence[i])) > 1e-6:  # 跳过填充的零
            # 设置关节角度
            for j, joint_id in enumerate(range(env.num_joints)):
                p.resetJointState(env.robot_id, joint_id, predicted_sequence[i, j].item())
            
            # 可视化
            p.stepSimulation()
            time.sleep(0.05)
    
    # 关闭环境
    env.close()
    print("训练和测试完成！")
