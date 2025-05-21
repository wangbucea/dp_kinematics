import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
import os
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
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
        
        # 为整个动作序列生成概率分布
        self.num_bins = 5  # 每个动作序列的离散化数量
        # (batch_size, seq_len, num_bins, action_dim) 的形状
        self.output_layer = nn.Linear(hidden_dim, self.num_bins * self.action_dim)
        
        # 定义每个动作维度的取值范围（可以根据实际情况调整）
        self.action_ranges = [(-1.0, 1.0) for _ in range(action_dim)]
        
        # 用于自回归生成的起始token
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.max_seq_len = max_seq_len
    
    def _discretize_action(self, action):
        """将连续动作转换为离散索引"""
        batch_size = action.size(0)
        seq_len = action.size(1) if len(action.size()) > 2 else 1
        
        # 如果是单个时间步的动作，添加时间维度
        if len(action.size()) == 2:
            action = action.unsqueeze(1)
            
        # 初始化结果张量 - 现在我们为整个动作向量生成一个索引
        indices = torch.zeros(batch_size, seq_len, dtype=torch.long, device=action.device)
        
        # 将整个动作向量映射到离散索引
        # 这里我们需要一个映射策略，将7维动作空间映射到num_bins个离散值
        # 一种简单的方法是使用向量的范数或某种哈希函数
        
        # 这里使用一个简化的方法：将每个动作维度归一化后求和，然后离散化
        normalized_sum = torch.zeros(batch_size, seq_len, device=action.device)
        
        for dim in range(self.action_dim):
            min_val, max_val = self.action_ranges[dim]
            # 归一化每个维度
            normalized = (action[:, :, dim] - min_val) / (max_val - min_val)
            normalized_sum += normalized
        
        # 将归一化和映射到[0, num_bins-1]
        normalized_sum = normalized_sum / self.action_dim  # 确保在[0,1]范围内
        bin_indices = (normalized_sum * self.num_bins).clamp(0, self.num_bins - 1).long()
        indices = bin_indices
        
        return indices
    
    def _continuous_from_logits(self, logits, sample=True, temperature=2.0):
        """从logits生成连续动作值
        
        Args:
            logits: 形状为[batch_size, seq_len, num_bins * action_dim]的logits
            sample: 是否从分布中采样，如果为False则使用argmax
            temperature: 采样温度，较低的温度使分布更加尖锐
        """
        batch_size = logits.size(0)
        seq_len = logits.size(1)
        
        # 重塑logits以便于处理 - 改为 [batch_size, seq_len, num_bins, action_dim]
        logits = logits.reshape(batch_size, seq_len, self.num_bins, self.action_dim)
        
        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature
        
        # 计算每个bin的概率 - 在num_bins维度上应用softmax
        probs = F.softmax(logits, dim=2)  # 在num_bins维度上应用softmax
        
        # 初始化结果张量
        continuous_actions = torch.zeros(batch_size, seq_len, self.action_dim, device=logits.device)
        
        if sample:
            # 从分布中采样 - 为每个时间步采样一个bin索引
            # 首先将probs重塑为 [batch_size * seq_len, num_bins, action_dim]
            flat_probs = probs.reshape(-1, self.num_bins, self.action_dim)
            
            # 对每个动作维度单独采样
            sampled_actions = torch.zeros(batch_size * seq_len, self.action_dim, device=logits.device)
            
            for dim in range(self.action_dim):
                # 获取当前维度的概率分布
                dim_probs = flat_probs[:, :, dim]  # [batch_size * seq_len, num_bins]
                
                # 从分布中采样
                bin_indices = torch.multinomial(dim_probs, 1).squeeze(-1)  # [batch_size * seq_len]
                
                # 将离散索引转换回连续值
                min_val, max_val = self.action_ranges[dim]
                bin_width = (max_val - min_val) / self.num_bins
                continuous_values = min_val + (bin_indices.float() + 0.5) * bin_width
                
                sampled_actions[:, dim] = continuous_values
            
            # 重塑回原始形状
            continuous_actions = sampled_actions.reshape(batch_size, seq_len, self.action_dim)
        else:
            # 使用最高概率的bin - 为每个时间步和每个动作维度选择最高概率的bin
            for dim in range(self.action_dim):
                # 获取当前维度的概率分布
                dim_probs = probs[:, :, :, dim]  # [batch_size, seq_len, num_bins]
                
                # 选择最高概率的bin
                bin_indices = torch.argmax(dim_probs, dim=2)  # [batch_size, seq_len]
                
                # 将离散索引转换回连续值
                min_val, max_val = self.action_ranges[dim]
                bin_width = (max_val - min_val) / self.num_bins
                continuous_values = min_val + (bin_indices.float() + 0.5) * bin_width
                
                continuous_actions[:, :, dim] = continuous_values
        
        return continuous_actions, probs
    
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
    
    def forward(self, joint_angles, joint_positions, target_position, target_trajectory=None, sample=True, temperature=1.0):
        """
        自回归前向传播
        Args:
            joint_angles: 初始关节角度 [batch_size, 7]
            joint_positions: 初始关节位置 [batch_size, 7, 3]
            target_position: 目标位置 [batch_size, 3]
            target_trajectory: 用于训练时的目标轨迹 [batch_size, seq_len, 7]，如果为None则为推理模式
            sample: 是否从分布中采样动作，如果为False则使用最高概率的动作
            temperature: 采样温度，较低的温度使分布更加尖锐
        """
        batch_size = joint_angles.size(0)
        device = joint_angles.device
        
        # 准备编码器记忆
        memory = self._prepare_inputs(joint_angles, joint_positions, target_position)
        
        # 初始化解码器输入（起始token）
        start_tokens = self.start_token.repeat(batch_size, 1, 1)  # [batch_size, 1, hidden_dim]
        decoder_input = start_tokens
        
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
            
            # 预测动作logits
            action_logits = self.output_layer(decoder_output)
            
            # 将logits转换为连续动作和概率
            predicted_trajectory, action_probs = self._continuous_from_logits(
                action_logits, sample=sample, temperature=temperature
            )
            
            return predicted_trajectory, action_probs
        
        # 推理模式（自回归生成）
        else:
            outputs = []
            all_probs = []
            
            for i in range(self.max_seq_len):
                # 添加位置编码
                pos_decoder_input = self.pos_encoder(decoder_input)
                
                # 创建注意力掩码
                seq_len = decoder_input.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                
                # 解码当前序列
                decoder_output = self.transformer_decoder(pos_decoder_input, memory, tgt_mask=tgt_mask)
                
                # 预测下一个动作的logits
                next_action_logits = self.output_layer(decoder_output[:, -1:])  # [batch_size, 1, action_dim * num_bins]
                
                # 将logits转换为连续动作和概率
                next_action, next_probs = self._continuous_from_logits(
                    next_action_logits, sample=sample, temperature=temperature
                )
                
                outputs.append(next_action)
                all_probs.append(next_probs)
                
                # 为下一步准备输入 - 嵌入预测的动作
                next_embedding = self.action_embed(next_action.reshape(batch_size, -1)).unsqueeze(1)
                decoder_input = torch.cat([decoder_input, next_embedding], dim=1)
            
            # 合并所有预测的动作和概率
            trajectory = torch.cat(outputs, dim=1)  # [batch_size, max_seq_len, action_dim]
            action_probs = torch.cat(all_probs, dim=1)  # [batch_size, max_seq_len, action_dim, num_bins]
            
            return trajectory, action_probs

def train_transformer2(model, dataset, num_epochs=100, batch_size=64, lr=1e-4, test_ratio=0.1, save_dir='model_results'):
    """
    训练Transformer模型，并将数据分为训练集和测试集
    
    Args:
        model: 要训练的模型
        dataset: 数据集
        num_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        test_ratio: 测试集比例
        save_dir: 保存结果的目录
    """
    # 创建保存目录
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # 使用MSE损失计算连续动作的误差
    mse_criterion = nn.MSELoss()
    
    # 使用交叉熵损失计算离散动作分布的误差
    ce_criterion = nn.CrossEntropyLoss()
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import os
    import matplotlib.pyplot as plt
    
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
    
    # 划分训练集和测试集
    dataset_size = len(tensor_dataset)
    test_size = int(dataset_size * test_ratio)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(tensor_dataset, [train_size, test_size])
    
    print(f"数据集总大小: {dataset_size}, 训练集: {train_size}, 测试集: {test_size}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 用于记录训练和测试损失
    train_losses = []
    test_losses = []
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for joint_angles, joint_positions, target_positions, target_trajectories in tqdm(train_dataloader):
            joint_angles = joint_angles.to(device)
            joint_positions = joint_positions.to(device)
            target_positions = target_positions.to(device)
            target_trajectories = target_trajectories.to(device)
            
            # 前向传播（使用教师强制）
            pred_trajectories, action_probs = model(
                joint_angles, 
                joint_positions, 
                target_positions, 
                target_trajectory=target_trajectories,
                sample=True  # 训练时采样以增加多样性
            )
            
            # 计算MSE损失（连续动作的误差）
            mse_loss = mse_criterion(pred_trajectories, target_trajectories)
            
            # 计算离散动作的目标索引
            target_indices = model._discretize_action(target_trajectories)
            
            # 计算交叉熵损失（离散动作分布的误差）
            ce_loss = 0
            
            # 修改交叉熵损失计算 - 现在我们为每个时间步和每个动作维度计算损失
            for dim in range(model.action_dim):
                # 获取当前维度的概率分布和目标
                dim_probs = action_probs[:, :, :, dim]  # [batch_size, seq_len, num_bins]
                
                # 重塑为二维张量以适应交叉熵损失
                dim_probs = dim_probs.reshape(-1, model.num_bins)  # [batch_size * seq_len, num_bins]
                dim_targets = target_indices.reshape(-1)  # [batch_size * seq_len]
                
                # 计算交叉熵损失
                ce_loss += ce_criterion(dim_probs, dim_targets)
            
            # 总损失 = MSE损失 + 交叉熵损失
            loss = mse_loss + ce_loss / model.action_dim
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # 在测试集上评估模型
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for joint_angles, joint_positions, target_positions, target_trajectories in test_dataloader:
                joint_angles = joint_angles.to(device)
                joint_positions = joint_positions.to(device)
                target_positions = target_positions.to(device)
                target_trajectories = target_trajectories.to(device)
                
                # 前向传播（推理模式）
                pred_trajectories, _ = model(
                    joint_angles, 
                    joint_positions, 
                    target_positions, 
                    target_trajectory=None,  # 推理模式
                    sample=False  # 测试时使用最高概率的动作
                )
                
                # 计算MSE损失
                loss = mse_criterion(pred_trajectories, target_trajectories)
                test_loss += loss.item()
        
        # 计算平均测试损失
        avg_test_loss = test_loss / len(test_dataloader)
        test_losses.append(avg_test_loss)
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], 训练损失: {avg_train_loss:.4f}, 测试损失: {avg_test_loss:.4f}")
            
            # 添加验证步骤
            model.eval()
            with torch.no_grad():
                # 随机选择一个测试样本进行可视化
                sample_idx = np.random.randint(0, len(test_dataset))
                sample_joint_angles, sample_joint_positions, sample_target_positions, sample_target_trajectories = test_dataset[sample_idx]
                
                # 将样本移动到设备上
                sample_joint_angles = sample_joint_angles.unsqueeze(0).to(device)
                sample_joint_positions = sample_joint_positions.unsqueeze(0).to(device)
                sample_target_positions = sample_target_positions.unsqueeze(0).to(device)
                
                # 生成轨迹（推理模式）- 使用采样
                generated_trajectory_sampled, _ = model(
                    sample_joint_angles,
                    sample_joint_positions,
                    sample_target_positions,
                    target_trajectory=None,  # 推理模式
                    sample=True,
                    temperature=2  # 使用较低的温度使分布更加尖锐
                )
                
                # 生成轨迹（推理模式）- 使用最高概率
                generated_trajectory_argmax, _ = model(
                    sample_joint_angles,
                    sample_joint_positions,
                    sample_target_positions,
                    target_trajectory=None,  # 推理模式
                    sample=False
                )
                
                # 打印一些统计信息
                # print(f"生成轨迹形状: {generated_trajectory_sampled.shape}")
                # print(f"目标轨迹形状: {sample_target_trajectories.shape}")
                
                # 计算生成轨迹与目标轨迹的MSE
                sample_target_trajectories = sample_target_trajectories.unsqueeze(0).to(device)
                val_mse_sampled = F.mse_loss(generated_trajectory_sampled, sample_target_trajectories[:, :model.max_seq_len]).item()
                val_mse_argmax = F.mse_loss(generated_trajectory_argmax, sample_target_trajectories[:, :model.max_seq_len]).item()
                print(f"采样轨迹验证MSE: {val_mse_sampled:.4f}")
                print(f"最大概率轨迹验证MSE: {val_mse_argmax:.4f}")
    
    # 保存模型
    model_path = os.path.join(save_dir, 'transformer_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}")
    
    # 绘制并保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='训练损失')
    plt.plot(range(1, num_epochs+1), test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和测试损失曲线')
    plt.legend()
    plt.grid(True)
    
    loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    print(f"损失曲线已保存至 {loss_curve_path}")
    
    # 计算测试集上的最终MSE和到达目标点的准确率
    model.eval()
    total_mse = 0
    total_success = 0
    
    with torch.no_grad():
        for joint_angles, joint_positions, target_positions, target_trajectories in test_dataloader:
            joint_angles = joint_angles.to(device)
            joint_positions = joint_positions.to(device)
            target_positions = target_positions.to(device)
            target_trajectories = target_trajectories.to(device)
            
            # 前向传播（使用最高概率的动作）
            pred_trajectories, _ = model(
                joint_angles, 
                joint_positions, 
                target_positions, 
                target_trajectory=None,  # 推理模式
                sample=True,  # 使用最高概率的动作
                temperature=2  # 温度为1以获得最高概率的动作
            )
            
            # 计算MSE
            batch_mse = F.mse_loss(pred_trajectories, target_trajectories).item()
            total_mse += batch_mse * joint_angles.size(0)
            
            # 计算到达目标点的准确率
            last_pred_actions = pred_trajectories[:, -1, :]
            last_target_actions = target_trajectories[:, -1, :]
            
            # 计算最后一个动作的误差
            action_errors = torch.norm(last_pred_actions - last_target_actions, dim=1)
            
            # 如果误差小于阈值，则认为成功到达目标
            success_count = (action_errors < 0.1).sum().item()
            total_success += success_count
    
    # 计算平均MSE和准确率
    avg_mse = total_mse / len(test_dataset)
    accuracy = total_success / len(test_dataset) * 100
    
    print(f"测试集上的平均MSE: {avg_mse:.4f}")
    print(f"估计的目标点到达准确率: {accuracy:.2f}%")
    
    # 保存测试结果
    test_results = {
        'avg_mse': avg_mse,
        'accuracy': accuracy,
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    
    import pickle
    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f)
    
    return model, test_results
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
                    sample=True,
                    temperature=2  # 温度为1以获得最高概率的动作
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
                    new_predicted_trajectory, _ = transformer_model(
                        joint_angles, 
                        joint_positions, 
                        target_position, 
                        target_trajectory=None,
                        sample=False
                    )
                    new_predicted_trajectory = new_predicted_trajectory.squeeze(0).cpu().numpy()
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


def train_transformer_with_sequence(model, dataset, num_epochs=100, batch_size=64, lr=1e-4, test_ratio=0.1, save_dir='model_results',
                                   early_stop_patience=20, lr_scheduler_type='step', weight_decay=1e-5, 
                                   mse_weight=1.0, ce_weight=1.0, data_augmentation=False, save_best_only=True):
    """
    使用序列数据训练Transformer模型，并将数据分为训练集和测试集
    
    Args:
        model: 要训练的模型
        dataset: 使用collect_sequence_data收集的数据集
        num_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        test_ratio: 测试集比例
        save_dir: 保存结果的目录
        early_stop_patience: 早停耐心值，连续多少个epoch测试损失没有改善就停止训练
        lr_scheduler_type: 学习率调度器类型，可选 'step', 'cosine', 'plateau'
        weight_decay: 权重衰减系数，用于L2正则化
        mse_weight: MSE损失的权重
        ce_weight: 交叉熵损失的权重
        data_augmentation: 是否使用数据增强
        save_best_only: 是否只保存最佳模型
    """
    # 创建保存目录
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用Adam优化器，添加权重衰减
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 根据选择配置学习率调度器
    if lr_scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    elif lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif lr_scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    else:
        raise ValueError(f"不支持的学习率调度器类型: {lr_scheduler_type}")
    
    # 使用MSE损失计算连续动作的误差，设置reduction='none'以便应用时间权重
    mse_criterion = nn.MSELoss(reduction='none')
    
    # 使用交叉熵损失计算离散动作分布的误差，设置reduction='none'以便应用时间权重
    ce_criterion = nn.CrossEntropyLoss(reduction='none')
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import matplotlib.pyplot as plt
    
    # 获取模型的最大序列长度
    max_seq_len = model.max_seq_len
    
    # 处理数据集
    processed_data = {
        'initial_joint_angles': [],
        'initial_joint_positions': [],
        'target_positions': [],
        'trajectories': []
    }
    
    # 处理每个轨迹
    for i in range(len(dataset['trajectories'])):
        traj = dataset['trajectories'][i]
        seq_joint_angles = dataset['sequence_joint_angles'][i]
        seq_joint_positions = dataset['sequence_joint_positions'][i]
        seq_end_effector_positions = dataset['sequence_end_effector_positions'][i]
        target_position = dataset['target_positions'][i]
        
        traj_len = traj.size(0)
        
        if traj_len <= max_seq_len:
            # 如果轨迹长度小于等于max_seq_len，进行填充
            padding = torch.zeros(max_seq_len - traj_len, model.action_dim)
            processed_traj = torch.cat([traj, padding], dim=0)
            
            # 使用初始状态
            processed_data['initial_joint_angles'].append(seq_joint_angles[0])
            processed_data['initial_joint_positions'].append(seq_joint_positions[0])
            processed_data['target_positions'].append(target_position)
            processed_data['trajectories'].append(processed_traj)
        else:
            # 如果轨迹长度大于max_seq_len，从随机位置开始截取max_seq_len长度的序列
            start_idx = torch.randint(0, traj_len - max_seq_len + 1, (1,)).item()
            
            # 截取轨迹
            processed_traj = traj[start_idx:start_idx + max_seq_len]
            
            # 使用截取位置的状态作为初始状态
            processed_data['initial_joint_angles'].append(seq_joint_angles[start_idx])
            processed_data['initial_joint_positions'].append(seq_joint_positions[start_idx])
            processed_data['target_positions'].append(target_position)
            processed_data['trajectories'].append(processed_traj)
            
            # 数据增强：如果启用，再添加一个不同起点的样本
            if data_augmentation and traj_len > max_seq_len + 10:
                # 选择不同的起点
                another_start_idx = (start_idx + 10) % (traj_len - max_seq_len + 1)
                
                # 截取另一段轨迹
                another_processed_traj = traj[another_start_idx:another_start_idx + max_seq_len]
                
                # 添加到数据集
                processed_data['initial_joint_angles'].append(seq_joint_angles[another_start_idx])
                processed_data['initial_joint_positions'].append(seq_joint_positions[another_start_idx])
                processed_data['target_positions'].append(target_position)
                processed_data['trajectories'].append(another_processed_traj)
    
    # 转换为张量
    processed_data['initial_joint_angles'] = torch.stack(processed_data['initial_joint_angles'])
    processed_data['initial_joint_positions'] = torch.stack(processed_data['initial_joint_positions'])
    processed_data['target_positions'] = torch.stack(processed_data['target_positions'])
    processed_data['trajectories'] = torch.stack(processed_data['trajectories'])
    
    # 创建数据集
    tensor_dataset = TensorDataset(
        processed_data['initial_joint_angles'],
        processed_data['initial_joint_positions'],
        processed_data['target_positions'],
        processed_data['trajectories']
    )
    
    # 划分训练集和测试集
    dataset_size = len(tensor_dataset)
    test_size = int(dataset_size * test_ratio)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(tensor_dataset, [train_size, test_size])
    
    print(f"数据集总大小: {dataset_size}, 训练集: {train_size}, 测试集: {test_size}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 用于记录训练和测试损失
    train_losses = []
    test_losses = []
    
    # 早停相关变量
    best_test_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # 创建时间步权重 - 越靠前的时间步权重越大
    # 使用指数衰减权重：w_t = alpha^t，其中alpha是衰减因子（0<alpha<1）
    alpha = 0.9  # 衰减因子，可以根据需要调整
    time_weights = torch.tensor([alpha ** t for t in range(max_seq_len)], device=device)
    time_weights = time_weights / time_weights.sum()  # 归一化权重，确保总和为1
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for joint_angles, joint_positions, target_positions, target_trajectories in tqdm(train_dataloader):
            joint_angles = joint_angles.to(device)
            joint_positions = joint_positions.to(device)
            target_positions = target_positions.to(device)
            target_trajectories = target_trajectories.to(device)
            
            # 前向传播（使用教师强制）
            pred_trajectories, action_probs = model(
                joint_angles, 
                joint_positions, 
                target_positions, 
                target_trajectory=target_trajectories,
                sample=True,  # 训练时采样以增加多样性
                temperature=0.1  # 温度为0.1以获得更确定的动作
            )
            
            # 计算MSE损失（连续动作的误差）- 应用时间步权重
            mse_loss_raw = mse_criterion(pred_trajectories, target_trajectories)  # [batch_size, seq_len, action_dim]
            # 对每个时间步的损失应用权重
            mse_loss_weighted = mse_loss_raw.mean(dim=2)  # [batch_size, seq_len]
            mse_loss = (mse_loss_weighted * time_weights.unsqueeze(0)).sum(dim=1).mean()  # 先在时间维度上加权求和，再在batch维度上求平均
            
            # 计算离散动作的目标索引
            target_indices = model._discretize_action(target_trajectories)
            
            # 计算交叉熵损失（离散动作分布的误差）
            ce_loss = 0
            
            # 修改交叉熵损失计算 - 现在我们为每个时间步和每个动作维度计算损失，并应用时间步权重
            for dim in range(model.action_dim):
                # 获取当前维度的概率分布和目标
                dim_probs = action_probs[:, :, :, dim]  # [batch_size, seq_len, num_bins]
                
                # 重塑为二维张量以适应交叉熵损失
                batch_size = dim_probs.size(0)
                dim_probs_flat = dim_probs.reshape(-1, model.num_bins)  # [batch_size * seq_len, num_bins]
                dim_targets_flat = target_indices.reshape(-1)  # [batch_size * seq_len]
                
                # 计算交叉熵损失
                ce_loss_raw = ce_criterion(dim_probs_flat, dim_targets_flat)  # [batch_size * seq_len]
                ce_loss_raw = ce_loss_raw.reshape(batch_size, max_seq_len)  # [batch_size, seq_len]
                
                # 应用时间步权重
                ce_loss_weighted = (ce_loss_raw * time_weights.unsqueeze(0)).sum(dim=1).mean()  # 先在时间维度上加权求和，再在batch维度上求平均
                ce_loss += ce_loss_weighted
            
            # 总损失 = MSE损失权重 * MSE损失 + 交叉熵损失权重 * 交叉熵损失
            loss = mse_weight * mse_loss + ce_weight * ce_loss / model.action_dim
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # 在测试集上评估模型
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for joint_angles, joint_positions, target_positions, target_trajectories in test_dataloader:
                joint_angles = joint_angles.to(device)
                joint_positions = joint_positions.to(device)
                target_positions = target_positions.to(device)
                target_trajectories = target_trajectories.to(device)
                
                # 前向传播（推理模式）
                pred_trajectories, _ = model(
                    joint_angles, 
                    joint_positions, 
                    target_positions, 
                    target_trajectory=None,  # 推理模式
                    sample=True,
                    temperature=0.1  # 推理模式下，温度为0.1以获得更确定的动作
                )
                
                # 计算MSE损失 - 应用时间步权重
                mse_loss_raw = mse_criterion(pred_trajectories, target_trajectories)  # [batch_size, seq_len, action_dim]
                mse_loss_weighted = mse_loss_raw.mean(dim=2)  # [batch_size, seq_len]
                loss = (mse_loss_weighted * time_weights.unsqueeze(0)).sum(dim=1).mean()  # 先在时间维度上加权求和，再在batch维度上求平均
                
                test_loss += loss.item()
        
        # 计算平均测试损失
        avg_test_loss = test_loss / len(test_dataloader)
        test_losses.append(avg_test_loss)
        
        # 更新学习率（如果使用ReduceLROnPlateau）
        if lr_scheduler_type == 'plateau':
            scheduler.step(avg_test_loss)
        else:
            scheduler.step()
        
        # 早停检查
        # if avg_test_loss < best_test_loss:
        #     best_test_loss = avg_test_loss
        #     best_model_state = model.state_dict().copy()
        #     patience_counter = 0
            
        #     # 如果设置了只保存最佳模型，则立即保存
        #     if save_best_only:
        #         best_model_path = os.path.join(save_dir, 'best_transformer_sequence_model.pth')
        #         torch.save(best_model_state, best_model_path)
        #         print(f"保存最佳模型，测试损失: {best_test_loss:.4f}")
        # else:
        #     patience_counter += 1
        #     if patience_counter >= early_stop_patience:
        #         print(f"早停触发！连续{early_stop_patience}个epoch没有改善。")
        #         break
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], 训练损失: {avg_train_loss:.4f}, 测试损失: {avg_test_loss:.4f}, 最佳测试损失: {best_test_loss:.4f}")
            
            # 添加验证步骤
            model.eval()
            with torch.no_grad():
                # 随机选择一个测试样本进行可视化
                sample_idx = np.random.randint(0, len(test_dataset))
                sample_joint_angles, sample_joint_positions, sample_target_positions, sample_target_trajectories = test_dataset[sample_idx]
                
                # 将样本移动到设备上
                sample_joint_angles = sample_joint_angles.unsqueeze(0).to(device)
                sample_joint_positions = sample_joint_positions.unsqueeze(0).to(device)
                sample_target_positions = sample_target_positions.unsqueeze(0).to(device)
                
                # 生成轨迹（推理模式）- 使用采样
                generated_trajectory_sampled, _ = model(
                    sample_joint_angles,
                    sample_joint_positions,
                    sample_target_positions,
                    target_trajectory=None,  # 推理模式
                    sample=True,
                    temperature=0.1  # 使用较低的温度使分布更加尖锐
                )
                
                # 生成轨迹（推理模式）- 使用最高概率
                generated_trajectory_argmax, _ = model(
                    sample_joint_angles,
                    sample_joint_positions,
                    sample_target_positions,
                    target_trajectory=None,  # 推理模式
                    sample=True,
                    temperature=0.1
                )
                
                # 计算生成轨迹与目标轨迹的MSE
                sample_target_trajectories = sample_target_trajectories.unsqueeze(0).to(device)
                val_mse_sampled = F.mse_loss(generated_trajectory_sampled, sample_target_trajectories).item()
                val_mse_argmax = F.mse_loss(generated_trajectory_argmax, sample_target_trajectories).item()
                print(f"采样轨迹验证MSE: {val_mse_sampled:.4f}")
                print(f"最大概率轨迹验证MSE: {val_mse_argmax:.4f}")
                
                # 可视化生成的轨迹
                plt.figure(figsize=(15, 5))
                
                # 绘制第一个关节的轨迹
                plt.subplot(1, 3, 1)
                plt.plot(sample_target_trajectories[0, :, 4].cpu().numpy(), label='目标')
                plt.plot(generated_trajectory_sampled[0, :, 4].cpu().numpy(), label='生成(采样)')
                plt.plot(generated_trajectory_argmax[0, :, 4].cpu().numpy(), label='生成(最大概率)')
                plt.title('关节1轨迹')
                plt.legend()
                
                # 绘制第二个关节的轨迹
                plt.subplot(1, 3, 2)
                plt.plot(sample_target_trajectories[0, :, 5].cpu().numpy(), label='目标')
                plt.plot(generated_trajectory_sampled[0, :, 5].cpu().numpy(), label='生成(采样)')
                plt.plot(generated_trajectory_argmax[0, :, 5].cpu().numpy(), label='生成(最大概率)')
                plt.title('关节2轨迹')
                plt.legend()
                
                # 绘制第三个关节的轨迹
                plt.subplot(1, 3, 3)
                plt.plot(sample_target_trajectories[0, :, 6].cpu().numpy(), label='目标')
                plt.plot(generated_trajectory_sampled[0, :, 6].cpu().numpy(), label='生成(采样)')
                plt.plot(generated_trajectory_argmax[0, :, 6].cpu().numpy(), label='生成(最大概率)')
                plt.title('关节3轨迹')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'trajectory_epoch_{epoch+1}.png'))
                plt.close()
    
    # 恢复最佳模型状态
    if best_model_state is not None and not save_best_only:
        model.load_state_dict(best_model_state)
    
    # 保存最终模型（如果没有设置只保存最佳模型）
    if not save_best_only:
        final_model_path = os.path.join(save_dir, 'transformer_sequence_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"最终模型已保存至 {final_model_path}")
    
    # 绘制并保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='训练损失')
    plt.plot(range(1, len(test_losses)+1), test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和测试损失曲线')
    plt.legend()
    plt.grid(True)
    
    loss_curve_path = os.path.join(save_dir, 'sequence_loss_curve.png')
    plt.savefig(loss_curve_path)
    print(f"损失曲线已保存至 {loss_curve_path}")
    
    return model, train_losses, test_losses
    
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
        obs = env.reset(random_init=True)
        
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
                predicted_trajectory,_ = transformer_model(
                    joint_angles, 
                    joint_positions, 
                    target_position, 
                    target_trajectory=None,  # 推理模式
                    sample=True,
                    temperature=0.1  # 温度为1以获得最高概率的动作
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
                    new_predicted_trajectory, _ = transformer_model(
                        joint_angles, 
                        joint_positions, 
                        target_position, 
                        target_trajectory=None,
                        sample=True,
                        temperature=0.1,
                    )

                    new_predicted_trajectory = new_predicted_trajectory.squeeze(0).cpu().numpy()
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
