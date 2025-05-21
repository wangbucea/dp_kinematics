import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
from tqdm import tqdm
import pybullet as p
import pybullet_data

class sinLU(nn.Module):
    """
     sinLU
    """
    def __init__(self):
        super(sinLU, self).__init__()
    def forward(self, x):
        return (x + torch.sin(x))*torch.sigmoid(x)

class MLPEncoder(nn.Module):
    """
    编码器：将机械臂状态和目标位置编码为隐藏表示
    参考MPNet中的Encoder结构
    """
    
    def __init__(self, input_size, output_size):
        super(MLPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512), nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(512, 384), nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(384, 256), nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(256, output_size)
        )
    def forward(self, x):
        return self.encoder(x)

class RotationMLP(nn.Module):
    """旋转矩阵生成模块：学习sin/cos的多项式近似"""
    def __init__(self, input_dim, hidden_size=64, degree=3):
        super().__init__()
        self.degree = degree
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.PReLU(),
            nn.Linear(hidden_size//2, degree*2)  # 输出sin和cos的系数
        )
        self.A = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # 初始化 a
        self.B = nn.Parameter(torch.tensor(0.0), requires_grad=True)  # 初始化 b
        self.A2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # 初始化 a
        self.B2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)  # 初始化 b
    def forward(self, theta):
        # theta形状: (batch_size, 1)
        B = theta.shape[0]
        coeffs = self.net(theta)  # (batch, degree*2)
        sin_coeffs = coeffs[..., :self.degree]
        cos_coeffs = coeffs[..., self.degree:]
        
        # 构建多项式近似
        # angles = theta.unsqueeze(-1)  # (batch, 1, 1)
        # powers = torch.arange(1, 2*self.degree+1, 2, device=theta.device).float()  # 奇次幂
        # # 计算多项式项：x^1, x^3, x^5...
        # # 通过sin(ax+b) 来近似
        # terms = angles.pow(powers)  # (batch, 1, degree)

        # 组合sin和cos近似
        # sin_approx = (sin_coeffs.unsqueeze(1) * terms).sum(dim=-1)  # (batch, 1)
        # cos_approx = (cos_coeffs.unsqueeze(1) * terms).sum(dim=-1)  # (batch, 1)
        sin_approx = self.A*sin_coeffs+self.B
        cos_approx = self.A2*cos_coeffs+self.B2
        # 构建2x2旋转矩阵
        rot_matrix = torch.cat([cos_approx, -sin_approx, sin_approx, cos_approx], dim=-1)
        
        return rot_matrix.view(B, -1)  # (batch, 4)

class sinLU(nn.Module):
    """
     sinLU
    """
    def __init__(self):
        super(sinLU, self).__init__()
    def forward(self, x):
        return (x + torch.sin(x))*torch.sigmoid(x)
class TranslationMLP(nn.Module):
    """平移向量生成模块：学习连杆参数到位移的映射"""
    def __init__(self, input_dim, hidden_size=64):
        super().__init__()
        self.ac = sinLU()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            self.ac,
            nn.Linear(hidden_size, hidden_size),
            self.ac,
            nn.Linear(hidden_size, 3)  # 输出dx, dy, dz
        )

    def forward(self, link_params):
        # link_params形状: (batch, 3) [d, a, alpha]
        B = link_params.shape[0]
        # 直接将link_params作为输入
        link_params = self.net(link_params)  # Flatten the lin
        link_params = link_params.view(B, -1)
        return link_params  # (batch, 3)

class MLPPathGenerator(nn.Module):
    """
    路径生成器：生成机械臂的运动轨迹
    """
    def __init__(self, input_size, output_size, hidden_sizes=[1280, 896, 512, 384, 256, 128, 64, 32]):
        super(MLPPathGenerator, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
   
class SequenceMLPPathModel(nn.Module):
    """
    序列编码的路径规划模型：能够处理和生成整个动作序列
    """
    def __init__(self, 
                 joint_dim=7,           # 关节角度维度
                 position_dim=3,        # 位置维度
                 hidden_dim=128,        # 隐藏层维度
                 seq_len=30,            # 生成轨迹的长度
                 dropout=0.1):          # Dropout比率
        super(SequenceMLPPathModel, self).__init__()
        
        # 计算输入维度：关节角度 + 关节位置 + 目标位置
        self.input_dim = joint_dim + joint_dim * position_dim + position_dim
        self.joint_dim = joint_dim
        self.position_dim = position_dim
        self.seq_len = seq_len
        
        # 编码器：将状态编码为隐藏表示
        self.encoder = MLPEncoder(self.input_dim, hidden_dim)
        # 运动学注意力
        self.RotationAttention = RotationMLP(hidden_dim)
        self.TranslationAttention = TranslationMLP(hidden_dim)

        self.upDim_RotationAttention = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        
        self.upDim_TranslationAttention = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        # 序列生成器：一次性生成整个序列的所有时间步
        # 输入是编码器输出
        self.sequence_generator = MLPPathGenerator(hidden_dim, joint_dim * seq_len)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _prepare_inputs(self, joint_angles, joint_positions, target_position):
        """准备编码器的输入"""
        batch_size = joint_angles.size(0)
        
        # 合并输入特征
        joint_positions_flat = joint_positions.reshape(batch_size, -1)
        inputs = torch.cat([joint_angles, joint_positions_flat, target_position], dim=1)
        
        return inputs
    
    def forward(self, joint_angles, joint_positions, target_position):
        """
        前向传播 - 一次性生成整个轨迹序列
        Args:
            joint_angles: 初始关节角度 [batch_size, joint_dim]
            joint_positions: 初始关节位置 [batch_size, joint_dim, position_dim]
            target_position: 目标位置 [batch_size, position_dim]
        Returns:
            predicted_trajectory: 预测的轨迹 [batch_size, seq_len, joint_dim]
        """
        batch_size = joint_angles.size(0)
        
        # 准备输入并编码
        inputs = self._prepare_inputs(joint_angles, joint_positions, target_position)
        encoded_state = self.encoder(inputs)  #  [B, 256]
        # 运动学注意力
        rotation_matrix = self.RotationAttention(encoded_state)
        rotation_matrix = self.upDim_RotationAttention(rotation_matrix)
        translation_vector = self.TranslationAttention(encoded_state)
        translation_vector = self.upDim_TranslationAttention(translation_vector)
        encoded_state = encoded_state+rotation_matrix+translation_vector

        encoded_state = self.dropout(encoded_state)
        
        # 生成整个轨迹序列
        trajectory_flat = self.sequence_generator(encoded_state)
        
        # 重塑为 [batch_size, seq_len, joint_dim]
        predicted_trajectory = trajectory_flat.view(batch_size, self.seq_len, self.joint_dim)
        
        return predicted_trajectory

def train_sequence_mlp_path_model(model, dataset, num_epochs=100, batch_size=64, lr=1e-4, 
                                 weight_decay=1e-5, test_ratio=0.1, save_dir='model_results'):
    """
    训练序列MLP路径模型
    
    Args:
        model: SequenceMLPPathModel实例
        dataset: 包含训练数据的字典，格式为：
                {
                    'initial_joint_angles': tensor [num_samples, joint_dim],
                    'initial_joint_positions': tensor [num_samples, joint_dim, position_dim],
                    'target_positions': tensor [num_samples, position_dim],
                    'trajectories': tensor [num_samples, seq_len, joint_dim]
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
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
        'trajectories': []
    }
    
    # 处理每个轨迹
    for i in range(len(dataset['trajectories'])):
        traj = dataset['trajectories'][i]
        seq_joint_angles = dataset['sequence_joint_angles'][i] if 'sequence_joint_angles' in dataset else None
        seq_joint_positions = dataset['sequence_joint_positions'][i] if 'sequence_joint_positions' in dataset else None
        target_position = dataset['target_positions'][i]
        
        traj_len = traj.size(0)
        
        if traj_len <= max_seq_len:
            # 如果轨迹长度小于等于max_seq_len，进行填充
            padding = torch.zeros(max_seq_len - traj_len, model.joint_dim)
            processed_traj = torch.cat([traj, padding], dim=0)
            
            # 使用初始状态
            if seq_joint_angles is not None and seq_joint_positions is not None:
                processed_data['initial_joint_angles'].append(seq_joint_angles[0])
                processed_data['initial_joint_positions'].append(seq_joint_positions[0])
            else:
                processed_data['initial_joint_angles'].append(dataset['initial_joint_angles'][i])
                processed_data['initial_joint_positions'].append(dataset['initial_joint_positions'][i])
                
            processed_data['target_positions'].append(target_position)
            processed_data['trajectories'].append(processed_traj)
        else:
            # 如果轨迹长度大于max_seq_len，从随机位置开始截取max_seq_len长度的序列
            start_idx = torch.randint(0, traj_len - max_seq_len + 1, (1,)).item()
            
            # 截取轨迹
            processed_traj = traj[start_idx:start_idx + max_seq_len]
            
            # 使用截取位置的状态作为初始状态
            if seq_joint_angles is not None and seq_joint_positions is not None:
                processed_data['initial_joint_angles'].append(seq_joint_angles[start_idx])
                processed_data['initial_joint_positions'].append(seq_joint_positions[start_idx])
            else:
                # 如果没有序列状态数据，使用初始状态
                processed_data['initial_joint_angles'].append(dataset['initial_joint_angles'][i])
                processed_data['initial_joint_positions'].append(dataset['initial_joint_positions'][i])
                
            processed_data['target_positions'].append(target_position)
            processed_data['trajectories'].append(processed_traj)
    
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
    early_stop_patience = 20
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for joint_angles, joint_positions, target_positions, target_trajectories in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            joint_angles = joint_angles.to(device)
            joint_positions = joint_positions.to(device)
            target_positions = target_positions.to(device)
            target_trajectories = target_trajectories.to(device)
            
            # 前向传播
            pred_trajectories = model(joint_angles, joint_positions, target_positions)
            
            # 计算损失
            loss = criterion(pred_trajectories, target_trajectories)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
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
                
                # 前向传播
                pred_trajectories = model(joint_angles, joint_positions, target_positions)
                
                # 计算损失
                loss = criterion(pred_trajectories, target_trajectories)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_dataloader)
        test_losses.append(avg_test_loss)
        
        # 更新学习率
        scheduler.step(avg_test_loss)
        
        # 早停检查
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(save_dir, 'best_sequence_mlp_model.pth')
            torch.save(best_model_state, best_model_path)
            print(f"保存最佳模型，测试损失: {best_test_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"早停触发！连续{early_stop_patience}个epoch没有改善。")
                break
        
        # 打印训练进度
        print(f"Epoch [{epoch+1}/{num_epochs}], 训练损失: {avg_train_loss:.6f}, 测试损失: {avg_test_loss:.6f}")
        
        # 每10个epoch可视化一次
        if (epoch + 1) % 10 == 0:
            # 可视化训练进度
            model.eval()
            with torch.no_grad():
                # 随机选择一个测试样本进行可视化
                sample_idx = np.random.randint(0, len(test_dataset))
                sample_joint_angles, sample_joint_positions, sample_target_positions, sample_target_trajectories = test_dataset[sample_idx]
                
                # 将样本移动到设备上
                sample_joint_angles = sample_joint_angles.unsqueeze(0).to(device)
                sample_joint_positions = sample_joint_positions.unsqueeze(0).to(device)
                sample_target_positions = sample_target_positions.unsqueeze(0).to(device)
                
                # 生成轨迹
                generated_trajectory = model(
                    sample_joint_angles,
                    sample_joint_positions,
                    sample_target_positions
                )
                
                # 可视化生成的轨迹
                plt.figure(figsize=(15, 5))
                
                # 绘制第一个关节的轨迹
                plt.subplot(1, 3, 1)
                plt.plot(sample_target_trajectories[:, 0].cpu().numpy(), label='target')
                plt.plot(generated_trajectory[0, :, 0].cpu().numpy(), label='generated')
                plt.title('arm 1 traj')
                plt.legend()
                
                # 绘制第二个关节的轨迹
                plt.subplot(1, 3, 2)
                plt.plot(sample_target_trajectories[:, 1].cpu().numpy(), label='target')
                plt.plot(generated_trajectory[0, :, 1].cpu().numpy(), label='generated')
                plt.title('arm 2 traj')
                plt.legend()
                
                # 绘制第三个关节的轨迹
                plt.subplot(1, 3, 3)
                plt.plot(sample_target_trajectories[:, 2].cpu().numpy(), label='target')
                plt.plot(generated_trajectory[0, :, 2].cpu().numpy(), label='generated')
                plt.title('arm 3 traj')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'sequence_mlp_trajectory_epoch_{epoch+1}.png'))
                plt.close()
    
    # 恢复最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 保存最终模型
    final_path = os.path.join(save_dir, 'sequence_mlp_model_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"最终模型已保存至 {final_path}")
    
    # 绘制并保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='train loss')
    plt.plot(range(1, len(test_losses)+1), test_losses, label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('train and test loss ')
    plt.legend()
    plt.grid(True)
    
    loss_curve_path = os.path.join(save_dir, 'sequence_mlp_loss_curve.png')
    plt.savefig(loss_curve_path)
    print(f"损失曲线已保存至 {loss_curve_path}")
    
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
    def create_weights(length, decay_factor=0.9):
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
        
        # 计算要执行的步数 - 最多执行预测的前10步
        steps_to_execute = min(10, len(predicted_trajectory))
        
        # 创建权重 - 前面的动作权重更大
        weights = create_weights(steps_to_execute)
        
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
                for j in range(min(5, steps_to_execute - i)):
                    action += weights[j] * predicted_trajectory[i + j]
            
            # 确保动作在合理范围内
            action = np.clip(action, -1.0, 1.0)
            
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
    obs = env.reset(random_init=True)
    
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

def test_model_visualization():
    """
    测试模型可视化函数
    """
    from robotEnv import RoboticArmEnv
    import torch
    import os
    import pybullet as p
    
    # 检查是否已有连接
    try:
        # 尝试断开现有连接
        p.disconnect()
    except:
        pass
    
    # 创建环境 - 使用非GUI模式
    env = RoboticArmEnv(use_gui=True)  # 使用非GUI模式
    
    # 加载模型
    model_path = os.path.join('model_results', 'best_sequence_mlp_model.pth')
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请先训练模型")
        env.close()  # 确保关闭环境
        return
    
    # 创建模型实例
    model = SequenceMLPPathModel(joint_dim=7, position_dim=3, hidden_dim=256, seq_len=30)
    
    # 加载模型参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    print("模型加载成功，开始生成轨迹...")
    
    try:
        # 可视化模型生成的轨迹
        visualize_model_trajectory(model, env, output_file="mlp_model_trajectory.gif", 
                                fps=10, max_steps=10000)
        print("轨迹生成完成！")
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


if __name__ == "__main__":
    # 创建序列MLP模型
    model = SequenceMLPPathModel(joint_dim=7, position_dim=3, hidden_dim=256, seq_len=30)
    
    # 加载数据集
    from robotEnv import RoboticArmEnv, collect_sequence_data
    env = RoboticArmEnv()
    dataset = collect_sequence_data(env, num_trajectories=1, max_steps=200)
    
    # 训练模型
    # model, train_losses, test_losses = train_sequence_mlp_path_model(
    #     model, 
    #     dataset, 
    #     num_epochs=1000, 
    #     batch_size=64, 
    #     lr=1e-5, 
    #     weight_decay=5e-4,
    #     test_ratio=0.1,
    #     save_dir='model_results'
    # )
    
    # 生成轨迹示例
    sample_idx = np.random.randint(0, len(dataset['initial_joint_angles']))
    initial_joint_angles = dataset['initial_joint_angles'][sample_idx]
    initial_joint_positions = dataset['initial_joint_positions'][sample_idx]
    target_position = dataset['target_positions'][sample_idx]
    
    trajectory = generate_sequence_trajectory(model, initial_joint_angles, initial_joint_positions, target_position)
    test_model_visualization()
    # 可视化轨迹
