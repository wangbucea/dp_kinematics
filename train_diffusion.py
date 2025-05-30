import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from diffusion_action_model import DiffusionActionModel, train_diffusion_model
from mlpPathModel import BidirectionalSequenceMLPPathModel
from robotEnv import RoboticArmEnv
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建扩散模型
diffusion_model = DiffusionActionModel(
    action_dim=7,
    seq_len=30,
    joint_dim=7,
    position_dim=3,
    hidden_dims=[128, 256, 512, 1024],
    condition_dim=512,
    time_emb_dim=128,
    num_heads=8,
    dropout=0.1,
    num_timesteps=1000
)

# 主训练流程
if __name__ == "__main__":
    # 创建环境
    env = RoboticArmEnv(use_gui=False)
    
    # 准备训练数据
    print("准备训练数据...")
    h5_data_path = r'C:/DiskD/trae_doc/robot_gym/result/robot_trajectory_data.h5'
    dataset = {
        'joint_angles': [],
        'joint_positions': [],
        'target_positions': [],
        'trajectories': [],
        'sequence_joint_angles': [],
        'sequence_joint_positions': [],
    }
    print("加载数据!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    with h5py.File(h5_data_path, 'r') as h5_file:
        for key in dataset.keys():
            for idx in tqdm(h5_file[key].keys()):
                data = h5_file[key][idx][:]
                dataset[key].append(data)
    print("加载数据完成！ 转换数据格式！！！！！！！！")
    for key in tqdm(dataset.keys()):
        dataset[key] = np.array(dataset[key])
    print("转换数据格式完成！！！！！！！！开始训练！！！！！！！！")
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
    max_seq_len = 30  # 假设最大序列长度为30
    batch_size = 1024  # 批次大小
    test_ratio = 0.1
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 训练扩散模型
    print("开始训练扩散模型...")
    trained_diffusion_model, train_losses, val_losses = train_diffusion_model(
        diffusion_model, 
        train_loader,  # 使用训练数据加载器
        val_dataloader=test_loader,  # 使用测试数据加载器作为验证集
        num_epochs=2000,
        lr=1e-5, 
        device=device,
        patience=10,  # 早停耐心值：100轮验证损失不改善就停止
        min_delta=1e-6,  # 最小改善阈值
        save_path='model_results/best_diffusion_model.pth'  # 最佳模型保存路径
    )
    
    # 保存最终模型和训练历史
    torch.save(trained_diffusion_model.state_dict(), 'model_results/final_diffusion_model.pth')
    
    # 保存训练历史
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    if val_losses:
        plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if val_losses:
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('轮数')
        plt.ylabel('验证损失')
        plt.title('验证损失曲线')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_results/diffusion_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("扩散模型训练完成并保存")
    
    # 创建混合模型
    # hybrid_model = HybridDiffusionMLPModel(mlp_model, trained_diffusion_model)
    
    # 测试生成效果
    print("测试生成效果...")
    test_obs = env.reset()
    test_joint_angles = torch.tensor(test_obs[:7], dtype=torch.float32).unsqueeze(0).to(device)
    test_joint_positions = torch.tensor(test_obs[7:-3], dtype=torch.float32).reshape(1, 7, 3).to(device)
    test_target_position = torch.tensor(test_obs[-3:], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 纯扩散模型生成
        diffusion_trajectory = trained_diffusion_model(test_joint_angles, test_joint_positions, test_target_position)
        
        # 混合模型生成
        # hybrid_trajectory = hybrid_model(test_joint_angles, test_joint_positions, test_target_position)
        
        print(f"扩散模型轨迹形状: {diffusion_trajectory.shape}")
        # print(f"混合模型轨迹形状: {hybrid_trajectory.shape}")
    
    print("训练和测试完成！")
