import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
from robotEnv import RoboticArmEnv, collect_data
from transformerModel2 import TrajectoryTransformer2, collect_data_with_transformer
import pybullet as p

def visualize_trajectory_static(trajectory, ax=None, title="机械臂轨迹"):
    """
    静态可视化单条轨迹
    
    参数:
        trajectory: 轨迹数据，包含关节位置
        ax: matplotlib 3D轴对象，如果为None则创建新的
        title: 图表标题
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # 提取末端执行器位置
    end_effector_positions = []
    for step in trajectory:
        # 假设joint_positions的最后一个是末端执行器
        if isinstance(step, dict) and 'joint_positions' in step:
            end_effector_positions.append(step['joint_positions'][-1])
        elif isinstance(step, np.ndarray):
            # 如果是数组，假设最后3个元素是末端执行器位置
            end_effector_positions.append(step[-3:])
    
    end_effector_positions = np.array(end_effector_positions)
    
    # 绘制轨迹
    if len(end_effector_positions) > 0:
        ax.plot(end_effector_positions[:, 0], 
                end_effector_positions[:, 1], 
                end_effector_positions[:, 2], 'b-', linewidth=2)
        
        # 标记起点和终点
        ax.scatter(end_effector_positions[0, 0], 
                  end_effector_positions[0, 1], 
                  end_effector_positions[0, 2], 
                  color='green', s=100, label='起点')
        
        ax.scatter(end_effector_positions[-1, 0], 
                  end_effector_positions[-1, 1], 
                  end_effector_positions[-1, 2], 
                  color='red', s=100, label='终点')
    
    # 设置图表属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    return ax

def visualize_arm_step(joint_positions, ax=None, target_position=None):
    """
    可视化机械臂的单个姿态
    
    参数:
        joint_positions: 关节位置数组
        ax: matplotlib 3D轴对象
        target_position: 目标位置（可选）
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # 清除当前轴
    ax.clear()
    
    # 绘制机械臂连杆
    x = joint_positions[:, 0]
    y = joint_positions[:, 1]
    z = joint_positions[:, 2]
    
    ax.plot(x, y, z, 'bo-', linewidth=2, markersize=6)
    
    # 如果有目标位置，绘制目标
    if target_position is not None:
        ax.scatter(target_position[0], target_position[1], target_position[2], 
                  color='red', s=100, label='目标位置')
    
    # 设置图表属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('机械臂姿态')
    
    # 设置坐标轴范围
    max_range = max(
        np.max(np.abs(x)),
        np.max(np.abs(y)),
        np.max(np.abs(z))
    ) * 1.5
    
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    if target_position is not None:
        ax.legend()
    
    return ax

def create_trajectory_animation(trajectory, output_file="trajectory_animation.gif", fps=10):
    """
    创建机械臂轨迹的动画并保存为GIF文件
    
    参数:
        trajectory: 轨迹数据列表，每个元素是包含关节位置的字典
        output_file: 输出文件路径
        fps: 每秒帧数
    """
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
                  color='red', s=150, marker='*', label='目标位置')
    
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
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('强化学习机械臂轨迹动画')
    
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

def compare_trajectories(traj1, traj2, title1="传统方法轨迹", title2="Transformer方法轨迹"):
    """
    比较两种方法生成的轨迹
    
    参数:
        traj1, traj2: 两种方法的轨迹数据
        title1, title2: 两个轨迹的标题
    """
    fig = plt.figure(figsize=(15, 7))
    
    # 第一个轨迹
    ax1 = fig.add_subplot(121, projection='3d')
    visualize_trajectory_static(traj1, ax1, title1)
    
    # 第二个轨迹
    ax2 = fig.add_subplot(122, projection='3d')
    visualize_trajectory_static(traj2, ax2, title2)
    
    plt.tight_layout()
    plt.savefig("trajectory_comparison.png")
    plt.show()

# def collect_and_visualize():
#     """
#     收集并可视化两种方法的轨迹数据
#     """
#     # 创建环境
#     env = RoboticArmEnv()
    
#     # 收集传统方法的轨迹
#     print("收集传统方法的轨迹数据...")
#     traditional_dataset = collect_data(env, num_trajectories=5, max_steps=200)
    
#     # 为传统方法创建轨迹可视化
#     print("创建传统方法轨迹动画...")
#     for i in range(min(3, len(traditional_dataset['trajectories']))):
#         # 构建完整轨迹
#         traj = []
        
#         # 初始关节位置
#         joint_angles = traditional_dataset['joint_angles'][i].numpy()
#         joint_positions = traditional_dataset['joint_positions'][i].numpy()
#         target_position = traditional_dataset['target_positions'][i].numpy()
        
#         # 初始状态
#         traj.append({
#             'joint_angles': joint_angles,
#             'joint_positions': joint_positions,
#             'target_position': target_position
#         })
        
#         # 执行轨迹中的每一步
#         current_angles = joint_angles.copy()
        
#         # 计算初始距离
#         initial_pos = joint_positions[-1]  # 末端执行器位置
#         initial_distance = np.linalg.norm(target_position - initial_pos)
        
#         # 记录最终距离
#         final_distance = initial_distance
        
#         # 注意这里的修改，适应新的数据结构
#         for action in traditional_dataset['trajectories'][i]:
#             action = action.numpy()
#             current_angles = current_angles + action
            
#             # 使用环境计算新的关节位置
#             env.reset()
#             for j, angle in enumerate(current_angles):
#                 if j < env.num_joints:
#                     p.resetJointState(env.robot_id, j, angle)
            
#             # 获取关节位置
#             joint_positions = []
#             for joint_id in range(env.num_joints):
#                 joint_info = p.getLinkState(env.robot_id, joint_id)
#                 joint_positions.append(joint_info[0])
#             joint_positions = np.array(joint_positions)
            
#             # 更新最终距离
#             final_distance = np.linalg.norm(target_position - joint_positions[-1])
            
#             traj.append({
#                 'joint_angles': current_angles,
#                 'joint_positions': joint_positions,
#                 'target_position': target_position
#             })
        
#         # 打印轨迹信息
#         print(f"轨迹 {i+1}:")
#         print(f"  初始距离: {initial_distance:.4f}")
#         print(f"  最终距离: {final_distance:.4f}")
#         print(f"  改进率: {(initial_distance - final_distance) / initial_distance * 100:.2f}%")
#         print(f"  步数: {len(traditional_dataset['trajectories'][i])}")
        
#         # 创建动画
#         create_trajectory_animation(traj, f"traditional_trajectory_{i+1}.gif")
    
#     env.close()
#     print("可视化完成！")

def collect_and_visualize():
    """
    收集并可视化两种方法的轨迹数据
    """
    # 创建环境
    env = RoboticArmEnv()
    
    # 加载Transformer模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 7 + 7*3 + 3  # 关节角度 + 关节位置 + 目标位置
    transformer_model = TrajectoryTransformer2(input_dim=state_dim).to(device)
    
    # 检查模型文件是否存在
    model_path = r'C:\DiskD\trae_doc\robot_gym\transformer_model.pth'
    if os.path.exists(model_path):
        transformer_model.load_state_dict(torch.load(model_path, map_location=device))
        print("Transformer模型加载成功")
    else:
        print("未找到Transformer模型，请先训练模型")
        return
    
    # 收集传统方法的轨迹
    print("收集传统方法的轨迹数据...")
    traditional_dataset = collect_data(env, num_trajectories=5, max_steps=200)
    
    # 收集Transformer方法的轨迹
    print("收集Transformer方法的轨迹数据...")
    transformer_dataset = collect_data_with_transformer(
            env, 
            transformer_model, 
            num_trajectories=5, 
            max_steps=200,
            use_rolling_prediction=True, 
            decay_factor=0.9
        )
    
    # 为传统方法创建轨迹可视化
    print("创建传统方法轨迹动画...")
    for i in range(min(3, len(traditional_dataset['trajectories']))):
        # 构建完整轨迹
        traj = []
        
        # 初始关节位置
        joint_angles = traditional_dataset['joint_angles'][i].numpy()
        joint_positions = traditional_dataset['joint_positions'][i].numpy()
        target_position = traditional_dataset['target_positions'][i].numpy()
        
        # 初始状态
        traj.append({
            'joint_angles': joint_angles,
            'joint_positions': joint_positions,
            'target_position': target_position
        })
        
        # 执行轨迹中的每一步
        current_angles = joint_angles.copy()
        for action in traditional_dataset['trajectories'][i]:
            action = action.numpy()
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
        create_trajectory_animation(traj, f"traditional_trajectory_{i+1}.gif")
    
    # 为Transformer方法创建轨迹可视化
    print("创建Transformer方法轨迹动画...")
    for i in range(min(3, len(transformer_dataset['observations']))):
        # 构建完整轨迹
        traj = []
        
        # 初始观测
        obs = transformer_dataset['observations'][i]
        joint_angles = obs[:7]
        joint_positions_flat = obs[7:-3]
        joint_positions = joint_positions_flat.reshape(7, 3)
        target_position = obs[-3:]
        
        # 初始状态
        traj.append({
            'joint_angles': joint_angles,
            'joint_positions': joint_positions,
            'target_position': target_position
        })
        
        # 执行轨迹中的每一步
        current_obs = obs
        for j in range(len(transformer_dataset['actions'])):
            if i+j >= len(transformer_dataset['observations']):
                break
                
            action = transformer_dataset['actions'][i+j]
            next_obs = transformer_dataset['next_observations'][i+j]
            
            # 提取关节位置
            next_joint_angles = next_obs[:7]
            next_joint_positions_flat = next_obs[7:-3]
            next_joint_positions = next_joint_positions_flat.reshape(7, 3)
            
            traj.append({
                'joint_angles': next_joint_angles,
                'joint_positions': next_joint_positions,
                'target_position': target_position
            })
            
            # 检查是否完成或到达另一条轨迹
            if transformer_dataset['dones'][i+j] or j > 200:
                break
        
        # 创建动画
        create_trajectory_animation(traj, f"transformer_trajectory_{i+1}.gif")
    
    # 比较两种方法的轨迹
    if len(traditional_dataset['trajectories']) > 0 and len(transformer_dataset['observations']) > 0:
        # 提取传统方法的末端执行器轨迹
        trad_traj = []
        for i in range(len(traditional_dataset['trajectories'][0])):
            trad_traj.append(traditional_dataset['joint_positions'][0][-1].numpy())
        
        # 提取Transformer方法的末端执行器轨迹
        trans_traj = []
        for i in range(min(200, len(transformer_dataset['observations']))):
            obs = transformer_dataset['observations'][i]
            joint_positions_flat = obs[7:-3]
            joint_positions = joint_positions_flat.reshape(7, 3)
            trans_traj.append(joint_positions[-1])
        
        # 比较轨迹
        compare_trajectories(trad_traj, trans_traj)
    
    env.close()
    print("可视化完成！")

if __name__ == "__main__":
    collect_and_visualize()
