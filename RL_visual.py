import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
import pickle
import time
from matplotlib.widgets import Slider, Button
from matplotlib import cm
import sys
import pybullet as p
import pybullet_data


# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入相关模块
from robotEnv import RoboticArmEnv
from RL_train import TransformerActor, TransformerCritic

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_rl_model(model_path, state_dim, action_dim):
    """
    加载训练好的强化学习模型
    
    参数:
        model_path: 模型文件路径
        state_dim: 状态空间维度
        action_dim: 动作空间维度
    
    返回:
        加载的模型，如果文件不存在则返回None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型 - 修改输入维度为31而不是当前的state_dim
    actor = TransformerActor(31, action_dim).to(device)
    
    # 加载模型参数
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        try:
            # 尝试使用 'actor' 键加载
            actor.load_state_dict(checkpoint['actor'])
        except KeyError:
            # 如果没有 'actor' 键，尝试直接加载
            try:
                actor.load_state_dict(checkpoint)
                print(f"模型已从 {model_path} 直接加载")
            except Exception as e:
                print(f"加载模型时出错: {e}")
                return None
        print(f"模型已从 {model_path} 加载")
        return actor
    else:
        print(f"未找到模型文件: {model_path}")
        return None

def generate_trajectory(model, env, initial_state=None, max_steps=100):
    """
    使用训练好的模型生成轨迹
    
    参数:
        model: 训练好的模型
        env: 环境实例
        initial_state: 初始状态，如果为None则随机生成
        max_steps: 最大步数
    
    返回:
        轨迹数据，包含关节角度、关节位置和目标位置
    """
    device = next(model.parameters()).device
    
    # 重置环境
    state = env.reset() if initial_state is None else env.reset(initial_state)
    
    # 记录轨迹
    trajectory = []
    joint_angles = []
    joint_positions = []
    
    # 记录初始状态
    # 从观测中提取关节角度和位置
    obs = env._get_observation()
    joint_angles.append(obs[:env.num_joints])  # 前num_joints个元素是关节角度
    
    # 获取关节位置 - 从观测中提取或直接获取
    joint_pos = []
    for joint_id in range(env.num_joints):
        joint_info = p.getLinkState(env.robot_id, joint_id)
        joint_pos.append(joint_info[0])  # 链接位置
    joint_positions.append(np.array(joint_pos))
    
    # 目标位置
    target_position = env.target_position
    
    for step in range(max_steps):
        # 转换状态为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # 使用模型选择动作
        with torch.no_grad():
            action, _, _ = model.sample(state_tensor)
            action = action.cpu().numpy().flatten()
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 记录轨迹 - 从观测中提取关节角度和位置
        obs = env._get_observation()
        joint_angles.append(obs[:env.num_joints])  # 前num_joints个元素是关节角度
        
        # 获取关节位置
        joint_pos = []
        for joint_id in range(env.num_joints):
            joint_info = p.getLinkState(env.robot_id, joint_id)
            joint_pos.append(joint_info[0])  # 链接位置
        joint_positions.append(np.array(joint_pos))
        
        # 更新状态
        state = next_state
        
        if done:
            break
    
    # 构建轨迹数据
    trajectory = {
        'joint_angles': np.array(joint_angles),
        'joint_positions': np.array(joint_positions),
        'target_position': target_position,
        'steps': len(joint_angles)
    }
    
    return trajectory

def visualize_trajectory_3d(trajectory, output_file=None, show=True):
    """
    3D可视化轨迹
    
    参数:
        trajectory: 轨迹数据
        output_file: 输出文件路径，如果为None则不保存
        show: 是否显示图形
    """
    joint_positions = trajectory['joint_positions']
    target_position = trajectory['target_position']
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置颜色映射
    cmap = cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 1, len(joint_positions))]
    
    # 绘制机械臂轨迹
    for i, positions in enumerate(joint_positions):
        # 透明度随时间增加
        alpha = 0.3 + 0.7 * (i / len(joint_positions))
        
        # 只绘制部分帧以避免过于密集
        if i % max(1, len(joint_positions) // 20) == 0 or i == len(joint_positions) - 1:
            # 绘制机械臂连杆
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'o-', color=colors[i], alpha=alpha, linewidth=2, markersize=4)
    
    # 绘制末端执行器轨迹
    end_positions = np.array([pos[-1] for pos in joint_positions])
    ax.plot(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], 
           '-', color='blue', alpha=0.7, linewidth=2, label='末端轨迹')
    
    # 标记起点和终点
    ax.scatter(end_positions[0, 0], end_positions[0, 1], end_positions[0, 2], 
              color='green', s=100, marker='o', label='起点')
    ax.scatter(end_positions[-1, 0], end_positions[-1, 1], end_positions[-1, 2], 
              color='red', s=100, marker='*', label='终点')
    
    # 绘制目标位置
    ax.scatter(target_position[0], target_position[1], target_position[2], 
              color='purple', s=150, marker='*', label='目标位置')
    
    # 设置图形属性
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('强化学习机械臂轨迹 (3D视图)')
    ax.legend()
    
    # 设置坐标轴等比例
    all_positions = np.vstack(joint_positions)
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
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图形
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图形已保存至 {output_file}")
    
    # 显示图形
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax

def create_trajectory_animation(trajectory, output_file="trajectory_animation.gif", fps=10):
    """
    创建机械臂轨迹的动画并保存为GIF文件
    
    参数:
        trajectory: 轨迹数据
        output_file: 输出文件路径
        fps: 每秒帧数
    """
    joint_positions = trajectory['joint_positions']
    target_position = trajectory['target_position']
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 初始化机械臂线条
    line, = ax.plot([], [], [], 'bo-', linewidth=2, markersize=6)
    
    # 初始化末端轨迹线条
    end_trajectory, = ax.plot([], [], [], 'g--', alpha=0.5)
    
    # 绘制目标位置
    ax.scatter(target_position[0], target_position[1], target_position[2], 
              color='red', s=150, marker='*', label='目标位置')
    
    # 设置坐标轴范围
    all_positions = np.vstack(joint_positions)
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
        positions = joint_positions[frame]
        
        # 更新机械臂线条
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        
        # 更新末端位置历史
        end_positions_history.append(positions[-1])
        
        # 更新末端轨迹线条
        if len(end_positions_history) > 1:
            end_x = [p[0] for p in end_positions_history]
            end_y = [p[1] for p in end_positions_history]
            end_z = [p[2] for p in end_positions_history]
            end_trajectory.set_data(end_x, end_y)
            end_trajectory.set_3d_properties(end_z)
        
        # 更新时间标签
        progress = frame / (len(joint_positions) - 1) * 100
        time_text.set_text(f'进度: {progress:.1f}%')
        
        return line, end_trajectory, time_text
    
    # 创建动画
    frames = min(len(joint_positions), 100)  # 限制帧数，避免过多计算
    step = max(1, len(joint_positions) // frames)
    ani = FuncAnimation(fig, update, frames=range(0, len(joint_positions), step),
                       init_func=init, interval=1000/fps, blit=True)
    
    # 添加图例
    ax.legend()
    
    # 保存动画
    ani.save(output_file, writer='pillow', fps=fps)
    print(f"动画已保存至 {output_file}")
    
    plt.close(fig)
    
    return ani

def interactive_trajectory_viewer(trajectory):
    """
    交互式轨迹查看器
    
    参数:
        trajectory: 轨迹数据
    """
    joint_positions = trajectory['joint_positions']
    target_position = trajectory['target_position']
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 当前帧
    current_frame = 0
    
    # 绘制初始状态
    positions = joint_positions[current_frame]
    line, = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'bo-', linewidth=2, markersize=6)
    
    # 绘制目标位置
    ax.scatter(target_position[0], target_position[1], target_position[2], 
              color='red', s=150, marker='*', label='目标位置')
    
    # 设置坐标轴范围
    all_positions = np.vstack(joint_positions)
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
    ax.set_title('强化学习机械臂轨迹 (交互式查看器)')
    
    # 添加时间标签
    time_text = ax.text2D(0.05, 0.95, f'帧: {current_frame}/{len(joint_positions)-1}', transform=ax.transAxes)
    
    # 添加轨迹滑块
    ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03])
    slider = Slider(ax=ax_slider, label='轨迹帧', valmin=0, valmax=len(joint_positions)-1, 
                   valinit=current_frame, valstep=1)
    
    # 添加末端轨迹切换按钮
    ax_button = plt.axes([0.8, 0.06, 0.15, 0.04])
    button = Button(ax_button, '显示/隐藏末端轨迹')
    
    # 末端轨迹线
    end_positions = np.array([pos[-1] for pos in joint_positions])
    end_line, = ax.plot(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], 
                       'g--', alpha=0.5, label='末端轨迹')
    
    # 显示末端轨迹标志
    show_end_trajectory = True
    
    # 更新函数
    def update(val):
        # 获取当前帧
        current_frame = int(slider.val)
        
        # 更新机械臂位置
        positions = joint_positions[current_frame]
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])
        
        # 更新时间标签
        time_text.set_text(f'帧: {current_frame}/{len(joint_positions)-1}')
        
        fig.canvas.draw_idle()
    
    # 末端轨迹切换函数
    def toggle_end_trajectory(event):
        nonlocal show_end_trajectory
        show_end_trajectory = not show_end_trajectory
        end_line.set_visible(show_end_trajectory)
        fig.canvas.draw_idle()
    
    # 注册更新函数
    slider.on_changed(update)
    button.on_clicked(toggle_end_trajectory)
    
    # 添加图例
    ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def main():
    """主函数"""
    # 模型路径
    model_path = r'sac_actor.pth'
    
    # 创建环境
    env = RoboticArmEnv()
    
    # 状态和动作空间维度
    # 注意：这里使用31作为state_dim，与保存的模型匹配
    state_dim = 31
    action_dim = env.action_space.shape[0]
    
    # 加载模型
    model = load_rl_model(model_path, state_dim, action_dim)
    
    # 检查模型是否成功加载
    if model is None:
        print("模型加载失败，无法继续执行。请确保模型文件存在。")
        return
    
    # 生成轨迹
    print("生成机械臂轨迹...")
    trajectory = generate_trajectory(model, env)
    
    # 保存轨迹数据
    with open('rl_trajectory.pkl', 'wb') as f:
        pickle.dump(trajectory, f)
    print("轨迹数据已保存至 rl_trajectory.pkl")
    
    # 可视化轨迹
    print("可视化轨迹...")
    visualize_trajectory_3d(trajectory, output_file='rl_trajectory_3d.png')
    
    # 创建动画
    print("创建轨迹动画...")
    create_trajectory_animation(trajectory, output_file='rl_trajectory_animation.gif')
    
    # 交互式查看器
    print("启动交互式轨迹查看器...")
    interactive_trajectory_viewer(trajectory)
    
    print("可视化完成！")

if __name__ == "__main__":
    main()
