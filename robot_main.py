from robotEnv import *
from transformerModel import *
from RL_train import *
import numpy as np
import torch
from tqdm import tqdm
import os
import time
import pybullet as p

def check_and_process_data(env, transformer_model, data_path='robot_trajectory_data.npy', num_trajectories=2000, max_steps=30):
    """
    检查是否有收集好的数据，如果有直接使用，如果没有则采集数据并保存为npy格式
    
    参数:
        env: 机器人环境
        transformer_model: 用于收集数据的Transformer模型
        data_path: 数据保存路径
        num_trajectories: 需要收集的轨迹数量
        max_steps: 每条轨迹的最大步数
        
    返回:
        dataset: 包含轨迹数据的字典
    """
    # 检查数据文件是否存在
    if os.path.exists(data_path):
        print(f"找到现有数据文件: {data_path}，正在加载...")
        # 加载现有数据
        dataset = np.load(data_path, allow_pickle=True).item()
        print(f"数据加载成功，包含 {len(dataset['observations'])} 个观测样本")
    else:
        print(f"未找到数据文件: {data_path}，开始收集新数据...")
        # 收集新数据
        dataset = collect_data_with_transformer(env, transformer_model, num_trajectories, max_steps)
        # 保存数据
        print(f"数据收集完成，正在保存到 {data_path}...")
        np.save(data_path, dataset)
        print(f"数据保存成功，包含 {len(dataset['observations'])} 个观测样本")
    
    return dataset

def check_and_train_model(env, model_path='transformer_model.pth', data_path='robot_trajectory_data.npy', 
                          input_dim=27, hidden_dim=128, num_layers=2, nhead=4, max_seq_len=30,
                          num_epochs=100, batch_size=64, lr=1e-4):
    """
    检查是否有训练好的模型权重，如果有则加载，没有则开始训练
    
    参数:
        env: 机器人环境
        model_path: 模型权重保存路径
        data_path: 数据文件路径
        input_dim, hidden_dim, num_layers, nhead, max_seq_len: 模型参数
        num_epochs, batch_size, lr: 训练参数
        
    返回:
        model: 加载或训练好的模型
    """
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryTransformer(input_dim, hidden_dim, num_layers, nhead, max_seq_len).to(device)
    
    # 检查模型权重文件是否存在
    if os.path.exists(model_path):
        print(f"找到现有模型权重: {model_path}，正在加载...")
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功")
    else:
        print(f"未找到模型权重: {model_path}，开始训练新模型...")
        
        # 检查并获取数据
        dataset = check_and_process_data(env, model, data_path)
        
        # 训练模型
        print("开始训练模型...")
        model = train_transformer(model, dataset, num_epochs, batch_size, lr)
        
        # 保存模型
        print(f"模型训练完成，正在保存到 {model_path}...")
        torch.save(model.state_dict(), model_path)
        print("模型保存成功")
    
    return model

def main():
    # 创建环境
    env = RoboticArmEnv()
    if not os.path.exists(r'C:\DiskD\trae_doc\robot_gym\transformer_model.pth'):
        print("Transformer模型未找到，开始训练...")
        # 步骤1: 使用传统方法收集初始数据
        print("收集初始数据...")
        dataset = collect_data(env, num_trajectories=100, max_steps=200)
        print("初始数据收集完成: ", len(dataset['trajectories']), "条轨迹")
        # 步骤2: 训练Transformer模型
        state_dim = 7 + 7*3 + 3  # 关节角度 + 关节位置 + 目标位置
        action_dim = 7  # 6个关节的动作
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("使用设备: ", device)
        transformer_model = TrajectoryTransformer(input_dim=state_dim).to(device)
        print(transformer_model)
        # torch.save(transformer_model.state_dict(), 'robot_gym\transformer_model.pth')
        print("训练Transformer模型...")
        transformer_model = train_transformer(transformer_model, dataset, num_epochs=1)
        print("Transformer模型训练完成")
        

    # 初始化Transformer模型并训练
    # transformer_model = check_and_train_model(
    #     env, 
    #     model_path='robot_model.pth',
    #     data_path='robot_data.npy'
    # )
    # transformer_model.to(device)
        
        
        # 步骤3: 使用Transformer模型收集RL训练数据
        print("使用Transformer模型收集RL数据...")
        rl_dataset = collect_data_with_transformer(env, transformer_model, num_trajectories=100)
        
        # 步骤4: 训练SAC模型
        print("训练SAC模型...")
        replay_buffer = ReplayBuffer(1000000)
        
        # 将RL数据填充到回FFER区
        for i in range(len(rl_dataset['observations'])):
            # 确保数据形状一致
            obs = np.array(rl_dataset['observations'][i])
            action = np.array(rl_dataset['actions'][i])
            
            # 确保奖励是标量
            if isinstance(rl_dataset['rewards'][i], (list, np.ndarray)):
                reward = float(rl_dataset['rewards'][i][0]) if len(rl_dataset['rewards'][i]) > 0 else 0.0
            else:
                reward = float(rl_dataset['rewards'][i])
            
            next_obs = np.array(rl_dataset['next_observations'][i])
            
            # 确保done是布尔值
            if isinstance(rl_dataset['dones'][i], (list, np.ndarray)):
                done = bool(rl_dataset['dones'][i][0]) if len(rl_dataset['dones'][i]) > 0 else False
            else:
                done = bool(rl_dataset['dones'][i])
            
            replay_buffer.push(obs, action, reward, next_obs, done)
        
        # 初始化SAC模型
        sac = SAC(state_dim, action_dim)
        
        # 在线训练SAC
        num_episodes = 1000
        max_steps = 30
        updates_per_step = 1
        total_steps = 0
        
        # 设置PyBullet渲染参数 - 修改渲染设置
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
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            
            # 更新目标位置可视化 - 修改：只在目标位置变化时更新
            target_position = env.target_position
            
            # 安全删除之前的目标对象
            if env.target_object_id is not None:
                try:
                    p.removeBody(env.target_object_id)
                except:
                    pass  # 忽略错误，避免打印大量警告
            
            # 创建新的目标对象
            env.target_object_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=target_visual_id,
                basePosition=target_position
            )
            
            # 如果是10的倍数的episode，启用特殊渲染
            if (episode + 1) % 10 == 0 and episode > 950:
                # 启用特殊渲染
                render_episode = True
            else:
                render_episode = False
            # 清除之前的轨迹标记 - 修改：使用更安全的方式清理标记
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
            
            if render_episode:
                print(f"Episode {episode+1}: 启用特殊渲染...")
                
                # 添加轨迹起点标记
                visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 1.0])
                marker_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_id,
                    basePosition=env.last_end_effector_pos
                )
                env.trajectory_markers.append(marker_id)
                
                # 添加目标点连接线 - 新增：显示从起点到目标点的连接线
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
            
            # 修改：在渲染episode时，先暂停一下让用户观察初始状态
            if render_episode:
                print(f"初始状态 - 目标位置: {target_position}")
                print(f"初始状态 - 末端执行器位置: {env.last_end_effector_pos}")
                print("按Enter键开始执行动作...")
                input()
            
            for step in range(max_steps):
                # 选择动作
                action = sac.select_action(obs)
                
                # 确保动作在合理范围内 - 修改：使用更小的动作范围，避免大幅度运动
                action = np.clip(action, -0.5, 0.5)
                
                # 执行动作前打印信息 - 新增：在渲染episode时打印动作信息
                if render_episode:
                    print(f"Step {step+1} - 执行动作: {action}")
                
                # 执行动作
                next_obs, reward, done, _ = env.step(action)
                
                # 获取当前末端执行器位置
                current_end_effector_pos = env._get_end_effector_position()
                trajectory_points.append(current_end_effector_pos)
                
                # 如果是渲染episode，添加轨迹可视化
                if render_episode:
                    # 计算移动距离 - 新增：检查是否有实际移动
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
                        if step % 3 == 0:  # 每3步添加一个标记点，避免过多标记
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
                            f"距离目标: {distance:.4f} | 步数: {step+1}/{max_steps}",
                            [0.0, 0.0, 0.8],  # 在机械臂上方显示
                            textColorRGB=[1, 1, 1],
                            textSize=1.5,
                            lifeTime=0.1
                        )
                    except:
                        pass
                    
                    # 打印调试信息
                    print(f"Step {step+1} - 末端位置: {current_end_effector_pos}, 移动距离: {move_distance:.6f}, 到目标距离: {distance:.4f}")
                    
                    # 添加短暂延迟以便观察 - 增加延迟时间，使动作更容易观察
                    time.sleep(0.2)
                
                # 更新上一个位置
                env.last_end_effector_pos = current_end_effector_pos
                
                # 存储经验
                replay_buffer.push(obs, action, reward, next_obs, done)
                
                obs = next_obs
                episode_reward += reward
                total_steps += 1
                
                # 更新SAC模型
                if len(replay_buffer) > 1000:  # 等待缓冲区积累足够数据
                    for _ in range(updates_per_step):
                        critic_loss, actor_loss = sac.update_parameters(replay_buffer)
                
                if done:
                    if render_episode:
                        # 添加终点标记
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
                    break
            
            # 打印训练进度
            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}, Steps: {step+1}")
                # 如果是渲染episode，暂停一下让用户观察最终状态
                if render_episode:
                    # 保存轨迹截图
                    try:
                        screenshot_path = f"C:\\DiskD\\trae_doc\\robot_gym\\trajectory_episode_{episode+1}.png"
                        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
                        img_arr = p.getCameraImage(1280, 720)[2]
                        import cv2
                        cv2.imwrite(screenshot_path, img_arr)
                        print(f"轨迹截图已保存至 {screenshot_path}")
                    except Exception as e:
                        print(f"保存截图失败: {e}")
                    
                    print("轨迹渲染完成，按Enter键继续下一个episode...")
                    input()

        # 保存模型
        torch.save(transformer_model.state_dict(), "transformer_model.pth")
        torch.save(sac.actor.state_dict(), "sac_actor.pth")
        torch.save(sac.critic.state_dict(), "sac_critic.pth")
    else:
        print("Transformer模型已存在，跳过训练步骤")
        state_dim = 7 + 7*3 + 3  # 关节角度 + 关节位置 + 目标位置
        action_dim = 7  # 6个关节的动作
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("使用设备: ", device)
        transformer_model = TrajectoryTransformer(input_dim=state_dim).to(device)
        transformer_model.load_state_dict(torch.load(r'C:\DiskD\trae_doc\robot_gym\transformer_model.pth', map_location=device))
        print("Transformer模型加载成功")
        # 初始化SAC模型
        sac = SAC(state_dim, action_dim)
        # 加载模型权重
        sac.actor.load_state_dict(torch.load(r'C:\DiskD\trae_doc\robot_gym\sac_actor.pth', map_location=device))
        sac.critic.load_state_dict(torch.load(r'C:\DiskD\trae_doc\robot_gym\sac_critic.pth', map_location=device))
        print("SAC模型加载成功")

    # 步骤5: 评估模型
    # 评估模型
    print("评估模型...")
    eval_episodes = 50
    eval_rewards = []
    
    for _ in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = sac.select_action(obs, evaluate=True)
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
            obs = next_obs
        
        eval_rewards.append(episode_reward)
    
    print(f"评估平均奖励: {np.mean(eval_rewards):.2f}")
    
    env.close()

if __name__ == "__main__":
    main()
