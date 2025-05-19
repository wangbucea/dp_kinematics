import torch
import numpy as np
from gym import spaces
import pybullet as p
import pybullet_data
from tqdm import tqdm

class RoboticArmEnv:
    def __init__(self):
        # 初始化PyBullet物理引擎
        p.connect(p.GUI)#or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-9.81)
        _ = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("C:\\DiskD\\vs_work_path\\pygame\\pybullet\\rm_75_6f_description\\urdf\\rm_75_6f_description.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        # 加载机械臂模型（使用通用UR5或自定义URDF）
        # self.robot_id = p.loadURDF("ur5/ur5.urdf", [0, 0, 0], useFixedBase=True)
        
        # 获取关节数量和ID
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_ids = range(self.num_joints)
        
        # 动作和观测空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_joints,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints+self.num_joints+3,))  # 关节角度+关节位置+目标位置
        # 停留时间计数器
        self.target_stay_time = 0

    def reset(self, target_position=None):
        # 重置机械臂到初始位置
        for joint_id in self.joint_ids:
            p.resetJointState(self.robot_id, joint_id, 0)
            
        # 设置随机目标位置或使用指定目标
        if target_position is None:
            self.target_position = np.random.uniform(low=[0.3, -0.3, 0.2], high=[0.5, 0.3, 0.4])
        else:
            self.target_position = target_position
            
        return self._get_observation()
        
    def step(self, action):
        # 将归一化动作转换为关节角度增量
        action = np.clip(action, -1, 1) * 0.05  # 限制每步移动幅度
        
        # 获取当前关节角度
        current_angles = np.array([p.getJointState(self.robot_id, joint_id)[0] for joint_id in range(self.num_joints)])
        
        # 应用动作 - 修复错误：确保是元素级加法而不是数组连接
        target_angles = np.add(current_angles, action)  # 使用np.add替代+运算符
        
        for i, joint_id in enumerate(range(self.num_joints)):
            p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, target_angles[i])
            
        # 模拟一步
        p.stepSimulation()
        
        # 计算奖励和完成状态
        observation = self._get_observation()
        end_effector_pos = self._get_end_effector_position()
        
        distance = np.linalg.norm(end_effector_pos - self.target_position)
        reward = -distance  # 负距离作为奖励
        if distance < 0.08:
            reward += 100  # 当距离小于阈值时给予额外奖励
            self.target_stay_time += 1
            if self.target_stay_time >= 5:  # 停留时间达到阈值时认为任务完成
                done = True
        elif distance < 0.1:
            reward += 10  # 当距离大于阈值时给予惩罚
        elif distance < 0.2:
            reward += 5  # 当距离大于阈值时给予惩罚
        
        done = distance < 0.08  # 当距离小于阈值时认为任务完成
        
        info = {"distance": distance}
        
        return observation, reward, done, info
    
    def _get_observation(self):
        # 获取关节角度
        joint_angles = np.array([p.getJointState(self.robot_id, joint_id)[0] for joint_id in range(self.num_joints)])
        
        # 获取关节位置
        joint_positions = []
        for joint_id in range(self.num_joints):
            joint_info = p.getLinkState(self.robot_id, joint_id)
            joint_positions.extend(joint_info[0])  # 链接位置
        joint_positions = np.array(joint_positions).reshape(self.num_joints, 3)
        
        # 展平关节位置为一维数组
        joint_positions_flat = joint_positions.flatten()
        
        # 组合观测
        observation = np.concatenate([joint_angles, joint_positions_flat, self.target_position])
        
        return observation
    
    def _get_end_effector_position(self):
        end_effector_state = p.getLinkState(self.robot_id, self.num_joints - 1)  # 假设末端执行器是第7个链接
        return np.array(end_effector_state[0])
    def get_end_effector_position(self, joint_angles):
        """
        根据给定的关节角度计算末端执行器位置
        
        参数:
            joint_angles: 关节角度数组
        返回:
            末端执行器位置
        """
        # 保存当前关节角度
        current_angles = np.array([p.getJointState(self.robot_id, joint_id)[0] for joint_id in range(self.num_joints)])
        
        # 临时设置新的关节角度
        for i, joint_id in enumerate(range(self.num_joints)):
            p.resetJointState(self.robot_id, joint_id, joint_angles[i])
        
        # 获取末端执行器位置
        end_effector_state = p.getLinkState(self.robot_id, self.num_joints - 1)
        end_effector_pos = np.array(end_effector_state[0])
        
        # 恢复原来的关节角度
        for i, joint_id in enumerate(range(self.num_joints)):
            p.resetJointState(self.robot_id, joint_id, current_angles[i])
        
        return end_effector_pos
    def close(self):
        p.disconnect()

def collect_data(env, num_trajectories=1000, max_steps=100):
    dataset = {
        'joint_angles': [],
        'joint_positions': [],
        'target_positions': [],
        'trajectories': []
    }
    
    # 记录成功到达目标的轨迹数量
    success_count = 0
    
    for traj_idx in tqdm(range(num_trajectories)):
        # 重置环境，获取初始观测
        obs = env.reset()
        
        # 解析初始观测
        joint_angles = obs[:env.num_joints]
        joint_positions = obs[env.num_joints:-3].reshape(env.num_joints, 3)
        target_position = obs[-3:]
        
        # 使用逆运动学生成参考轨迹
        trajectory = []
        current_pos = env._get_end_effector_position()
        
        # 记录初始距离
        initial_distance = np.linalg.norm(target_position - current_pos)
        min_distance = initial_distance
        steps_without_progress = 0
        
        # 检查目标是否可达 - 使用更严格的阈值
        if initial_distance > 1.0:
            print(f"轨迹 {traj_idx}: 目标距离过远 ({initial_distance:.4f})，可能不可达")
            continue
        
        # 直接使用逆运动学尝试求解目标位置
        target_orn = p.getQuaternionFromEuler([0, 0, 0])
        direct_ik_solution = p.calculateInverseKinematics(
            env.robot_id, env.num_joints - 1, target_position, 
            targetOrientation=target_orn, 
            maxNumIterations=1000,
            residualThreshold=0.0001
        )
        
        # 验证直接IK解的质量
        direct_solution_angles = np.array(direct_ik_solution[:env.num_joints])
        direct_solution_pos = env.get_end_effector_position(direct_solution_angles)
        direct_solution_error = np.linalg.norm(direct_solution_pos - target_position)
        
        # 如果直接IK解足够好，直接使用
        if direct_solution_error < 0.05:
            print(f"轨迹 {traj_idx}: 直接IK解足够好，误差: {direct_solution_error:.4f}")
            current_angles = np.array([p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
            action = direct_solution_angles - current_angles
            
            # 将动作分解为多个小步骤
            num_substeps = 10
            for i in range(num_substeps):
                sub_action = action / num_substeps
                trajectory.append(sub_action)
                
            # 将轨迹保存到数据集
            dataset['joint_angles'].append(joint_angles)
            dataset['joint_positions'].append(joint_positions)
            dataset['target_positions'].append(target_position)
            dataset['trajectories'].append(trajectory)
            success_count += 1
            continue
        
        # 如果直接IK解不够好，使用迭代方法
        for step in range(max_steps):
            # 计算从当前位置到目标的方向
            direction = target_position - current_pos
            distance = np.linalg.norm(direction)
            
            # 更新最小距离
            if distance < min_distance:
                min_distance = distance
                steps_without_progress = 0
            else:
                steps_without_progress += 1
            
            # 自适应步长：距离越近，步长越小
            if distance > 0.5:
                step_size = 0.05
            elif distance > 0.2:
                step_size = 0.03
            elif distance > 0.1:
                step_size = 0.02
            else:
                step_size = 0.01
            
            # 如果连续多步没有进展，尝试随机扰动
            if steps_without_progress > 10:
                # 添加随机扰动
                random_direction = np.random.randn(3)
                random_direction = random_direction / np.linalg.norm(random_direction)
                direction = 0.7 * direction + 0.3 * random_direction
                direction = direction / np.linalg.norm(direction)
                step_size = 0.02
                steps_without_progress = 0
                print(f"轨迹 {traj_idx}: 添加随机扰动以跳出局部最小值")
            
            # 到达目标或无法继续前进时终止
            if distance < 0.05:  # 终止条件
                success_count += 1
                print(f"轨迹 {traj_idx}: 成功到达目标，最终距离: {distance:.4f}")
                break
            
            # 如果长时间没有进展且距离仍然很远，放弃当前轨迹
            if steps_without_progress > 20 and distance > 0.2:
                print(f"轨迹 {traj_idx}: 无法继续接近目标，最小距离: {min_distance:.4f}")
                break
                
            # 标准化方向
            if distance > 0:
                direction = direction / distance
                
            # 使用逆运动学求解，尝试多个姿态
            target_pos = current_pos + direction * min(step_size, distance)
            
            # 尝试不同的末端执行器姿态
            orientations = [
                p.getQuaternionFromEuler([0, 0, 0]),  # 默认姿态
                p.getQuaternionFromEuler([0, np.pi/6, 0]),  # 绕Y轴旋转30度
                p.getQuaternionFromEuler([np.pi/6, 0, 0]),  # 绕X轴旋转30度
                p.getQuaternionFromEuler([0, -np.pi/6, 0]),  # 绕Y轴反向旋转30度
                p.getQuaternionFromEuler([0, 0, np.pi/6]),  # 绕Z轴旋转30度
            ]
            
            best_solution = None
            min_solution_error = float('inf')
            
            for target_orn in orientations:
                # 增加迭代次数以提高求解精度
                joint_positions_target = p.calculateInverseKinematics(
                    env.robot_id, env.num_joints - 1, target_pos, 
                    targetOrientation=target_orn, 
                    maxNumIterations=500,
                    residualThreshold=0.0005
                )
                
                # 验证解的质量 - 前向运动学检查
                temp_angles = np.array(joint_positions_target[:env.num_joints])
                
                # 检查关节限制
                joint_limits_ok = True
                for i, angle in enumerate(temp_angles):
                    # 假设关节限制为 ±π
                    if angle < -np.pi or angle > np.pi:
                        joint_limits_ok = False
                        break
                
                if not joint_limits_ok:
                    continue
                
                # 使用前向运动学验证解的质量
                test_pos = env.get_end_effector_position(temp_angles)
                solution_error = np.linalg.norm(test_pos - target_pos)
                
                if solution_error < min_solution_error:
                    min_solution_error = solution_error
                    best_solution = temp_angles
            
            # 如果没有找到合适的解，尝试减小步长
            if best_solution is None or min_solution_error > 0.1:
                # 减小步长重试
                target_pos = current_pos + direction * min(step_size * 0.5, distance)
                joint_positions_target = p.calculateInverseKinematics(
                    env.robot_id, env.num_joints - 1, target_pos, 
                    targetOrientation=p.getQuaternionFromEuler([0, 0, 0]), 
                    maxNumIterations=500
                )
                best_solution = np.array(joint_positions_target[:env.num_joints])
            
            # 转换为动作（关节角度增量）
            current_angles = np.array([p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
            action = best_solution - current_angles
            
            # 限制动作幅度
            action = np.clip(action, -0.05, 0.05)
            
            # 记录动作
            trajectory.append(action)
            
            # 执行动作
            obs, reward, done, _ = env.step(action)
            current_pos = env._get_end_effector_position()
            
            if done:
                success_count += 1
                print(f"轨迹 {traj_idx}: 环境标记为完成，最终距离: {distance:.4f}")
                break
        
        # 将轨迹保存到数据集
        dataset['joint_angles'].append(joint_angles)
        dataset['joint_positions'].append(joint_positions)
        dataset['target_positions'].append(target_position)
        dataset['trajectories'].append(trajectory)
    
    # 打印成功率
    print(f"成功到达目标的轨迹: {success_count}/{num_trajectories} ({success_count/num_trajectories*100:.2f}%)")
    
    # 转换为张量 - 修改这部分代码
    for key in dataset:
        if key != 'trajectories':  # 对于轨迹以外的数据，直接转换为张量
            dataset[key] = torch.tensor(np.array(dataset[key]), dtype=torch.float32)
        else:
            # 对于轨迹数据，保持列表形式，每个元素是一个张量
            dataset[key] = [torch.tensor(np.array(traj), dtype=torch.float32) for traj in dataset[key]]
        
    return dataset
