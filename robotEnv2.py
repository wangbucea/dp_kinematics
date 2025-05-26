import torch
import numpy as np
from gym import spaces
import pybullet as p
import pybullet_data
from tqdm import tqdm

class RoboticArmEnv:
    def __init__(self, use_gui=True):
        # 初始化PyBullet物理引擎
        try:
            # 尝试断开现有连接
            p.disconnect()
        except:
            pass
        
        # 根据参数选择连接模式
        if use_gui:
            try:
                p.connect(p.GUI)  # 图形界面模式
            except:
                print("无法使用GUI模式，切换到DIRECT模式")
                p.connect(p.DIRECT)  # 如果GUI连接失败，使用DIRECT模式
        else:
            p.connect(p.DIRECT)  # 非图形界面模式
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-9.81)
        _ = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("C:\\DiskD\\vs_work_path\\pygame\\pybullet\\rm_75_6f_description\\urdf\\rm_75_6f_description.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        
        # 启用碰撞检测
        p.setRealTimeSimulation(0)  # 确保非实时模式，以便精确控制
        
        # 获取关节数量和ID
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_ids = range(self.num_joints)
        
        # 动作和观测空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_joints,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints+self.num_joints+3,))  # 关节角度+关节位置+目标位置
        # 停留时间计数器
        self.target_stay_time = 0
        # 记录上一步的末端执行器位置，用于判断是否停留
        self.last_end_effector_pos = None
        # 停留阈值 - 如果两步之间的移动距离小于此值，认为是停留
        self.stay_threshold = 0.05
        # 停留奖励所需的最小时间步
        self.min_stay_steps = 5
        
        # 碰撞检测相关
        self.collision_occurred = False

    def check_collision(self):
        """
        检查机械臂是否发生碰撞
        
        返回:
            bool: 是否发生碰撞
        """
        # 获取所有碰撞对
        contact_points = p.getContactPoints(bodyA=self.robot_id)
        
        # 检查机械臂与自身的碰撞
        self_collision = False
        for point in contact_points:
            # 如果碰撞对中的两个物体都是机械臂，则发生了自碰撞
            if point[1] == self.robot_id and point[2] == self.robot_id:
                self_collision = True
                break
        
        # 检查机械臂与环境的碰撞（排除与地面的正常接触）
        env_collision = False
        for point in contact_points:
            # 如果碰撞对中有机械臂和其他物体（非地面），则发生了环境碰撞
            if (point[1] == self.robot_id and point[2] != self.robot_id) or \
               (point[2] == self.robot_id and point[1] != self.robot_id):
                # 排除与地面的正常接触
                if point[1] != 0 and point[2] != 0:  # 地面的ID通常为0
                    env_collision = True
                    break
        
        return self_collision or env_collision
    
    def reset(self, target_position=None, random_init=False):
        # 重置机械臂到初始位置或随机位置
        if random_init:
            # 为每个关节设置随机角度，范围在合理的关节限制内
            random_angles = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=self.num_joints)
            for joint_id in range(self.num_joints):
                p.resetJointState(self.robot_id, joint_id, random_angles[joint_id])
        else:
            # 默认行为：重置到初始位置（所有关节角度为0）
            for joint_id in self.joint_ids:
                p.resetJointState(self.robot_id, joint_id, 0)
            
        # 设置随机目标位置或使用指定目标
        if target_position is None:
            # 生成机械臂可达范围内的随机目标位置
            # 确定机械臂基座位置
            base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            base_pos = np.array(base_pos)
            
            # 估计机械臂最大可达距离（根据机械臂链接长度总和估计）
            # 对于RM-75机械臂，最大可达距离约为0.8米
            max_reach = 0.8
            
            # 生成球形区域内的随机点
            while True:
                # 随机生成方向（单位球面上的点）
                phi = np.random.uniform(0, np.pi/2)  # 限制在前方半球
                theta = np.random.uniform(0, np.pi/2)  # 限制水平范围
                
                direction = np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
                
                # 随机生成距离（在最大可达范围内）
                distance = np.random.uniform(0.2, 0.8)
                
                # 计算目标位置
                target_pos = base_pos + direction * distance
                
                # 确保目标位置在工作空间内（高度大于0，避免目标位置在地面以下）
                if target_pos[2] > 0.05:
                    # 尝试使用IK检查该位置是否可达
                    target_orn = p.getQuaternionFromEuler([0, 0, 0])
                    ik_solution = p.calculateInverseKinematics(
                        self.robot_id, 
                        self.num_joints - 1, 
                        target_pos,
                        targetOrientation=target_orn,
                        maxNumIterations=100,
                        residualThreshold=0.01
                    )
                    
                    # 验证IK解的质量
                    test_angles = np.array(ik_solution[:self.num_joints])
                    test_pos = self.get_end_effector_position(test_angles)
                    error = np.linalg.norm(test_pos - target_pos)
                    
                    # 如果误差在可接受范围内，则认为该位置可达
                    if error < 0.1:
                        self.target_position = target_pos
                        break
                    
                    # 如果尝试了多次仍找不到可达位置，则使用默认范围
                    if np.random.random() < 0.1:  # 有10%的概率使用默认范围
                        self.target_position = np.random.uniform(low=[0.3, -0.3, 0.2], high=[0.5, 0.3, 0.4])
                        break
        else:
            self.target_position = target_position
        
        # 重置碰撞状态
        self.collision_occurred = False
        self.target_stay_time = 0
        self.last_end_effector_pos = None
            
        return self._get_observation()
        
    def step(self, action):
        # 将归一化动作转换为关节角度增量
        action = np.clip(action, -0.05, 0.05)  # 限制每步移动幅度
        
        # 获取当前关节角度
        current_angles = np.array([p.getJointState(self.robot_id, joint_id)[0] for joint_id in range(self.num_joints)])
        
        # 应用动作 - 修复错误：确保是元素级加法而不是数组连接
        target_angles = np.add(current_angles, action)  # 使用np.add替代+运算符
        
        for i, joint_id in enumerate(range(self.num_joints)):
            p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, target_angles[i])
            
        # 模拟一步
        p.stepSimulation()
        
        # 检查碰撞
        self.collision_occurred = self.check_collision()
        
        # 计算奖励和完成状态
        observation = self._get_observation()
        end_effector_pos = self._get_end_effector_position()
        
        distance = np.linalg.norm(end_effector_pos - self.target_position)
        done = False
        
        # 如果发生碰撞，给予大的负奖励
        if self.collision_occurred:
            reward = -9999  # 碰撞惩罚
            done = True
            self.target_stay_time = 0
        else:
            # 基础奖励为负距离
            reward = -(distance**2)  
            # 检查是否在目标附近
            if distance < 0.01:
                # 检查是否停留（与上一步位置相比变化很小）
                # if self.last_end_effector_pos is not None:
                #     movement = np.linalg.norm(end_effector_pos - self.last_end_effector_pos)
                #     if movement < self.stay_threshold:
                #         self.target_stay_time += 1
                #     else:
                #         # 如果移动了，重置停留计数器
                #         self.target_stay_time = 0
                reward += 999
                # 只有在停留足够时间后才给予额外奖励
                if self.target_stay_time >= self.min_stay_steps:
                    done = True
            elif distance < 0.05:
                # 接近目标但未达到停留奖励条件，重置停留计数器
                # self.target_stay_time = 0
                # 仍然给予一些接近奖励，但不是主要奖励
                reward += 199
            elif distance < 0.08:
                # 更远的距离，重置停留计数器
                self.target_stay_time = 0
                # 较小的接近奖励
                reward += 99
            elif distance < 0.1:
                # 更远的距离，重置停留计数器
                self.target_stay_time = 0
                # 较小的接近奖励
                reward += 100
            elif distance < 0.12:
                # 更远的距离，重置停留计数器
                self.target_stay_time = 0
                # 较小的接近奖励
                reward += 50
            elif distance < 0.15:
                # 更远的距离，重置停留计数器
                self.target_stay_time = 0
                # 较小的接近奖励
                reward += 10
            elif distance < 0.18:
                # 更远的距离，重置停留计数器
                self.target_stay_time = 0
                # 较小的接近奖励
                reward += 5
            elif distance < 0.2:
                # 更远的距离，重置停留计数器
                self.target_stay_time = 0
                # 较小的接近奖励
                reward += 1
            else:
                # 距离较远，重置停留计数器
                self.target_stay_time = 0
        
        # 更新上一步的末端执行器位置
        self.last_end_effector_pos = end_effector_pos.copy()
                    
        info = {
            "distance": distance,
            "collision": self.collision_occurred,
            "stay_time": self.target_stay_time
        }
        
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
            current_angles = np.array(
                [p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
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
                p.getQuaternionFromEuler([0, np.pi / 6, 0]),  # 绕Y轴旋转30度
                p.getQuaternionFromEuler([np.pi / 6, 0, 0]),  # 绕X轴旋转30度
                p.getQuaternionFromEuler([0, -np.pi / 6, 0]),  # 绕Y轴反向旋转30度
                p.getQuaternionFromEuler([0, 0, np.pi / 6]),  # 绕Z轴旋转30度
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
            current_angles = np.array(
                [p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
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
    print(f"成功到达目标的轨迹: {success_count}/{num_trajectories} ({success_count / num_trajectories * 100:.2f}%)")

    # 转换为张量 - 修改这部分代码
    for key in dataset:
        if key != 'trajectories':  # 对于轨迹以外的数据，直接转换为张量
            dataset[key] = torch.tensor(np.array(dataset[key]), dtype=torch.float32)
        else:
            # 对于轨迹数据，保持列表形式，每个元素是一个张量
            dataset[key] = [torch.tensor(np.array(traj), dtype=torch.float32) for traj in dataset[key]]

    return dataset


def collect_sequence_data(env, num_trajectories=1000, max_steps=100):
    """
    收集机械臂轨迹数据，包括每个时刻的状态
    
    Args:
        env: 机械臂环境
        num_trajectories: 要收集的轨迹数量
        max_steps: 每个轨迹的最大步数
        
    Returns:
        包含轨迹数据的字典，每个轨迹包含每个时刻的状态
    """
    dataset = {
        'initial_joint_angles': [],      # 初始关节角度
        'initial_joint_positions': [],   # 初始关节位置
        'target_positions': [],          # 目标位置
        'trajectories': [],              # 动作轨迹
        'sequence_joint_angles': [],     # 每个时刻的关节角度序列
        'sequence_joint_positions': [],  # 每个时刻的关节位置序列
        'sequence_end_effector_positions': []  # 每个时刻的末端执行器位置序列
    }

    # 记录成功到达目标的轨迹数量
    success_count = 0

    for traj_idx in tqdm(range(num_trajectories)):
        # 重置环境，获取初始观测
        obs = env.reset(random_init=False)

        # 解析初始观测
        joint_angles = obs[:env.num_joints]
        joint_positions = obs[env.num_joints:-3].reshape(env.num_joints, 3)
        target_position = obs[-3:]

        # 初始化序列数据
        sequence_joint_angles = [joint_angles.copy()]
        sequence_joint_positions = [joint_positions.copy()]
        sequence_end_effector_positions = [env._get_end_effector_position()]

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
            current_angles = np.array(
                [p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
            action = direct_solution_angles - current_angles

            # 将动作分解为多个小步骤
            num_substeps = 10
            for i in range(num_substeps):
                sub_action = action / num_substeps
                trajectory.append(sub_action)
                
                # 执行动作并记录状态
                obs, reward, done, _ = env.step(sub_action)
                
                # 记录每个时刻的状态
                current_joint_angles = obs[:env.num_joints]
                current_joint_positions = obs[env.num_joints:-3].reshape(env.num_joints, 3)
                current_end_effector_pos = env._get_end_effector_position()
                
                sequence_joint_angles.append(current_joint_angles.copy())
                sequence_joint_positions.append(current_joint_positions.copy())
                sequence_end_effector_positions.append(current_end_effector_pos.copy())

            # 将轨迹保存到数据集
            dataset['initial_joint_angles'].append(joint_angles)
            dataset['initial_joint_positions'].append(joint_positions)
            dataset['target_positions'].append(target_position)
            dataset['trajectories'].append(np.array(trajectory))
            dataset['sequence_joint_angles'].append(np.array(sequence_joint_angles))
            dataset['sequence_joint_positions'].append(np.array(sequence_joint_positions))
            dataset['sequence_end_effector_positions'].append(np.array(sequence_end_effector_positions))
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
                p.getQuaternionFromEuler([0, np.pi / 6, 0]),  # 绕Y轴旋转30度
                p.getQuaternionFromEuler([np.pi / 6, 0, 0]),  # 绕X轴旋转30度
                p.getQuaternionFromEuler([0, -np.pi / 6, 0]),  # 绕Y轴反向旋转30度
                p.getQuaternionFromEuler([0, 0, np.pi / 6]),  # 绕Z轴旋转30度
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
            current_angles = np.array(
                [p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
            action = best_solution - current_angles

            # 限制动作幅度
            action = np.clip(action, -0.05, 0.05)

            # 记录动作
            trajectory.append(action)

            # 执行动作
            obs, reward, done, _ = env.step(action)
            
            # 记录每个时刻的状态
            current_joint_angles = obs[:env.num_joints]
            current_joint_positions = obs[env.num_joints:-3].reshape(env.num_joints, 3)
            current_end_effector_pos = env._get_end_effector_position()
            
            sequence_joint_angles.append(current_joint_angles.copy())
            sequence_joint_positions.append(current_joint_positions.copy())
            sequence_end_effector_positions.append(current_end_effector_pos.copy())
            
            current_pos = env._get_end_effector_position()

            if done:
                success_count += 1
                print(f"轨迹 {traj_idx}: 环境标记为完成，最终距离: {distance:.4f}")
                break

        # 将轨迹保存到数据集
        dataset['initial_joint_angles'].append(joint_angles)
        dataset['initial_joint_positions'].append(joint_positions)
        dataset['target_positions'].append(target_position)
        dataset['trajectories'].append(np.array(trajectory))
        dataset['sequence_joint_angles'].append(np.array(sequence_joint_angles))
        dataset['sequence_joint_positions'].append(np.array(sequence_joint_positions))
        dataset['sequence_end_effector_positions'].append(np.array(sequence_end_effector_positions))

    # 打印成功率
    print(f"成功到达目标的轨迹: {success_count}/{num_trajectories} ({success_count / num_trajectories * 100:.2f}%)")

    # 转换为张量
    for key in dataset:
        if key != 'trajectories' and key != 'sequence_joint_angles' and key != 'sequence_joint_positions' and key != 'sequence_end_effector_positions':
            # 对于非序列数据，直接转换为张量
            dataset[key] = torch.tensor(np.array(dataset[key]), dtype=torch.float32)
        else:
            # 对于轨迹和序列数据，保持列表形式，每个元素是一个张量
            dataset[key] = [torch.tensor(np.array(item), dtype=torch.float32) for item in dataset[key]]

    return dataset

def collect_sequence_data2(env, num_trajectories=1000, max_steps=100):
    dataset = {
        'joint_angles': [],
        'joint_positions': [],
        'target_positions': [],
        'trajectories': [],
        'sequence_joint_angles': [],  # 添加这个键
        'sequence_joint_positions': []  # 添加这个键
    }

    # 记录成功到达目标的轨迹数量
    success_count = 0
    attempted_count = 0

    # 使用tqdm创建进度条
    pbar = tqdm(total=num_trajectories)
    pbar.set_description("收集成功轨迹")

    while success_count < num_trajectories and attempted_count < num_trajectories * 3:
        attempted_count += 1
        
        # 重置环境，获取初始观测
        obs = env.reset()

        # 解析初始观测
        joint_angles = obs[:env.num_joints]
        joint_positions = obs[env.num_joints:-3].reshape(env.num_joints, 3)
        target_position = obs[-3:]

        # 使用逆运动学生成参考轨迹
        trajectory = []
        sequence_joint_angles_list = []  # 添加这个列表来记录每一步的关节角度
        sequence_joint_positions_list = []  # 添加这个列表来记录每一步的关节位置
        
        current_pos = env._get_end_effector_position()
        current_angles = np.array([p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
        
        # 记录初始状态
        # sequence_joint_angles_list.append(current_angles.copy())
        # sequence_joint_positions_list.append(joint_positions.copy())

        # 记录初始距离
        initial_distance = np.linalg.norm(target_position - current_pos)
        min_distance = initial_distance
        steps_without_progress = 0
        
        # 标记当前轨迹是否成功
        is_success = False

        # 检查目标是否可达 - 使用更严格的阈值
        if initial_distance > 0.8:
            print(f"尝试 {attempted_count}: 目标距离过远 ({initial_distance:.4f})，可能不可达")
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
            print(f"尝试 {attempted_count}: 直接IK解足够好，误差: {direct_solution_error:.4f}")
            current_angles = np.array(
                [p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
            action = direct_solution_angles - current_angles

            # 将动作分解为多个小步骤
            num_substeps = 30
            for i in range(num_substeps):
                sub_action = action / num_substeps
                trajectory.append(sub_action)
                
                # 更新当前关节角度和位置
                current_angles = current_angles + sub_action
                # 设置关节状态以获取新的关节位置
                for j, joint_id in enumerate(range(env.num_joints)):
                    p.resetJointState(env.robot_id, joint_id, current_angles[j])
                
                # 获取新的关节位置
                joint_positions_new = []
                for joint_id in range(env.num_joints):
                    joint_info = p.getLinkState(env.robot_id, joint_id)
                    joint_positions_new.extend(joint_info[0])  # 链接位置
                joint_positions_new = np.array(joint_positions_new).reshape(env.num_joints, 3)
                
                # 记录序列数据
                sequence_joint_angles_list.append(current_angles.copy())
                sequence_joint_positions_list.append(joint_positions_new.copy())

            # 标记为成功
            is_success = True
        
        # 只有成功的轨迹才保存到数据集
        if is_success:
            dataset['joint_angles'].append(joint_angles)
            dataset['joint_positions'].append(joint_positions)
            dataset['target_positions'].append(target_position)
            dataset['trajectories'].append(trajectory)
            dataset['sequence_joint_angles'].append(sequence_joint_angles_list)  # 添加序列关节角度
            dataset['sequence_joint_positions'].append(sequence_joint_positions_list)  # 添加序列关节位置
            success_count += 1
            pbar.update(1)
            pbar.set_postfix({"成功率": f"{success_count}/{attempted_count} ({success_count/attempted_count*100:.1f}%)"})

    pbar.close()
    print(f"总共尝试了 {attempted_count} 次轨迹，成功收集了 {success_count} 条轨迹")
    print(f"成功率: {success_count/attempted_count*100:.2f}%")

    return dataset


def collect_sequence_data3(env, num_trajectories=1000, max_steps=30, output_file="robot_trajectory_data.h5", batch_size=10000):
    """
    收集机器人轨迹数据并保存为HDF5格式
    
    参数:
        env: 机器人环境
        num_trajectories: 需要收集的轨迹总数
        max_steps: 每条轨迹的最大步数
        output_file: 输出的HDF5文件路径
        batch_size: 每批处理的轨迹数量，用于控制内存使用
    
    返回:
        成功收集的轨迹数量
    """
    import h5py
    import os
    import time
    import traceback
    from datetime import datetime
    
    # 检查是否已存在部分完成的数据文件
    resume_collection = False
    current_trajectory_count = 0
    
    if os.path.exists(output_file):
        try:
            with h5py.File(output_file, 'r') as f:
                if 'metadata' in f and 'completed_trajectories' in f['metadata']:
                    current_trajectory_count = f['metadata']['completed_trajectories'][()]
                    if current_trajectory_count < num_trajectories:
                        resume_collection = True
                        print(f"发现已有数据文件，已完成 {current_trajectory_count}/{num_trajectories} 条轨迹，将继续收集")
                    else:
                        print(f"已完成所有 {num_trajectories} 条轨迹的收集，无需继续")
                        return current_trajectory_count
        except Exception as e:
            print(f"读取现有数据文件时出错: {e}")
            print("将创建新的数据文件")
    
    # 创建或打开HDF5文件
    try:
        if resume_collection:
            h5_file = h5py.File(output_file, 'a')
        else:
            h5_file = h5py.File(output_file, 'w')
            
            # 创建数据集结构
            h5_file.create_group('joint_angles')
            h5_file.create_group('joint_positions')
            h5_file.create_group('target_positions')
            h5_file.create_group('trajectories')
            h5_file.create_group('sequence_joint_angles')
            h5_file.create_group('sequence_joint_positions')
            
            # 创建元数据组
            metadata = h5_file.create_group('metadata')
            metadata.create_dataset('total_trajectories', data=num_trajectories)
            metadata.create_dataset('completed_trajectories', data=0)
            metadata.create_dataset('creation_time', data=str(datetime.now()))
            metadata.create_dataset('last_update', data=str(datetime.now()))
    except Exception as e:
        print(f"创建HDF5文件时出错: {e}")
        return 0
    
    # 记录成功到达目标的轨迹数量
    success_count = current_trajectory_count
    attempted_count = 0
    batch_count = 0
    
    # 临时存储当前批次的数据
    batch_data = {
        'joint_angles': [],
        'joint_positions': [],
        'target_positions': [],
        'trajectories': [],
        'sequence_joint_angles': [],
        'sequence_joint_positions': []
    }
    
    # 使用tqdm创建进度条
    pbar = tqdm(total=num_trajectories, initial=current_trajectory_count)
    pbar.set_description("收集成功轨迹")
    
    try:
        while success_count < num_trajectories and attempted_count < num_trajectories * 3:
            attempted_count += 1
            
            # 重置环境，获取初始观测
            obs = env.reset()

            # 解析初始观测
            joint_angles = obs[:env.num_joints]
            joint_positions = obs[env.num_joints:-3].reshape(env.num_joints, 3)
            target_position = obs[-3:]

            # 使用逆运动学生成参考轨迹
            trajectory = []
            sequence_joint_angles_list = []  # 添加这个列表来记录每一步的关节角度
            sequence_joint_positions_list = []  # 添加这个列表来记录每一步的关节位置
            
            current_pos = env._get_end_effector_position()
            current_angles = np.array([p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
            
            # 记录初始距离
            initial_distance = np.linalg.norm(target_position - current_pos)
            min_distance = initial_distance
            steps_without_progress = 0
            
            # 标记当前轨迹是否成功
            is_success = False

            # 检查目标是否可达 - 使用更严格的阈值
            if initial_distance > 0.8:
                if attempted_count % 100 == 0:  # 减少日志输出频率
                    print(f"尝试 {attempted_count}: 目标距离过远 ({initial_distance:.4f})，可能不可达")
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
                if attempted_count % 100 == 0:  # 减少日志输出频率
                    print(f"尝试 {attempted_count}: 直接IK解足够好，误差: {direct_solution_error:.4f}")
                current_angles = np.array(
                    [p.getJointState(env.robot_id, joint_id)[0] for joint_id in range(env.num_joints)])
                action = direct_solution_angles - current_angles

                # 将动作分解为多个小步骤
                num_substeps = 30
                for i in range(num_substeps):
                    sub_action = action / num_substeps
                    trajectory.append(sub_action)
                    
                    # 更新当前关节角度和位置
                    current_angles = current_angles + sub_action
                    # 设置关节状态以获取新的关节位置
                    for j, joint_id in enumerate(range(env.num_joints)):
                        p.resetJointState(env.robot_id, joint_id, current_angles[j])
                    
                    # 获取新的关节位置
                    joint_positions_new = []
                    for joint_id in range(env.num_joints):
                        joint_info = p.getLinkState(env.robot_id, joint_id)
                        joint_positions_new.extend(joint_info[0])  # 链接位置
                    joint_positions_new = np.array(joint_positions_new).reshape(env.num_joints, 3)
                    
                    # 记录序列数据
                    sequence_joint_angles_list.append(current_angles.copy())
                    sequence_joint_positions_list.append(joint_positions_new.copy())

                # 标记为成功
                is_success = True
            
            # 只有成功的轨迹才保存到数据集
            if is_success:
                # 将数据添加到当前批次
                batch_data['joint_angles'].append(joint_angles)
                batch_data['joint_positions'].append(joint_positions)
                batch_data['target_positions'].append(target_position)
                batch_data['trajectories'].append(trajectory)
                batch_data['sequence_joint_angles'].append(sequence_joint_angles_list)
                batch_data['sequence_joint_positions'].append(sequence_joint_positions_list)
                
                success_count += 1
                batch_count += 1
                pbar.update(1)
                pbar.set_postfix({"成功率": f"{success_count}/{attempted_count} ({success_count/attempted_count*100:.1f}%)"})
                
                # 当达到批次大小或完成所有轨迹时，将数据写入HDF5文件
                if batch_count >= batch_size or success_count >= num_trajectories:
                    # 将批次数据保存到HDF5文件
                    save_batch_to_hdf5(h5_file, batch_data, current_trajectory_count)
                    
                    # 更新元数据
                    h5_file['metadata']['completed_trajectories'][()] = success_count
                    h5_file['metadata']['last_update'][()] = str(datetime.now())
                    
                    # 确保数据写入磁盘
                    h5_file.flush()
                    
                    # 重置批次数据和计数器
                    for key in batch_data:
                        batch_data[key] = []
                    current_trajectory_count = success_count
                    batch_count = 0
                    
                    # 打印进度信息
                    print(f"已保存 {success_count}/{num_trajectories} 条轨迹到 {output_file}")
    
    except KeyboardInterrupt:
        print("\n用户中断数据收集过程")
    except Exception as e:
        print(f"\n数据收集过程中出错: {e}")
        traceback.print_exc()
    finally:
        # 确保保存最后一批数据（如果有）
        if batch_count > 0:
            try:
                save_batch_to_hdf5(h5_file, batch_data, current_trajectory_count)
                h5_file['metadata']['completed_trajectories'][()] = success_count
                h5_file['metadata']['last_update'][()] = str(datetime.now())
                h5_file.flush()
            except Exception as e:
                print(f"保存最后一批数据时出错: {e}")
        
        # 关闭HDF5文件
        h5_file.close()
        pbar.close()
    
    print(f"总共尝试了 {attempted_count} 次轨迹，成功收集了 {success_count} 条轨迹")
    print(f"成功率: {success_count/attempted_count*100:.2f}%")
    print(f"数据已保存到: {output_file}")
    
    return success_count

def save_batch_to_hdf5(h5_file, batch_data, start_idx):
    """
    将一批数据保存到HDF5文件
    
    参数:
        h5_file: 打开的HDF5文件对象
        batch_data: 包含批次数据的字典
        start_idx: 起始索引
    """
    # 获取批次大小
    batch_size = len(batch_data['joint_angles'])
    if batch_size == 0:
        return
    
    # 遍历批次中的每条轨迹
    for i in range(batch_size):
        idx = start_idx + i
        idx_str = str(idx)
        
        # 保存关节角度
        h5_file['joint_angles'].create_dataset(idx_str, data=np.array(batch_data['joint_angles'][i]))
        
        # 保存关节位置
        h5_file['joint_positions'].create_dataset(idx_str, data=np.array(batch_data['joint_positions'][i]))
        
        # 保存目标位置
        h5_file['target_positions'].create_dataset(idx_str, data=np.array(batch_data['target_positions'][i]))
        
        # 保存轨迹
        trajectory_data = np.array(batch_data['trajectories'][i])
        h5_file['trajectories'].create_dataset(idx_str, data=trajectory_data)
        
        # 保存序列关节角度
        seq_joint_angles = np.array(batch_data['sequence_joint_angles'][i])
        h5_file['sequence_joint_angles'].create_dataset(idx_str, data=seq_joint_angles)
        
        # 保存序列关节位置
        seq_joint_positions = np.array(batch_data['sequence_joint_positions'][i])
        h5_file['sequence_joint_positions'].create_dataset(idx_str, data=seq_joint_positions)
