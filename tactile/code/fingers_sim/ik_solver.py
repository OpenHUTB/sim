"""
改进的逆运动学求解器 - 使用阻尼最小二乘法 (Damped Least Squares)

本模块实现了改进的IK算法，相比原算法有以下改进：
1. 使用阻尼最小二乘法 (DLS) 替代简单的伪逆，提高数值稳定性
2. 动态调整阻尼系数，在奇异点附近自动增大阻尼
3. 添加关节限位约束
4. 使用任务优先级方法处理多目标
5. 优化计算效率

作者: [你的名字]
日期: 2025
"""

import numpy as np
from math import pi, cos, sin, sqrt
from typing import List, Tuple, Optional


class ImprovedIKSolver:
    """改进的逆运动学求解器"""
    
    def __init__(self, 
                 damping_base: float = 0.01,
                 damping_max: float = 0.5,
                 singular_threshold: float = 1e-4,
                 joint_limit_weight: float = 10.0):
        """
        初始化IK求解器
        
        Args:
            damping_base: 基础阻尼系数
            damping_max: 最大阻尼系数
            singular_threshold: 奇异点检测阈值
            joint_limit_weight: 关节限位权重
        """
        self.damping_base = damping_base
        self.damping_max = damping_max
        self.singular_threshold = singular_threshold
        self.joint_limit_weight = joint_limit_weight
        
        # 关节限位 (弧度)
        self.joint_limits = {
            'J1': (0, 5*pi/16),      # 近端关节
            'J2': (0, 3*pi/8),       # 中端关节
            'J3': (0, 5*pi/16),      # 远端关节
        }
    
    def compute_jacobian(self, q: np.ndarray, links: List[float], R: np.ndarray) -> np.ndarray:
        """
        计算任务雅可比矩阵
        
        Args:
            q: 关节角度 [q1, q2, q3]
            links: 连杆长度参数
            R: 旋转矩阵
            
        Returns:
            Jp: 3x3 位置雅可比矩阵
        """
        q1, q2, q3 = q
        
        # 提取连杆长度
        L1y, L1z = links[3], links[4]
        L2y, L2z = links[5], links[6]
        L3y, L3z = links[7], links[8]
        
        # 计算雅可比矩阵 (相对于本地坐标系)
        Jp_local = np.array([
            [0, 0, 0],
            [L1y*sin(q1) - L1z*cos(q1) + 
             L2y*sin(q1+q2) - L2z*cos(q1+q2) + 
             L3y*sin(q1+q2+q3) - L3z*cos(q1+q2+q3),
             L2y*sin(q1+q2) - L2z*cos(q1+q2) + 
             L3y*sin(q1+q2+q3) - L3z*cos(q1+q2+q3),
             L3y*sin(q1+q2+q3) - L3z*cos(q1+q2+q3)],
            [-L1y*cos(q1) - L1z*sin(q1) - 
             L2y*cos(q1+q2) - L2z*sin(q1+q2) - 
             L3y*cos(q1+q2+q3) - L3z*sin(q1+q2+q3),
             -L2y*cos(q1+q2) - L2z*sin(q1+q2) - 
             L3y*cos(q1+q2+q3) - L3z*sin(q1+q2+q3),
             -L3y*cos(q1+q2+q3) - L3z*sin(q1+q2+q3)]
        ], dtype=float)
        
        # 转换到世界坐标系
        Jp = R @ Jp_local
        
        return Jp
    
    def compute_distance_jacobian(self, p: np.ndarray, Jp: np.ndarray) -> np.ndarray:
        """
        计算距离雅可比矩阵 (标量距离对关节角度的导数)
        
        Args:
            p: 指尖位置 (3x1)
            Jp: 位置雅可比矩阵 (3x3)
            
        Returns:
            Jd: 1x3 距离雅可比矩阵
        """
        d = np.linalg.norm(p)
        if d < 1e-6:
            d = 1e-6
        Jd = (p.T @ Jp) / d
        return Jd
    
    def compute_damping(self, singular_value: float) -> float:
        """
        动态计算阻尼系数
        
        在奇异点附近自动增大阻尼以提高稳定性
        
        Args:
            singular_value: 最小奇异值
            
        Returns:
            damping: 阻尼系数
        """
        if singular_value > self.singular_threshold:
            return self.damping_base
        else:
            # 在奇异点附近平滑增加阻尼
            ratio = singular_value / self.singular_threshold
            damping = self.damping_base + (self.damping_max - self.damping_base) * (1 - ratio)**2
            return damping
    
    def compute_null_space_projection(self, J: np.ndarray) -> np.ndarray:
        """
        计算零空间投影矩阵
        
        Args:
            J: 雅可比矩阵
            
        Returns:
            N: 零空间投影矩阵
        """
        # 使用SVD计算伪逆
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        
        # 计算阻尼伪逆
        damping = self.compute_damping(np.min(S))
        S_damped = S / (S**2 + damping**2)
        J_pinv = Vt.T @ np.diag(S_damped) @ U.T
        
        # 零空间投影矩阵
        N = np.eye(3) - J_pinv @ J
        
        return N
    
    def compute_joint_limit_avoidance(self, q: np.ndarray) -> np.ndarray:
        """
        计算关节限位避免速度
        
        Args:
            q: 当前关节角度 [q1, q2, q3]
            
        Returns:
            q_dot_avoid: 关节限位避免速度
        """
        q_dot_avoid = np.zeros(3)
        
        for i, (joint_name, (q_min, q_max)) in enumerate(self.joint_limits.items()):
            qi = q[i]
            
            # 计算到限位的距离
            dist_to_min = qi - q_min
            dist_to_max = q_max - qi
            
            # 如果接近限位，产生远离限位的速度
            margin = (q_max - q_min) * 0.1  # 10% 边界
            
            if dist_to_min < margin:
                # 接近下限，向正方向移动
                q_dot_avoid[i] = self.joint_limit_weight * (1 - dist_to_min/margin)
            elif dist_to_max < margin:
                # 接近上限，向负方向移动
                q_dot_avoid[i] = -self.joint_limit_weight * (1 - dist_to_max/margin)
        
        return q_dot_avoid
    
    def solve(self, 
              target_distance: float,
              current_q: np.ndarray,
              current_p: np.ndarray,
              links: List[float],
              R: np.ndarray,
              Kp: float = 100.0) -> np.ndarray:
        """
        求解逆运动学
        
        Args:
            target_distance: 目标距离
            current_q: 当前关节角度 [q1, q2, q3]
            current_p: 当前指尖位置 (3x1)
            links: 连杆长度参数
            R: 旋转矩阵
            Kp: 比例增益
            
        Returns:
            q_dot: 关节速度 [q1_dot, q2_dot, q3_dot]
        """
        # 计算雅可比矩阵
        Jp = self.compute_jacobian(current_q, links, R)
        Jd = self.compute_distance_jacobian(current_p, Jp)
        
        # 当前距离
        d = np.linalg.norm(current_p)
        
        # 距离误差
        distance_error = target_distance - d
        
        # 使用SVD计算阻尼伪逆
        U, S, Vt = np.linalg.svd(Jd, full_matrices=False)
        
        # 动态阻尼
        damping = self.compute_damping(np.min(S))
        
        # 阻尼伪逆
        S_inv = S / (S**2 + damping**2)
        Jd_pinv = Vt.T @ np.diag(S_inv) @ U.T
        
        # 主任务：距离控制
        q_dot_primary = Jd_pinv * Kp * distance_error
        
        # 计算零空间投影
        N = self.compute_null_space_projection(Jd)
        
        # 次要任务1：关节限位避免
        q_dot_limit = self.compute_joint_limit_avoidance(current_q)
        
        # 次要任务2：姿态优化 (使关节保持在中间位置)
        q_center = np.array([5*pi/32, 3*pi/16, 5*pi/32])  # 关节中间位置
        q_dot_posture = 0.5 * (q_center - current_q)
        
        # 组合次要任务
        q_dot_secondary = q_dot_limit + q_dot_posture
        
        # 总关节速度 = 主任务 + 零空间中的次要任务
        q_dot = q_dot_primary.flatten() + (N @ q_dot_secondary)
        
        # 限制最大速度
        max_speed = 5.0  # rad/s
        speed = np.linalg.norm(q_dot)
        if speed > max_speed:
            q_dot = q_dot * max_speed / speed
        
        return q_dot


# 全局求解器实例
_ik_solver = ImprovedIKSolver()


def move_finger_improved(finger_type: str,
                         target_distance: float,
                         data,
                         model,
                         joint_ids: dict,
                         joint_qpos_adr: dict,
                         actuator_ids: dict,
                         palm_pos: np.ndarray,
                         links: List[float],
                         R: np.ndarray,
                         hand_prefix: str):
    """
    改进的手指运动控制函数
    
    Args:
        finger_type: 手指类型 ("index", "middle", "annular", "pinky")
        target_distance: 目标距离
        data: MuJoCo data
        model: MuJoCo model
        joint_ids: 关节ID字典
        joint_qpos_adr: 关节qpos地址字典
        actuator_ids: 执行器ID字典
        palm_pos: 手掌位置
        links: 连杆长度
        R: 旋转矩阵
        hand_prefix: 手前缀 ("Left_" 或 "Right_")
    """
    # 构建关节名称
    finger_name = finger_type.capitalize()
    joint_names = [f"{hand_prefix}{finger_name}_J{i}" for i in range(1, 4)]
    
    # 获取当前关节角度
    try:
        q = np.array([
            data.qpos[joint_qpos_adr[joint_names[0]]],
            data.qpos[joint_qpos_adr[joint_names[1]]],
            data.qpos[joint_qpos_adr[joint_names[2]]]
        ])
    except KeyError as e:
        print(f"Error: Joint not found - {e}")
        return
    
    # 获取当前指尖位置
    try:
        tip_body_name = f"{hand_prefix}{finger_name}_Tip"
        tip_pos = data.xpos[model.body(tip_body_name).id]
        # 相对于手掌的位置
        p = tip_pos - palm_pos
        p = p.reshape(3, 1)
    except KeyError:
        # 如果找不到Tip body，使用计算位置
        p = compute_fingertip_position(q, links, R)
    
    # 求解IK
    q_dot = _ik_solver.solve(target_distance, q, p, links, R)
    
    # 应用控制命令
    for i, joint_name in enumerate(joint_names):
        if joint_name in actuator_ids:
            actuator_id = actuator_ids[joint_name]
            data.ctrl[actuator_id] = q_dot[i]


def move_thumb_improved(target_distance: float,
                        abduction: float,
                        data,
                        model,
                        joint_ids: dict,
                        joint_qpos_adr: dict,
                        actuator_ids: dict,
                        palm_pos: np.ndarray,
                        links: List[float],
                        R: np.ndarray,
                        hand_side: int):
    """
    改进的拇指运动控制函数
    
    拇指有额外的自由度（外展/内收），需要特殊处理
    """
    # 拇指关节名称
    hand_prefix = "Right_" if hand_side == 1 else "Left_"
    joint_names = [f"{hand_prefix}Thumb_J{i}" for i in range(1, 4)]
    
    # 获取当前关节角度
    try:
        q = np.array([
            data.qpos[joint_qpos_adr[joint_names[0]]],
            data.qpos[joint_qpos_adr[joint_names[1]]],
            data.qpos[joint_qpos_adr[joint_names[2]]]
        ])
    except KeyError as e:
        print(f"Error: Thumb joint not found - {e}")
        return
    
    # 求解IK (拇指使用不同的连杆参数)
    # 这里简化处理，实际应该使用拇指特定的运动学模型
    tip_body_name = f"{hand_prefix}Thumb_Tip"
    try:
        tip_pos = data.xpos[model.body(tip_body_name).id]
        p = tip_pos - palm_pos
        p = p.reshape(3, 1)
    except KeyError:
        p = compute_fingertip_position(q, links, R)
    
    q_dot = _ik_solver.solve(target_distance, q, p, links, R)
    
    # 应用控制命令
    for i, joint_name in enumerate(joint_names):
        if joint_name in actuator_ids:
            actuator_id = actuator_ids[joint_name]
            data.ctrl[actuator_id] = q_dot[i]


def compute_fingertip_position(q: np.ndarray, links: List[float], R: np.ndarray) -> np.ndarray:
    """
    计算指尖位置（正向运动学）
    
    Args:
        q: 关节角度 [q1, q2, q3]
        links: 连杆长度
        R: 旋转矩阵
        
    Returns:
        p: 指尖位置 (3x1)
    """
    q1, q2, q3 = q
    
    # 提取连杆长度
    offset_x, offset_y, offset_z = links[0], links[1], links[2]
    L1y, L1z = links[3], links[4]
    L2y, L2z = links[5], links[6]
    L3y, L3z = links[7], links[8]
    
    # 计算关节位置 (本地坐标系)
    py_j2 = -L1y*cos(q1) - L1z*sin(q1)
    pz_j2 = -L1y*sin(q1) + L1z*cos(q1)
    
    py_j3 = py_j2 - L2y*cos(q1+q2) - L2z*sin(q1+q2)
    pz_j3 = pz_j2 - L2y*sin(q1+q2) + L2z*cos(q1+q2)
    
    py_tip = py_j3 - L3y*cos(q1+q2+q3) - L3z*sin(q1+q2+q3)
    pz_tip = pz_j3 - L3y*sin(q1+q2+q3) + L3z*cos(q1+q2+q3)
    
    # 本地坐标
    pos_local = np.array([[0], [py_tip], [pz_tip + 0.02]])
    
    # 转换到世界坐标
    offset = np.array([[offset_x], [offset_y], [offset_z]])
    p = R @ pos_local + offset
    
    return p
