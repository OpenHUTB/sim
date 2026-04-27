"""
逆运动学求解器测试

测试改进的IK算法的正确性和性能
"""

import unittest
import numpy as np
from math import pi
import sys
sys.path.insert(0, '..')

from fingers_sim.ik_solver import ImprovedIKSolver, compute_fingertip_position


class TestImprovedIKSolver(unittest.TestCase):
    """测试改进的IK求解器"""
    
    def setUp(self):
        """测试前设置"""
        self.solver = ImprovedIKSolver()
        
        # 测试用的连杆参数
        self.test_links = [
            0.025, 0.007, 0.018,  # 偏移量
            0.005, 0.007,         # L1
            0.005, 0.007,         # L2
            0.002, 0.002          # L3
        ]
        
        # 单位旋转矩阵
        self.R = np.eye(3)
    
    def test_jacobian_computation(self):
        """测试雅可比矩阵计算"""
        q = np.array([pi/4, pi/4, pi/4])
        
        Jp = self.solver.compute_jacobian(q, self.test_links, self.R)
        
        # 检查维度
        self.assertEqual(Jp.shape, (3, 3))
        
        # 检查第一列（x方向）应该为0（手指在yz平面运动）
        self.assertAlmostEqual(Jp[0, 0], 0.0, places=5)
        self.assertAlmostEqual(Jp[0, 1], 0.0, places=5)
        self.assertAlmostEqual(Jp[0, 2], 0.0, places=5)
    
    def test_distance_jacobian(self):
        """测试距离雅可比计算"""
        q = np.array([pi/4, pi/4, pi/4])
        p = compute_fingertip_position(q, self.test_links, self.R)
        Jp = self.solver.compute_jacobian(q, self.test_links, self.R)
        
        Jd = self.solver.compute_distance_jacobian(p, Jp)
        
        # 检查维度
        self.assertEqual(Jd.shape, (1, 3))
        
        # 检查是否非零
        self.assertTrue(np.any(Jd != 0))
    
    def test_damping_computation(self):
        """测试阻尼系数计算"""
        # 正常情况
        damping_normal = self.solver.compute_damping(0.1)
        self.assertEqual(damping_normal, self.solver.damping_base)
        
        # 奇异点附近
        damping_singular = self.solver.compute_damping(1e-5)
        self.assertGreater(damping_singular, self.solver.damping_base)
        self.assertLessEqual(damping_singular, self.solver.damping_max)
    
    def test_joint_limit_avoidance(self):
        """测试关节限位避免"""
        # 正常位置
        q_normal = np.array([pi/8, pi/8, pi/8])
        avoidance = self.solver.compute_joint_limit_avoidance(q_normal)
        self.assertTrue(np.allclose(avoidance, 0))
        
        # 接近下限
        q_near_min = np.array([0.01, pi/8, pi/8])
        avoidance = self.solver.compute_joint_limit_avoidance(q_near_min)
        self.assertGreater(avoidance[0], 0)
        
        # 接近上限
        q_near_max = np.array([5*pi/16 - 0.01, pi/8, pi/8])
        avoidance = self.solver.compute_joint_limit_avoidance(q_near_max)
        self.assertLess(avoidance[0], 0)
    
    def test_ik_solution_convergence(self):
        """测试IK解的收敛性"""
        # 初始关节角度
        q_current = np.array([pi/8, pi/8, pi/8])
        
        # 计算当前位置
        p_current = compute_fingertip_position(q_current, self.test_links, self.R)
        d_current = np.linalg.norm(p_current)
        
        # 目标距离（稍微小一点）
        target_distance = d_current * 0.8
        
        # 迭代求解
        for _ in range(50):
            q_dot = self.solver.solve(target_distance, q_current, p_current, 
                                      self.test_links, self.R, Kp=50.0)
            q_current = q_current + q_dot * 0.01  # 模拟时间步
            q_current = np.clip(q_current, 0, pi/2)  # 限位
            p_current = compute_fingertip_position(q_current, self.test_links, self.R)
        
        # 检查是否收敛
        d_final = np.linalg.norm(p_current)
        error = abs(d_final - target_distance)
        self.assertLess(error, 0.01, f"IK did not converge: error = {error}")
    
    def test_null_space_projection(self):
        """测试零空间投影"""
        J = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        
        N = self.solver.compute_null_space_projection(J)
        
        # 检查维度
        self.assertEqual(N.shape, (3, 3))
        
        # 检查投影性质: J @ N 应该接近0
        JN = J @ N
        self.assertTrue(np.allclose(JN, 0, atol=1e-10))
    
    def test_speed_limit(self):
        """测试速度限制"""
        q = np.array([pi/4, pi/4, pi/4])
        p = compute_fingertip_position(q, self.test_links, self.R)
        
        # 设置一个很大的目标距离差，产生大速度
        target_distance = np.linalg.norm(p) + 1.0
        
        q_dot = self.solver.solve(target_distance, q, p, self.test_links, self.R, Kp=1000.0)
        
        # 检查速度是否被限制
        speed = np.linalg.norm(q_dot)
        self.assertLessEqual(speed, 5.0 + 1e-6)


class TestForwardKinematics(unittest.TestCase):
    """测试正向运动学"""
    
    def test_fingertip_position(self):
        """测试指尖位置计算"""
        links = [0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005]
        R = np.eye(3)
        
        # 零角度时
        q_zero = np.array([0, 0, 0])
        p_zero = compute_fingertip_position(q_zero, links, R)
        
        # 检查z坐标应该是正的（指尖在手掌上方）
        self.assertGreater(p_zero[2], 0)
        
        # 弯曲时
        q_bent = np.array([pi/4, pi/4, pi/4])
        p_bent = compute_fingertip_position(q_bent, links, R)
        
        # 弯曲后y坐标应该变化
        self.assertNotEqual(p_zero[1], p_bent[1])


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def test_ik_computation_speed(self):
        """测试IK计算速度"""
        import time
        
        solver = ImprovedIKSolver()
        links = [0.025, 0.007, 0.018, 0.005, 0.007, 0.005, 0.007, 0.002, 0.002]
        R = np.eye(3)
        
        q = np.array([pi/4, pi/4, pi/4])
        p = compute_fingertip_position(q, links, R)
        target = np.linalg.norm(p) * 0.9
        
        # 预热
        for _ in range(10):
            solver.solve(target, q, p, links, R)
        
        # 计时
        start = time.time()
        iterations = 1000
        for _ in range(iterations):
            solver.solve(target, q, p, links, R)
        elapsed = time.time() - start
        
        avg_time = elapsed / iterations * 1000  # ms
        print(f"\nAverage IK solve time: {avg_time:.3f} ms")
        
        # 应该小于1ms
        self.assertLess(avg_time, 1.0)


if __name__ == '__main__':
    unittest.main()
