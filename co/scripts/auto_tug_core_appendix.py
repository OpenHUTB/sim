#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动牵引作业核心控制程序

文件用途：
    本程序用于论文附录或工程说明中展示自动牵引作业的核心算法逻辑。
    程序保留了自动牵引任务状态机、关键目标点生成、相对位姿误差控制、
    路径曲率计算以及前后转向桥角度映射等核心内容。

说明：
    1. 本文件为核心算法脚本，省略 ROS 节点初始化、Gazebo 服务调用、
       控制器话题发布等工程外围代码。
    2. 实际仿真运行时，可将 publish_motion() 替换为 ROS 控制器发布函数。
    3. 飞机位姿 aircraft_pose 可由 Gazebo 真值、视觉识别、激光雷达定位
       或其他感知模块提供。
"""

import math
import enum
from dataclasses import dataclass
from typing import Tuple


# ============================================================
# 基础数学工具
# ============================================================

def clamp(value: float, lower: float, upper: float) -> float:
    """将数值限制在指定范围内。"""
    return max(lower, min(upper, value))


def wrap_to_pi(angle: float) -> float:
    """将角度归一化到 [-pi, pi]。"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def distance_2d(p1: "Pose2D", p2: "Pose2D") -> float:
    """计算两个二维位姿点之间的平面距离。"""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.sqrt(dx * dx + dy * dy)


@dataclass
class Pose2D:
    """
    二维位姿数据结构。

    参数：
        x   : 世界坐标系下的 x 坐标
        y   : 世界坐标系下的 y 坐标
        yaw : 世界坐标系下的航向角，单位 rad
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


@dataclass
class ControlCommand:
    """
    车辆控制指令。

    参数：
        speed       : 车辆线速度，单位 m/s
        front_delta : 前转向桥角度，单位 rad
        back_delta  : 后转向桥角度，单位 rad
        reached     : 是否到达当前目标点
    """

    speed: float
    front_delta: float
    back_delta: float
    reached: bool = False


class TaskState(enum.Enum):
    """
    自动牵引作业状态机。
    """

    SEARCH_TARGET = 0          # 搜索并确认飞机目标
    APPROACH_PRE_DOCK = 1      # 前往预对接点
    PRECISE_DOCK = 2           # 低速精确对接
    ATTACH = 3                 # 建立牵引连接
    PUSHBACK = 4               # 推出飞机
    RELEASE = 5                # 解除牵引连接
    RETREAT = 6                # 车辆撤离
    DONE = 7                   # 作业完成
    ABORT = 8                  # 任务中止


# ============================================================
# 自动牵引核心算法
# ============================================================

class AutoTugCore:
    """
    自动牵引作业核心控制类。

    主要功能：
        1. 根据飞机位姿生成预对接点、对接点、推出点和撤离点；
        2. 根据牵引车与目标点之间的相对位姿误差生成速度和曲率；
        3. 将路径曲率映射为前后转向桥角度；
        4. 通过有限状态机组织完整自动牵引作业流程。
    """

    def __init__(self):
        # -----------------------------
        # 车辆几何参数
        # -----------------------------
        self.wheelbase = 1.20
        self.max_steer_angle = 0.32

        # 牵引车前牵引钩相对车体中心的偏移
        self.hitch_offset_x = 0.65
        self.hitch_offset_y = 0.00

        # 飞机模型原点到前起落架目标点的偏移
        self.nose_gear_offset_x = 2.50
        self.nose_gear_offset_y = 0.00

        # -----------------------------
        # 作业路径参数
        # -----------------------------
        self.pre_dock_distance = 2.00
        self.pushback_distance = 5.00
        self.retreat_offset_x = -1.50
        self.retreat_offset_y = -2.00

        # -----------------------------
        # 控制参数
        # -----------------------------
        self.k_rho = 0.75
        self.k_alpha = 1.80
        self.k_yaw = 0.90

        self.position_tolerance = 0.12
        self.yaw_tolerance = 0.10

        # -----------------------------
        # 分阶段速度限制
        # -----------------------------
        self.max_approach_speed = 0.70
        self.max_dock_speed = 0.22
        self.max_pushback_speed = 0.35
        self.max_retreat_speed = 0.55
        self.min_motion_speed = 0.06

        # 当前任务状态
        self.state = TaskState.SEARCH_TARGET

    # ========================================================
    # 坐标变换与任务点生成
    # ========================================================

    def transform_local_to_world(
        self,
        base_pose: Pose2D,
        local_x: float,
        local_y: float,
        local_yaw: float = 0.0
    ) -> Pose2D:
        """
        将局部坐标系下的点转换到世界坐标系。

        参数：
            base_pose : 基准位姿
            local_x   : 局部坐标系 x 方向偏移
            local_y   : 局部坐标系 y 方向偏移
            local_yaw : 相对航向角

        返回：
            转换后的世界坐标系位姿
        """

        c = math.cos(base_pose.yaw)
        s = math.sin(base_pose.yaw)

        world_x = base_pose.x + c * local_x - s * local_y
        world_y = base_pose.y + s * local_x + c * local_y
        world_yaw = wrap_to_pi(base_pose.yaw + local_yaw)

        return Pose2D(world_x, world_y, world_yaw)

    def generate_task_points(
        self,
        aircraft_pose: Pose2D
    ) -> Tuple[Pose2D, Pose2D, Pose2D, Pose2D]:
        """
        根据飞机位姿生成自动牵引作业所需关键目标点。

        生成点包括：
            pre_dock_pose : 预对接点
            dock_pose     : 精确对接点
            pushback_pose : 推出目标点
            retreat_pose  : 车辆撤离点

        设计依据：
            对接时牵引车前牵引钩应与飞机前起落架位置重合。
        """

        # 飞机前起落架位置
        nose_gear_pose = self.transform_local_to_world(
            aircraft_pose,
            self.nose_gear_offset_x,
            self.nose_gear_offset_y,
            0.0
        )

        # 对接点：牵引车前牵引钩与飞机前起落架重合
        dock_pose = self.transform_local_to_world(
            nose_gear_pose,
            -self.hitch_offset_x,
            -self.hitch_offset_y,
            0.0
        )

        # 预对接点：沿飞机航向反方向后退一定距离
        pre_dock_pose = self.transform_local_to_world(
            dock_pose,
            -self.pre_dock_distance,
            0.0,
            0.0
        )

        # 推出点：沿飞机航向反方向推出
        pushback_pose = self.transform_local_to_world(
            dock_pose,
            -self.pushback_distance,
            0.0,
            0.0
        )

        # 撤离点：车辆完成解挂后向侧后方撤离
        retreat_pose = self.transform_local_to_world(
            pushback_pose,
            self.retreat_offset_x,
            self.retreat_offset_y,
            0.0
        )

        return pre_dock_pose, dock_pose, pushback_pose, retreat_pose

    # ========================================================
    # 相对位姿误差控制
    # ========================================================

    def compute_pose_control(
        self,
        current_pose: Pose2D,
        target_pose: Pose2D,
        max_speed: float
    ) -> Tuple[float, float, float, float]:
        """
        基于相对位姿误差的车辆运动控制律。

        控制步骤：
            1. 计算目标点相对牵引车的位置误差；
            2. 将世界坐标误差转换到车辆自身坐标系；
            3. 根据距离误差生成线速度；
            4. 根据横向误差和航向误差生成角速度；
            5. 将角速度转换为参考路径曲率。

        返回：
            speed     : 车辆线速度
            curvature : 参考路径曲率
            distance  : 车辆到目标点的距离
            yaw_error : 航向角误差绝对值
        """

        dx = target_pose.x - current_pose.x
        dy = target_pose.y - current_pose.y

        # 世界坐标误差转换到车辆坐标系
        c = math.cos(current_pose.yaw)
        s = math.sin(current_pose.yaw)

        ex = c * dx + s * dy
        ey = -s * dx + c * dy

        distance = math.sqrt(dx * dx + dy * dy)
        target_angle_body = math.atan2(ey, ex)

        # 判断目标点位于车辆前方还是后方
        direction = 1.0
        alpha = target_angle_body

        if abs(target_angle_body) > math.pi / 2.0:
            direction = -1.0
            alpha = wrap_to_pi(target_angle_body - math.pi)

        yaw_error = wrap_to_pi(target_pose.yaw - current_pose.yaw)

        # 距离误差生成线速度
        speed = self.k_rho * distance
        speed = clamp(speed, self.min_motion_speed, max_speed)
        speed *= direction

        # 远距离主要朝向目标点，近距离同时修正最终航向
        if distance > 0.8:
            omega = self.k_alpha * alpha
        else:
            omega = self.k_alpha * alpha + self.k_yaw * yaw_error

        # 倒车接近时修正角速度方向
        if direction < 0.0:
            omega = -omega

        # 角速度转换为路径曲率
        if abs(speed) < 1e-5:
            curvature = 0.0
        else:
            curvature = omega / speed

        return speed, curvature, distance, abs(yaw_error)

    # ========================================================
    # 四轮转向映射
    # ========================================================

    def curvature_to_steering(
        self,
        curvature: float,
        mode: str
    ) -> Tuple[float, float]:
        """
        将参考路径曲率转换为前后转向桥角度。

        参数：
            curvature : 参考路径曲率
            mode      : 转向模式

        转向模式：
            front_only    : 前桥转向，后桥回中；
            counter_phase : 前后桥反相转向，适合小半径转弯和精确对接；
            crab          : 前后桥同相转向，适合斜行或横向微调。

        返回：
            front_delta : 前桥转角
            back_delta  : 后桥转角
        """

        if abs(curvature) < 1e-6:
            return 0.0, 0.0

        if mode == "front_only":
            front_delta = math.atan(self.wheelbase * curvature)
            back_delta = 0.0

        elif mode == "counter_phase":
            front_delta = math.atan(0.5 * self.wheelbase * curvature)
            back_delta = -front_delta

        elif mode == "crab":
            front_delta = math.atan(0.5 * self.wheelbase * curvature)
            back_delta = front_delta

        else:
            front_delta = math.atan(self.wheelbase * curvature)
            back_delta = 0.0

        front_delta = clamp(
            front_delta,
            -self.max_steer_angle,
            self.max_steer_angle
        )

        back_delta = clamp(
            back_delta,
            -self.max_steer_angle,
            self.max_steer_angle
        )

        return front_delta, back_delta

    def go_to_pose_step(
        self,
        current_pose: Pose2D,
        target_pose: Pose2D,
        max_speed: float,
        steering_mode: str
    ) -> ControlCommand:
        """
        执行一次目标点跟踪控制。

        实际 ROS 工程中，该函数应被周期性调用，
        并将返回的速度和转角发送至车辆控制器。
        """

        speed, curvature, distance, yaw_error = self.compute_pose_control(
            current_pose,
            target_pose,
            max_speed
        )

        if distance < self.position_tolerance and yaw_error < self.yaw_tolerance:
            return ControlCommand(
                speed=0.0,
                front_delta=0.0,
                back_delta=0.0,
                reached=True
            )

        front_delta, back_delta = self.curvature_to_steering(
            curvature,
            steering_mode
        )

        return ControlCommand(
            speed=speed,
            front_delta=front_delta,
            back_delta=back_delta,
            reached=False
        )

    # ========================================================
    # 自动牵引作业状态机
    # ========================================================

    def run_task_state_machine(
        self,
        aircraft_pose: Pose2D,
        current_tug_pose: Pose2D,
        target_valid: bool = True
    ) -> ControlCommand:
        """
        自动牵引作业状态机主逻辑。

        参数：
            aircraft_pose     : 飞机当前位姿
            current_tug_pose  : 牵引车当前位姿
            target_valid      : 飞机目标识别是否有效

        返回：
            当前周期的车辆控制指令
        """

        if not target_valid or aircraft_pose is None:
            self.state = TaskState.SEARCH_TARGET
            return ControlCommand(0.0, 0.0, 0.0, reached=False)

        pre_dock_pose, dock_pose, pushback_pose, retreat_pose = \
            self.generate_task_points(aircraft_pose)

        # 1. 搜索并确认飞机目标
        if self.state == TaskState.SEARCH_TARGET:
            self.state = TaskState.APPROACH_PRE_DOCK
            return ControlCommand(0.0, 0.0, 0.0, reached=False)

        # 2. 前往预对接点
        if self.state == TaskState.APPROACH_PRE_DOCK:
            cmd = self.go_to_pose_step(
                current_tug_pose,
                pre_dock_pose,
                self.max_approach_speed,
                "front_only"
            )

            if cmd.reached:
                self.state = TaskState.PRECISE_DOCK

            return cmd

        # 3. 低速精确对接
        if self.state == TaskState.PRECISE_DOCK:
            cmd = self.go_to_pose_step(
                current_tug_pose,
                dock_pose,
                self.max_dock_speed,
                "counter_phase"
            )

            if cmd.reached:
                self.state = TaskState.ATTACH

            return cmd

        # 4. 建立牵引连接
        if self.state == TaskState.ATTACH:
            # 实际工程中，此处可建立虚拟牵引约束或机械连接模型。
            self.state = TaskState.PUSHBACK
            return ControlCommand(0.0, 0.0, 0.0, reached=False)

        # 5. 飞机推出
        if self.state == TaskState.PUSHBACK:
            cmd = self.go_to_pose_step(
                current_tug_pose,
                pushback_pose,
                self.max_pushback_speed,
                "counter_phase"
            )

            if cmd.reached:
                self.state = TaskState.RELEASE

            return cmd

        # 6. 解除牵引连接
        if self.state == TaskState.RELEASE:
            # 实际工程中，此处解除虚拟牵引约束或机械连接。
            self.state = TaskState.RETREAT
            return ControlCommand(0.0, 0.0, 0.0, reached=False)

        # 7. 车辆撤离
        if self.state == TaskState.RETREAT:
            cmd = self.go_to_pose_step(
                current_tug_pose,
                retreat_pose,
                self.max_retreat_speed,
                "front_only"
            )

            if cmd.reached:
                self.state = TaskState.DONE

            return cmd

        # 8. 作业完成
        if self.state == TaskState.DONE:
            return ControlCommand(0.0, 0.0, 0.0, reached=True)

        # 9. 异常中止
        self.state = TaskState.ABORT
        return ControlCommand(0.0, 0.0, 0.0, reached=False)

    # ========================================================
    # 工程接口占位函数
    # ========================================================

    def publish_motion(self, command: ControlCommand) -> None:
        """
        工程接口占位函数。

        在实际 ROS + Gazebo 工程中，应将该函数替换为控制器话题发布逻辑：
            1. 将 command.speed 转换为四个车轮的角速度；
            2. 将 command.front_delta 发布给前转向桥位置控制器；
            3. 将 command.back_delta 发布给后转向桥位置控制器。
        """

        print(
            "speed = {:.3f} m/s, front_delta = {:.3f} rad, "
            "back_delta = {:.3f} rad, reached = {}".format(
                command.speed,
                command.front_delta,
                command.back_delta,
                command.reached
            )
        )


# ============================================================
# 示例调用
# ============================================================

def main():
    """
    示例主函数。

    注意：
        该示例仅用于说明核心算法调用方式。
        实际仿真中 current_tug_pose 和 aircraft_pose 应由 Gazebo 或感知节点实时提供。
    """

    controller = AutoTugCore()

    # 示例：飞机位姿
    aircraft_pose = Pose2D(
        x=6.0,
        y=0.0,
        yaw=0.0
    )

    # 示例：牵引车当前位姿
    current_tug_pose = Pose2D(
        x=0.0,
        y=-1.0,
        yaw=0.0
    )

    # 执行一次状态机计算
    command = controller.run_task_state_machine(
        aircraft_pose=aircraft_pose,
        current_tug_pose=current_tug_pose,
        target_valid=True
    )

    controller.publish_motion(command)


if __name__ == "__main__":
    main()
