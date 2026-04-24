#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import threading
import rospy
from std_msgs.msg import Float64

if sys.platform != 'win32':
    import tty
    import termios
    import select


# =========================
# 参数
# =========================
STEER_ANGLE = 0.32

SPEED_STEP = 0.8
MAX_SPEED = 15

TURN_SPEED_LIMIT = 2.0
WHEEL_ACCEL_STEP = 0.12

STEER_SPEED = 0.05
STEER_TIMEOUT = 0.30          # 转向保持时间
DRIVE_HOLD_TIMEOUT = 0.20     # 动力键保持时间
STEER_SETTLE_TIME = 0.25

RATE_HZ = 50


# =========================
# 状态变量
# =========================
target_speed = 0.0
current_speed = 0.0

front_cmd = 0.0
back_cmd = 0.0

current_front = 0.0
current_back = 0.0

rear_unlocked = False

last_steer_time = 0.0
last_drive_time = 0.0
last_steer_cmd_time = 0.0

running = True


# =========================
# 键盘线程
# =========================
def keyboard_thread():
    global target_speed, front_cmd, back_cmd
    global rear_unlocked, last_steer_time, last_drive_time, last_steer_cmd_time, running

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        while running and not rospy.is_shutdown():
            dr, _, _ = select.select([sys.stdin], [], [], 0.01)
            if not dr:
                continue

            key = sys.stdin.read(1).lower()
            now = time.time()

            if key == 'w':
                target_speed += SPEED_STEP
                if target_speed > MAX_SPEED:
                    target_speed = MAX_SPEED
                last_drive_time = now

                # 如果当前已经有转向命令，按动力键时继续保持转向
                if abs(front_cmd) > 1e-3 or abs(back_cmd) > 1e-3:
                    last_steer_time = now

            elif key == 's':
                target_speed -= SPEED_STEP
                if target_speed < -MAX_SPEED:
                    target_speed = -MAX_SPEED
                last_drive_time = now

                if abs(front_cmd) > 1e-3 or abs(back_cmd) > 1e-3:
                    last_steer_time = now

            elif key == 'a':
                front_cmd = +STEER_ANGLE
                if rear_unlocked:
                    back_cmd = -STEER_ANGLE
                else:
                    back_cmd = 0.0
                last_steer_time = now
                last_steer_cmd_time = now

            elif key == 'd':
                front_cmd = -STEER_ANGLE
                if rear_unlocked:
                    back_cmd = +STEER_ANGLE
                else:
                    back_cmd = 0.0
                last_steer_time = now
                last_steer_cmd_time = now

            elif key == 'j':
                back_cmd = +STEER_ANGLE
                last_steer_time = now
                last_steer_cmd_time = now

            elif key == 'k':
                back_cmd = -STEER_ANGLE
                last_steer_time = now
                last_steer_cmd_time = now

            elif key == 'l':
                rear_unlocked = not rear_unlocked
                if not rear_unlocked:
                    back_cmd = 0.0
                print("\n后桥{}".format("解锁" if rear_unlocked else "锁定"))

            elif key == ' ':
                target_speed = 0.0

            elif key == 'c':
                front_cmd = 0.0
                back_cmd = 0.0

            elif key == 'q':
                running = False
                break

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)


# =========================
# 主程序
# =========================
def main():
    global current_speed, target_speed
    global current_front, current_back
    global front_cmd, back_cmd
    global last_steer_time, last_drive_time

    rospy.init_node("keyboard_vehicle_control_real")

    pub_front = rospy.Publisher(
        "/zongzhuang/front_steer_position_controller/command",
        Float64, queue_size=1)

    pub_back = rospy.Publisher(
        "/zongzhuang/back_steer_position_controller/command",
        Float64, queue_size=1)

    pub_fl = rospy.Publisher(
        "/zongzhuang/front_left_wheel_velocity_controller/command",
        Float64, queue_size=1)

    pub_fr = rospy.Publisher(
        "/zongzhuang/front_right_wheel_velocity_controller/command",
        Float64, queue_size=1)

    pub_bl = rospy.Publisher(
        "/zongzhuang/back_left_wheel_velocity_controller/command",
        Float64, queue_size=1)

    pub_br = rospy.Publisher(
        "/zongzhuang/back_right_wheel_velocity_controller/command",
        Float64, queue_size=1)

    print("\n控制说明：")
    print("w/s 前进后退 | a/d 转向 | l 后桥锁/解锁 | j/k 后桥单独 | 空格刹车 | c回中 | q退出\n")

    th = threading.Thread(target=keyboard_thread)
    th.daemon = True
    th.start()

    rate = rospy.Rate(RATE_HZ)

    while not rospy.is_shutdown() and running:
        now = time.time()

        # =========================
        # 自动回中逻辑
        # =========================
        steering_active = (abs(front_cmd) > 1e-3 or abs(back_cmd) > 1e-3)
        drive_active = abs(target_speed) > 1e-3 and (now - last_drive_time) < DRIVE_HOLD_TIMEOUT

        # 只有“既没有继续转向输入，也没有继续动力输入辅助保持”时才回中
        if steering_active and (now - last_steer_time > STEER_TIMEOUT) and (not drive_active):
            front_cmd = 0.0
            back_cmd = 0.0

        # =========================
        # 转向平滑
        # =========================
        if front_cmd > current_front:
            current_front += STEER_SPEED
            if current_front > front_cmd:
                current_front = front_cmd
        elif front_cmd < current_front:
            current_front -= STEER_SPEED
            if current_front < front_cmd:
                current_front = front_cmd

        if back_cmd > current_back:
            current_back += STEER_SPEED
            if current_back > back_cmd:
                current_back = back_cmd
        elif back_cmd < current_back:
            current_back -= STEER_SPEED
            if current_back < back_cmd:
                current_back = back_cmd

        current_front = max(min(current_front, STEER_ANGLE), -STEER_ANGLE)
        current_back = max(min(current_back, STEER_ANGLE), -STEER_ANGLE)

        # =========================
        # 轮速控制
        # =========================
        allowed_speed = target_speed

        # 转向时限速
        if abs(front_cmd) > 1e-3 or abs(back_cmd) > 1e-3:
            if allowed_speed > TURN_SPEED_LIMIT:
                allowed_speed = TURN_SPEED_LIMIT
            elif allowed_speed < -TURN_SPEED_LIMIT:
                allowed_speed = -TURN_SPEED_LIMIT

        # 刚打方向后，先让桥到位，再给动力
        if now - last_steer_cmd_time < STEER_SETTLE_TIME:
            allowed_speed = 0.0

        # 轮速平滑
        if current_speed < allowed_speed:
            current_speed += WHEEL_ACCEL_STEP
            if current_speed > allowed_speed:
                current_speed = allowed_speed
        elif current_speed > allowed_speed:
            current_speed -= WHEEL_ACCEL_STEP
            if current_speed < allowed_speed:
                current_speed = allowed_speed

        # =========================
        # 发布命令
        # =========================
        pub_front.publish(current_front)
        pub_back.publish(current_back)

        pub_fl.publish(current_speed)
        pub_fr.publish(current_speed)
        pub_bl.publish(current_speed)
        pub_br.publish(current_speed)

        rate.sleep()

    # 退出时回中+停车
    pub_fl.publish(0.0)
    pub_fr.publish(0.0)
    pub_bl.publish(0.0)
    pub_br.publish(0.0)
    pub_front.publish(0.0)
    pub_back.publish(0.0)

    print("\n已停止")


if __name__ == "__main__":
    main()
