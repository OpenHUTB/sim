#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyboard Control Module for AirSim Vehicle Simulation
OpenHUTB Simulation Platform

@author: MAXRainJay
@date: 2026-01-12
@version: 1.0.0
"""

import pygame
import time
import math
import random
from datetime import datetime


class VehicleSimulator:
    """Simple vehicle physics simulator"""

    def __init__(self):
        self.position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.speed = 0.0
        self.throttle = 0.0
        self.steering = 0.0
        self.brake = 0.0
        self.heading = 0.0  # vehicle heading in radians
        self.timestamp = datetime.now()

    def update(self, controls):
        """Update vehicle state based on controls"""
        dt = 0.1  # time step

        # Apply controls
        self.throttle = controls.get('throttle', 0.0)
        self.steering = controls.get('steering', 0.0)
        self.brake = controls.get('brake', 0.0)

        # Simple physics simulation
        acceleration = self.throttle * 15.0  # m/s^2
        self.speed += acceleration * dt

        # Drag and friction
        self.speed *= 0.98

        # Speed limits
        max_speed = 40.0  # m/s
        min_speed = -8.0  # m/s (reverse)
        self.speed = max(min_speed, min(self.speed, max_speed))

        # Brake effect
        if self.brake > 0:
            brake_force = self.brake * 20.0
            if self.speed > 0:
                self.speed = max(0, self.speed - brake_force * dt)
            elif self.speed < 0:
                self.speed = min(0, self.speed + brake_force * dt)

        # Update heading based on steering (only when moving)
        if abs(self.speed) > 0.5:
            turn_rate = self.steering * 2.0  # radians per second
            self.heading += turn_rate * dt

        # Update position based on speed and heading
        velocity_x = self.speed * math.cos(self.heading) * dt
        velocity_y = self.speed * math.sin(self.heading) * dt

        self.position['x'] += velocity_x
        self.position['y'] += velocity_y

        self.timestamp = datetime.now()

        return {
            'position': self.position,
            'speed': self.speed,
            'heading': self.heading,
            'controls': {
                'throttle': self.throttle,
                'steering': self.steering,
                'brake': self.brake
            }
        }

    def reset(self):
        """Reset vehicle to origin"""
        self.position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.speed = 0.0
        self.throttle = 0.0
        self.steering = 0.0
        self.brake = 0.0
        self.heading = 0.0


class AirSimKeyboardController:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 700))
        pygame.display.set_caption("🚗 AirSim Keyboard Controller - Simulator Mode")

        # Vehicle simulator
        self.vehicle = VehicleSimulator()
        self.controls = {'throttle': 0.0, 'steering': 0.0, 'brake': 0.0}

        # Control parameters
        self.max_throttle = 0.6
        self.max_steering = 0.6

        # UI fonts
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 32)

        print("🎮 AirSim Keyboard Controller Started!")
        print("Controls:")
        print("  W/S - Forward/Backward")
        print("  A/D - Turn Left/Right")
        print("  SPACE - Brake")
        print("  R - Reset Vehicle")
        print("  Q/ESC - Exit")
        print("  UP/DOWN - Adjust Max Throttle")
        print("  LEFT/RIGHT - Adjust Steering Sensitivity")

    def handle_input(self):
        """Handle keyboard input"""
        keys = pygame.key.get_pressed()

        # Reset controls
        self.controls['throttle'] = 0.0
        self.controls['steering'] = 0.0
        self.controls['brake'] = 0.0

        # Direction control
        if keys[pygame.K_w]:
            self.controls['throttle'] = self.max_throttle
        elif keys[pygame.K_s]:
            self.controls['throttle'] = -self.max_throttle * 0.4  # Reverse is slower

        # Steering control
        if keys[pygame.K_a]:
            self.controls['steering'] = -self.max_steering
        elif keys[pygame.K_d]:
            self.controls['steering'] = self.max_steering

        # Brake
        if keys[pygame.K_SPACE]:
            self.controls['brake'] = 1.0

        # Parameter adjustment
        if keys[pygame.K_UP]:
            self.max_throttle = min(1.0, self.max_throttle + 0.01)
        elif keys[pygame.K_DOWN]:
            self.max_throttle = max(0.1, self.max_throttle - 0.01)

        if keys[pygame.K_LEFT]:
            self.max_steering = min(1.0, self.max_steering + 0.01)
        elif keys[pygame.K_RIGHT]:
            self.max_steering = max(0.1, self.max_steering - 0.01)

        return self.controls

    def draw_dashboard(self, vehicle_state):
        """Draw professional dashboard"""
        # Background gradient
        for y in range(700):
            color_intensity = int(20 + (y / 700) * 15)
            pygame.draw.line(self.screen, (color_intensity, color_intensity + 5, color_intensity + 10), (0, y),
                             (1000, y))

        # Title
        title = self.title_font.render("🚗 AirSim Keyboard Controller - Simulator Mode", True, (100, 200, 255))
        self.screen.blit(title, (30, 25))

        # Vehicle Status Panel
        pygame.draw.rect(self.screen, (40, 50, 70), (30, 80, 400, 200), border_radius=10)
        pygame.draw.rect(self.screen, (60, 80, 120), (30, 80, 400, 200), 2, border_radius=10)

        status_texts = [
            f"📍 Position: X={vehicle_state['position']['x']:8.2f}, Y={vehicle_state['position']['y']:8.2f}",
            f"⚡ Speed: {vehicle_state['speed']:7.2f} m/s",
            f"🧭 Heading: {math.degrees(vehicle_state['heading']):6.1f}°",
            f"🎛️  Controls: Throttle={vehicle_state['controls']['throttle']:6.2f}, Steering={vehicle_state['controls']['steering']:6.2f}, Brake={vehicle_state['controls']['brake']:6.2f}",
            f"🔧 Parameters: Max Throttle={self.max_throttle:5.2f}, Steering Sensitivity={self.max_steering:5.2f}"
        ]

        y_pos = 100
        for text in status_texts:
            surface = self.small_font.render(text, True, (220, 230, 240))
            self.screen.blit(surface, (50, y_pos))
            y_pos += 28

        # Control Instructions Panel
        pygame.draw.rect(self.screen, (40, 60, 80), (30, 320, 400, 180), border_radius=10)
        pygame.draw.rect(self.screen, (60, 100, 140), (30, 320, 400, 180), 2, border_radius=10)

        help_texts = [
            "🎮 Controls:",
            "  W/S: Forward/Backward  |  A/D: Turn Left/Right",
            "  SPACE: Brake           |  R: Reset Vehicle",
            "  UP/DOWN: Adjust Throttle  |  LEFT/RIGHT: Adjust Steering",
            "  Q/ESC: Exit"
        ]

        y_pos = 340
        for text in help_texts:
            surface = self.small_font.render(text, True, (180, 200, 230))
            self.screen.blit(surface, (50, y_pos))
            y_pos += 25

        # Speedometer
        speed = abs(vehicle_state['speed'])
        max_display_speed = 45.0
        speed_ratio = min(speed / max_display_speed, 1.0)

        # Speedometer background
        speedo_x, speedo_y = 650, 400
        pygame.draw.circle(self.screen, (50, 60, 90), (speedo_x, speedo_y), 120)
        pygame.draw.circle(self.screen, (70, 90, 130), (speedo_x, speedo_y), 115, 3)
        pygame.draw.circle(self.screen, (30, 40, 70), (speedo_x, speedo_y), 100)

        # Speed scale (0-45 m/s)
        for i in range(0, 46, 5):
            angle = (i / 45.0) * 270 - 135  # -135 to 135 degrees
            rad_angle = math.radians(angle)
            x1 = speedo_x + 85 * math.cos(rad_angle)
            y1 = speedo_y + 85 * math.sin(rad_angle)
            x2 = speedo_x + 95 * math.cos(rad_angle)
            y2 = speedo_y + 95 * math.sin(rad_angle)
            pygame.draw.line(self.screen, (120, 140, 180), (x1, y1), (x2, y2), 2)

            # Numbers
            num_x = speedo_x + 70 * math.cos(rad_angle)
            num_y = speedo_y + 70 * math.sin(rad_angle)
            num_text = self.small_font.render(str(i), True, (200, 220, 240))
            num_rect = num_text.get_rect(center=(num_x, num_y))
            self.screen.blit(num_text, num_rect)

        # Speed needle
        speed_angle = (speed_ratio * 270 - 135) * math.pi / 180
        needle_length = 80
        needle_x = speedo_x + needle_length * math.cos(speed_angle)
        needle_y = speedo_y + needle_length * math.sin(speed_angle)

        # Needle shadow
        pygame.draw.line(self.screen, (255, 100, 100), (speedo_x, speedo_y), (needle_x, needle_y), 6)
        pygame.draw.circle(self.screen, (255, 150, 150), (speedo_x, speedo_y), 8)

        # Speed display
        speed_text = self.font.render(f"{speed:5.1f}", True, (255, 255, 255))
        speed_rect = speed_text.get_rect(center=(speedo_x, speedo_y + 30))
        self.screen.blit(speed_text, speed_rect)

        speed_unit = self.small_font.render("m/s", True, (180, 200, 230))
        unit_rect = speed_unit.get_rect(center=(speedo_x, speedo_y + 55))
        self.screen.blit(speed_unit, unit_rect)

        # Status indicators
        status_color = (0, 255, 0) if vehicle_state['speed'] > 0.1 else (255, 255, 0) if vehicle_state[
                                                                                             'speed'] < -0.1 else (100,
                                                                                                                   100,
                                                                                                                   100)
        pygame.draw.circle(self.screen, status_color, (speedo_x, speedo_y - 80), 10)

        status_text = "MOVING" if abs(vehicle_state['speed']) > 0.1 else "STOPPED"
        status_surface = self.small_font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(center=(speedo_x, speedo_y - 100))
        self.screen.blit(status_surface, status_rect)

        # Throttle and brake bars
        bar_width, bar_height = 200, 25
        bar_x, bar_y = 500, 550

        # Throttle bar
        throttle_ratio = abs(vehicle_state['controls']['throttle'])
        pygame.draw.rect(self.screen, (80, 80, 100), (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        if vehicle_state['controls']['throttle'] > 0:
            pygame.draw.rect(self.screen, (0, 255, 100), (bar_x, bar_y, int(bar_width * throttle_ratio), bar_height),
                             border_radius=5)
        else:
            pygame.draw.rect(self.screen, (255, 150, 0), (bar_x, bar_y, int(bar_width * throttle_ratio), bar_height),
                             border_radius=5)

        throttle_text = self.small_font.render(f"Throttle: {vehicle_state['controls']['throttle']:6.2f}", True,
                                               (200, 220, 240))
        self.screen.blit(throttle_text, (bar_x, bar_y - 25))

        # Brake bar
        bar_y += 40
        brake_ratio = vehicle_state['controls']['brake']
        pygame.draw.rect(self.screen, (80, 80, 100), (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, (255, 50, 50), (bar_x, bar_y, int(bar_width * brake_ratio), bar_height),
                         border_radius=5)

        brake_text = self.small_font.render(f"Brake: {vehicle_state['controls']['brake']:6.2f}", True, (200, 220, 240))
        self.screen.blit(brake_text, (bar_x, bar_y - 25))

        # Steering indicator
        steer_x, steer_y = 200, 580
        pygame.draw.circle(self.screen, (60, 70, 90), (steer_x, steer_y), 50, 2)

        # Steering wheel visualization
        steering_angle = vehicle_state['controls']['steering'] * math.pi * 0.5
        wheel_radius = 40
        for i in range(4):
            angle = steering_angle + i * math.pi / 2
            x1 = steer_x + wheel_radius * 0.7 * math.cos(angle)
            y1 = steer_y + wheel_radius * 0.7 * math.sin(angle)
            x2 = steer_x + wheel_radius * math.cos(angle)
            y2 = steer_y + wheel_radius * math.sin(angle)
            pygame.draw.line(self.screen, (150, 170, 200), (x1, y1), (x2, y2), 3)

        steer_text = self.small_font.render(f"Steering: {vehicle_state['controls']['steering']:6.2f}", True,
                                            (200, 220, 240))
        steer_rect = steer_text.get_rect(center=(steer_x, steer_y + 70))
        self.screen.blit(steer_text, steer_rect)

    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True

        try:
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            self.vehicle.reset()
                            print("🔄 Vehicle Reset")

                # Handle input and update vehicle
                controls = self.handle_input()
                vehicle_state = self.vehicle.update(controls)

                # Draw dashboard
                self.draw_dashboard(vehicle_state)

                # Update display
                pygame.display.flip()

                # Control frame rate
                clock.tick(60)

        except KeyboardInterrupt:
            print("\n🛑 Program interrupted")
        except Exception as e:
            print(f"\n💥 Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up resources...")
        pygame.quit()
        print("✅ Program exited")


def main():
    """Main entry point"""
    try:
        controller = AirSimKeyboardController()
        controller.run()
    except Exception as e:
        print(f"💥 Startup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()