#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for keyboard control functionality
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyboard_control import VehicleSimulator


class TestVehicleSimulator(unittest.TestCase):
    """Test cases for VehicleSimulator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.vehicle = VehicleSimulator()

    def test_initial_state(self):
        """Test initial vehicle state"""
        self.assertEqual(self.vehicle.position['x'], 0.0)
        self.assertEqual(self.vehicle.position['y'], 0.0)
        self.assertEqual(self.vehicle.speed, 0.0)
        self.assertEqual(self.vehicle.throttle, 0.0)
        self.assertEqual(self.vehicle.steering, 0.0)
        self.assertEqual(self.vehicle.brake, 0.0)
        self.assertEqual(self.vehicle.heading, 0.0)

    def test_throttle_control(self):
        """Test throttle control"""
        controls = {'throttle': 0.5, 'steering': 0.0, 'brake': 0.0}
        state = self.vehicle.update(controls)

        self.assertGreater(state['speed'], 0.0)
        self.assertEqual(state['controls']['throttle'], 0.5)

    def test_brake_control(self):
        """Test brake functionality"""
        # First accelerate
        self.vehicle.update({'throttle': 1.0, 'steering': 0.0, 'brake': 0.0})

        # Then brake
        state = self.vehicle.update({'throttle': 0.0, 'steering': 0.0, 'brake': 1.0})

        self.assertLess(state['speed'], 1.0)  # Speed should decrease

    def test_steering_control(self):
        """Test steering functionality"""
        # First get some speed
        self.vehicle.update({'throttle': 1.0, 'steering': 0.0, 'brake': 0.0})

        # Then steer
        initial_heading = self.vehicle.heading
        state = self.vehicle.update({'throttle': 0.5, 'steering': 0.5, 'brake': 0.0})

        self.assertNotEqual(state['heading'], initial_heading)

    def test_reset_functionality(self):
        """Test vehicle reset"""
        # Modify vehicle state
        self.vehicle.update({'throttle': 1.0, 'steering': 0.5, 'brake': 0.0})

        # Reset
        self.vehicle.reset()

        # Check if reset worked
        self.assertEqual(self.vehicle.position['x'], 0.0)
        self.assertEqual(self.vehicle.position['y'], 0.0)
        self.assertEqual(self.vehicle.speed, 0.0)
        self.assertEqual(self.vehicle.throttle, 0.0)
        self.assertEqual(self.vehicle.steering, 0.0)
        self.assertEqual(self.vehicle.brake, 0.0)
        self.assertEqual(self.vehicle.heading, 0.0)

    def test_speed_limits(self):
        """Test speed limiting functionality"""
        # Try to exceed max speed
        for _ in range(100):
            state = self.vehicle.update({'throttle': 1.0, 'steering': 0.0, 'brake': 0.0})

        self.assertLessEqual(state['speed'], 40.0)  # Max speed limit

        # Test reverse speed limit
        self.vehicle.reset()
        for _ in range(50):
            state = self.vehicle.update({'throttle': -1.0, 'steering': 0.0, 'brake': 0.0})

        self.assertGreaterEqual(state['speed'], -8.0)  # Min speed limit


if __name__ == '__main__':
    unittest.main()