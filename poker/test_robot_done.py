#!/usr/bin/env python3
"""
Helper script to simulate robot completion.
Run this after manually placing the chip.

Usage: python3 -m poker.test_robot_done
"""
from poker.robot_interface import robot_done, is_robot_executing

print(f"Robot executing: {is_robot_executing()}")
if is_robot_executing():
    robot_done()
    print("Called robot_done() - chip verification will run on next poll")
else:
    print("Robot not executing - nothing to do")
