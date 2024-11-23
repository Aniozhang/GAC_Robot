import asyncio
import numpy as np
import asyncio
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from robot import RobotArm, JointAngles, Recording
from robotConfigs import RobotData



robot_name = "WX250"  # Replace with the actual robot name in your configuration
robot = RobotArm(robot_name, port="COM3")  # Adjust port as needed

r = Recording.load("recording_1.rec")

robot.playBack(r)


