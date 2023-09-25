import numpy as np

from my_panda_gym.envs.core import RobotTaskEnv
from my_panda_gym.envs.robots.panda_l3 import PandaL3
from my_panda_gym.envs.tasks.push3 import Push3
from my_panda_gym.pybullet import PyBullet

from my_panda_gym.utils import distance
from typing import Any, Dict, Union


class PandaL3PushEnv3(RobotTaskEnv):
    """Push task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = PandaL3(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Push3(sim, reward_type=reward_type)
        super().__init__(robot, task)
