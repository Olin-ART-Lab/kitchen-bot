import os

import numpy as np
import pybullet as p

from .env import ClutteredPushGrasp
from .ur5push2 import Ur5Push2
from .robot import Panda, UR5Robotiq85, UR5Robotiq140
from .utilities import YCBModels, Camera
import time
import math


def user_control_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    UR5Robotiq85_pos = (-0.48, -0.1095, 0)
    UR5Robotiq85_ori = (0, 0, 0)
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq85(UR5Robotiq85_pos, UR5Robotiq85_ori)
    env = Ur5Push2(robot, ycb_models, camera, vis=True)
    all_obs = []
    # env._create_scene()
    # env.SIMULATION_STEP_DELAY = 0
    # while True:
    #     # action = env.action_space.sample()  # random action
    #     action = np.random.rand(8)
    #     # obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
    #     obs, reward, done, info = env.step(action, 'joint')
    #     # print(obs, reward, done, info)

    # for j in range(1000):
    #     # action = env.action_space.sample()  # random action
    #     action = 0.1 * (2 * np.random.rand(7) - 1)
    #     # action = 0.1 * np.zeros(7)
    #     # obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
    #     obs, reward, done, info = env.step(action, 'joint')
    #     # print(obs, reward, done, info)

    for i in range(10):
        obs = env.reset()
        for j in range(50):
            action = (2 * np.random.rand(7) - 1)
            # action = [0, 0, 0, 0, 0, 0, 0.0]
            # Obs comes from get_obs which contains task state info
            obs, reward, done, info = env.step(action, 'end')
            observation = np.concatenate((obs["observation"],
                                         obs['desired_goal']), axis=0)
            all_obs.append(observation)
            # print(obs["joint_pos"][-1])
    #breakpoint()
    print(f"all_obs single index shape: {all_obs[1].size}") # all_obs is list of dictionaries
    np.save("robot_env/task_states_push_new2.npy", all_obs, allow_pickle=True)



if __name__ == '__main__':
    user_control_demo()
