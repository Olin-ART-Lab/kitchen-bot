import os
import pickle, yaml
import numpy as np
import pybullet as p

from .env import ClutteredPushGrasp
from .ur5push1 import Ur5Push1
from .robot import Panda, UR5Robotiq85, UR5Robotiq140
from .utilities import YCBModels, Camera
from matplotlib import pyplot as plt
import time
import math
from robot_env.load_model import load_model, load_transforms
import time
from robot_env.utilities import distance
from utils import Namespace

def user_control_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)

    UR5Robotiq85_pos = (-0.48, -0.1095, 0)
    UR5Robotiq85_ori = (0, 0, 0)
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq85(UR5Robotiq85_pos, UR5Robotiq85_ori)
    env = Ur5Push1(robot, ycb_models, camera, vis=True)
    all_obs_regular = []
    all_obs_embeddings = []
    with open('/home/jess/kitchen-bot/robot_env/train_bc.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = Namespace(cfg)
        vision_model = load_model(cfg)
        vision_model.to("cuda:0")
        vision_model.eval()
    transforms = load_transforms(cfg)
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
    # 10 tajectories? 
    for i in range(161):
        obs = env.reset(vision_model=vision_model, transforms=transforms)
        #rgb, depth, seg = camera.shot()
        #plt.imshow(rgb)
        #plt.show()
        #breakpoint()
        # 50 steps? 
        for j in range(50):
            action = (2 * np.random.rand(7) - 1)
            # action = [0, 0, 0, 0, 0, 0, 0.0]
            # Obs comes from get_obs which contains task state info
            obs, reward, done, info = env.step(action, 'end', model=vision_model, transforms=transforms)
            #print(f"obs: {obs['observation_img']}")
            #rgb, depth, seg = camera.shot()
            #plt.imshow(rgb)
            #plt.show()
            #breakpoint()
            #print(f"obs: {obs['object_state']}")
            #print(f"desired: {obs['desired_goal']}")
            #breakpoint()
            rgb = {"rgb":obs["embedded_img"].squeeze(),"obj_state":obs["object_state"], "desired_state":obs["desired_goal"]}
            #print(rgb.shape)
            joints = obs["joint_pos"]
            #print(joints.shape)
            all_obs_embeddings.append(rgb)
            observation = np.concatenate((obs["embedded_img"].squeeze(),obs["object_state"],
                                         obs['desired_goal']), axis=0)
            all_obs_regular.append(observation)
            # print(obs["joint_pos"][-1])
    #breakpoint()
    #print(f"all_obs single index shape: {all_obs[1].size}") # all_obs is list of dictionaries
    np.save("robot_env/task_states_for_embeddings_push1", all_obs_embeddings, allow_pickle=True)
    np.save("robot_env/task_states_push1", all_obs_regular, allow_pickle=True)
    with open("push4_rgb_plus.pickle", "wb") as f:
        pickle.dump(all_obs_embeddings, f)



if __name__ == '__main__':
    user_control_demo()
