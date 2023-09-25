#!/usr/bin/env python3
# -*- encoding: utf-8 -*-



import numpy as np
from env.ur5_env_ddpg import Ur5Env
from spinup.utils.mpi_tools import mpi_fork
#import DDPG.core as core
import SAC.core_HER as core
import torch

# import os
# print(os.getcwd())

env=Ur5Env(is_good_view=True,is_render=True)

#ac=torch.load("E:/logs/ddpg/ddpg_s0/pyt_save/model.pt")

ac=torch.load("E:/logs/SAC_HER/SAC_HER_s0/pyt_save/model.pt")
print('ac={}'.format(ac))



sum_reward=0
for i in range(50):
    
    obs=env.reset()
    for step in range(200):
        goals = env.goal_state()
        obs = np.concatenate((obs, goals))
        actions=ac.act(torch.as_tensor(obs,dtype=torch.float32))
        obs,reward,done,info=env.step(actions)
        
        sum_reward+=reward
        k=env.human()
        if done:
            break

    print('sum reward={}'.format(sum_reward))
    sum_reward = 0

