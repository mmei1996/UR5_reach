#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   core.py
@Time    :   2021/03/20 14:32:33
@Author  :   Yan Wen 
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib


import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from ppo.logx import Logger
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


IS_DEBUG=False
core_logger=Logger(output_dir="../logs/",is_debug=IS_DEBUG)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)



def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        core_logger.log("layers={}".format(layers),'green')
        core_logger.log("nn.Sequential={}".format(nn.Sequential(*layers)))

    return nn.Sequential(*layers)

#模型中可学习参数的总数量。
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Actor(nn.Module):
    #具体的生成过程由子类实现，因此在基类中抛出了 NotImplementedError 异常，需要在子类中进行重写。
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32) #初始化方差的对数
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        core_logger.log("mu_net={}".format(self.mu_net))

    def _distribution(self, obs):
        mu = self.mu_net(obs) #均值
        std = torch.exp(self.log_std) #方差
        core_logger.log("mu={},std={},Normal(mu,std)={}".format(mu,std,Normal(mu,std)))
        return Normal(mu, std) #将会生成1000个三维的随机数样本，其中每个维度的值服从均值为loc对应维度的值，标准差为scale对应维度的值的正态分布

    def _log_prob_from_distribution(self, pi, act):
        core_logger.log("pi.log_prob(act)={}".format(pi.log_prob(act)),'red')
        core_logger.log("pi.log_prob(act).sum(axis=-1)={}".format(pi.log_prob(act).sum(axis=-1)))

        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution 
                                              #动作的对数概率之和
                                              


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        core_logger.log("v_net={}".format(self.v_net))


    def forward(self, obs):
        core_logger.log("torch.squeeze(self.v_net(obs), -1)".format(torch.squeeze(self.v_net(obs), -1)),'green')
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.
                                                   # 去除最后一个维度


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box): #如果动作空间是连续空间
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation) #实例化actor
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value functionp
        self.v = MLPCritic(obs_dim, hidden_sizes, activation) #实例化criticor

    def step(self, obs): #输出为 通过Actor的网络输出的动作 通过critic的网络输出的价值 以及动作的指数概率
        with torch.no_grad():
            pi = self.pi._distribution(obs) # 有act_dimention维度的Normal(mu, std) 
            core_logger.log("pi={}".format(pi))
            a = pi.sample() #采样动作
            core_logger.log("a={}".format(a))
            logp_a = self.pi._log_prob_from_distribution(pi, a) #获取动作的指数概率 动作的对数概率之和
            core_logger.log("logp_a={}".format(logp_a))

            v = self.v(obs) #MLP估计的这个state的价值
            """
            在代码中，self.v是一个MLPCritic类的实例对象，
            而MLPCritic类继承自nn.Module类，而nn.Module类中定义了__call__方法。
            因此，当我们调用self.v(obs)时，实际上是在调用MLPCritic类对象的父类nn.Module的__call__方法即是前向传播方法。
            """                                                   
            core_logger.log("v={}".format(v),'blue')
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0] #获取MLP根据state输出的动作

