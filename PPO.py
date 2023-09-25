#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   ppo.py
@Time    :   2021/03/20 14:32:27
@Author  :   Yan Wen 
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import ppo.core as core
from ppo.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from env.kuka_reach_env import KukaReachEnv
from env.ur5_env import Ur5Env
from ppo.logx import Logger
import sys
from colorama import Fore, Back, init

init(autoreset=True)

IS_DEBUG = False
ppo_logger = Logger(output_dir="../logs", is_debug=IS_DEBUG)


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim),
                                dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim),
                                dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr) #slice(2, 7)从索引 2 开始（包括索引 2）到索引 7 结束（不包括索引 7）的切片范围
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # r_t + gamma*v_t+1 - v_t 与 A=Q_t - V_t一样的想法。每个state的采取的动作产生
        #的回报相对于平均回报怎么样.如果大于0梯度上升，reward变大，那么智能体就会优于选择这些动作
        #计算了每个state的adv



        self.adv_buf[path_slice] = core.discount_cumsum( #每个state的discounted adv
            deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1] #每个state的discounted reward并且去掉最后一个

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf,
                    act=self.act_buf,
                    ret=self.ret_buf,
                    adv=self.adv_buf,
                    logp=self.logp_buf)
        return {
            k: torch.as_tensor(v, dtype=torch.float32)  #将转换后的torch.Tensor对象存储在一个新的字典中，
                                                         #其中键不变，对应的值为转换后的torch.Tensor对象
            for k, v in data.items()
        }


def ppo(env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(),
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        target_kl=0.01,
        logger_kwargs=dict(),
        save_freq=10):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.2维
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.1维
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.1维度
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    print('called ppo.')
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    # EpochLogger是一个用于记录训练过程中的日志的工具类。
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment

    # modified this to satisfy the custom env
    #env = env_fn()
    env = env
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    #actor_critic=core.MLPActorCritic
    #**ac_kwargs 字典类型的参数，用于传递给PPO中的ActorCritic对象的构造函数。具体来说是隐藏层参数
    #**ac_kwargs是一个关键字参数，它允许在创建actor-critic模型时传递额外的配置参数。这些额外的参数可以包括神经网络的隐藏层大小、激活函数类型等
    # ac_kwargs=dict(hidden_sizes=[args.hid] * args.l)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    #ppo_logger = Logger(output_dir="../logs", is_debug=IS_DEBUG)
    #在实例化Logger对象时，需要提供output_dir参数，指定日志文件的输出目录。
    #此外，还可以通过is_debug参数来设置是否启用调试模式，以决定是否打印额外的调试信息。
    ppo_logger.log("ac={}".format(ac), 'green')

    print('ac={}'.format(ac))
    """
    ac=MLPActorCritic(
  (pi): MLPGaussianActor(
    (mu_net): Sequential(
      (0): Linear(in_features=3, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
      (4): Linear(in_features=64, out_features=3, bias=True)
      (5): Identity()
    )
  )
  (v): MLPCritic(
    (v_net): Sequential(
      (0): Linear(in_features=3, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
      (4): Linear(in_features=64, out_features=1, bias=True)
      (5): Identity()
    )
  )
)
    """
    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    """
    通过将steps_per_epoch除以num_procs()，可以计算出每个进程/计算节点需要执行的步数，
    这就是local_steps_per_epoch的值。
    这样可以确保每个进程/计算节点在训练时期中处理相同数量的数据，以保持训练的一致性。
    """
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    ppo_logger.log("buf={}".format(buf))

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        #获取logp_old!!!!!!!!!!!!!!!!!
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data[
            'logp']
        ppo_logger.log("obs={}\nact={}\nadv={}\nlogp_old={}".format(
            obs, act, adv, logp_old))

        # Policy loss
        #MLPGaussianActor是actor的子类，actor是nn.model的子类，所以会自动执行Actor中的forward函数
        #获取新logp!!!!!!!!!!!!!!!!!!!!!!!
        pi, logp = ac.pi(obs, act) #获取正态分布pi（1000组每组有3维度数据，每个维度都服从正太分布）以及1000组数据每组数据3个动作的对数概率和logp
        ppo_logger.log("pi={},logp={}".format(pi, logp))

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        #目标是求价值函数的最大值，使用梯度上升。但是可以转化求负的价值函数（梯度取负号）的最小值，使用梯度下降
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        ppo_logger.log("ratio={},clip_adv={},loss_pi={}".format(
            ratio, clip_adv, loss_pi))

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        ppo_logger.log("obs={},ret={},loss_v={}".format(
            obs, ret, ((ac.v(obs) - ret)**2).mean())) #discounted reward和MLP估计的reward的差
        return ((ac.v(obs) - ret)**2).mean() #用于准确估计每个state的reward

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()
        """
        data = dict(obs=self.obs_buf,
                    act=self.act_buf,
                    ret=self.ret_buf,
                    adv=self.adv_buf,
                    logp=self.logp_buf)
        
        在使用这批数据训练actor时,data中的logp即logp_old是不变的
        """

        pi_l_old, pi_info_old = compute_loss_pi(data)
        #ppo_logger.log("pi_l_old={},pi_info_old={}".format(pi_l_old,pi_info_old))

        pi_l_old = pi_l_old.item() #z转化为标量
        v_l_old = compute_loss_v(data).item()
        #ppo_logger.log("pi_l_old={},v_l_old={}".format(pi_l_old,v_l_old))

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            #ppo_logger.log("loss_pi={},pi_info={}".format(loss_pi,pi_info))

            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log(
                    'Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            ppo_logger.log("loss_v={}".format(loss_v))
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old,
                     LossV=v_l_old,
                     KL=kl,
                     Entropy=ent,
                     ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    # ppo_logger.log("o={},ep_ret={},ep_len={}".format(o,ep_ret,ep_len))

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch): #4000
            #输出是 接下来的动作，critic网络估计的这个state的价值以及动作的指数概率
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            #  ppo_logger.log("a={},v={},logp={}".format(a,v,logp))

            # print('a={}'.format(a))
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            # print(Back.RED+'o={},\na={},\nr={},\nv={},\nlogp={}'.format(o,a,r,v,logp))
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len #True or Falsch 1000
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' %
                          ep_len,
                          flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:#if terminted
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        # 通过logger.store储存
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':

    env = Ur5Env(is_render=False, is_good_view=False)

    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')

    #modified this to satisfy the custom env
    parser.add_argument('--env', type=str, default=env)

    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='PPO') #存储训练信息的文件名
    parser.add_argument('--log_dir', type=str, default="../logs") #存储目录
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name,
                                        args.seed,
                                        data_dir=args.log_dir)

    ppo(env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),#列表[args.hid]被复制了args.l了次。[64，64]
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=env.max_steps_one_episode * args.cpu,
        epochs=4000,
        logger_kwargs=logger_kwargs)