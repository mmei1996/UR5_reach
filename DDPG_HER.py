from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import DDPG.core as core
from spinup.utils.logx import EpochLogger
from env.ur5_env_ddpg import Ur5Env
import random
import math
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, 6), dtype=np.float32) # obs_dim + goal_dim
        self.obs2_buf = np.zeros(core.combined_shape(size, 6), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        #self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        """
        具体操作为，将self.ptr的值加1，并将结果与self.max_size取模（即取余数），
        这样可以保证指针位置在0到self.max_size-1之间循环更新。当指针位置达到最大容量时，下一次更新会将指针位置重置为0，实现环形循环的效果。
        """
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def ddpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    
    obs_dim = 6
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
        q = ac.q(o,a)
        
        # Bellman backup for Q function
        
        # maxQ由target_q网络得出，Q由q网络得出。
        

        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        # 类似于TD的思想 V(s) = r + γ * V(s')，立即奖励加上未来的折扣奖励（G（s'）reward to go, 用下一个状态的值来跟新现在的状态
        # r + lambd * maxQ(s',a') 就是这个state的Q值即target。Q（s,a）来逼近这个目标以此来训练critict
        # 更新公式 Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a)) 
        loss_q = ((q - backup)**2).mean() 

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    # Actor(pi)的功能是，输出一个动作A，这个动作A输入到Crititc后，能够获得最大的Q值。
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean() #求最大值是梯度下降的反方向

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q（critict）.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi（actor）.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        # 参数的平滑更新
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()): # 将原始网络（ac）和目标网络（ac_targ）的参数进行配对。
                # NB: We use an in-place operations "mul_", "add_" to update target # 
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak) # 表示将目标网络的参数乘以 polyak，即进行衰减。
                p_targ.data.add_((1 - polyak) * p.data) # 表示将原始网络的参数乘以 (1 - polyak)，然后加到目标网络的参数上，即进行更新。
        """
        在给出的代码中，目标网络的参数是在每次迭代中进行更新的，这是一种常见的做法。
        通过在每次迭代中更新目标网络的参数，可以更及时地将原始网络的最新参数传播到目标网络中，从而加快学习的过程。
        """

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1

            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    def generate_goals(i, episode_cache, sample_num, sample_range = 200):
        '''
        Input: current steps, current episode transition's cache, sample number 
        Return: new goals sets
        notice here only "future" sample policy
        '''
        end = (i+sample_range) if i+sample_range < len(episode_cache) else len(episode_cache)
        epi_to_go = episode_cache[i:end]
        if len(epi_to_go) < sample_num:
            sample_trans = epi_to_go
        else:
            sample_trans = random.sample(epi_to_go, sample_num)
        return [np.array(trans[3][:3]) for trans in sample_trans]
    

    def calcu_reward(new_goal, state, action, mode='her'):
    # direcly use observation as goal
        if mode == 'shaping':
            # shaping reward
            goal_cos, goal_sin, goal_thdot = new_goal[0], new_goal[1], new_goal[2]
            cos_th, sin_th, thdot = state[0], state[1], state[2]
            costs = (goal_cos - cos_th)**2 + (goal_sin - sin_th)**2 + (goal_thdot-thdot)**2
            reward = -costs
        elif mode  == 'her':
            # binary reward 
            tolerance = 0.1
            goal_x, goal_y, goal_z = new_goal[0], new_goal[1], new_goal[2]
            x, y, z = state[0], state[1], state[2]
            costs = (goal_x - x)**2 + (goal_y - y)**2 + (goal_z - z)**2
            reward = 0 if math.sqrt(costs) < tolerance else -1
        return reward

    def gene_new_sas(new_goals, transition):
        state, new_state = transition[0][:3], transition[3][:3]
        action = transition[1]
        state = np.concatenate((state, new_goals))
        new_state = np.concatenate((new_state, new_goals))
        return state, action, new_state
    
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret_real, ep_len,ep_ret_her = env.reset(), 0, 0, 0
    episode_cache = []
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        goals = env.goal_state()
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        o = np.concatenate((o, goals))
        a = get_action(o, act_noise)
        

        # Step the env
        o2, r_real, d, _ = env.step(a)
        
        r = calcu_reward(goals, o, a)

        o2 = np.concatenate((o2, goals))
        
        ep_ret_real += r_real
        ep_ret_her += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer

        replay_buffer.store(o, a, r, o2)
        episode_cache.append((o, a, r, o2))

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2[:3]

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet_real=ep_ret_real, EpLen=ep_len, EpRet_her=ep_ret_her)
            o, ep_ret_her, ep_len, ep_ret_real= env.reset(), 0, 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for i, transition in enumerate(episode_cache):
                    new_goals = generate_goals(i, episode_cache, args.HER_sample_num)
                    for new_goal in new_goals:
                        reward = calcu_reward(new_goal, transition[0][:3], transition[1]) #new goal state action
                        state, action, new_state = gene_new_sas(new_goal, transition)
                        replay_buffer.store(state, action, reward, new_state)

            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)
            episode_cache = []
        
            

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            #test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet_real', with_min_and_max=True)
            logger.log_tabular('EpRet_her', with_min_and_max=True)
            #logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            #logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    env = Ur5Env(is_render=True, is_good_view=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=env)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--exp_name', type=str, default='DDPG_HER')
    parser.add_argument("--HER_sample_num", default=4, type = int )
    parser.add_argument('--log_dir', type=str, default="../logs") #存储目录
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed,data_dir=args.log_dir)

    ddpg(env, actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
