from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import SAC.core_HER as core
from spinup.utils.logx import EpochLogger

from env.ur5_env_ddpg import Ur5Env
import random
import math
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50,act_noise=0.1, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

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

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

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


    target_entropy = 1.0 
    #target_entropy *= np.log(act_dim)
    target_entropy = -act_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_log = torch.tensor((-np.log(act_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True)


    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data, alpha):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2) 

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data, alpha):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac_targ.q1(o, pi)
        q2_pi = ac_targ.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi) #?????

        # Entropy-regularized policy loss

        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info
    
    def compute_alpha(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        obj_alpha = (alpha_log * (-logp_pi - target_entropy).detach()).mean()

        return obj_alpha
    
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
            reward = 0 if math.sqrt(costs) < tolerance and (z-goal_z)>0.03 else -1
        return reward

    def gene_new_sas(new_goals, transition):
        state, new_state = transition[0][:3], transition[3][:3]
        action = transition[1]
        state = np.concatenate((state, new_goals))
        new_state = np.concatenate((new_state, new_goals))
        return state, action, new_state
    
        


    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    alpha_optimizer = Adam(params=[alpha_log], lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        alpha = alpha_log.exp().detach()
        # First run one gradient descent step for Q1 and Q2

        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data, alpha)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        #Automating Entropy Adjustment
        obj_alpha = compute_alpha(data)
        alpha_optimizer.zero_grad()
        obj_alpha.backward()
        alpha_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        alpha = alpha_log.exp().detach()
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data, alpha)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), Alpha=alpha.item(),**pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
    """
    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)
    """
    
    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    #def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
     # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret_real, ep_len,ep_ret_her = env.reset(), 0, 0, 0
    episode_cache = []

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        
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
            logger.log_tabular('EpRet_her', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    env = Ur5Env(is_render=False, is_good_view=False)
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default=env)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument("--HER_sample_num", default=4, type = int )
    parser.add_argument('--exp_name', type=str, default='SAC_HER')
    parser.add_argument('--log_dir', type=str, default="../logs") #存储目录
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=args.log_dir)

    torch.set_num_threads(torch.get_num_threads())
    
    sac(env, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
