import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import safety_gym
from scripts.model import RegressionModel
from scripts.constraint_model import costModel, rewardModel
import tqdm
import torch.optim as optim

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # memory.states.append(state)
        # memory.actions.append(action)
        # memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.rsample()

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action, action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, gamma_cost, gamma_rew):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.gamma_cost = gamma_cost = 0.99
        self.gamma_reward = gamma_rew = 0.99

        # self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def rollout(self, initial_state, dynamics_model, cost_model, reward_model, H=30):
        curr_state = torch.tensor(initial_state)
        exp_reward = 0
        exp_cost = 0

        actions_raw = torch.zeros((H, 2))
        logp_list = torch.zeros((H, 1))
        for i in range(H):
            a, logp_pi, _, _ = self.policy.evaluate(curr_state)
            # print ("shapes: ", a.shape, curr_state.shape)
            next_state = dynamics_model.predict(torch.cat((curr_state, a)))
            next_state = next_state[0, :]
            cost = cost_model(next_state)
            reward = reward_model(next_state)
            curr_state = next_state
            exp_cost += (self.gamma_cost ** i) * cost
            exp_reward += (self.gamma_reward ** i) * reward
            actions_raw[i] = a[:]
            logp_list[i] = logp_pi

        return curr_state, logp_list, actions_raw, exp_reward, exp_cost

    def CCEM(self, env, policy, dynamics_model, cost_model, reward_model, I, N, C=10, E=50, H=30):
        initial_state = env.observation_space.sample()
        obj = 0
        for i in range(I):
            cost_traj = torch.zeros((N))
            reward_traj = torch.zeros((N))
            action_sequence = torch.zeros((N, H, 2))
            feasible_set_idx = []
            loglist = torch.zeros((N, H))
            for traj in range(N):
                curr_state, logp_list, actions_raw, exp_reward, exp_cost = self.rollout(initial_state, dynamics_model,
                                                                                        cost_model, reward_model)
                if (exp_cost <= C):
                    feasible_set_idx.append(traj)
                cost_traj[traj] = exp_cost
                reward_traj[traj] = exp_reward
                action_sequence[traj, :] = actions_raw
                loglist[traj, :] = logp_list[:, 0]

            # cost_traj                                                   =   np.array(cost_traj)
            # reward_traj                                                 =   np.array(reward_traj)
            # action_sequence                                             =   np.array(action_sequence)
            feasible_set_idx = np.array(feasible_set_idx)
            # elite_set_idxs                                              =   np.zeros((E,))
            # logp_list                                                   =   np.array(loglist)

            if (len(feasible_set_idx) < E):
                sorted_idxs = torch.argsort(cost_traj)
                elite_set_idxs = sorted_idxs[:E]

            else:
                sorted_idxs = torch.argsort(-reward_traj)
                elite_set_idxs = torch.where(cost_traj[sorted_idxs] < C)[0][:E]

            baseline = reward_traj[elite_set_idxs].mean()
            ret = reward_traj[elite_set_idxs] - baseline
            loglist_elite = loglist[elite_set_idxs]
            obj += -((torch.diag(ret) @ loglist_elite).mean(axis=1)).mean()

        obj /= I
        return (obj)

    def update(self, env, memory, dynamics_model, cost_model, reward_model, C=10):
        # Monte Carlo estimate of rewards:
        '''
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        '''
        # Optimize policy for K epochs:
        for _ in tqdm.tqdm(range(10)):
            loss = self.CCEM(env, self.policy, dynamics_model, cost_model, reward_model, I=1, N=50, C=C, E=50, H=30)
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def generate_action(env, random_flag, state=None, ppo_object=None):
    if random_flag:
        return env.action_space.sample()
    else:
        return ppo_object.policy.act(torch.Tensor(state)).detach().numpy()


def fill_buffer(env, dynamics_model, cost_model, reward_model, random_flag=False, ppo=None, Horizon=500,
                num_episodes=10):
    data_num = 0
    for epi in tqdm.tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        i = 0
        while not done and i < Horizon:
            action = generate_action(env, random_flag, obs, ppo)
            obs_next, reward, done, info = env.step(action)
            if not done:  # otherwise the goal position will change
                x, y = np.concatenate((obs, action)), obs_next
                dynamics_model.add_data_point(x, y)
                cost = info["cost"]
                reward_model.add_data_point(obs_next, reward)
                cost_model.add_data_point(obs_next, cost)
                data_num += 1
                i += 1
            obs = obs_next
    print("Finished collecting %i steps of data " % data_num)


def main():
    ############## Hyperparameters ##############
    env_name = "Safexp-PointGoal1-v0"
    render = True
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval

    max_timesteps = 1500  # max timesteps in one episode

    update_timestep = 4000  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    # creating environment
    env = gym.make(env_name)
    env_test = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    ###########Initial Buffer Fill##############################
    print ("###########Initial Buffer Fill##############################")
    dynamics_model = RegressionModel(state_dim + action_dim, state_dim)  # .to(device)
    reward_model = rewardModel(state_dim)  # .to(device)
    cost_model = costModel(state_dim)  # .to(device)
    pretrain_eps = 10
    Horizon = 500
    gamma_cost = 0.99
    gamma_rew = 0.99

    fill_buffer(env, dynamics_model, cost_model, reward_model, random_flag=True, ppo=None, Horizon=Horizon,
                num_episodes=pretrain_eps)

    ##############Model Pre-Training Loop#######################
    print ("#############Model Pre-Training Loop#################")
    # optimizer_dynamics =  optim.Adam(dynamics_model.parameters(),lr=0.01)
    optimizer_reward = optim.Adam(reward_model.parameters(), lr=0.01)
    optimizer_cost = optim.Adam(cost_model.parameters(), lr=0.01)
    epochs = 5  # change this when running

    dynamics_model.fit(epochs=epochs)
    cost_model.fit(epochs=epochs, optimizer=optimizer_cost)
    reward_model.fit(epochs=epochs, optimizer=optimizer_reward)

    ####################### Main Loop ###########################
    print ("####################### Main Loop ###########################")
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, gamma_cost, gamma_rew)
    epochs = 5
    fill_buffer_eps = 10
    kappa = 1  # initially high to be more conservative
    alpha = 0.1  # learning rate for kappa
    C = 10
    eval_freq = 1
    eval_traj_num = 5
    save_every = 100
    max_episodes = 10  # max training episodes

    eval_cost = []
    eval_rew = []
    kappa_list = []

    for i_episode in tqdm.tqdm(range(1, max_episodes + 1)):
        print("Main loop eps: : ", i_episode)

        dynamics_model.fit(epochs=epochs)
        cost_model.fit(epochs=epochs, optimizer=optimizer_cost)
        reward_model.fit(epochs=epochs, optimizer=optimizer_reward)

        print ("Dynamics model fit complete")

        ppo.update(env, memory, dynamics_model, cost_model, reward_model, C)

        print ("Policy Gradient update complete")

        fill_buffer(env, dynamics_model, cost_model, reward_model, random_flag=False, ppo=ppo, Horizon=Horizon,
                    num_episodes=fill_buffer_eps)
        _, _, _, _, Jc = ppo.rollout(env.observation_space.sample(), dynamics_model, cost_model, reward_model, Horizon)

        print ("Fill Buffer with updated policy complete")

        kappa = kappa + alpha * (Jc - C)
        kappa_list.append(kappa)
        print("Kappa Update Complete")

        if i_episode % eval_freq == 0:
            avg_cost = 0
            avg_reward = 0
            for t in range(eval_traj_num):
                c, r = evaluate(env_test, ppo, Horizon, gamma_cost, gamma_rew)
                avg_cost += c
                avg_reward += r

            avg_cost = avg_cost / eval_traj_num
            avg_reward = avg_reward / eval_traj_num
            eval_rew.append(avg_reward)
            eval_cost.append(avg_cost)
            print('Eval: Episode {} \t Avg Cost {} \t Avg Reward: {}'.format(i_episode, avg_cost, avg_reward))

        # save every 500 episodes
        if i_episode % save_every == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))

    # save final trained policy
    torch.save(ppo.policy.state_dict(), './PPO_continuous_fin_{}.pth'.format(env_name))

    plt.figure()
    plt.plot(np.arange(len(eval_rew)), eval_rew, label='eval rewards')
    plt.savefig('eval_rewards.png')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(eval_cost)), eval_cost, label='eval costs')
    plt.savefig('eval_cost.png')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(kappa_list)), kappa_list, label='kappa')
    plt.savefig('kappa_plot.png')
    plt.show()

    print ("##########################Complete!##################################")
    ################################################################################



def evaluate(env, ppo, Horizon, gamma_cost, gamma_rew):
    obs = env.reset()
    exp_cost, exp_rew = 0, 0
    for i in range(Horizon):
        act = generate_action(env, random_flag=False, state=obs, ppo_object=ppo)
        obs_next, rew, done, info = env.step(act)
        exp_cost += gamma_cost ** i * info['cost']
        exp_rew += gamma_rew ** i * rew
        if done:
            break
        obs = obs_next

    return exp_cost, exp_rew


if __name__ == '__main__':
    main()
