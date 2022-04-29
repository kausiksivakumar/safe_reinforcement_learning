import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import tqdm

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, device):
        super(ActorCritic, self).__init__()
        self.device = device
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
        self.action_var = torch.full((action_dim,), action_std * action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach()

    def evaluate(self, state):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.rsample()

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action, action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr, action_std, betas, gamma_cost, gamma_rew, device):
        self.betas = betas
        self.lr = lr

        self.policy = ActorCritic(state_dim, action_dim, action_std, device).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std,device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.gamma_cost = gamma_cost
        self.gamma_reward = gamma_rew
        self.device = device

        # self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def rollout(self, initial_belief, initial_state, transition_model, cost_model, reward_model, planning_horizon):
        curr_state = initial_state.clone().detach().requires_grad_(True).unsqueeze(0)
        curr_belief = initial_belief.clone().detach().requires_grad_(True).unsqueeze(0)
        exp_reward = 0
        costs = torch.zeros(planning_horizon)

        actions_raw = torch.zeros((planning_horizon, 2))
        logp_list = torch.zeros((planning_horizon, 1))
        belief_list = torch.zeros((planning_horizon, initial_belief.shape[0]))
        for i in range(planning_horizon):
            a, logp_pi, _, _ = self.policy.evaluate(curr_state)
            # print ("shapes: ", a.shape, curr_state.shape, curr_belief.shape, a.unsqueeze(0).shape)

            next_belief, next_state, _, _ = transition_model(curr_state, a.unsqueeze(dim=0), curr_belief)
            # next_state = dynamics_model.predict(torch.cat((curr_state, a)))
            # next_state = next_state[0, :]
            next_belief = next_belief.squeeze(0)
            next_state = next_state.squeeze(0)
            cost = cost_model(next_belief, next_state)
            reward = reward_model(next_belief, next_state)
            belief_list[i, :] = curr_belief
            curr_state = next_state
            curr_belief = next_belief
            costs[i] = cost
            # exp_cost += (self.gamma_cost ** i) * cost
            exp_reward += (self.gamma_reward ** i) * reward
            actions_raw[i] = a[:]
            logp_list[i] = logp_pi

        return belief_list, logp_list, actions_raw, exp_reward, costs
    #
    # def CCEM(self, env, policy, dynamics_model, cost_model, reward_model, kappa, I, N, C=10, E=50, H=30):
    #     initial_state = env.observation_space.sample()
    #     obj = 0
    #     for i in range(I):
    #         cost_traj = torch.zeros((N))
    #         reward_traj = torch.zeros((N))
    #         action_sequence = torch.zeros((N, H, 2))
    #         feasible_set_idx = []
    #         loglist = torch.zeros((N, H))
    #         for traj in range(N):
    #             curr_state, logp_list, actions_raw, exp_reward, exp_cost = self.rollout(initial_state, dynamics_model,
    #                                                                                     cost_model, reward_model, H=H)
    #             if exp_cost <= C-kappa:
    #                 feasible_set_idx.append(traj)
    #             cost_traj[traj] = exp_cost
    #             reward_traj[traj] = exp_reward
    #             action_sequence[traj, :] = actions_raw
    #             loglist[traj, :] = logp_list[:, 0]
    #
    #         # cost_traj                                                   =   np.array(cost_traj)
    #         # reward_traj                                                 =   np.array(reward_traj)
    #         # action_sequence                                             =   np.array(action_sequence)
    #         feasible_set_idx = np.array(feasible_set_idx)
    #         # elite_set_idxs                                              =   np.zeros((E,))
    #         # logp_list                                                   =   np.array(loglist)
    #
    #         if len(feasible_set_idx) < E:
    #             sorted_idxs = torch.argsort(cost_traj)
    #             elite_set_idxs = sorted_idxs[:E]
    #
    #         else:
    #             sorted_idxs = torch.argsort(-reward_traj)
    #             elite_set_idxs = torch.where(cost_traj[sorted_idxs] < C)[0][:E]
    #
    #         baseline = reward_traj[elite_set_idxs].mean()
    #         ret = reward_traj[elite_set_idxs] - baseline
    #         loglist_elite = loglist[elite_set_idxs]
    #         obj += -((torch.diag(ret) @ loglist_elite).mean(axis=1)).mean()
    #
    #     obj /= I
    #     return (obj)
    #
    # def update(self, env, dynamics_model, cost_model, reward_model, kappa, K_epochs = 5, I =1, N =50,C=10,E=50, H=30):
    #     # Monte Carlo estimate of rewards:
    #     '''
    #     rewards = []
    #     discounted_reward = 0
    #     for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
    #         if is_terminal:
    #             discounted_reward = 0
    #         discounted_reward = reward + (self.gamma * discounted_reward)
    #         rewards.insert(0, discounted_reward)
    #
    #     # Normalizing the rewards:
    #     rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    #
    #     # convert list to tensor
    #     old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
    #     old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
    #     old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
    #     '''
    #     # Optimize policy for K epochs:
    #     for _ in tqdm.tqdm(range(K_epochs)):
    #         loss = self.CCEM(env, self.policy, dynamics_model, cost_model, reward_model, kappa=kappa, I=I, N=N, C=C, E=E, H=H)
    #         # take gradient step
    #         self.optimizer.zero_grad()
    #         loss.mean().backward()
    #         torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0000)
    #         self.optimizer.step()
    #
    #     # Copy new weights into old policy:
    #     self.policy_old.load_state_dict(self.policy.state_dict())