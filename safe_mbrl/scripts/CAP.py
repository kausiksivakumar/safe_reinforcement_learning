import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import safety_gym
from scripts.model import RegressionModel
from scripts.constraint_model import costModel,rewardModel
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
        self.actor =  nn.Sequential(
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
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

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

        return action,action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def rollout(self,policy,initial_state,dynamics_model,cost_model,reward_model, H=30):
        curr_state                                                          = torch.tensor(initial_state)
        exp_reward                                                          = 0
        exp_cost                                                            = 0
        gamma_cost                                                          = 0.99
        gamma_reward                                                        = 0.99

        actions_raw = torch.zeros((H, 2))
        logp_list = torch.zeros((H, 1))
        for i in range(H): 
            a, logp_pi,_,_                                                  = self.policy.evaluate(curr_state)
            # print ("shapes: ", a.shape, curr_state.shape)
            next_state = dynamics_model.predict(torch.cat((curr_state, a)))
            next_state = next_state[0, :]
            cost                                                            = cost_model(next_state)
            reward                                                          = reward_model(next_state)
            curr_state                                                      = next_state
            exp_cost                                                        += (gamma_cost**i)*cost
            exp_reward                                                      += (gamma_reward**i)*reward
            actions_raw[i] = a[:]
            logp_list[i] = logp_pi

        return curr_state ,logp_list  ,actions_raw,exp_reward ,exp_cost
    
    def CCEM(self,env,policy,dynamics_model,cost_model,reward_model, I, N, C = 10,E = 50, H=30):
        initial_state                                                   =   env.observation_space.sample()
        obj                                                             =   0
        for i in range(I):
            cost_traj                                                   =   torch.zeros((N))
            reward_traj                                                 =   torch.zeros((N))
            action_sequence                                             =   torch.zeros((N, H, 2))
            feasible_set_idx                                            =   []
            loglist                                                     =   torch.zeros((N, H))
            for traj in range(N):
                curr_state ,logp_list ,actions_raw,exp_reward ,exp_cost = self.rollout(policy,initial_state,dynamics_model,cost_model,reward_model)
                if(exp_cost<=C):
                    feasible_set_idx.append(traj)
                cost_traj[traj] = exp_cost
                reward_traj[traj] = exp_reward
                action_sequence[traj, :]  = actions_raw
                loglist[traj, :] = logp_list[:, 0]

            # cost_traj                                                   =   np.array(cost_traj)
            # reward_traj                                                 =   np.array(reward_traj)
            # action_sequence                                             =   np.array(action_sequence)
            feasible_set_idx                                              =   np.array(feasible_set_idx)
            # elite_set_idxs                                              =   np.zeros((E,))
            # logp_list                                                   =   np.array(loglist)
            
            if(len(feasible_set_idx)<E):
                sorted_idxs                                             =   torch.argsort(cost_traj)
                elite_set_idxs                                          =   sorted_idxs[:E]
                
            else:
                sorted_idxs                                             =   torch.argsort(-reward_traj)
                elite_set_idxs                                          =   torch.where(cost_traj[sorted_idxs]<C)[0][:E]
             
            baseline                                                    =   reward_traj[elite_set_idxs].mean()
            ret                                                         =   reward_traj[elite_set_idxs] - baseline
            loglist_elite                                               =   loglist[elite_set_idxs]
            obj                                                         +=   -((torch.diag(ret)@loglist_elite).mean(axis=1)).mean()
            
        obj/=I
        return(obj)    

    def update(self,env, memory,dynamics_model,cost_model,reward_model):
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
            loss = self.CCEM(env, self.policy,dynamics_model,cost_model,reward_model, I=1, N=50, C = 10, E = 50, H=30)
            # Evaluating old actions and values :
            # logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            
            # Finding the ratio (pi_theta / pi_theta__old):
            # ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            # advantages = rewards - state_values.detach()
            # surr1 = ratios * advantages
            # surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def main():

    
    ############## Hyperparameters ##############
    env_name = "Safexp-PointGoal1-v0"
    render = True
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode

    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
        
        
    ###########Initial Buffer Training##############################
    data_num = 0
    dynamics_model = RegressionModel(state_dim+action_dim,state_dim)#.to(device)
    reward_model   = rewardModel(state_dim)#.to(device)
    cost_model     = costModel(state_dim)#.to(device)
    pretrain_eps = 10
    Horizon = 500
    
    for epi in tqdm.tqdm(range(pretrain_eps)):
        obs = env.reset()
        done = False
        i = 0
        while not done and i<Horizon:
            action = env.action_space.sample()
            obs_next, reward, done, info = env.step(action)
            if not done:  # otherwise the goal position will change
                x, y = np.concatenate((obs, action)), obs_next
                dynamics_model.add_data_point(x, y)
                cost = info["cost"] 
                reward_model.add_data_point(obs_next,reward)
                cost_model.add_data_point(obs_next, cost)
                data_num += 1
                i += 1
            obs = obs_next
    print("Finish to collect %i data "%data_num)
    
    ##############Training Loop#################################################
    # optimizer_dynamics =  optim.Adam(dynamics_model.parameters(),lr=0.01)
    optimizer_reward   =  optim.Adam(reward_model.parameters(),lr=0.01)  
    optimizer_cost     =  optim.Adam(cost_model.parameters(),lr=0.01)  
    epochs = 5 #change this when running

    dynamics_model.fit(epochs=epochs)
    cost_model.fit(epochs=epochs,optimizer=optimizer_cost)
    reward_model.fit(epochs=epochs,optimizer=optimizer_reward)

    #################################################################################
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    epochs = 5

    for i_episode in tqdm.tqdm(range(1, max_episodes + 1)):
        dynamics_model.fit(epochs=epochs)
        cost_model.fit(epochs=epochs, optimizer=optimizer_cost)
        reward_model.fit(epochs=epochs, optimizer=optimizer_reward)

        ppo.update(env, memory, dynamics_model, cost_model, reward_model)
        print ("eps: ", i_episode)






    #################################################################################
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()