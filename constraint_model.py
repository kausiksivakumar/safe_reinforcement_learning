import torch
import torch.nn as nn
from safe_rl.mbrl.models.base import MLPRegression, MLPCategorical, CUDA, CPU, combined_shape, DataBuffer

DEFAULT_CONFIG = dict(
                n_epochs=100,
                learning_rate=0.001,
                batch_size=256,
                hidden_sizes=(1024, 1024, 1024),
                buffer_size=500000,

                save=False,
                save_folder=None,
                load=False,
                load_folder=None,
                test_freq=2,
                test_ratio=0.1,
                activation="relu",
            )

class costModel(nn.Module):
    def __init__(self,env) -> None:
        super(costModel,self).__init__()
        self.state_dim  = env.observation_space.shape[0]
        self.network = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.epochs = 100
        self.data_buf = DataBuffer(self.state_dim,1, max_len=500000)
        
    def forward(self,x):
        cost = self.network(x)
        return(cost)
    
    def add_data_point(self,state,cost):
        self.data_buf.store(state, cost)
        
    
        
    
        
class rewardModel(nn.Module):
    def __init__(self,env) -> None:
        super(costModel,self).__init__()
        self.state_dim  = env.observation_space.shape[0]
        # self.action_dim = env.action_space.shape[0]
        self.network = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.data_buf = DataBuffer(self.state_dim,1, max_len=500000)
        
        
    def forward(self,x):
        reward = self.network(x)
        
        return(reward)
    
    def add_data_point(self,state,reward):
        self.data_buf.store(state, reward)
        
