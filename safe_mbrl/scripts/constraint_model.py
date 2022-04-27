import os

import joblib
import torch
import torch.nn as nn
import os.path as osp
from safe_rl.mbrl.models.base import MLPRegression, MLPCategorical, CUDA, CPU, combined_shape, DataBuffer
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import lightgbm as lgb
from safe_rl.mbrl.utils.run_utils import color2num, colorize


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class costModel:
    def __init__(self, env_wrap, config=None):
        super().__init__()
        self.env_wrap = env_wrap
        self.idx_goal = env_wrap.key_to_slice["goal"]
        self.state_dim = env_wrap.observation_size
        self.action_dim = env_wrap.action_size

        self.safe_data_buf = DataBuffer(self.state_dim, max_len=config["safe_buffer_size"])
        self.unsafe_data_buf = DataBuffer(self.state_dim, max_len=config["unsafe_buffer_size"])

        self.max_ratio = config["max_ratio"]
        self.batch = int(config["batch"])

        self.model = self.load_data(config["load_folder"]) if config["load"] else None
        if self.model is None:
            self.model = lgb.LGBMClassifier(**config["model_param"])
        else:
            self.model.set_params(**config["model_param"])

        self.folder = config["save_folder"]
        if osp.exists(self.folder):
            print("Warning: Saving dir %s already exists! Storing model and buffer there anyway." % self.folder)
        else:
            os.makedirs(self.folder)
        self.safe_data_buf_path = osp.join(self.folder, "safe_data_buf.pkl")
        self.unsafe_data_buf_path = osp.join(self.folder, "unsafe_data_buf.pkl")
        self.model_path = osp.join(self.folder, "cost_model.pkl")

    def add_data_point(self, state, cost):
        '''
        This method is used for streaming data setting, where one data will be added at each time.
        @param state [list or ndarray, (state_dim)]
        @param cost [int, (1)]: 0 - safe data; 1 - unsafe data;
        '''
        x = np.array(state).reshape(self.state_dim)
        if cost == 0:
            self.safe_data_buf.store(x, 0)
        elif cost > 0:
            self.unsafe_data_buf.store(x, 1)

    def predict(self, state):
        '''
        This method perform regression with ndarray data and output ndarray data
        @param state [ndarray, (batch, input_dim)]
        @return cost [ndarray, (batch,)]
        '''
        goal_pos = state[:, self.idx_goal]  # [batch, 2]
        dist = np.sqrt(np.sum(goal_pos ** 2, axis=1))  # [batch], cost = x^2+y^2
        dist[dist < 0.2] += -5
        dist[dist < 0.1] += -8
        cost = dist  # Goal reward part
        constraints_violation = self._predict_cost(state)
        cost[constraints_violation == 1] += 500
        return cost

    def _predict_reward(self, state):
        '''
        This method perform regression with ndarray data and output ndarray data
        @param state [ndarray, (batch, input_dim)]
        @return cost [ndarray, (batch,)]
        '''
        goal_pos = state[:, self.idx_goal]  # [batch, 2]
        dist = np.sqrt(np.sum(goal_pos ** 2, axis=1))  # [batch], cost = x^2+y^2
        dist[dist < 0.2] += -5
        dist[dist < 0.1] += -8
        cost_reward = dist  # Goal reward part
        return cost_reward

    def _predict_cost(self, state):
        data_num = state.shape[0]
        if data_num > self.batch:
            constraints_violation = np.zeros(data_num)
            forward_num = int(np.ceil(data_num / self.batch))
            for i in range(forward_num):
                idx = slice(i * self.batch, (i + 1) * self.batch)
                constraints_violation[idx] = self.model.predict(state[idx])
        else:
            constraints_violation = self.model.predict(state)
        return constraints_violation

    def fit(self, x=None, y=None, use_data_buf=True):
        '''
        Train the model either from external data or internal data buf.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch)]
        '''
        if use_data_buf:
            x0, y0 = self.safe_data_buf.get_all()
            x1, y1 = self.unsafe_data_buf.get_all()
            num0, num1 = x0.shape[0], x1.shape[0]
            max_num0 = int(num1 * self.max_ratio + 1)  # balance the safe and unsafe data ratio
            num0 = max_num0 if num0 > max_num0 else num0
            X = np.concatenate((x0[-num0:], x1), axis=0)
            Y = np.concatenate((y0[:num0], y1), axis=0)
        else:  # use external data loader
            X, Y = x, y
        self.model.fit(X, Y)
        if use_data_buf:
            if x0.shape[0] != 0:
                y_pred = self.model.predict(x0)
                acc0 = np.equal(y0, y_pred)
                print("safe data accuracy: %.2f%%" % (100 * acc0.mean()))
            if x1.shape[0] != 0:
                y_pred = self.model.predict(x1)
                acc1 = np.equal(y1, y_pred)
                print("unsafe data accuracy: %.2f%%" % (100 * acc1.mean()))
        # if self.save:
        #     self.save_data()

    def transform(self, x):
        '''
        @param x - [ndarray, (batch, input_dim)]
        @return out - [ndarray, [batch, 2]]
        '''
        x = x.reshape(-1, self.state_dim)
        out = self.model.predict_proba(x)
        return out

    # def save_data(self):
    #     joblib.dump(self.model, self.model_path)
    #     self.safe_data_buf.save(self.safe_data_buf_path)
    #     self.unsafe_data_buf.save(self.unsafe_data_buf_path)
    #     print(colorize("Successfully save model and data buffer to %s" % self.folder, 'green', bold=False))
    #
    # def load_data(self, path):
    #     model_path = osp.join(path, "cost_model.pkl")
    #     if osp.exists(model_path):
    #         model = joblib.load(model_path)
    #         print(colorize("Loading model from %s ." % model_path, 'magenta', bold=True))
    #     else:
    #         model = None
    #         print(colorize("We can not find the model from %s" % model_path, 'red', bold=True))
    #     unsafe_data_buf_path = osp.join(path, "unsafe_data_buf.pkl")
    #     if osp.exists(unsafe_data_buf_path):
    #         print(colorize("Loading data buffer from %s ." % unsafe_data_buf_path, 'magenta', bold=True))
    #         self.unsafe_data_buf.load(unsafe_data_buf_path)
    #     safe_data_buf_path = osp.join(path, "safe_data_buf.pkl")
    #     if osp.exists(safe_data_buf_path):
    #         print(colorize("Loading data buffer from %s ." % safe_data_buf_path, 'magenta', bold=True))
    #         self.safe_data_buf.load(safe_data_buf_path)
    #     return model
            
    def make_dataloader(self, x, y, normalize = True):
        '''
        This method is used to generate dataloader object for training.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch, output_dim)]
        '''

        tensor_x = x if torch.is_tensor(x) else torch.tensor(x).float()

        tensor_y = y if torch.is_tensor(y) else torch.tensor(y).float()
        num_data = tensor_x.shape[0]

        if normalize:
            self.mu = torch.mean(tensor_x, dim=0, keepdims=True)
            self.sigma = torch.std(tensor_x, dim=0, keepdims=True)
            # self.label_mu = torch.mean(tensor_y, dim=0, keepdims=True)
            # self.label_sigma = torch.std(tensor_y, dim=0, keepdims=True)
            
            self.sigma[self.sigma<self.eps] = 1
            # self.label_sigma[self.label_sigma<self.eps] = 1

            print("data normalized")
            print("mu: ", self.mu)
            print("sigma: ", self.sigma)
            # print("label mu: ", self.label_mu)
            # print("label sigma: ", self.label_sigma)

            tensor_x = (tensor_x-self.mu) / (self.sigma)
            # tensor_y = (tensor_y-self.label_mu) / (self.label_sigma)

        dataset = TensorDataset(tensor_x, tensor_y)

        testing_len = int(self.test_ratio * num_data)
        training_len = num_data - testing_len

        train_set, test_set = random_split(dataset, [training_len, testing_len])
        train_loader = DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=self.batch_size)

        return train_loader, test_loader
    
    def test_model(self, testloader):
        '''
        Test the model with normalized test dataset.
        @param test_loader [torch.utils.data.DataLoader]
        '''
        self.model = self.network
        self.model.eval()
        loss_test = 0
        for datas, labels in testloader:
            # datas = CUDA(datas)
            # labels = CUDA(labels)
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss_test += loss.item()*datas.shape[0]
        loss_test /= len(testloader.dataset)
        self.model.train()
        return loss_test
    
    def test(self, data, label):
        '''
        Test the model with unnormalized ndarray test dataset.
        @param data [list or ndarray, (batch, input_dim)]
        @param label [list or ndarray, (batch, output_dim)]
        '''
        pred = self.predict(data) 
        #mse = np.mean((pred-label)**2)
        #print("MSE: ", mse)
        pred = torch.tensor(pred).float()
        labels = torch.tensor(label).float()
        loss = self.criterion(pred, labels)
        return loss.item()

    ###########SAVE DATA#################################################
    def save_model(self):
        joblib.dump(self.model, self.model_path)
        # checkpoint = {"model_state_dict":  self.model.state_dict()}
        # torch.save(checkpoint, path)

    def save_data(self):
        # self.save_model(self.model_path)
        self.safe_data_buf.save(self.safe_data_buf_path)
        self.unsafe_data_buf.save(self.unsafe_data_buf_path)
        print("Successfully save model and data buffer to %s"%self.folder)

    #########LOAD DATA######################################################
    def load_model(self):
        # checkpoint = torch.load(path)
        # self.model = checkpoint["model_state_dict"]
        # model_path = osp.join(self.model_path, "cost_model.pkl")
        if osp.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(colorize("Loading model from %s ." % self.model_path, 'magenta', bold=True))
        else:
            self.model = None
            print(colorize("We can not find the model from %s" % self.model_path, 'red', bold=True))
        # return model

    def load_data(self):
        if osp.exists(self.safe_data_buf_path):
            print("Loading dynamic data buffer from %s ."%self.safe_data_buf_path)
            self.safe_data_buf.load(self.safe_data_buf_path)
        else:
            print("We can not find the dynamic data buffer from %s"%self.safe_data_buf_path)

        if osp.exists(self.unsafe_data_buf_path):
            print("Loading dynamic data buffer from %s ."%self.unsafe_data_buf_path)
            self.unsafe_data_buf.load(self.unsafe_data_buf_path)
        else:
            print("We can not find the dynamic data buffer from %s"%self.unsafe_data_buf_path)
        
class rewardModel(nn.Module):
    def __init__(self,state_dim,config=None) -> None:
        super().__init__()
        self.state_dim  = state_dim
        self.network = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.data_buf = DataBuffer(self.state_dim,1, max_len=500000)
        self.eps = 1e-3
        self.n_epochs = config["n_epochs"]
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]

        self.test_freq = config["test_freq"]
        self.test_ratio = config["test_ratio"]
        activ = config["activation"]
        self.criterion = nn.MSELoss(reduction='mean')
        self.save = config["save"]
        if self.save:
            self.folder = config["save_folder"]
            if osp.exists(self.folder):
                print("Warning: Saving dir %s already exists! Storing model and buffer there anyway." % self.folder)
            else:
                os.makedirs(self.folder)
            self.data_buf_path = osp.join(self.folder, "reward_data_buf.pkl")
            self.model_path = osp.join(self.folder, "reward_model.pkl")
        
        
        
    def forward(self,x):
        reward = self.network(x)
        return(reward)
    
    def add_data_point(self,state,reward):
        self.data_buf.store(state, reward)
        
    def fit(self, x=None, y=None, use_data_buf=True, normalize=True,epochs=100,optimizer=None):
        '''
    Train the model either from external data or internal data buf.
    @param x [list or ndarray, (batch, input_dim)]
    @param y [list or ndarray, (batch, output_dim)]
    '''
        self.model = self.network

        # early stopping
        patience = 6
        best_loss = 1e3
        loss_increase = 0

        if use_data_buf:
            x, y = self.data_buf.get_all()
            train_loader, test_loader =  self.make_dataloader(x, y, normalize = normalize)
        else: # use external data loader
            train_loader, test_loader = self.make_dataloader(x, y, normalize = normalize)
        
        for epoch in range(epochs):
            self.model.train()
            loss_train = 0
            loss_test = 1e5
            for datas, labels in train_loader:
                # datas = CUDA(datas)
                # labels = CUDA(labels)
                optimizer.zero_grad()

                outputs = self.model(datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()*datas.shape[0] # sum of the loss
            
            # if self.save and (epoch+1) % self.save_freq == 0:
            #     self.save_model(self.save_path)
                
            if (epoch+1) % self.test_freq == 0:
                loss_train /= len(train_loader.dataset)
                loss_test = -0.1234
                if len(test_loader) > 0:
                    loss_test = self.test_model(test_loader)
                    print(f"training epoch Reward [{epoch}/{epochs}],loss train Reward: {loss_train:.4f}, loss test Reward {loss_test:.4f}")
                else:
                    print(f"training epoch Reward[{epoch}/{epochs}],loss train Reward: {loss_train:.4f}, no testing data")
                loss_unormalized = self.test(x[::50], y[::50])
                print("loss unnormalized Reward: ", loss_unormalized)

                if loss_test < best_loss:
                    best_loss = loss_test
                    loss_increase = 0
                else:
                    loss_increase += 1

                if loss_increase > patience:
                    break
        
        # if self.save:
        #     self.save_data()
    
    def make_dataloader(self, x, y, normalize = True):
        '''
        This method is used to generate dataloader object for training.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch, output_dim)]
        '''

        tensor_x = x if torch.is_tensor(x) else torch.tensor(x).float()

        tensor_y = y if torch.is_tensor(y) else torch.tensor(y).float()
        num_data = tensor_x.shape[0]

        if normalize:
            self.mu = torch.mean(tensor_x, dim=0, keepdims=True)
            self.sigma = torch.std(tensor_x, dim=0, keepdims=True)
            self.label_mu = torch.mean(tensor_y, dim=0, keepdims=True)
            self.label_sigma = torch.std(tensor_y, dim=0, keepdims=True)
            
            self.sigma[self.sigma<self.eps] = 1
            self.label_sigma[self.label_sigma<self.eps] = 1

            print("data normalized")
            print("mu: ", self.mu)
            print("sigma: ", self.sigma)
            print("label mu: ", self.label_mu)
            print("label sigma: ", self.label_sigma)

            tensor_x = (tensor_x-self.mu) / (self.sigma)
            tensor_y = (tensor_y-self.label_mu) / (self.label_sigma)

        dataset = TensorDataset(tensor_x, tensor_y)

        testing_len = int(self.test_ratio * num_data)
        training_len = num_data - testing_len

        train_set, test_set = random_split(dataset, [training_len, testing_len])
        train_loader = DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=self.batch_size)

        return train_loader, test_loader
    
    def test_model(self, testloader):
        '''
        Test the model with normalized test dataset.
        @param test_loader [torch.utils.data.DataLoader]
        '''
        self.model = self.network
        self.model.eval()
        loss_test = 0
        for datas, labels in testloader:
            # datas = CUDA(datas)
            # labels = CUDA(labels)
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss_test += loss.item()*datas.shape[0]
        loss_test /= len(testloader.dataset)
        self.model.train()
        return loss_test
    
    def test(self, data, label):
        '''
        Test the model with unnormalized ndarray test dataset.
        @param data [list or ndarray, (batch, input_dim)]
        @param label [list or ndarray, (batch, output_dim)]
        '''
        pred = self.predict(data) 
        #mse = np.mean((pred-label)**2)
        #print("MSE: ", mse)
        pred = torch.tensor(pred).float()
        labels = torch.tensor(label).float()
        loss = self.criterion(pred, labels)
        return loss.item()

    ###########SAVE DATA#################################################
    def save_model(self, path):
        checkpoint = {"model_state_dict": self.network.state_dict()}
        torch.save(checkpoint, path)

    def save_data(self):
        # self.save_model(self.model_path)
        self.data_buf.save(self.data_buf_path)
        print("Successfully save model and data buffer to %s"%self.folder)

    #########LOAD DATA######################################################
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint["model_state_dict"]
        # self.model = CUDA(self.model)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.mu = checkpoint["mu"]
        # self.sigma = checkpoint["sigma"]
        # self.label_mu = checkpoint["label_mu"]
        # self.label_sigma = checkpoint["label_sigma"]

    def load_data(self, path):
        # model_path = osp.join(path, "dynamic_model.pkl")
        # if osp.exists(model_path):
        #     self.load_model(model_path)
        #     print("Loading dynamic model from %s ."%model_path)
        # else:
        #     print("We can not find the model from %s"%model_path)
        data_buf_path = osp.join(path, "reward_data_buf.pkl")
        if osp.exists(data_buf_path):
            print("Loading dynamic data buffer from %s ."%data_buf_path)
            self.data_buf.load(data_buf_path)
        else:
            print("We can not find the dynamic data buffer from %s"%data_buf_path)
    
    def predict(self, data):
        '''
        This method perform regression with ndarray data and output ndarray data
        @param data [list or ndarray or tensor, (batch, input_dim)]
        @return out [list or ndarray, (batch, output_dim)]
        '''
        self.model.eval()
        inputs = data if torch.is_tensor(data) else torch.tensor(data).float()
        inputs = (inputs-self.mu) / (self.sigma)
        # inputs = CUDA(inputs)
        with torch.no_grad():
            out = self.model(inputs)
            # out = CPU(out)
            out = out * (self.label_sigma) + self.label_mu
            out = out.numpy()
        return out
            
