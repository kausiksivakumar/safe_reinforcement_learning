import torch
import torch.nn as nn
from safe_rl.mbrl.models.base import MLPRegression, MLPCategorical, CUDA, CPU, combined_shape, DataBuffer
from torch.utils.data import TensorDataset, DataLoader, random_split

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
    def __init__(self,state_dim,config=DEFAULT_CONFIG) -> None:
        super().__init__()
        self.state_dim  = state_dim
        self.network = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.epochs = 100
        self.data_buf = DataBuffer(self.state_dim,1, max_len=500000)
        self.save = True
        # self.env=env
        self.eps = 1e-3
        self.n_epochs = config["n_epochs"]
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]

        self.test_freq = config["test_freq"]
        self.test_ratio = config["test_ratio"]
        activ = config["activation"]
        self.criterion = nn.MSELoss(reduction='mean')
        
        
    def forward(self,x):
        cost = self.network(x)
        return(cost)
    
    def add_data_point(self,state,cost):
        self.data_buf.store(state, cost)
    
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
                    print(f"training epoch cost[{epoch}/{self.n_epochs}],loss train cost: {loss_train:.4f}, loss test cost  {loss_test:.4f}")
                else:
                    print(f"training epoch cost [{epoch}/{self.n_epochs}],loss train cost : {loss_train:.4f}, no testing data")
                loss_unormalized = self.test(x[::50], y[::50])
                print("loss unnormalized cost: ", loss_unormalized)

                if loss_test < best_loss:
                    best_loss = loss_test
                    loss_increase = 0
                else:
                    loss_increase += 1

                if loss_increase > patience:
                    break
                #print(loss_increase)

            # #if self.save and (epoch+1) % self.save_freq == 0 and loss_test == best_loss:
            # if self.save and loss_test <= best_loss:
            #     self.save_model(self.save_path)
            #     print("Saving model..... with loss", loss_test)
        
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

        
    
        
    
        
class rewardModel(nn.Module):
    def __init__(self,state_dim,config=DEFAULT_CONFIG) -> None:
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
        self.save = True
        self.eps = 1e-3
        self.n_epochs = config["n_epochs"]
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]

        self.test_freq = config["test_freq"]
        self.test_ratio = config["test_ratio"]
        activ = config["activation"]
        self.criterion = nn.MSELoss(reduction='mean')
        
        
        
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
                    print(f"training epoch Reward [{epoch}/{self.n_epochs}],loss train Reward: {loss_train:.4f}, loss test Reward {loss_test:.4f}")
                else:
                    print(f"training epoch Reward[{epoch}/{self.n_epochs}],loss train Reward: {loss_train:.4f}, no testing data")
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
            
