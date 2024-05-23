import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
from torch.optim import Adam


#Global variables
spread = 10 
num_neuron = 35
nx=4
nu=2

class MyNueralNetworkA(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,4))
        
    def forward(self,x1):
        return self.network(x1).squeeze()

    
class MyNueralNetworkB(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,4))
    
    def forward(self,x1):
        return self.network(x1).squeeze()

#loading the models
f=MyNueralNetworkA()
f2=MyNueralNetworkB()
    
def train_model(x,y,f,n_epochs=100):
    if os.path.exists('A_matrix_V5_online.pth'):
        f.load_state_dict(torch.load('A_matrix_V5_online.pth'))

    opt=Adam(f.parameters(),lr=0.01)
    L=nn.MSELoss()

    losses=[]
    epochs=[]
    for epoch in range(n_epochs):
        #print(f'Epoch{epoch}')
        loss_value=L(f(x),y)
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        epochs.append(epoch)
        losses.append(loss_value.item())
    torch.save(f.state_dict(),'A_matrix_V5_online.pth')
    return np.array(epochs),np.array(losses)



def cycle_dynamics_pontry_train_A(nn_data):

    y= nn_data[:,6:]
    x1= nn_data[:,0:6]

    x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=42)

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()

    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    epoch_data, loss_data = train_model(x_train,y_train,f)

    L=nn.MSELoss()
    with torch.no_grad():
        y_eval=f.forward(x_test)
        loss_test=L(y_eval,y_test)


    print (f'train loss: {loss_data[-1]}')
    print (f'test loss:{loss_test}')

    
    # plt.plot(epoch_data,loss_data)
    # plt.show()


def cycle_dynamics_nn_diff_A(x,u,avg_data,std_data):

    arr=[x[0],x[1],x[2],x[3],u[0],u[1]]
    x1=torch.tensor(arr).float()
    nn_pred_A=f(x1).detach().numpy()
    nn_pred_A=((nn_pred_A/spread) * std_data) + avg_data
    fin_nn_pred_A=np.zeros([nx,nx])

    for i in range(nx):
        for j in range(nx):
            if i == j:
                fin_nn_pred_A[i,j]=1

    var1=0

    for i in range(nx):
        for j in range(nx):
            if i>j and j!=2 and i!=1:
                fin_nn_pred_A[j,i]=nn_pred_A[var1]
                var1=var1+1
    
    return fin_nn_pred_A

# Training Nueral network B     
def train_model_B(x,y,f2,n_epochs=100):
    if os.path.exists('B_matrix_V5_online.pth'):
        f2.load_state_dict(torch.load('B_matrix_V5_online.pth'))

    opt=Adam(f2.parameters(),lr=0.01)
    L=nn.MSELoss()

    losses=[]
    epochs=[]
    for epoch in range(n_epochs):
        #print(f'Epoch{epoch}')
        loss_value=L(f2(x),y)
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        epochs.append(epoch)
        losses.append(loss_value.item())
    torch.save(f2.state_dict(),'B_matrix_V5_online.pth')
    return np.array(epochs),np.array(losses)


def cycle_dynamics_pontry_train_B(nn_data_B):

    y= nn_data_B[:,6:]
    x1= nn_data_B[:,0:6]

    x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=42)

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()

    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    epoch_data, loss_data = train_model_B(x_train,y_train,f2)

    L=nn.MSELoss()
    with torch.no_grad():
        y_eval=f2.forward(x_test)
        loss_test=L(y_eval,y_test)

    #print (loss_test)
    #plt.plot(epoch_data,loss_data)
    #plt.show()

def cycle_dynamics_nn_diff_B(x,u):

    arr=[x[0],x[1],x[2],x[3],u[0],u[1]]
    x1=torch.tensor(arr).float()
    nn_pred_B=f2(x1).detach().numpy()

    fin_nn_pred_B=np.zeros([nx,nu])

    var1=0

    for i in range(nx):
        for j in range(nu):
            if i>1:
                fin_nn_pred_B[i,j]=nn_pred_B[var1]
                var1=var1+1
    
    return fin_nn_pred_B

    
def normalize_A(data_mat):
        norm_mat=data_mat[:,6:] # we only need the last 4 columns
        xy_mat=data_mat[:,0:6]
        norm_avg=np.mean(norm_mat,axis=0)
        norm_std=np.std(norm_mat,axis=0)
        standardized_mat=spread*((norm_mat-norm_avg)/norm_std)
        new_data_mat=np.concatenate((xy_mat,standardized_mat),axis=1)
        return new_data_mat,norm_avg,norm_std
