import numpy as np
import torch
import torch.nn as nn
from torch.nn import utils
from torch.autograd.functional import jacobian as jc
import torch.autograd as autograd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from jupyterthemes import jtplot
from torch import optim
from ipywidgets import IntProgress
from IPython.display import display
from casadi import *

h = 0.1
Tfinal = 2 # final time
Nt =int((Tfinal/h)+1) # number of time steps
nx = 4
nu = 2
dim_hidden=8
M=20
Q=1*np.eye(nx)
R=0.1*np.eye(nu)
Qn= 10*np.eye(nx)
learning_rate = 0.001

class Dynamics(nn.Module):

    """
        action[0] = acceleration
        action[1] = steering angle

        state[0] = x
        state[1] = y
        state[2] = velocity
        state[3] = theta

    """

    def __init__(self):
        super(Dynamics,self).__init__()
        self.nx = nx
        self.nu = nu
        self.Nt = Nt

    def forward(self,state,action):
        
        state_tensor = torch.zeros(4)
        state_tensor[0] = torch.cos(state[3])
        state_tensor[1] = torch.sin(state[3])
        velocity_mat = torch.mul(torch.tensor([1.,1.,0.,0.]),state[2])
        inter_mat = torch.mul(velocity_mat,state_tensor)
        accel_tensor = torch.mul(torch.tensor([0., 0., 1., 0.]),action[0])
        angle_tensor = torch.mul(torch.tensor([0., 0., 0., 1.]),action[1])
        final_state = h*(inter_mat+accel_tensor+angle_tensor)
        state = state + final_state

        return state
    
    def rollout(self,x,u):
        x = torch.from_numpy(x)
        u = torch.from_numpy(u)
        x_new=torch.ones([self.nx,self.Nt])
        x_new[:,0] = x[:,0]
        for i in range(self.Nt-1):
            x_new[:,i+1]= self.forward(x_new[:,i],u[:,i])
        return x_new.numpy()


class Controller(nn.Module):
    def __init__(self,dim_input,dim_hidden,dim_output):
        """
        dim_input = No. of state (4)
        dim_output = No. of actions (2)
        dim_hidden = User choice

        """
        super(Controller,self).__init__()
        

        self.network = nn.Sequential(
            nn.Linear(dim_input,dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden,dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden,dim_output))
            #nn.ReLU())

    def forward(self,state):
            action = self.network(state)
            #action = (action - torch.tensor([0.5, 0.5]))*2
            #print (action)
            return action
    
class Simulation(nn.Module):
    def __init__(self,controller,dynamics,Nt,n_params):
        super(Simulation,self).__init__()
        #self.state=self.initialize_state()
        self.controller=controller
        self.dynamics=dynamics
        #self.T=T
        self.theta_trajectory=torch.empty((1,0))
        self.u_trajectory=torch.empty((1,0))
        self.Nt = Nt
        self.n_params = n_params


    def forward(self,state):
        self.action_trajectory=[]
        self.state_trajectory=[]
        self.state_trajectory.append(state)
        dpi_dx = torch.zeros((self.Nt-1,nu,nx))
        dpi_dtheta = torch.zeros((self.Nt-1,nu,self.n_params))
        for i in range (self.Nt-1):
            action = self.controller(state)
            # find dpi/dtheta and dpi/dx
            dpi_dx[i,:,:] = jc(self.controller, state).squeeze(0).squeeze(1)

            grads=[]
            for j in range(action.shape[0]):
                self.controller.zero_grad()
                grad  = autograd.grad(outputs=action[j],inputs=self.controller.parameters(),retain_graph=True)
                grad_flat = torch.cat([g.view(-1) for g in grad])
                grads.append(grad_flat)
            dpi_dtheta[i,:,:] = torch.stack(grads)
            
            state=self.dynamics(state,action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)

        self.action_trajectory = torch.transpose(torch.stack(self.action_trajectory),0,1).detach().numpy()
        self.state_trajectory = torch.transpose(torch.stack(self.state_trajectory),0,1).detach().numpy()
        dpi_dx = dpi_dx.numpy()
        dpi_dtheta = dpi_dtheta.numpy()
        return self.action_trajectory,self.state_trajectory,dpi_dx,dpi_dtheta
    
class find_A_and_B():

    def __init__(self,states,control,total_time_steps,step_size,traj,dynamics):
        self.nx = states
        self.nu = control
        self.Nt = total_time_steps
        self.h = step_size
        self.M = traj 
        self.dynamics = dynamics
        self.Q=1*np.eye(self.nx)
        self.R=0.1*np.eye(self.nu)
        self.Qn= 10*np.eye(self.nx)
        self.qp_solver_func()

    def qp_solver_func(self):
        data=[]
        A_var=SX.sym('A',self.nx,self.nx)
        B_var=SX.sym('B',self.nx,self.nu)
        cost = 0
        for i in range(self.M):
            x_t = SX.sym('x_t',self.nx)
            u_t = SX.sym('u_t',self.nu)
            x_tp1 = SX.sym('x_tp1',self.nx)
            data.append(x_t)
            data.append(u_t)
            data.append(x_tp1)
            cost = cost + sumsqr(A_var @ x_t + B_var @ u_t - x_tp1)

        self.qp_param = vvcat(data)
        self.qp_var  =vvcat([A_var,B_var])
        self.qp_cost = cost
        self.qp_program = {'x':self.qp_var, 'f': self.qp_cost, 'p':self.qp_param}

        opts={'osqp': {'verbose': False}}
        self.qp_solver = qpsol('qp_solver','osqp',self.qp_program,opts)

    def jacobian_estimation(self,x,u):
        
        A_var=SX.sym('A',self.nx,self.nx)
        B_var=SX.sym('B',self.nx,self.nu)
        
        ep=0.02 
        u_new=np.zeros([self.M,self.nu,self.Nt-1])
        x_new=np.zeros([self.M,self.nx,self.Nt])
        x_new[0,:,0]=x[:,0]
        del_x=np.zeros([self.M,self.nx,self.Nt])
        del_u=np.zeros([self.M,self.nu,self.Nt-1])
        for j in range(self.M):
            z1 = np.random.default_rng().standard_normal(size=(self.nu,self.Nt-1))
            u_new[j,:,:]=u+ep*z1
            x_new[j,:,:]=self.dynamics.rollout(x,u_new[j,:,:])
            del_u[j,:,:] = u_new[j,:,:]-u
            del_x[j,:,:] = x_new[j,:,:]-x 
        
        A_fin=np.zeros([self.Nt-1,self.nx,self.nx])
        B_fin=np.zeros([self.Nt-1,self.nx,self.nu])

        for i in range(self.Nt-1):
            
            param_value=[]
            for j in range(self.M):
                x_val = del_x[j,:,i]
                u_val=del_u[j,:,i]
                x_tp1_val = del_x[j,:,i+1]
                param_value.append(x_val)
                param_value.append(u_val)
                param_value.append(x_tp1_val)
            
            param_value_fin= vvcat(param_value)
            sol = self.qp_solver(p=param_value_fin)

            vec_AB = vvcat([A_var, B_var])
            AB_mapping_fn = Function('AB_Mapping_fn', [vec_AB], [A_var, B_var])
            A_fin[i,:,:]=AB_mapping_fn(sol['x'])[0]
            B_fin[i,:,:]=AB_mapping_fn(sol['x'])[1]
        
        return A_fin,B_fin


def calculate_djdt(x,u,A,B,dpi_dx,dpi_dtheta,n_params):
    dx_dtheta = np.zeros([Nt,nx,n_params])
    du_dtheta = np.zeros([Nt-1,nu,n_params])
    for i in range(Nt-1):
        du_dtheta[i,:,:] = dpi_dtheta[i,:,:] + (dpi_dx[i,:,:] @ dx_dtheta[i,:,:])
        dx_dtheta[i+1,:,:] = (A[i,:,:] @ dx_dtheta[i,:,:]) + (B[i,:,:] @ du_dtheta[i,:,:])

    dj_dtheta = np.zeros([1,n_params])
    for i in range(Nt-1):
        dj_dtheta = dj_dtheta + ((Q @ x[:,i]).T @ dx_dtheta [i,:,:]) + (R @ u[:,i]).T @ du_dtheta[i,:,:] 
    dj_dtheta = dj_dtheta + (Qn @ x[:,-1]).T @ dx_dtheta[-1,:,:]

    return dj_dtheta

def update_params(c,dj_dtheta):
    params_flattened = np.concatenate([p.detach().cpu().numpy().flatten() for p in c.parameters()])
    dj_dtheta = dj_dtheta.reshape(-1)
    #print (dj_dtheta)
    new_params = params_flattened - learning_rate*(dj_dtheta)
    new_params = new_params.astype(np.float32)
    #print (new_params)
    idx = 0
    for param in c.parameters():
        param_size = param.numel()
        param_shape = param.shape
        param.data = torch.from_numpy(new_params[idx:idx + param_size]).reshape(param_shape)    
        idx+= param_size

def cost_function(x,u):
        cost=0.5*x[:,-1].T@Qn@x[:,-1] # terminal cost
        for i in range(Nt-1):
            cost=cost+(0.5*x[:,i].T@Q@x[:,i])+(0.5*u[:,i].T@R@u[:,i]) # stage cost
        return cost

d = Dynamics()
c = Controller(nx,dim_hidden,nu)

# find the number of parameters for the 
n_params=sum(p.numel() for p in c.parameters())

s = Simulation(c,d,Nt,n_params)
x0=torch.tensor([10.,12.,0.,0.], requires_grad=True)
i = find_A_and_B(nx,nu,Nt,h,M,d)

for _ in range(10):
    u,x,dpi_dx,dpi_dtheta=s.forward(x0)
    J = cost_function(x,u)
    print (f"cost: {J}")
    A,B = i.jacobian_estimation(x,u)
    dj_dtheta = calculate_djdt(x,u,A,B,dpi_dx,dpi_dtheta,n_params)
    update_params(c,dj_dtheta)
    # u,x,_,_=s.forward(x0)
    # J = cost_function(x,u)
    # print (f" Cost after one update:{J}")
    #print ("-----------------------------------------------------------------------")




