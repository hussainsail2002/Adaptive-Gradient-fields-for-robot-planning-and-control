import numpy as np
import matplotlib.pyplot as plt
import os
from casadi import *
from AG_OC_ilqr import Ilqr

# Global variables
h=0.1 # time step
nx=4 # number of state  
nu=2 # number of control
Tfinal = 5 # final time 
Nt =int((Tfinal/h)+1) # number of time steps
M = 20 # number of perturbed trajectories  

x0=np.array([[10],[15],[2],[0]]) # Initial state:
u=np.random.default_rng().standard_normal(size=(nu,Nt-1)) # Assume random control values
x=np.zeros([nx,Nt]) #initialize the state vector
x[:,0:1]=x0 

#Load the ilqr class
ilqr=Ilqr(nx,nu,Nt,h,M)
x = ilqr.forward_rollout(x,u)

#ilqr / DDP algorithim
# We do the forward pass in this codeblock

u_fin=ilqr.forward_pass(x,u)
x_fin=ilqr.forward_rollout(x,u_fin)

if os.path.exists('A_matrix_V5_online.pth'):
    os.remove('A_matrix_V5_online.pth')

if os.path.exists('B_matrix_V5_online.pth'):
    os.remove('B_matrix_V5_online.pth')

if os.path.exists('ml_trainingdata_V4_R2_A.csv'):
    os.remove('ml_trainingdata_V4_R2_A.csv')

plt.plot(x_fin[0,:],x_fin[1,:],color='b')
plt.title('X and Y axis position of Bicycle')
plt.xlabel('X axis Position ')
plt.ylabel('Y axis Position')
plt.show()
