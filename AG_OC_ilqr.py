import numpy as np
from AG_OC_pontry_nn import MyNueralNetworkA, MyNueralNetworkB


# Loading the models
f=MyNueralNetworkA()
f2=MyNueralNetworkB()

class Ilqr():

    def __init__(self,states,control,total_time_steps,step_size):
        self.nx = states
        self.nu = control
        self.Nt = total_time_steps
        self.h = step_size
        self.Q=1*np.eye(self.nx)
        self.R=0.1*np.eye(self.nu)
        self.Qn= 100*np.eye(self.nx)

    # This is the forward dynamics of the uni-cycle model
    def cycle_dynamics(self,x,u):
        x_new=np.zeros(self.nx)
        x_new[0] = x[0] + x[2]*np.cos(x[3])*self.h
        x_new[1] = x[1] + x[2]*np.sin(x[3])*self.h
        x_new[2] = x[2] + u[0]*self.h
        x_new[3] = x[3] + u[1]*self.h
        return x_new
    
    def forward_rollout(self,x,u):
        x_new=np.ones([self.nx,self.Nt])
        x_new[:,0] = x[:,0]
        for i in range(self.Nt-1):
            x_new[:,i+1]= self.cycle_dynamics(x_new[:,i],u[:,i])
        return x_new
        
    def cost_function(self,x,u):
        cost=0.5*x[:,-1].T@self.Qn@x[:,-1] # terminal cost
        for i in range(self.Nt-1):
            cost=cost+(0.5*x[:,i:i+1].T@self.Q@x[:,i:i+1])+(0.5*u[:,i:i+1].T@self.R@u[:,i:i+1]) # stage cost
        return cost[0,0]

    def backward_pass(self):
        pass

    def forward_pass(self):

        p=np.ones([self.nx,self.Nt])
        P=np.zeros([self.Nt,self.nx,self.nx])
        d=np.ones([self.nu,self.Nt-1])
        K = np.zeros([self.Nt-1,self.nu,self.nx])
        del_j=1.0

        xn=np.zeros([self.nx,self.Nt])
        un=np.zeros([self.nu,self.Nt-1])

        gx=np.zeros([self.nx,1])
        gu=np.zeros([self.nu,1])
        Gxx=np.zeros([self.nx,self.nx])
        Guu=np.zeros([self.nu,self.nu])
        Gxu=np.zeros([self.nx,self.nu])
        Gux=np.zeros([self.nu,self.nx])

        nn_data=np.empty((0,10))
        nn_data_B=np.empty((0,10))

        iter = 0
        flag=0
        nn_count=0 # how many times am I training the nn ?
        while iter<65:
            
            del_j,K,d,A,nn_data,nn_data_B = self.backward_pass(p,P,d,K,x,u,flag,nn_data,nn_data_B)
            xn[:,0]=x[:,0]
            alpha = 0.1

            for i in range(self.Nt-1):
                un[:,i]=u[:,i] - alpha*d[:,i] - np.matmul(K[i,:,:],(xn[:,i]-x[:,i]))
                xn[:,i+1] = self.cycle_dynamics(xn[:,i],un[:,i])
            Jn = self.cost_function(xn,un)

            loop_count=0
            
            while Jn>J:
                J=Jn
                nn_count=nn_count+1
                print (f"training time:{nn_count}")
                flag =1
                del_j,K,d,A,nn_data,nn_data_B = self.backward_pass(p,P,d,K,x,u,flag,nn_data,nn_data_B)
                xn[:,0]=x[:,0]
                alpha = 0.1
                for i in range(self.Nt-1):
                    un[:,i]=u[:,i] - alpha*d[:,i] - np.matmul(K[i,:,:],(xn[:,i]-x[:,i]))
                    xn[:,i+1] = self.cycle_dynamics(xn[:,i],un[:,i],self.h)
                Jn = self.cost_function(xn,un)
                loop_count=loop_count+1
                if loop_count>=3:
                    break
            flag=0
            
            #while Jn > (J - 0.001*alpha*del_j):
                #alpha = 0.5*alpha
                #for i in range(Nt-1):
                    #un[:,i]=u[:,i] - alpha*d[:,i] - np.matmul(K[i,:,:],(xn[:,i]-x[:,i]))
                    #xn[:,i+1] = cycle_dynamics(xn[:,i],un[:,i],h)
                #Jn = cost_function(xn,un)
                #if alpha < 0.05:
                    #break
            
            J = Jn
            x=xn
            u=un
            xn=np.zeros([self.nx,self.Nt])
            print (J)
            iter=iter+1
    
        return u

