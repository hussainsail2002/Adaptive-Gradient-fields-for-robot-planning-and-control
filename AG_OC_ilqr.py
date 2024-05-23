import numpy as np
from AG_OC_pontry_nn import normalize_A, cycle_dynamics_pontry_train_A, cycle_dynamics_nn_diff_A
from AG_OC_pontry_nn import cycle_dynamics_pontry_train_B, cycle_dynamics_nn_diff_B
from casadi import *
from scipy import linalg

class Ilqr():

    def __init__(self,states,control,total_time_steps,step_size,traj):
        self.nx = states
        self.nu = control
        self.Nt = total_time_steps
        self.h = step_size
        self.M = traj 
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
    

    def cycle_dynamics_pontryagin(self,x,u):
        ep=0.15 
        u_new=np.zeros([self.M,self.nu,self.Nt-1])
        x_new=np.zeros([self.M,self.nx,self.Nt])
        x_new[0,:,0]=x[:,0]
        del_x=np.zeros([self.M,self.nx,self.Nt])
        del_u=np.zeros([self.M,self.nu,self.Nt-1])
        for j in range(self.M):
            z1 = np.random.default_rng().standard_normal(size=(self.nu,self.Nt-1))
            u_new[j,:,:]=u+ep*z1
            x_new[j,:,:]=self.forward_rollout(x,u_new[j,:,:])
            del_u[j,:,:] = u_new[j,:,:]-u
            del_x[j,:,:] = x_new[j,:,:]-x 
        
        A_fin=np.zeros([self.Nt-1,self.nx,self.nx])
        B_fin=np.zeros([self.Nt-1,self.nx,self.nu])

        for i in range(self.Nt-1):
            loss_function=0
            A=SX.sym('A',self.nx,self.nx)
            B=SX.sym('B',self.nx,self.nu)
            for j in range(self.M):
                loss_function=loss_function+sumsqr(A @ del_x[j,:,i]+ B @ del_u[j,:,i] - del_x[j,:,i+1])
            qp = {'x':vertcat(reshape(A,-1,1),reshape(B,-1,1)),'f':loss_function}
            S=qpsol('S','proxqp',qp)
            solve=S()
            vec_AB = vvcat([A, B])
            AB_mapping_fn = Function('AB_Mapping_fn', [vec_AB], [A, B])
            A_fin[i,:,:]=AB_mapping_fn(solve['x'])[0]
            B_fin[i,:,:]=AB_mapping_fn(solve['x'])[1]
        
        return A_fin,B_fin,x_new,u_new

    def backward_pass(self,p,P,d,K,x,u,flag,nn_data,nn_data_B):
        del_j=0
        p[:,-1] = self.Qn@x[:,-1]
        P[-1,:,:] = self.Qn
        avg_data=0
        std_data=1
        # _,B,_,_=cycle_dynamics_pontryagin(x,u)
        A1,_,_,_=self.cycle_dynamics_pontryagin(x,u)
        if flag==1:
            A1,B,x_traj,u_traj=self.cycle_dynamics_pontryagin(x,u)
            for i in range(self.M):
                for j in range(self.Nt-1):
                    x_flat=x_traj[i,:,j]
                    u_flat=u_traj[i,:,j].flatten()
                    A_flat=[]
                    B_flat=[]
                    for l in range(self.nx):
                        for m in range(self.nx):
                            if l>m:
                                A_flat.append(A1[j,m,l])
                    A_flat=np.array(A_flat)
                    A_flat=np.delete(A_flat,0)
                    A_flat=np.delete(A_flat,-1)
                    for p1 in range(self.nx):
                        for q in range(self.nu):
                            if p1>1:
                                B_flat.append(B[j,p1,q])
                    B_flat=np.array(B_flat)

                    new_data=np.matrix(np.hstack([x_flat,u_flat,A_flat]))
                    new_data_B=np.matrix(np.hstack([x_flat,u_flat,B_flat]))
                    nn_data=np.append(nn_data,new_data,axis=0)
                    nn_data_B=np.append(nn_data_B,new_data_B,axis=0)
            
            print (nn_data.shape)
            if nn_data.shape[0] > 1600:
                nn_data_R1,avg_data,std_data = normalize_A(nn_data[-1600:,:])
                cycle_dynamics_pontry_train_A(nn_data_R1)
                cycle_dynamics_pontry_train_B(nn_data_B)
                avg_data=np.ravel(avg_data)
                std_data=np.ravel(std_data)
            else:
                nn_data,avg_data,std_data = normalize_A(nn_data)
                cycle_dynamics_pontry_train_A(nn_data)
                cycle_dynamics_pontry_train_B(nn_data_B)
                avg_data=np.ravel(avg_data)
                std_data=np.ravel(std_data)

        for i in reversed(range(self.Nt-1)):
            q = self.Q@x[:,i]
            r = self.R@u[:,i]

            A = cycle_dynamics_nn_diff_A(x[:,i],u[:,i],avg_data,std_data)
            B = cycle_dynamics_nn_diff_B(x[:,i],u[:,i])
            #A,B=cycle_dynamics_diff(x[:,i],u[:,i])

            gx = q + A.T@p[:,i+1]
            gu = r + B.T@p[:,i+1]

            Gxx = self.Q + A.T@P[i+1,:,:]@A
            Guu = self.R + B.T@P[i+1,:,:]@B
            Gxu = A.T@P[i+1,:,:]@B
            Gux = B.T@P[i+1,:,:]@A

            det = np.linalg.det(Guu)
            if det < 0.01:
                print(det)

            d[:,i] = linalg.inv(Guu) @ gu
            K[i,:,:] = linalg.inv(Guu) @ Gux

            p[:,i] = gx - K[i,:,:].T@gu + K[i,:,:].T@Guu@d[:,i] - Gxu@d[:,i]
            P[i,:,:] = Gxx + K[i,:,:].T@Guu@K[i,:,:] - Gxu@K[i,:,:] - K[i,:,:].T@Gux

            del_j = del_j + gu.T@d[:,i]
        
        
        return del_j,K,d,A,nn_data,nn_data_B

    def forward_pass(self,x,u):

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
        J = self.cost_function(x,u)


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
                    xn[:,i+1] = self.cycle_dynamics(xn[:,i],un[:,i])
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
