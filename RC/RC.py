"""
Reservoir Computing Code

Reservoir Computer as a Controller for Selective Inhibition
@author: michaelmccreesh
"""

## Import Modules
import numpy as np
import scipy as sp
import scipy.linalg as splin
import scipy.io as spio
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

class Reservoir(object):
    # Generate internal reservoir matrices and the additive constant
    def __init__(self,N,n):
        J_init = 4*np.random.rand(N*n,N*n) - 2
        # Normalize reservoir
        scale = np.linalg.norm(splin.eigh(J_init, eigvals_only = True, subset_by_index=[n, n]))
        self.J = 0.25 * (J_init / scale)
        self.Jy = 2*(np.random.rand(N*n,n)-0.5)
        self.Jr = 2*(np.random.rand(N*n,n)-0.5)
        self.Jrdot = 2*(np.random.rand(N*n,n)-0.5)
        self.d = 0.1
        
# Only use if rest of code is set up to accomodate
class Deep_Reservoir(object):
    def __init__(self,N,n,num_layers):
        J_init = 4*np.random.rand(N*n,N*n,num_layers) - 2
        # Normalize reservoir
        scale = np.linalg.norm(splin.eigh(J_init, subset_by_index=[n, n]))*1.05
        self.J = J_init / scale
        self.Jy = 2*(np.random.rand(N*n,n,num_layers)-0.5)
        self.Jr = 2*(np.random.rand(N*n,n,num_layers)-0.5)
        self.Jrdot = 2*(np.random.rand(N*n,n,num_layers)-0.5)
    

class RC_Controller(object):
    def __init__(self,W,res,m,dt,Tau_net,Tau_res,delta):
        self.W = W
        self.res = res 
        self.m = m # Threshold
        self.dt = dt # Timestep
        self.Tau_net = Tau_net # network timescale
        self.Tau_res = Tau_res # reservoir timescale
        self.delta = delta # Steps ahead to look for tracking
        self.x = np.ones((np.size(W,axis=0),1))*m # Current reservoir state
        self.X = np.ones((np.size(res.J,axis=0),1))*m
        self.u = np.ones((np.size(W,axis=0),1)) # Current control value
        
    #### Learning Phase
    def train(self,v):
        x_driven = np.zeros((np.size(v,0),np.size(v,1)))
        X_driven = np.zeros((np.size(self.res.J,0),np.size(v,1)))
        
        # Propagate the network with the training signal
        for i in range(0,np.size(v,1)):
            self.propagate_system(v[:,i,None])
            x_driven[:,i,None] = self.x
        
        for i in range(0,np.size(v,1)-self.delta):
            xdelta = x_driven[:,i+self.delta]
            x_deriv = (xdelta - x_driven[:,i])/(self.delta*self.dt)
            self.propagate_reservoir(xdelta, x_deriv)
            X_driven[:,i,None] = self.X
        return X_driven, x_driven
    
    # Integrators for Learning Phase
    def propagate_system(self,u):
        k1 = self.LTN_sys(self.x,u)
        k2 = self.LTN_sys(self.x+k1/2,u)
        k3 = self.LTN_sys(self.x+k2/2,u)
        k4 = self.LTN_sys(self.x+k3,u)
        self.x = self.x + (k1 + 2*k2 + 2*k3 + k4)/6
        return
    
    def LTN_sys(self,x,u):
        val = self.W @ x + u
        val[val <= 0] = 0
        val[val > self.m] = self.m
        return self.Tau_net @ (-x + val)
    
    def propagate_reservoir(self,xdelta,x_deriv):
        k1 = self.LTN_reservoir(self.X,self.x,xdelta,x_deriv)
        k2 = self.LTN_reservoir(self.X+k1/2,self.x,xdelta,x_deriv)
        k3 = self.LTN_reservoir(self.X+k2/2,self.x,xdelta,x_deriv)
        k4 = self.LTN_reservoir(self.X+k3,self.x,xdelta,x_deriv)
        self.X = self.X + (k1 + 2*k2 + 2*k3 + k4)/6
        return
    
    def LTN_reservoir(self,X,x,xdelta,x_deriv):
        val = self.res.J @ X + self.res.Jy @ x + (self.res.Jr @ xdelta).reshape(-1,1) + (self.res.Jrdot @ x_deriv).reshape(-1,1) + self.res.d
        val[val <= 0] = 0
        val[val > self.m] = self.m
        return self.Tau_res * (-X + val)
    
    
    #### Control Phase
    def control_system(self,J_out,u0,T):
        times = np.arange(0,T,self.dt)
        sys_traject = np.zeros((np.size(u0),np.size(times)))
        control_vals = np.zeros((np.size(u0),np.size(times)))
        self.x = np.ones((np.size(W,axis=0),1))*m/2 # Current reservoir state
        self.X = np.ones((np.size(res.J,axis=0),1))*m/2
        counter = 0
        for i in times:
            self.propagate_system(self.u)
            self.propagate_reservoir_control(i)
            if i < 25: 
                self.u = 0.5*u0 # Don't immediately apply reservoir controller   
            else:
                self.u = J_out @ self.X
            sys_traject[:,counter,None] = self.x
            control_vals[:,counter ,None] = self.u
            counter += 1
        return times, sys_traject, control_vals
    
    ## Reservoir RK Integrators for Control
    def propagate_reservoir_control(self,t):
        k1 = self.LTN_reservoir_control(self.X,t)
        k2 = self.LTN_reservoir_control(self.X+k1/2,t)
        k3 = self.LTN_reservoir_control(self.X+k2/2,t)
        k4 = self.LTN_reservoir_control(self.X+k3,t)
        self.X = self.X + (k1 + 2*k2 + 2*k3 + k4)/6
        return
    
    def LTN_reservoir_control(self,X,t):
        val = self.res.J @ X + self.res.Jy @ self.x + self.res.Jr @ reference(t+self.delta*self.dt) + self.res.Jrdot @ reference_deriv(t) + self.res.d
        val[val <= 0] = 0
        val[val > self.m] = self.m
        return self.Tau_res * (-X + val)
            


#### Main Code
## RC Parameters
m = 10  # threshold
dt = 0.02 # time step
Tau_res = 1.4 # reservoir time constant
delta = 1 # Number of steps ahead to look in reference tracking

# Times and samples for training
T_l = 1000 # learning phase
T_t = 100 # transient phase
T_p = 1000 # prediction phase

n_train = int(T_l/dt)
n_trans = int(T_t/dt)
n_pred = int(T_p/dt)
## Will need something for indices for training time but come back to ##
n_samples = n_trans + n_train + n_pred


## Network for tracking
W = np.array([[0.0112, -0.9903],[0.4101, -0.5115]])
tau_net = 1/4*np.identity(2)

## Generate Reservoir
N = 50 # scaling of reservoir from network size
n = np.size(W,0) # Size of network
num_layers = 1 # Number of layers if deep RC

res = Reservoir(N,n) 

## Import MATLAB reservoir and data (used for testing)
mat = spio.loadmat('data.mat', squeeze_me = True)

res.J = mat['J']
res.Jy = mat['Jy']
res.Jr = mat['Jr']
res.Jrdot = mat['Jrdot']
v_mat = mat['v']
x_driven_mat = mat['x_driven']
X_driven_mat = mat['X_driven']
vT_mat = mat['vT']
X_drivenT_mat = mat['X_drivenT']
J_out_mat = mat['J_out']
## Create the Reservoir Controller (Object)
R = RC_Controller(W,res,m,dt,tau_net,Tau_res,delta)

## Generate Training Data
# This could be done in almost any way. Here is an arbitrarily constructed one
# that was found to work. Different training data can change prediction
# performance
idx = np.arange(0,n_samples+1,1,dtype=int)
scaling = np.sin(2*idx/1000)
v = scaling*np.array([3*np.ones((n_samples+delta)),-2*np.ones((n_samples+delta))])

## Train the Reservoir Controller
X_driven, x_driven = R.train(v)

# Discard start of data
vT = v[:,(n_train+n_trans):]
X_drivenT = X_driven[:,(n_train+n_trans):]

def compute_output_vector(X_driven,vT,beta):        
    J_out_transpose = tikhonov_reg(np.transpose(X_driven),np.transpose(vT),beta)
    return np.transpose(J_out_transpose)

#Tikhonov Regularization
def tikhonov_reg(A,b,beta):
    U,S,Vt = np.linalg.svd(A,full_matrices = False)
    V = Vt.T
    true_s = np.zeros((U.shape[1], V.shape[0]))
    true_s[:S.size, :S.size] = np.diag(S)
    btilde = U.T @ b
    Shold = true_s.T @ true_s
    hold2 = Shold + (beta**2)*np.identity(np.shape(true_s)[0])
    Xtilde = np.linalg.inv(hold2) @ (np.diag(S.T) @ btilde)
    X = V @ Xtilde
    return X


    
beta = 0.5
J_out = compute_output_vector(X_drivenT, vT, beta)


## Define Reference Signal
def reference(t):
    r1 = np.sin(2*np.pi*t/200)+2
    r2 = 0
    return np.array([[r1],[r2]])

def reference_deriv(t):
    r1_deriv = np.cos(2*np.pi*t/200)*2*np.pi/200
    r2_deriv = 0
    return np.array([[r1_deriv],[r2_deriv]])
    
# Dimensions must match the network size



## Track the Reference Signal
u0 = np.array([[2],[2]])
T = 750
times, sys_traject, control_vals = R.control_system(J_out, u0, T)

## Plot Results

times = np.arange(0,750,dt)
reference_vals = np.zeros((2,np.size(times,0)))
counter = 0
for i in times:
    reference_vals[:,counter,None] = reference(i)
    counter += 1
fig = plt.figure()
plt.plot(times,reference_vals[0,:],'--r',linewidth = 2)
plt.plot(times,reference_vals[1,:],'--k', linewidth = 2)
plt.plot(times,sys_traject[0,:],'-b',linewidth = 2)
plt.plot(times,sys_traject[1,:],'-g',linewidth = 2)
plt.xlim([0,750])
plt.ylim([-0.1,3.05])
plt.show()


# fig2 = plt.figure()
# plt.plot(times,control_vals[:,0],'-r')
# plt.plot(times,control_vals[:,1],'-b')
# plt.show()


    