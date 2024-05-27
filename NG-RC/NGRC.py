#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Next-Generation Reservoir Computing Code

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

class NGRC_Controller(object):
    
    def __init__(self,W,m,tau,K,dt,c,beta):
        self.W = W
        self.m = m
        self.tau = tau
        self.K = -5
        self.c = 0.5
        self.beta = beta
        self.dt = dt
        self.x = np.ones((np.size(W,axis=0),1))*m/2 # Current reservoir state
        self.u = np.ones((np.size(W,axis=0),1)) # Current control value
        self.X_l = None # Linear feature vector
        self.X_nl = None # Nonlinear feature vector
    
    def train(self,v):
        # Initialize to zeros variables to return
        x_driven = np.zeros((np.size(v,0),np.size(v,1)))
        X_total = np.zeros((int((np.size(v,0)+1 + np.size(self.W,1) + np.size(self.W,1)*(np.size(self.W,1)+1)/2)),np.size(v,1)))
        
        for i in range(0,np.size(v,1)):
            self.X_l = self.x # set linear feature vector
            self.X_nl = self.nonlin_feature_vector(self.X_l)
            X_total[:,i,None] = np.concatenate((np.append(v[:,i],self.c)[:,None],self.X_l,self.X_nl))
            # Move system forward
            self.propagate_system(v[:,i,None])
            x_driven[:,i,None] = self.x
        
        J_total = self.compute_output_vector(X_total, x_driven)
        J_u = J_total[:,0:np.size(v,0)]
        J_X = J_total[:,np.size(v,0):]
        return J_X, J_u        
    
    def compute_output_vector(self,X_driven,vT):        
        J_out_transpose = self.tikhonov_reg(np.transpose(X_driven),np.transpose(vT))
        return np.transpose(J_out_transpose)

    #Tikhonov Regularization
    def tikhonov_reg(self,A,b):
        U,S,Vt = np.linalg.svd(A,full_matrices = False)
        V = Vt.T
        true_s = np.zeros((U.shape[1], V.shape[0]))
        true_s[:S.size, :S.size] = np.diag(S)
        btilde = U.T @ b
        Shold = true_s.T @ true_s
        hold2 = Shold + (self.beta**2)*np.identity(np.shape(true_s)[0])
        Xtilde = np.linalg.inv(hold2) @ (np.diag(S.T) @ btilde)
        X = V @ Xtilde
        return X
                           
    def nonlin_feature_vector(self,X_l):
        X_nl = np.zeros((int(np.size(self.W,0)*(np.size(self.W,0)+1)/2),1))
        for i in range(0,np.size(X_l,0)):
            for j in range(0,np.size(X_l,0)):
                X_nl[i+j] = X_l[i]*X_l[j]
        return X_nl
    
    def control(self,J_X,J_u,T):
        times = np.arange(0,T,self.dt)
        sys_traject = np.zeros((np.size(self.x),np.size(times)+1))
        control_vals = np.zeros((np.size(self.u),np.size(times)+1))
        
        counter = 1
        self.x = np.ones((np.size(self.W,0),1))
        self.u = np.zeros((np.size(self.W,0),1))
        
        for i in times:
            self.propagate_system(self.u)
            self.X_l = self.x
            self.X_nl = self.nonlin_feature_vector(self.X_l)
            X = np.append(self.c,np.concatenate((self.X_l,self.X_nl)))
            if i < 25: # Don't start control immediately
                self.u = np.zeros((np.size(self.W,0),1))
            else:
                self.u = np.linalg.inv(J_u) @ (reference(i+self.dt) - (J_X @ X).reshape(2,1) + self.K *(self.x - reference(i)))
            
            sys_traject[:,counter,None] = self.x
            control_vals[:,counter,None] = self.u
            counter +=1
        
        return sys_traject, control_vals
    # RK Integrators for Propagating System
    def propagate_system(self,u):
        k1 = self.dt * self.LTN_sys(self.x,u)
        k2 = self.dt * self.LTN_sys(self.x+k1/2,u)
        k3 = self.dt * self.LTN_sys(self.x+k2/2,u)
        k4 = self.dt * self.LTN_sys(self.x+k3,u)
        self.x = self.x + (k1 + 2*k2 + 2*k3 + k4)/6
        return
    
    def LTN_sys(self,x,u):
        val = self.W @ x + u
        val[val <= 0] = 0
        val[val > self.m] = self.m
        return self.tau @ (-x + val)
    
    
        
        
#### Main Code
## RC Parameters
m = 10  # threshold
dt = 0.02 # time step
delta = 1 # Number of steps ahead to look in reference tracking

## Network for tracking
W = np.array([[0.0112, -0.9903],[0.4101, -0.5115]])
tau = 1/4*np.identity(2)

## NG-RC Parameters
K = -5 # Proportional Control Value - determine experimentally
c = 0.5 # NG-RC feature vector scalar term
beta = 0.3 # regularization parameter

## Initialize NG-RC Controller
R = NGRC_Controller(W, m, tau, K, dt, c, beta)




## Import training signal from MATLAB file and other data for debugging
mat = spio.loadmat('data.mat', squeeze_me = True)
v = mat['v_train']

J_X, J_u = R.train(v)

## Define reference signal
def reference(t):
    r1 = np.sin(2*np.pi*t/200)+2
    r2 = 0
    return np.array([[r1],[r2]])

T = 750
sys_traject, control_traject = R.control(J_X, J_u, T)

## Plot Results

times = np.arange(0,750+dt,dt)
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
