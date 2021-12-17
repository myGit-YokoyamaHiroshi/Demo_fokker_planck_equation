# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:36:31 2021

@author: yokoyama
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')


import os
current_path = os.path.dirname(__file__)
os.chdir(current_path)

import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#"IPAexGothic"
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams['xtick.direction']  = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams["font.size"]        = 12 # 全体のフォントサイズが変更されます。
plt.rcParams['lines.linewidth']  = 2.0
plt.rcParams['figure.dpi']       = 300
plt.rcParams['savefig.dpi']      = 300 
#%%
from copy import deepcopy
from numpy.matlib import repmat
from numpy.random import *
import numpy as np
#%%
def func_kuramoto(theta, K, omega):
    Nosc = theta.shape[0]
    phase_diff = np.zeros(K.shape)
    phase_dynamics = np.zeros(theta.shape)
    
    for n in range(len(theta)):
        phase_diff = theta - theta[n]
        phase_dynamics[n] = omega[n] + np.sum(K[n,:] * np.sin(phase_diff)) 
    return phase_dynamics


def euler_maruyama(h, func, theta_now, K, omega, noise_scale):
    dt = h
    p  = noise_scale
    dw = np.random.randn(theta_now.shape[0])
    
    theta      = theta_now + func(theta_now, K, omega) * dt
    theta_next = theta + np.sqrt(dt) * p * dw
    theta_next = np.mod(theta_next, 2*np.pi)
    return theta_next

def solve_fokker_planck_2osc(omega, kappa, dt, noise_scale):
    _, _      = kappa.shape
    N         = 100
    t         = np.arange(0, 100+dt, dt)
    Nt        = len(t)
    phi_diff  = np.linspace(0, 2*np.pi, N)
    D         = (noise_scale*np.sqrt(dt))
    
    G21 = omega[1] - omega[0] - (kappa[1, 0] + kappa[0, 1]) * np.sin(phi_diff) # d(θ2-θ1)/dt  
    G12 = omega[0] - omega[1] - (kappa[0, 1] + kappa[1, 0]) * np.sin(phi_diff) # d(θ1-θ2)/dt 
    
    P = np.zeros((Nt, N, 2))
    
    delta    = (2*np.pi-0)/N
    P[0,:,0] = np.ones(N)
    P[0,:,1] = np.ones(N)
    for t in range(1, Nt):
        P21_now = P[t-1,:,0]
        P12_now = P[t-1,:,1]
        
        dP21    = np.zeros(N)
        dP12    = np.zeros(N)
        for i in range(1, N-1):
            term1   = P21_now[i]* ((G21[i+1] - G21[i-1])/(2*delta)) # numerical approximation with center difference
            term2   = G21[i] * ((P21_now[i+1] - P21_now[i-1])/(2*delta)) # numerical approximation with center difference
            term3   = D * ((P21_now[i+1] - 2*P21_now[i] + P21_now[i-1])/delta**2) # numerical approximation of second derivative
            dP21[i] = -term1 - term2 + term3 # dP(θ2-θ1)/dt
            
            term1   = P12_now[i]* ((G12[i+1] - G12[i-1])/(2*delta)) # numerical approximation with center difference
            term2   = G12[i] * ((P12_now[i+1] - P12_now[i-1])/(2*delta)) # numerical approximation with center difference
            term3   = D * ((P12_now[i+1] - 2*P12_now[i] + P12_now[i-1])/delta**2) # numerical approximation of second derivative
            dP12[i] = -term1 - term2 + term3 # dP(θ1-θ2)/dt
        
        P[t,:,0] = P21_now + dP21*dt # numerical integral with Euler method
        P[t,:,1] = P12_now + dP12*dt # numerical integral with Euler method
    
    # normalization
    prob = np.array([P[-1,:,i]/(P[-1,:,i].sum() * delta) for i in range(2)]).T
    return phi_diff, prob
#%% generate synthetic phase data with 2 coupled kuramoto model 
time        = 200# measurement time
h           = 0.01    # micro time
Nosc        = 2
Nt          = int(time/h)
K           = np.ones((Nosc, Nosc)) - np.eye(Nosc)

cnt         = 0

omega       = 2 * np.pi * np.array([1.06, 1.11])##
kappa       = 1 / Nosc * np.array([[0.0, 0.55], [0.55, 0.0]])
dtheta      = np.zeros((Nt, Nosc))
theta       = np.zeros((Nt, Nosc))

theta[0, :] = np.array([0, np.pi])
order       = np.zeros(Nt)

phse_dynamics       = np.zeros((Nt, Nosc))
phse_dynamics[0, :] = func_kuramoto(theta[0, :], kappa, omega)#[0,:])
order[0]            = abs(np.mean(np.exp(1j*theta[0, :])))

noise_scale = 0.1
for t in range(1, Nt):    
    theta_now  = theta[t-1, :] 
    theta_next = euler_maruyama(h, func_kuramoto, theta_now, kappa, omega, noise_scale)#runge_kutta(h, func_kuramoto, theta_now, kappa, omega)#[t, :])
    
    theta[t, :]         = theta_next
    phse_dynamics[t, :] = func_kuramoto(theta[t, :], kappa, omega)#[t, :])
    
    for i in range(Nosc):
        theta_unwrap = np.unwrap(deepcopy(theta[t-1:t+1, i]))
        
        dtheta[t, i] = (theta_unwrap[1] - theta_unwrap[0])/h
#%% solve Fokker-Planck equation (estimate probability density of phase difference)
xaxis, prob = solve_fokker_planck_2osc(omega, kappa, h, noise_scale)

fig_save_dir = current_path + '/figures/' 
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)
    
plt.hist(np.mod(theta[:,1]-theta[:,0], 2*np.pi), bins=100, density=True, range=(0, 2*np.pi)); 
plt.plot(xaxis, prob[:,0])
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi, 2 * np.pi], ['$0$', '$\\pi$', '$2 \\pi$'])
plt.xlabel('$\\theta_1 - \\theta_0$')
plt.ylabel('density')
plt.savefig(fig_save_dir + 'density_phase_diff10.png', bbox_inches="tight")
plt.savefig(fig_save_dir + 'density_phase_diff10.svg', bbox_inches="tight")
plt.show()

plt.hist(np.mod(theta[:,0]-theta[:,1], 2*np.pi), bins=100, density=True, range=(0, 2*np.pi)); 
plt.plot(xaxis, prob[:,1])
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi, 2 * np.pi], ['$0$', '$\\pi$', '$2 \\pi$'])
plt.xlabel('$\\theta_0 - \\theta_1$')
plt.ylabel('density')
plt.savefig(fig_save_dir + 'density_phase_diff01.png', bbox_inches="tight")
plt.savefig(fig_save_dir + 'density_phase_diff01.svg', bbox_inches="tight")
plt.show()

