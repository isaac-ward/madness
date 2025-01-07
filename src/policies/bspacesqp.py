## TODO convert this code from LP to QP - more from HW8 A11.8 from Convex Optimization I\
# import matplotlib.pyplot as plt
import cvxpy as cvx
import cupy as cp
import numpy as np

import utils.geometric
from utils.general import log_softmax, gradient_log_softmax, Cacher
import policies.costs
import policies.samplers

from dynamics_jax import DynamicsQuadcopter3D
from sdf import Environment_SDF, SDF_Types

class Trajectory:
    def __init__(
                self,
                state=None,
                action=None
            ):
        self.state = state
        self.action = action

class BSpaceSolver:
    def __init__(self,
            K,
            dynamics: DynamicsQuadcopter3D,
            trajInit: Trajectory,
            sdf:Environment_SDF,
            cost_tol = 1e-3,
            maxiter = 50.,
            sig = 10.,
            eps_dyn = 1.,
            eps_sdf = 1e-4,
            eta = 1.,
            pull_from_cache=False):
        
        self.K = K
        self.dynamics = dynamics
        self.dt = dynamics.dt
        self.cost_tol = cost_tol
        self.maxiter = maxiter
        self.sig = sig
        self.eps_dyn = eps_dyn
        self.eps_sdf = eps_sdf
        self.eta = eta

        self.action_prev = trajInit.action
        self.state_prev = trajInit.state
        self.sdf = sdf

        self.pull_from_cache = pull_from_cache

    def f0(self, x):
        '''
        Boiler plate objective function
        '''
        return x
    
    def barrier(self, x):
        '''
        Boiler plate barrier
        '''
        return x
    
    def constructA(self):
        '''
        Boiler plate to construct stacked equality consraint matrix
        '''
        A = None
        return A
    
    

# A11.8
def gradFun(x,c):
    gradF = c - 1/x
    return gradF

def rFun(x,v,A,b,c):
    r = np.vstack([gradFun(x,c) + A.T @ v, A @ x - b])
    # print(r.shape)
    return r

def isNewton(A,b,c,x0, maxiter = 50, alpha = 0.4, beta = 0.8, eps = 1e-6):

    v = np.zeros((m,1))
    x = x0.copy()
    ns = 0
    rs = [np.linalg.norm(rFun(x,v,A,b,c))]

    while ns < maxiter:
        ns += 1

        g = gradFun(x,c) + A.T @ v
        h = A @ x - b 

        Hinv = np.diag((x**2)[:,0])
                       
        s = A @ (Hinv @ A.T)
        
        btild = h - A @ (Hinv @ g)
        
        delv = np.linalg.solve(s, btild)
        delx = -Hinv @ (g + A.T @ delv)

        t = 1

        xstep = x + t*delx
        vstep = v + t*delv

        while np.any(xstep <= 0):
            t = beta*t 
            xstep = x + t*delx
            vstep = v + t*delv
        
        while np.linalg.norm(rFun(xstep,vstep,A,b,c)) > (1 - alpha*t)*np.linalg.norm(rFun(x,v,A,b,c)) or np.any(xstep <= 0):
            t = beta*t 
            xstep = x + t*delx
            vstep = v + t*delv

        x = xstep
        v = vstep

        rs.append(np.linalg.norm(rFun(x,v,A,b,c)))

        if np.all(np.abs(A @ x - b) <= eps) and np.linalg.norm(rFun(x,v,A,b,c)) <= eps:
            break
    
    xstar = x
    vstar = v

    rs = np.array(rs)
    ks = np.linspace(0,ns,ns+1)

    return xstar, vstar, ns, rs, ks

def barrierMethod(A,b,c,x0,t0,mu = 2,eps = 1e-3):
    xstar = x0
    t = t0
    lamstar = 1/(t*xstar)
    n = xstar.size

    cs = 0
    maxcs = 100

    nslist = []
    dglist = []

    while cs < maxcs:
        cs += 1
        [xstar, vstar, ns, _, _] = isNewton(A,b,t*c,xstar)

        lamstar = 1/(t*xstar)

        nslist.append(ns)
        dglist.append(n/t)

        if n/t <= eps:
            break

        t = mu*t 

    history = np.array([nslist, dglist])
    print(history.shape)
    print(cs)

    
    return xstar, vstar, lamstar, history

# # Various starting conditions

# fig, ax = plt.subplots()
# for i in range(8):
#     n = 20 # dimensions of x and c
#     m = 10 # dimensions of A (mxn) and v

#     rng = np.random.default_rng(i)
#     A = rng.normal(size=(m,n))
#     A[0,:] = np.abs(A[0,:])
#     p = np.abs(rng.normal(size=(n,1)))
#     b = A @ p
#     c = rng.normal(size=(n,1))

#     x0 = np.abs(rng.normal(size=(n,1)))
#     # print(b)
#     # print(x0)

#     [xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0)
#     ax.semilogy(ks, rs)


# ax.set_xlabel('k')
# ax.set_ylabel('2-Norm of Residual')
# ax.set_title('Convergence for Various Starting Conditions (n = 20, m = 10)')

# # Fixed Problem with Fixed Alpha and Varying Beta
# n = 20 # dimensions of x and c
# m = 10 # dimensions of A (mxn) and v

# rng = np.random.default_rng(10)
# A = rng.normal(size=(m,n))
# p = np.abs(rng.normal(size=(n,1)))
# b = A @ p
# c = rng.normal(size=(n,1))
# x0 = np.abs(rng.normal(size=(n,1)))

# betas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# fig, ax = plt.subplots()
# for i in range(betas.size):
#     [xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0, beta=betas[i])
#     ax.semilogy(ks, rs, label=r'$\beta = $' + str(betas[i]))
# ax.legend()

# ax.set_xlabel('k')
# ax.set_ylabel('2-Norm of Residual')
# ax.set_title(r'Convergence for Fixed $\alpha$ Values and Varying $\beta$ (n = 20, m = 10)')

# #Fixed Problem with Fixed Beta and Verying Alpha
# alphas = np.array([0.1, 0.2, 0.3, 0.4])
# fig, ax = plt.subplots()
# for i in range(alphas.size):
#     [xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0, alpha=alphas[i])
#     ax.semilogy(ks, rs, label=r'$\alpha = $' + str(alphas[i]))
# ax.legend()

# ax.set_xlabel('k')
# ax.set_ylabel('2-Norm of Residual')
# ax.set_title(r'Convergence for Fixed $\beta$ Values and Varying $\alpha$ (n = 20, m = 10)')

# # Test Unbounded Below
# m = 3
# n = 4
# A = np.hstack([np.eye(m), np.zeros((m, 1))])
# b = np.ones((m,1))
# c = -np.ones((n,1))
# x0 = np.abs(rng.normal(size=(n,1)))

# [xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0)
# fig, ax = plt.subplots()
# ax.semilogy(ks,rs)

# ax.set_xlabel('k')
# ax.set_ylabel('2-Norm of Residual')
# ax.set_title('Convergence for Unbounded Below Case (n = 4, m = 3)')

# # Test Infeasible
# b = -np.ones((m,1))

# [xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0)
# fig, ax = plt.subplots()
# ax.semilogy(ks,rs)

# ax.set_xlabel('k')
# ax.set_ylabel('2-Norm of Residual')
# ax.set_title('Convergence for Infeasible Case (n = 4, m = 3)')

# # Large m and n values
# n = 200 # dimensions of x and c
# m = 150 # dimensions of A (mxn) and v

# rng = np.random.default_rng(10)
# A = rng.normal(size=(m,n))
# p = np.abs(rng.normal(size=(n,1)))
# b = A @ p
# c = rng.normal(size=(n,1))

# x0 = np.abs(rng.normal(size=(n,1)))
# [xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0)
# fig, ax = plt.subplots()
# ax.semilogy(ks,rs)

# ax.set_xlabel('k')
# ax.set_ylabel('2-Norm of Residual')
# ax.set_title('Convergence for High Dimensional Problem (n = 200, m = 150)')
