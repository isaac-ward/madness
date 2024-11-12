## TODO convert this code from LP to QP - more from HW7 A10.4 from Convex Optimization I
import numpy as np
import cvxpy as cp

# A10.4
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
        # print(ns)

        g = gradFun(x,c) + A.T @ v
        h = A @ x - b 

        # if ns == 1:
            # print(np.diag((x**2)[:,0]))
            # print('g = ', g)
            # print('h = ', h)
            # print('r = ', rFun(x,v,A,b,c))

        Hinv = np.diag((x**2)[:,0])
                       
        s = A @ (Hinv @ A.T)

        # print(np.linalg.matrix_rank(s))
        
        btild = h - A @ (Hinv @ g)
        
        delv = np.linalg.solve(s, btild)
        delx = -Hinv @ (g + A.T @ delv)

        # print(delx)
        # print(delv)

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

        # print(x)
        # print(v)

        rs.append(np.linalg.norm(rFun(x,v,A,b,c)))

        if np.all(np.abs(A @ x - b) <= eps) and np.linalg.norm(rFun(x,v,A,b,c)) <= eps:
            break
    
    xstar = x
    vstar = v

    rs = np.array(rs)
    ks = np.linspace(0,ns,ns+1)

    return xstar, vstar, ns, rs, ks

# Various starting conditions

fig, ax = plt.subplots()
for i in range(8):
    n = 20 # dimensions of x and c
    m = 10 # dimensions of A (mxn) and v

    rng = np.random.default_rng(i)
    A = rng.normal(size=(m,n))
    A[0,:] = np.abs(A[0,:])
    p = np.abs(rng.normal(size=(n,1)))
    b = A @ p
    c = rng.normal(size=(n,1))

    x0 = np.abs(rng.normal(size=(n,1)))
    # print(b)
    # print(x0)

    [xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0)
    ax.semilogy(ks, rs)
ax.set_xlabel('k')
ax.set_ylabel('2-Norm of Residual')
ax.set_title('Convergence for Various Starting Conditions (n = 20, m = 10)')

# Fixed Problem with Fixed Alpha and Varying Beta
n = 20 # dimensions of x and c
m = 10 # dimensions of A (mxn) and v

rng = np.random.default_rng(10)
A = rng.normal(size=(m,n))
p = np.abs(rng.normal(size=(n,1)))
b = A @ p
c = rng.normal(size=(n,1))
x0 = np.abs(rng.normal(size=(n,1)))

betas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
fig, ax = plt.subplots()
for i in range(betas.size):
    [xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0, beta=betas[i])
    ax.semilogy(ks, rs, label=r'$\beta = $' + str(betas[i]))
ax.legend()

ax.set_xlabel('k')
ax.set_ylabel('2-Norm of Residual')
ax.set_title(r'Convergence for Fixed $\alpha$ Values and Varying $\beta$ (n = 20, m = 10)')

#Fixed Problem with Fixed Beta and Verying Alpha
alphas = np.array([0.1, 0.2, 0.3, 0.4])
fig, ax = plt.subplots()
for i in range(alphas.size):
    [xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0, alpha=alphas[i])
    ax.semilogy(ks, rs, label=r'$\alpha = $' + str(alphas[i]))
ax.legend()

ax.set_xlabel('k')
ax.set_ylabel('2-Norm of Residual')
ax.set_title(r'Convergence for Fixed $\beta$ Values and Varying $\alpha$ (n = 20, m = 10)')

# Test Unbounded Below
m = 3
n = 4
A = np.hstack([np.eye(m), np.zeros((m, 1))])
b = np.ones((m,1))
c = -np.ones((n,1))
x0 = np.abs(rng.normal(size=(n,1)))

[xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0)
fig, ax = plt.subplots()
ax.semilogy(ks,rs)

ax.set_xlabel('k')
ax.set_ylabel('2-Norm of Residual')
ax.set_title('Convergence for Unbounded Below Case (n = 4, m = 3)')

# Test Infeasible
b = -np.ones((m,1))

[xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0)
fig, ax = plt.subplots()
ax.semilogy(ks,rs)

ax.set_xlabel('k')
ax.set_ylabel('2-Norm of Residual')
ax.set_title('Convergence for Infeasible Case (n = 4, m = 3)')

# Large m and n values
n = 200 # dimensions of x and c
m = 150 # dimensions of A (mxn) and v

rng = np.random.default_rng(10)
A = rng.normal(size=(m,n))
p = np.abs(rng.normal(size=(n,1)))
b = A @ p
c = rng.normal(size=(n,1))

x0 = np.abs(rng.normal(size=(n,1)))
[xstar, vstar, ns, rs, ks] = isNewton(A,b,c,x0)
fig, ax = plt.subplots()
ax.semilogy(ks,rs)

ax.set_xlabel('k')
ax.set_ylabel('2-Norm of Residual')
ax.set_title('Convergence for High Dimensional Problem (n = 200, m = 150)')
