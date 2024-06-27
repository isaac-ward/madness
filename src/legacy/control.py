import numpy as np
import dynamics
from scipy.linalg import cho_factor, cho_solve

def optimal_control(path):

    # TODO: get nominal controls and trajectories through state space

    # TODO: do (i)LQR to produce control sequence with dynamics model

    # TODO: execute control (on actual simuation) with true dynamics

    pass

def ilqr(x_track, u_track, N, quadrotor: dynamics.Quadrotor2D, Q, R, QN, eps=1e-3, max_iters=1000):
    """
    Compute controls to track trajectory with iLQR
    Based on code from AA203 HW2
    TODO Need to adjust ilqr algorithm to match our problem, determine cost function for tracking trajectory
    """
    # Check valid setup
    if max_iters <= 1:
        raise ValueError("Argument `max_iters` must be at least 1.")

    # Get variables
    n = quadrotor.n_dim  # state dimension
    m = quadrotor.m_dim  # control dimension

    # Initialize control gains Y and offsets y
    Y = np.zeros((N, m, n))
    y = np.zeros((N, m))

    # Initialize the nominal trajectory and deviations `(dx, du)`
    x_bar = np.zeros(np.shape(x_track))
    u_bar = np.copy(u_track)
    x_bar[0] = np.copy(x_track[0])
    dt = np.zeros(u_track.shape[0])
    for k in range(N):
        dt[k] = np.linalg.norm(x_track[k+1,[0,2]]-x_track[k,[0,2]])/np.linalg.norm(x_track[k,[1,3]])
        x_bar[k + 1] = quadrotor.dynamics_true_no_disturbances(x_bar[k], u_bar[k], dt=dt[k])
    dx = np.zeros((N + 1, n))
    du = np.zeros((N, m))

    # iLQR loop
    converged = False
    for _i in range(max_iters):

        # Backwards Pass
        qN = QN@(x_bar[N]-x_track[-1])
        V = np.copy(QN)
        vbar = np.copy(qN)

        for k in range(N-1,-1,-1):
            # Get Ak and Bk
            Ak,Bk = quadrotor.linearize(x_bar[k], u_bar[k], dt[k])

            # Define cost functions
            qk = Q@(x_bar[k]-x_track[k])
            rk = R@u_bar[k]
            
            # Define S
            reg = 1e-9
            Su = rk + vbar.T@Bk
            Suu = R + Bk.T@V@Bk + reg * np.eye(m)
            Sux = Bk.T@V@Ak

            # Define Y, y
            Y[k] = -np.linalg.pinv(Suu)@Sux
            y[k] = -np.linalg.pinv(Suu)@Su

            # Update V, vbar
            V = Q + Ak.T@V@Ak - Y[k].T@Suu@Y[k]
            vbar = qk + Ak.T@vbar + Sux.T@y[k]

        # Forwards Pass
        u = np.zeros((N, m))
        x = np.zeros((N + 1, n))
        x[0] = np.copy(x_track[0])
        for k in range(N):
            dx[k] = x[k] - x_bar[k]
            du[k] = y[k] + Y[k]@dx[k]
            u[k] = u_bar[k] + du[k]
            dt[k] = np.linalg.norm(x_bar[k+1,[0,2]]-x_bar[k,[0,2]])/np.linalg.norm(x_bar[k,[1,3]])
            x[k + 1] = quadrotor.dynamics_true_no_disturbances(x[k], u[k], dt=dt[k])
        x_bar = np.copy(x)
        u_bar = np.copy(u)

        print("iLQR iteration: " + str(_i) + "\ndu: " + str(np.max(np.abs(du))) + "\n")

        if np.max(np.abs(du)) < eps:
            converged = True
            break

    # Verify solution found
    if not converged:
        raise RuntimeError("iLQR did not converge!")
    
    return x_bar, u_bar, Y, y
