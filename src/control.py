import numpy as np
import dynamics as dy

def optimal_control(path):

    # TODO: get nominal controls and trajectories through state space

    # TODO: do (i)LQR to produce control sequence with dynamics model

    # TODO: execute control (on actual simuation) with true dynamics

    pass

def ilqr(x0, x_goal, N, dt, Q, R, QN, eps=1e-3, max_iters=1000):
    """
    Compute controls to track trajectory with iLQR
    Based on code from AA203 HW2
    TODO Need to adjust ilqr algorithm to match our problem, determine cost function for tracking trajectory
    """
    # Check valid setup
    if max_iters <= 1:
        raise ValueError("Argument `max_iters` must be at least 1.")
    
    # Create 2D dynamics model
    quadrotor = dy.Quadrotor2D(dt)

    # Get variables
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize control gains Y and offsets y
    Y = np.zeros((N, m, n))
    y = np.zeros((N, m))

    # Initialize the nominal trajectory `(x_bar, u_bar`), and the
    # deviations `(dx, du)`
    # TODO get this from path planning
    u_bar = np.zeros((N, m))
    x_bar = np.zeros((N + 1, n))
    x_bar[0] = x0
    for k in range(N):
        x_bar[k + 1] = quadrotor.dynamics_true(x_bar[k], u_bar[k])
    dx = np.zeros((N + 1, n))
    du = np.zeros((N, m))

    # iLQR loop
    converged = False
    for _ in range(max_iters):

        # Backwards Pass
        qN = QN@(x_bar[N]-x_goal)
        V = np.copy(QN)
        vbar = np.copy(qN)
        for k in range(N-1,-1,-1):

            # Get Ak and Bk
            Ak,Bk = quadrotor.linearize(x_bar, u_bar)

            # Define cost functions
            qk = Q@(x_bar[k]-x_goal)
            rk = R@u_bar[k]
            
            # Define S
            Su = rk + vbar.T@Bk
            Suu = R + Bk.T@V@Bk
            Sux = Bk.T@V@Ak

            # Define Y, y
            Y[k] = -np.linalg.inv(Suu)@Sux
            y[k] = -np.linalg.inv(Suu)@Su

            # Update V, vbar
            V = Q + Ak.T@V@Ak - Y[k].T@Suu@Y[k]
            vbar = qk + Ak.T@vbar + Sux.T@y[k]

        # Forwards Pass
        u = np.zeros((N, m))
        x = np.zeros((N + 1, n))
        x[0] = x0
        for k in range(N):
            dx[k] = x[k] - x_bar[k]
            du[k] = y[k] + Y[k]@dx[k]
            u[k] = u_bar[k] + du[k]
            x[k + 1] = quadrotor.dynamics_true(x[k], u[k])
        x_bar = np.copy(x)
        u_bar = np.copy(u)

        if np.max(np.abs(du)) < eps:
            converged = True
            break

    # Verify solution found
    if not converged:
        raise RuntimeError("iLQR did not converge!")
    
    return x_bar, u_bar, Y, y
