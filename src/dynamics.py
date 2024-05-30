import numpy as np
import scipy.interpolate
import visuals
import utils
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
import globals

class Quadrotor2D:

    def __init__(self,dt):
        # Dynamics constants (sourced from AA274A)
        self.n_dim = 6 # state dimension
        self.m_dim = 2 # control dimension
        self.g = globals.g # gravity (m/s**2)
        self.m = globals.DRONE_MASS # mass (kg)
        self.l = globals.DRONE_HALF_LENGTH # half-length (m)
        self.Iyy = 1.0 # moment of inertia about the out-of-plane axis (kg * m**2)
        self.CD_v = 0#0.25 # translational drag coefficient
        self.CD_phi = 0#0.02255 # rotational drag coefficient
        self.dt = dt # time interval between steps

        # Control constraints (sourced from AA274A)
        self.max_thrust_per_prop = globals.MAX_THRUST_PER_PROP  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0

        # Wind variables
        self.wx = 0 # wind velocity in x-dir #TODO Implement Dryden wind model
        self.wy = 0 # wind velocity in y-dir
    
    def dynamics_test(self,log_folder,xtrue,ytrue,obstacles,v_desired,spline_alpha):
        """
        Function to perform simple diagnostic tests on the dynamics
        Check if trajectory extrapolation works
        Check if dynamics propigation works
        """
        # Get dimensions
        T = np.shape(xtrue)[0]

        # Extrapolate state and control data from trajectory
        """state,control = self.differential_flatness_trajectory(xtrue,ytrue,v_desired,spline_alpha)
        T = np.shape(state)[0]
        xnom = np.zeros((T,1))
        ynom = np.zeros((T,1))
        for i in range(T):
            xnom[i] = state[i,0]
            ynom[i] = state[i,2]"""

        # Extrapolate state and control data from trajectory
        astar_path = np.column_stack((xtrue,ytrue))
        state,control = self.SCP_nominal_trajectory(astar_path=astar_path,obstacles=obstacles,R=np.eye(self.m_dim),Q=np.eye(self.n_dim),QN=10*np.eye(self.n_dim),v_desired=v_desired,spline_alpha=spline_alpha)
        T = np.shape(state)[0]
        xnom = np.zeros((T,1))
        ynom = np.zeros((T,1))
        for i in range(T):
            xnom[i] = state[i,0]
            ynom[i] = state[i,2]

        # Test controls on true dynamic model
        truestate = np.array([state[0]])
        xcont = np.zeros((T,1))
        ycont = np.zeros((T,1))
        x_next = state[0]
        xcont[0] = state[0,0]
        ycont[0] = state[0,2]
        for i in range(1,T):
            x_next = self.dynamics_true_no_disturbances(x_next, control[i-1])
            xcont[i] = x_next[0]
            ycont[i] = x_next[2]
            truestate = np.append(truestate,x_next)
        
        # fig, (ax) = plt.subplots()
        # ax.plot(xtrue,ytrue,label='true')
        # ax.plot(xcont,ycont,label='fit')
        # ax.grid()
        # ax.legend()
        # plt.show()
        truestate = np.reshape(truestate,(T,6))
        return truestate,control
    
    def wind_model(self,x):
        """
        model wind disturbances using the Dryden model for low altitude with gusts
        sourced from "Small Unmanned Aircraft" by Randal W. Beard and Timothy W. McLain
        """
        alt = 50 # (m)
        Lu = 200 # (m)
        Lv = Lu
        Lw = 50 # (m)
        sigu = 2.12 # (m/s)
        sigv = sigu
        sigw = 1.4 # (m/s)

    def dynamics_true_no_disturbances(self, xk, uk):
        """
        Compute the true next state with nonlinear dynamics
        TODO: Add wind and drag into model. Will likely also entail rederiving differential flatness terms
        """
        # Breakup state x(k) and control vector u(k)
        x = xk[0]
        vx = xk[1]
        y = xk[2]
        vy = xk[3]
        phi = xk[4]
        om = xk[5]
        T1 = uk[0]
        T2 = uk[1]

        # Compute x(k+1)
        x_next = np.zeros((6,1))
        x_next[0] = x + self.dt*vx
        x_next[1] = vx + self.dt*((-(T1+T2)*np.sin(phi))/self.m) # - self.CD_v*vx + self.wx
        x_next[2] = y + self.dt*vy
        x_next[3] = vy + self.dt*(((T1+T2)*np.cos(phi))/self.m - self.g) # - self.CD_v*vy + self.wy
        x_next[4] = phi + self.dt*om
        x_next[5] = om + self.dt*((T2-T1)*self.l)/self.Iyy # - self.CD_phi*om

        return x_next
    
    def linearize(self, x_bar, u_bar):
        """
        Linearize dynamics about nominal state and control vectors
        Sourced from AA274A
        TODO Optional rewrite with jax for more efficiency
        """
        # Breakup state x(k) and control vector u(k)
        x = x_bar[0]
        vx = x_bar[1]
        y = x_bar[2]
        vy = x_bar[3]
        phi = x_bar[4]
        om = x_bar[5]
        T1 = u_bar[0]
        T2 = u_bar[1]
        
        # Compute A and B
        A = np.array([[1., self.dt, 0., 0., 0., 0.],
                      [0., 1.-self.dt*self.CD_v/self.m, 0., 0., -self.dt*(T1+T2)*np.cos(phi)/self.m, 0.],
                      [0., 0., 1., self.dt, 0., 0.],
                      [0., 0., 0., 1.-self.dt*self.CD_v/self.m, -self.dt*(T1+T2)*np.sin(phi)/self.m, 0.],
                      [0., 0., 0., 0., 1., self.dt],
                      [0., 0., 0., 0., 0., 1.-self.dt*self.CD_phi/self.Iyy]])
        
        B = np.array([[0., 0.],
                      [-self.dt*np.sin(phi)/self.m, -self.dt*np.sin(phi)/self.m],
                      [0., 0.],
                      [self.dt*np.cos(phi)/self.m, self.dt*np.cos(phi)/self.m],
                      [0., 0.],
                      [-self.dt*self.l/self.Iyy, self.dt*self.l/self.Iyy]])
        
        return A,B
    
    def affinize(self, x_bar, u_bar):
        """
        Affinize dynamics about nominal state and control vectors
        Sourced from AA274A
        TODO Optional rewrite with jax for more efficiency
        """
        # Breakup state x(k) and control vector u(k)
        x = x_bar[0]
        vx = x_bar[1]
        y = x_bar[2]
        vy = x_bar[3]
        phi = x_bar[4]
        om = x_bar[5]
        T1 = u_bar[0]
        T2 = u_bar[1]
        
        # Compute A and B
        A = np.array([[1., self.dt, 0., 0., 0., 0.],
                      [0., 1.-self.dt*self.CD_v/self.m, 0., 0., -self.dt*(T1+T2)*np.cos(phi)/self.m, 0.],
                      [0., 0., 1., self.dt, 0., 0.],
                      [0., 0., 0., 1.-self.dt*self.CD_v/self.m, -self.dt*(T1+T2)*np.sin(phi)/self.m, 0.],
                      [0., 0., 0., 0., 1., self.dt],
                      [0., 0., 0., 0., 0., 1.-self.dt*self.CD_phi/self.Iyy]])
        
        B = np.array([[0., 0.],
                      [-self.dt*np.sin(phi)/self.m, -self.dt*np.sin(phi)/self.m],
                      [0., 0.],
                      [self.dt*np.cos(phi)/self.m, self.dt*np.cos(phi)/self.m],
                      [0., 0.],
                      [-self.dt*self.l/self.Iyy, self.dt*self.l/self.Iyy]])
        
        C = np.reshape(self.dynamics_true_no_disturbances(x_bar, u_bar),(6,)) - A@x_bar - B@u_bar
        
        return A,B,C

    def dynamics_model(self, x, u, x_bar, u_bar):
        """
        Compute next state given current state and action
        based on our model of the world (no disturbances)
        """
        # Get linearized jacobians
        A,B = self.linearize(x_bar, u_bar)

        # Compute x(k+1)
        x_next = self.dynamics_true_no_disturbances(x_bar, u_bar) + A@(x-x_bar) + B@(u-u_bar)

        return x_next
    
    @staticmethod
    def spline_trajectory_smoothing(path,v_desired=0.15,spline_alpha=0.05):
        """
        Use a 5th order spline to smooth the desired trajectory
        Sourced from AA274A
        """
        ts = np.array([0])
        path_x = np.array([])
        path_y = np.array([])
        
        # Separate path into x and y components
        for i in range(0, len(path)):
            path_x = np.append(path_x,path[i][0])
            path_y = np.append(path_y,path[i][1])
        
        # Calculate cumulative time for each waypoint
        for i in range(0,len(path)-1):
            ts = np.append(ts,(np.linalg.norm(path[i+1]-path[i])/v_desired)+ts[-1])
        
        # Fit 5th degree polynomial splines for x and y
        path_x_spline = scipy.interpolate.splrep(ts, path_x, k=5, s=spline_alpha)
        path_y_spline = scipy.interpolate.splrep(ts, path_y, k=5, s=spline_alpha)
        #path_x_spline = scipy.interpolate.CubicSpline(ts, path_x, bc_type='clamped')
        #path_y_spline = scipy.interpolate.CubicSpline(ts, path_y, bc_type='clamped')

        return path_x_spline, path_y_spline, ts[-1]
    
    def bezier_trajectory_fitting(self,astar_path,boxes):
        """
        Fit a smooth trajectory to the A* path using 5th order bezier functions 
        constrained by bounding boxes
        """
        # Beta function
        def beta(t):
            return np.array([(t - 1)**8, -8*t*(t - 1)**7, 28*t**2*(t - 1)**6, -56*t**3*(t - 1)**5, 70*t**4*(t - 1)**4, -56*t**5*(t - 1)**3, 28*t**6*(t - 1)**2, -t**7*(8*t - 8), t**8])
        def beta_2d(t):
            return np.array([56*(t - 1)**6, -112*(4*t - 1)*(t - 1)**5, 56*(t - 1)**4*(28*t**2 - 14*t + 1), -112*t*(t - 1)**3*(28*t**2 - 21*t + 3), 280*t**2*(t - 1)**2*(14*t**2 - 14*t + 3), -112*t**3*(28*t**3 - 63*t**2 + 45*t - 10), 56*t**4*(28*t**2 - 42*t + 15), -112*t**5*(4*t - 3), 56*t**6])
        def beta_3d(t):
            return np.array([336*(t - 1)**5, -336*(8*t - 3)*(t - 1)**4, 336*(t - 1)**3*(28*t**2 - 21*t + 3), -336*(t - 1)**2*(56*t**3 - 63*t**2 + 18*t - 1), 1680*t*(14*t**4 - 35*t**3 + 30*t**2 - 10*t + 1), -336*t**2*(56*t**3 - 105*t**2 + 60*t - 10), 336*t**3*(28*t**2 - 35*t + 10), -336*t**4*(8*t - 5), 336*t**5])
        def beta_4d(t):
            return np.array([1680*(t - 1)**4, -6720*(2*t - 1)*(t - 1)**3, 3360*(t - 1)**2*(14*t**2 - 14*t + 3), - 94080*t**4 + 235200*t**3 - 201600*t**2 + 67200*t - 6720, 117600*t**4 - 235200*t**3 + 151200*t**2 - 33600*t + 1680, -6720*t*(14*t**3 - 21*t**2 + 9*t - 1), 3360*t**2*(14*t**2 - 14*t + 3), -6720*t**3*(2*t - 1), 1680*t**4])
        

        # Constants
        L = np.shape(boxes)[0]
        N = 8

        # Solve convex fitting problem
        # Declare Convex Variables
        s = cp.Variable((L*(N+1),2))

        # Define Objective
        objective = 0
        n = 100  # discretization points for the integral
        t = np.linspace(0,1,n)
        dt = (1-0)/(n-1)
        for i in range(L):
            # Integral term using the trapezoidal rule
            for j in range(n-1):
                objective += dt/2*(cp.norm(beta_4d(t[j])*s[i*(N+1):i*(N+1)+N+1],2)**2 + cp.norm(beta_4d(t[j+1])*s[i*(N+1):i*(N+1)+N+1],2)**2)
            objective += cp.sum([cp.norm(s[i*(N+1)+k,:]-s[i*(N+1)+k+1,:],2)**2 for k in range(N)])


        # Define Constraints
        constraints = [beta(0)@s[0:N+1] == astar_path[0],
                       beta(1)@s[(L-1)*(N+1):(L-1)*(N+1)+N+1] == astar_path[-1]]

        for i in range(L):
            for k in range(N+1):
                constraints += [s[i*(N+1)+k,1] <= boxes[i,0], # up
                                s[i*(N+1)+k,0] <= boxes[i,1], # right
                                s[i*(N+1)+k,1] >= boxes[i,2], # down
                                s[i*(N+1)+k,0] >= boxes[i,3]] # left
        for i in range(L-1):
            constraints += [beta(1)@s[i*(N+1):i*(N+1)+N+1] == beta(0)@s[(i+1)*(N+1):(i+1)*(N+1)+N+1],
                            beta_2d(1)@s[i*(N+1):i*(N+1)+N+1] == beta_2d(0)@s[(i+1)*(N+1):(i+1)*(N+1)+N+1],
                            beta_3d(1)@s[i*(N+1):i*(N+1)+N+1] == beta_3d(0)@s[(i+1)*(N+1):(i+1)*(N+1)+N+1],
                            beta_4d(1)@s[i*(N+1):i*(N+1)+N+1] == beta_4d(0)@s[(i+1)*(N+1):(i+1)*(N+1)+N+1]]

        # Problem
        print(objective.curvature)
        problem = cp.Problem(cp.Minimize(objective),constraints)

        # Solve
        problem.solve()
        s_val = s.value

        # Calc trajectories
        n = 10
        t = np.linspace(0,1,n)
        path = np.zeros(2)
        for i in range(L):
            for _t in t:
                path = np.vstack([path,beta(_t)@s_val[i*(N+1):i*(N+1)+N+1]])

        return path[1:]

    
    def differential_flatness_trajectory(self,x,y,v_desired=0.15,spline_alpha=0.05):
        """
        Compute the nominal trajectory from the planned path
        taking advantage of the differential flatness of the
        model
        TODO: Discuss initial state not being static and not at initial location??
        """
        # Smooth given trajectory and gather derivatives
        path = np.column_stack((x,y))
        x_spline,y_spline,duration = self.spline_trajectory_smoothing(path,v_desired,spline_alpha)
        ts = np.arange(0.,duration,self.dt)

        x_smooth = scipy.interpolate.splev(ts,x_spline,der=0)
        xd_smooth = scipy.interpolate.splev(ts,x_spline,der=1)
        xdd_smooth = scipy.interpolate.splev(ts,x_spline,der=2)
        xddd_smooth = scipy.interpolate.splev(ts,x_spline,der=3)
        xdddd_smooth = scipy.interpolate.splev(ts,x_spline,der=4)#np.zeros(np.shape(x_smooth))

        y_smooth = scipy.interpolate.splev(ts,y_spline,der=0)
        yd_smooth = scipy.interpolate.splev(ts,y_spline,der=1)
        ydd_smooth = scipy.interpolate.splev(ts,y_spline,der=2)
        yddd_smooth = scipy.interpolate.splev(ts,y_spline,der=3)
        ydddd_smooth = scipy.interpolate.splev(ts,y_spline,der=4)#np.zeros(np.shape(x_smooth))

        # Create nominal state and control vectors
        T = len(ts)
        state = np.zeros((T,6))
        control = np.zeros((T-1,2))

        # Calc state and control vectors given differential flatness
        for k in range(0,T-1):
            xd = xd_smooth[k]
            xdd = xdd_smooth[k]
            xddd = xddd_smooth[k]
            xdddd = xdddd_smooth[k]

            yd = yd_smooth[k]
            ydd = ydd_smooth[k]
            yddd = yddd_smooth[k]
            ydddd = ydddd_smooth[k]

            phi = np.arctan(-xdd/(ydd+self.g))

            omega = (yddd*xdd-(ydd+self.g)*xddd)/((ydd+self.g)**2+xdd**2)
            
            state[k,0] = x_smooth[k]
            state[k,1] = xd
            state[k,2] = y_smooth[k]
            state[k,3] = yd
            state[k,4] = phi
            state[k,5] = omega
            
            control[k,0] = -0.5*(self.Iyy/self.l*((xdd*ydddd-xdddd*(ydd+self.g))*((ydd+self.g)**2+xdd**2)+2*(xddd*(ydd+self.g)-xdd*yddd)*(yddd*(ydd+self.g)+xdd*xddd))/((ydd+self.g)**2+xdd**2)**2+self.m*xdd/np.sin(phi))
            control[k,1] = 0.5*(self.Iyy/self.l*((xdd*ydddd-xdddd*(ydd+self.g))*((ydd+self.g)**2+xdd**2)+2*(xddd*(ydd+self.g)-xdd*yddd)*(yddd*(ydd+self.g)+xdd*xddd))/((ydd+self.g)**2+xdd**2)**2-self.m*xdd/np.sin(phi))

            # Check control within bounds
            if control[k,0] > self.max_thrust_per_prop:
                control[k,0] = self.max_thrust_per_prop
            elif control[k,0] < self.min_thrust_per_prop:
                control[k,0] = self.min_thrust_per_prop
            
            if control[k,1] > self.max_thrust_per_prop:
                control[k,1] = self.max_thrust_per_prop
            elif control[k,1] < self.min_thrust_per_prop:
                control[k,1] = self.min_thrust_per_prop

        # Set final state
        state[T-1,0] = x_smooth[-1]
        state[T-1,1] = 0
        state[T-1,2] = y_smooth[-1]
        state[T-1,3] = 0
        state[T-1,4] = 0
        state[T-1,5] = 0
            
        return state,control
    
    def SCP_nominal_trajectory(self,astar_path,obstacles,R,Q,QN,max_iters=100,eps=5e-1,rho=1.0,v_desired=0.15,spline_alpha=0.05):
        """
        Use SCP techniques to better optimize the trajectory path
        Sourced from AA203
        """
        # Break down path into x and y coords
        path_x = np.array([])
        path_y = np.array([])
        for i in range(0, len(astar_path)):
            path_x = np.append(path_x,astar_path[i][0])
            path_y = np.append(path_y,astar_path[i][1])
        
        # Get initial and final points
        start_point = np.array([path_x[0],0,path_y[0],0,0,0])
        end_point = np.array([path_x[-1],0,path_y[-1],0,0,0])

        # Get a initial trajectory using differential flatness
        state,control = self.differential_flatness_trajectory(path_x,path_y,v_desired=0.15,spline_alpha=0.05)

        # Run nominal trajectory through SCP
        N = np.shape(state)[0]
        converged = False
        J = np.zeros(max_iters + 1)
        J[0] = np.inf

        for i in (prog_bar := tqdm(range(max_iters))):
            # Convex optimization
            # Declare Convex Variables
            x_cvx = cp.Variable((N, self.n_dim))
            u_cvx = cp.Variable((N-1, self.m_dim))

            # Define Objective
            objective = cp.quad_form(x_cvx[N-1]-state[N-1],QN) + cp.sum([cp.quad_form(x_cvx[k]-state[k],Q)+cp.quad_form(u_cvx[k,:],R) for k in range(0,N-1)])

            # Define Constraints
            constraints = [x_cvx[0] == start_point, x_cvx[-1] == end_point]

            for k in range(N-1):
                A,B,C = self.affinize(state[k], control[k])
                constraints += [x_cvx[k+1] == A@x_cvx[k] + B@u_cvx[k] + C,
                        u_cvx[k,0] >= self.min_thrust_per_prop,
                        u_cvx[k,1] >= self.min_thrust_per_prop,
                        u_cvx[k,0] <= self.max_thrust_per_prop,
                        u_cvx[k,1] <= self.max_thrust_per_prop,
                        cp.norm(u_cvx[k]-control[k],np.inf) <= rho,
                        cp.norm(x_cvx[k]-state[k],np.inf) <= rho]
                for p in range(np.shape(obstacles)[0]):
                    constraints += [cp.norm(cp.hstack([x_cvx[k,0],x_cvx[k,2]])-obstacles[p],2) >= self.l*1.5]

            # Problem
            problem = cp.Problem(cp.Minimize(objective),constraints)

            # Solve
            problem.solve()
            J[i + 1] = problem.objective.value
            state = np.copy(x_cvx.value)
            control = np.copy(u_cvx.value)
            
            dJ = np.abs(J[i + 1] - J[i])
            prog_bar.set_postfix({"objective change": "{:.5f}".format(dJ)})
            if dJ < eps:
                converged = True
                print("SCP converged after {} iterations.".format(i))
                break
        if not converged:
            raise RuntimeError("SCP did not converge!")
        J = J[1:i+1]
        
        return state,control
