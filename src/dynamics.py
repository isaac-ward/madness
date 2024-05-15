import numpy as np
import scipy.interpolate
import visuals
import utils
import matplotlib.pyplot as plt

class Quadrotor2D:

    def __init__(self,dt):
        # Dynamics constants (sourced from AA274A)
        self.n = 6 # state dimension
        self.m = 2 # control dimension
        self.g = 9.81 # gravity (m/s**2)
        self.m = 2.5 # mass (kg)
        self.l = 1.0 # half-length (m)
        self.Iyy = 1.0 # moment of inertia about the out-of-plane axis (kg * m**2)
        self.CD_v = 0.25 # translational drag coefficient
        self.CD_phi = 0.02255 # rotational drag coefficient
        self.dt = dt # time interval between steps

        # Control constraints (sourced from AA274A)
        self.max_thrust_per_prop = 0.75 * self.m * self.g  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0

        # Wind variables
        self.wx = 0 # wind velocity in x-dir #TODO Implement Dryden wind model
        self.wy = 0 # wind velocity in y-dir
    
    def dynamics_test(self, log_folder, xtrue, ytrue):
        """
        Function to perform simple diagnostic tests on the dynamics
        Check if trajectory extrapolation works
        Check if dynamics propigation works
        """
        # Get dimensions
        T = np.shape(xtrue)[0]

        # Extrapolate state and control data from trajectory
        state,control = self.nominal_trajectory(xtrue,ytrue,v_desired=1,spline_alpha=0.00001)
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

        truestate = np.reshape(truestate,(T,6))

        visuals.plot_trajectory(
                filepath=f"{log_folder}/dynamicstest.mp4",
                state_trajectory=truestate,
                state_element_labels=[],
                action_trajectory=control,
                action_element_labels=[],
                dt=self.dt
            )
    
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
    
    def smooth_trajectory(self,path,v_desired=0.15,spline_alpha=0.05):
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

        return path_x_spline, path_y_spline, ts[-1]
    
    def nominal_trajectory(self,x,y,v_desired=0.15,spline_alpha=0.05):
        """
        Compute the nominal trajectory from the planned path
        taking advantage of the differential flatness of the
        model
        TODO: Discuss initial state not being static and not at initial location??
        """
        # Smooth given trajectory and gather derivatives
        path = np.column_stack((x,y))
        x_spline,y_spline,duration = self.smooth_trajectory(path,v_desired,spline_alpha)
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
