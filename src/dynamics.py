import numpy as np
from scipy.interpolate import BarycentricInterpolator
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
    
    def dynamics_test(self):
        """
        Function to perform simple diagnostic tests on the dynamics
        Check if trajectory extrapolation works
        Check if dynamics propigation works
        """
        # Generate a sample trajectory
        T = 100
        xtrue = np.linspace(0,np.pi,T)
        ytrue = np.zeros((T,1))
        for i in range(T):
            ytrue[i] = np.sin(xtrue[i])

        # Extrapolate state and control data from trajectory
        xnom = np.zeros((T,1))
        ynom = np.zeros((T,1))
        state,control = self.nominal_trajectory(xtrue,ytrue)
        for i in range(T):
            xnom[i] = state[i,0]
            ynom[i] = state[i,2]

        # Test controls on true dynamic model
        print(control)
        print(state)
        xcont = np.zeros((T,1))
        ycont = np.zeros((T,1))
        x_next = state[0]
        xcont[0] = state[0,0]
        ycont[0] = state[0,2]
        for i in range(1,T-1):
            x_next = self.dynamics_true(x_next, control[i])
            xcont[i] = x_next[0]
            ycont[i] = x_next[2]

        fig, ax = plt.subplots()
        ax.plot(xtrue,ytrue,label='True Path')
        ax.plot(xnom,ynom,label='Nominal Path')
        ax.plot(xcont,ycont,label='Controlled Path')
        ax.grid()
        ax.legend()
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_title("Dynamics Verification")
        plt.show()
    
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

    def dynamics_true(self, xk, uk):
        """
        Compute the true next state with nonlinear dynamics
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
        x_next[1] = vx + self.dt * ((-(T1+T2)*np.sin(phi) - self.CD_v*vx)/self.m + self.wx)
        x_next[2] = y + self.dt*vy
        x_next[3] = vy + self.dt*(((T1+T2)*np.cos(phi) - self.CD_v*vy)/self.m - self.g + self.wy)
        x_next[4] = phi + self.dt*om
        x_next[5] = om + self.dt*((T2-T1)*self.l - self.CD_phi*om)/self.Iyy

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
        x_next = self.dynamics_true(x_bar, u_bar) + A@(x-x_bar) + B@(u-u_bar)

        return x_next
    
    def nominal_trajectory(self,x,y):
        """
        Compute the nominal trajectory from the planned path
        taking advantage of the differential flatness of the
        model
        """
        # Create nominal state and control vectors
        T = len(x)
        state = np.zeros((T,6))
        control = np.zeros((T-1,2))
        X = BarycentricInterpolator(np.array(np.linspace(0,T,T)),x)
        Y = BarycentricInterpolator(np.array(np.linspace(0,T,T)),y)

        for k in range(0,T-1):
            # TODO: Fix this so it's interpolating between more than just 2 points

            #X = BarycentricInterpolator(np.array([self.dt*(k),self.dt*(k+1)]),x[k:k+2])
            xd = X.derivative(x[k],der=1)
            xdd = X.derivative(x[k],der=2)
            xddd = X.derivative(x[k],der=3)
            xdddd = X.derivative(x[k],der=4)

            #Y = BarycentricInterpolator(np.array([self.dt*(k),self.dt*(k+1)]),y[k:k+2])
            yd = Y.derivative(y[k],der=1)
            ydd = Y.derivative(y[k],der=2)
            yddd = Y.derivative(y[k],der=3)
            ydddd = Y.derivative(y[k],der=4)

            phi = np.arctan(-xdd/(ydd+self.g))
            sin_phi = (-xdd/(ydd+self.g))/np.sqrt(1+(-xdd/(ydd+self.g))**2)

            omega = (yddd*xdd-(ydd+self.g)*xddd)/((ydd+self.g)**2+xdd**2)
            omega_dot = (ydddd*xdd-xdddd*(ydd+self.g))/((ydd+self.g)**2+xdd**2) - (2*xdd*yddd**2*(ydd+self.g)+2*xdd**2*xddd*yddd-2*xddd*yddd*(ydd+self.g)**2-2*xdd*xddd**2*(ydd+self.g))/(xdd**4+2*xdd**2*(ydd+self.g)**2+(ydd+self.g)**4)
            
            state[k,0] = x[k]
            state[k,1] = xd
            state[k,2] = y[k]
            state[k,3] = yd
            state[k,4] = phi
            state[k,5] = omega
            
            control[k,0] = -(self.CD_v*self.l*xd+self.l*self.m*xdd+self.CD_phi*omega*sin_phi+self.Iyy*omega_dot*sin_phi)/(2*self.l*sin_phi)
            control[k,1] = (-self.l*self.m*xdd-self.CD_v*self.l*xd+self.CD_phi*omega*sin_phi+self.Iyy*omega_dot*sin_phi)/(2*self.l*sin_phi)
            
            # Control within bounds
            if control[k,0] > self.max_thrust_per_prop:
                control[k,0] = self.max_thrust_per_prop
            elif control[k,0] < self.min_thrust_per_prop:
                control[k,0] = self.min_thrust_per_prop
            
            if control[k,1] > self.max_thrust_per_prop:
                control[k,1] = self.max_thrust_per_prop
            elif control[k,1] < self.min_thrust_per_prop:
                control[k,1] = self.min_thrust_per_prop
            
        state[T-1,0] = x[T-1]
        state[T-1,1] = 0
        state[T-1,2] = y[T-1]
        state[T-1,3] = 0
        state[T-1,4] = 0
        state[T-1,5] = 0
        
        return state,control
