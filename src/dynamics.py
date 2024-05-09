import numpy as np

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