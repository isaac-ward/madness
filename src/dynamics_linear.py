import jax.numpy as jnp
from dynamics import DynamicsQuadcopter3D

class DynamicsQuadcopter3DLinear(DynamicsQuadcopter3D):
    """
    This class modifies the DynamicsQuadcopter3D class with a simpler state_delta function
    """    

    def action_ranges(self):
        # If you're finding that state space isn't adequately explored,
        # consider increasing the size of the action space
        magnitude_lo = 5
        magnitude_hi = 5
        return jnp.array([
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
        ])
    
    def state_delta(self, state, action):
        """
        Function to calculate the continuous nonlinear state derivative given a particular
        state and action. Usable with jax
        
        Parameters
        ----------
        state: numpy.ndarray
            State vector shaped (12,)
        action: numpy.ndarray
            Control vector shaped (4,)
        
        Returns
        -------
        state_delta: jax.numpy.ndarray
            Continuous state derivative vector at given state and action. Shaped (12,)
        """

        # Assert that the shapes of the states and actions are correct
        assert state.shape == (12,), f"State shape is {state.shape}, should be (12,)"
        assert action.shape == (4,), f"Action shape is {action.shape}, should be (4,)"
        
        # the first three actions control the position
        w1, w2, w3     = action[0], action[1], action[2]
        
        # Compute the change in state, noting that we might use this function
        # with jax's vmap
        state_delta = jnp.zeros_like(state)

        # Positions change according to velocity
        state_delta = state_delta.at[0].set(w1) 
        state_delta = state_delta.at[1].set(w2)
        state_delta = state_delta.at[2].set(w3)

        return state_delta
