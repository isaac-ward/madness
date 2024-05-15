import numpy as np
from tqdm import tqdm

import mapping 
import planning
import control 
import dynamics
import visuals
import utils 
import globals

def policy_none(xk):
    """
    A policy that does nothing
    """
    return np.zeros((2,))

def policy_hover_left(xk):
    """
    Hover more on the left than right
    """
    return 1.5 * 9.81 * np.array([1, 0.999])

def policy_hover_right(xk):
    """
    Hover more on the right than left
    """
    return 1.5 * 9.81 * np.array([0.999, 1])

def policy_hover(xk):
    """
    A policy that applies a constant force to hover on both
    """
    return 1.414 * 9.81 * np.array([1, 1])

if __name__ == "__main__":

    # Will log everything to here
    log_folder = utils.make_log_folder(name="test")

    # Create a 2d dynamics object
    dt = 0.05
    D = dynamics.Quadrotor2D(dt=dt)

    # Define an initial state  
    n = int(D.n)
    x0 = np.zeros((n,1))

    # We want to test the true an linearized dynamics rollouts
    # for some basic control policies. Recall that the action
    # space is 2D, with T1 and T2 as the two actions corresponding
    # to the two rotors
    policy_fns = [ 
        policy_none, 
        policy_hover_left, 
        policy_hover_right, 
        policy_hover 
    ]
    dynamics_fns = [ D.dynamics_true ]#, D.dynamics_model ]

    # Perform the simulation for each policy with true and approx dynamics
    T = 8 # seconds
    N = int(T / dt) # steps
    results = {}
    for f in dynamics_fns:
        results[f.__name__] = {}
        for policy in policy_fns:
            state_trajectory = [x0]
            action_trajectory = []
            for t in tqdm(range(N), desc=f"Simulating {f.__name__} with {policy.__name__}"):
                # Get the latest state
                x_curr = state_trajectory[-1]
                # Get the control action
                u = policy(x_curr)
                # Compute the next state with true dynamics
                x_next = f(x_curr, u)
                # Store the state and action
                state_trajectory.append(x_next)
                action_trajectory.append(u)

            # States are 6x1, want them to just be vectors
            state_trajectory = np.array(state_trajectory).squeeze()

            # Store the results
            results[f.__name__][policy.__name__] = {
                "state_trajectory": state_trajectory,
                "action_trajectory": action_trajectory
            }

    
    # Now plot all the results. Recall that state is
    # [x, vx, y, vy, phi (angle), om (angle rate of change)]
    # Plot all of them as time series
    state_element_labels = ["x", "vx", "y", "vy", "phi", "om"]
    action_labels = ['left rotor', 'right rotor']
    for f in dynamics_fns:
        for policy in policy_fns:
            run = results[f.__name__][policy.__name__]
            visuals.plot_trajectory(
                filepath=f"{log_folder}/{f.__name__}_{policy.__name__}.mp4",
                state_trajectory=run["state_trajectory"],
                state_element_labels=state_element_labels,
                action_trajectory=run["action_trajectory"],
                action_element_labels=action_labels,
                dt=dt
            )
    


    