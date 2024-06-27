import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from sobol_seq import i4_sobol_generate

from path import Path

class MPPI:
    """
    Implementation of model predictive path integral control
    """

    def __init__(
        self, 
        dynamics_fn, 
        control_bounds_lower,
        control_bounds_upper,
        K,
        H,
        lambda_,
        nominal_xy_path,
        map,
    ):

        """
        dynamics_fn: function that takes in a state and action and returns the next state
        control_bounds_lower: lower bounds of the action space
        control_bounds_upper: upper bounds of the action space
        K: number of samples to take
        H: length of the horizon
        lambda_: temperature parameter:
            - if lambda_ << 0 then the smallest control sequence scores are emphasized in the weights 
            - if lambda_ = 0 then the weights are uniform (all equal)
            - if lambda_ >> 0 then the largest control sequence scores are emphasized in the weights
        """
        self.dynamics_fn = dynamics_fn
        self.control_bounds_lower = control_bounds_lower
        self.control_bounds_upper = control_bounds_upper
        self.K = K
        self.H = H
        self.control_dimensions = len(control_bounds_lower)
        self.lambda_ = lambda_
        self.nominal_xy_path = nominal_xy_path
        self.map = map

        # TODO we can configure different hyperparameter/behavioral parameters
        # to score random trajectories differently, thus changing the behavior
        # of the controller in different situations
        # Potential modes:
        # - general
        # - near to/within sample region
        # - near to obstacles (conservative)
        # - aggressive

        # TODO could be useful to draw from a low dimensional space
        # and then map to the high dimensional space using learning,
        # so that it's easier to explore/learn a good distribution
        # with flows in the low dimensional space (recall that this
        # is over U). flow can still have transformations learned
        # based on context


    def update_nominal_xy_path(self, nominal_xy_path):
        self.nominal_xy_path = nominal_xy_path

    def sample_all_action_sequences(self):
        """
        Random sampling in high dimensions trends towards
        the center of the space. To avoid this, we'll sample using 
        a sampling plan
        """

        # Maximally fill the space
        bounds = np.array([self.control_bounds_lower, self.control_bounds_upper])
        samples = np.zeros((self.K, self.H, self.control_dimensions))

        # Generate Sobol sequences
        for i in range(self.H):
            # I want some randomness or an offset in the sequence between timesteps
            # so that we don't get stuck in a local minimum
            # TODO
            sobol_samples = i4_sobol_generate(self.control_dimensions, self.K)
            for j in range(self.control_dimensions):
                sobol_samples[:, j] = sobol_samples[:, j] * (bounds[1, j] - bounds[0, j]) + bounds[0, j]
            samples[:, i, :] = sobol_samples

        return samples

    def sample_action_sequence(self):
        # This is actually the MPOPI simulation - we'll sample a sequence
        # of actions out to the horizon 
        action_sequence = np.random.uniform(
            low=self.control_bounds_lower,
            high=self.control_bounds_upper,
            size=(self.H, self.control_dimensions)
        )
        return action_sequence

    def rollout(self, x0, U):
        """
        Rollout the dynamics function from an initial state with a sequence of actions,
        and return the resulting trajectory of states
        """
        X = np.zeros((self.H+1, len(x0)))
        X[0] = x0
        # That the u here is a sequence of controls, u_0, u_1, ..., u_{H-1}
        for i, u in enumerate(U):
            X[i+1] = self.dynamics_fn(X[i], u)
        return X
    
    def score(self, prev_X, X, prev_U, U):
        """
        Score a trajectory of states and controls, with the previous states and controls
        known for context
        """
        
        # Optimize some things here:
        # - the positions that we go through should overlap with the nominal positions (use
        #   path comparison implementation for this
        # - we should be going forward along the path
        # - we shouldn't go too fast or too slow
        # - we should avoid obstacles
        # - we should not put in too much control effort
        # - stay upright (no rotations - penalize upside down states and high angular velocities)
        # - continuous controls

        # Recall that x and y at 0 and 2
        actual_xy_positions = X[:,[0,2]]
        actual_xy_positions = actual_xy_positions.reshape(-1, 2)
        actual_path = Path(actual_xy_positions)

        # If we hit a wall get negative infinity
        hit_boundary, collision_index =  self.map.does_path_hit_boundary(actual_path, return_earliest_hit_index=True)
        # index will be zero if we immediately hit the boundary
        # and will be H if we hit the boundary at the end of our 
        # projected path, so the following variable will be (H - 0) / H = 1 if
        # we hit the boundary immediately and (H - (H-1)) / H = 1/H if 
        # we hit the boundary as late as possible
        collision_closer_along_path = ( self.H - collision_index ) / self.H

        #print(f"Collision: {hit_boundary}, index: {collision_index}, closer: {collision_closer_along_path}")
        
        # If you hit boundary in the first X% of the path then it's -inf
        if collision_closer_along_path > 1 - 0.4 and hit_boundary:
            return -np.inf

        # How much does the actual path deviate from the nominal path?
        path_deviation = self.nominal_xy_path.deviation_from_path(actual_path)

        # Are we going in the right direction?
        forwardness = self.nominal_xy_path.forwardness_wrt_other_path(actual_path)

        # Stay at a certain speed throughout the path
        target_speed = 1 # m/s
        vxs = X[:,1]
        vys = X[:,3]
        speeds = np.linalg.norm(np.array([vxs, vys]).T, axis=1)
        speed_deviation = np.mean(np.abs(speeds - target_speed))   

        # Stay upright
        angles = X[:,4]
        angle_deviation = np.linalg.norm(angles) 
        # Don't have a high angular velocity
        angular_velocities = X[:,5]
        angular_velocity_deviation = np.linalg.norm(angular_velocities)

        # Control effort should be minimized
        control_effort = np.linalg.norm(U)

        # Control should be continuous, i.e., adjacent control actions should be similar,
        # so we measure the mean difference between adjacent control actions
        entire_U = np.concatenate((prev_U, U)) 
        adjacent_control_differences = np.mean(np.abs(np.diff(entire_U, axis=0)))

        # Score is a weighted sum - note the negatives mean that we minimize
        # deviations, differences, and efforts as desired, and maximize the
        # correct direction
        # Notes from playing around:
        # - the distance_to_end is important to emphasize, otherwise
        #   we'll go backwards along the path, even sometimes 
        #   crashing into the wall in the wrong direction from the 
        #   start
        # - the path_length_deviation is important to emphasize, otherwise
        #   we'll go too fast and crash
        score = - 5000 * path_deviation \
                + 500 * forwardness \
                - 1500 * speed_deviation \
                - 80 * angle_deviation \
                - 80 * angular_velocity_deviation \
                - 0 * control_effort \
                - 500 * adjacent_control_differences \
                - 10000 * (collision_closer_along_path if hit_boundary else 0)

        return score

    def optimal_control_sequence(self, prev_X, prev_U, return_scored_rollouts=False):
        """
        Optimize the control sequence to minimize the cost of the trajectory
        """

        # Having a handle to the most recent state is convienient
        x0 = prev_X[-1]

        # If we're doing adaptive improvement, then we'll 
        # 

        # Define a function for parallel execution of a sample
        Us = self.sample_all_action_sequences()
        def process_control_sequence(i):
            #U = self.sample_action_sequence()
            U = Us[i]
            X = self.rollout(x0, U)
            score = self.score(prev_X, X, prev_U, U)
            return U, X, score

        # Number of processes to run in parallel
        num_processes = min(self.K, os.cpu_count() - 4)
        num_processes = max(num_processes, 1)

         # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(process_control_sequence, range(self.K)))

        # Extract results from parallel execution
        Us, Xs, scores = zip(*results)
        
        # MPPI weighted with softmax + lambda
        weights = np.exp(self.lambda_ * np.array(scores))
        weights /= np.sum(weights)

        # Get the optimal by multiplying the weights by the control sequences
        # and summing
        #opt_U = np.sum(weights[:,np.newaxis,np.newaxis] * Us, axis=0)
        # Take the best control sequence
        opt_U = Us[np.argmax(scores)]

        # Return the optimal control sequence
        if return_scored_rollouts:
            return opt_U, (Xs, scores)
        else:   
            return opt_U


