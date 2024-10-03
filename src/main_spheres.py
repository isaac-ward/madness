import numpy as np
from tqdm import tqdm
import scipy
import os
import pickle
import csv
import time
import copy
import matplotlib.pyplot as plt
import cupy as cp

import utils.general
import utils.logging
import utils.geometric
import dynamics
from environment import Environment
from mapping import Map
from agent import Agent
from visual import Visual
from policies.simple import PolicyNothing, PolicyRandom, PolicyConstant
from policies.mppi import PolicyMPPI
import policies.samplers
import standard

class SDF:
    def __init__(
        self,
        center_metres_xyz,
        radius_metres,
    ):
        self.center_metres_xyz = center_metres_xyz
        self.radius_metres = radius_metres

    @staticmethod
    def find_max_non_collision_radius(
        center_metres_xyz,
        mapping,
    ):
        # Begin by converting from metres to voxel
        center_voxel_xyz = mapping.metres_to_voxel_coords(center_metres_xyz)

        # Start with a zero radius (just the start point voxel)
        radius_voxels = 0
        in_collision = False
        maximum_radius = 100
        while radius_voxels < maximum_radius and not in_collision:
            # Increase the radius by one voxel
            radius_voxels += 1

            # Get all voxels within the current radius
            voxels_to_check = mapping.get_voxels_within_radius(center_voxel_xyz, radius_voxels)

            # Check if any of the voxels are occupied
            in_collision = any(mapping.batch_is_collision_voxel_coords(voxels_to_check, radius_voxels))

        # Reduce the radius by one to get the maximum non-collision radius
        radius_voxels -= 1

        # Convert the radius back to metres
        radius_metres = radius_voxels * (1 / mapping.voxel_per_x_metres)

        return radius_metres 
    
    @staticmethod
    def get_optimal_sdf(
        center_metres_xyz,
        mapping,
    ):
        # Get the maximum radius
        radius = SDF.find_max_non_collision_radius(center_metres_xyz, mapping)

        # Create the SDF object
        sdf = SDF(center_metres_xyz,radius)

        return sdf
    
    def __str__(self):
        return f"SDF(center={self.center_metres_xyz}, radius={self.radius_metres})"


if __name__ == "__main__":

    # Seed everything
    utils.general.random_seed(42)

    # Are we using GPU? 
    # NOTE: suggest false for now because it's slow
    use_gpu_if_available = False
    keep_policy_logs = True

    # Will log everything to here
    log_folder = utils.logging.make_log_folder(name="run")

    # The environment follows some true dynamics, and the agent
    # has an internal model of the environment
    dyn = standard.get_standard_dynamics_quadcopter_3d()

    # Create a map representation
    #map_ = standard.get_standard_map()
    map_ = standard.get_28x28x28_at_111()
    #map_ = standard.get_28x28x28_at_111_with_obstacles()

    # Start and goal states
    # NOTE: The following utility finds two random points - it doesn't check for collisions!
    # If you're using a map with invalid positions then you might need to specify the start and goal states manually
    state_initial, state_goal = Environment.get_two_states_separated_by_distance(map_, min_distance=26)

    # # Generate a path from the initial state to the goal state
    """xyz_initial = state_initial[0:3]
    xyz_goal = state_goal[0:3]
    path_xyz = np.array([xyz_initial, xyz_goal])
    path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*4) # Ultra safe
    path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz)"""
    
    # Get the current starting position
    start_point = map_.metres_to_voxel_coords(state_initial[:3])

    # Create a list to hold centers and radii
    sdfs = [ SDF.get_optimal_sdf(start_point, map_) ]

    print(sdfs[0])

    # ----------------------------------------------------------------
    # Logging from here on
    # ----------------------------------------------------------------

    # Save the cubes
    utils.logging.pickle_to_filepath(
        os.path.join(log_folder, "signed_distance_function.pkl"),
        sdfs,
    )

    # Create the environment
    num_seconds = 16
    num_steps = int(num_seconds / dyn.dt)
    environment = Environment(
        state_initial=state_initial,
        state_goal=state_goal,
        dynamics=dyn,
        map_=map_,
        episode_length=num_steps,
    )

    # ----------------------------------------------------------------

    # Build a sphere at the start
    pbar = tqdm(total=num_steps, desc="Running simulation")
    
    pbar.close()

    # ----------------------------------------------------------------

    # Log everything of interest
    environment.log(log_folder)

    # Render visuals
    visual = Visual(log_folder)
    #visual.plot_histories()
    #visual.render_video(desired_fps=25)
    visual.plot_environment()
