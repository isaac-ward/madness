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
        center_voxel_coords,
        radius_voxels,
        interior_voxel_coords,
        voxel_per_x_metres,
    ):
        self.center_metres_xyz = center_metres_xyz
        self.radius_metres = radius_metres
        self.center_voxel_coords = center_voxel_coords
        self.radius_voxels = radius_voxels
        self.interior_voxel_coords = interior_voxel_coords     
        self.voxel_per_x_metres = voxel_per_x_metres   

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
        # Get the maximum radius (in metres)
        radius_metres = SDF.find_max_non_collision_radius(center_metres_xyz, mapping)

        # To get the interior voxels, create a meshgrid at the correct resolution,
        # the same size as the map, and then check if each point is within the sphere
        interior_voxel_coords = mapping.get_voxels_within_radius(
            center_metres_xyz,
            radius_metres,
        )
        
        # Create the SDF object
        sdf = SDF(
            center_metres_xyz=center_metres_xyz,
            radius_metres=radius_metres,
            center_voxel_coords=mapping.metres_to_voxel_coords(center_metres_xyz),
            radius_voxels=radius_metres / mapping.voxel_per_x_metres,
            interior_voxel_coords=interior_voxel_coords,
            voxel_per_x_metres=mapping.voxel_per_x_metres,
        )

        return sdf
    
    def __str__(self):
        s = "SDF:\n"
        s += f"  center_metres_xyz: {self.center_metres_xyz}\n"
        s += f"  radius_metres: {self.radius_metres}\n"
        s += f"  center_voxel_coords: {self.center_voxel_coords}\n"
        s += f"  radius_voxels: {self.radius_voxels}\n"
        s += f"  num_interior_voxels: {len(self.interior_voxel_coords)}\n"
        s += f"  voxel_per_x_metres: {self.voxel_per_x_metres}\n"
        return s

    def astar_within_sphere(
            self,
            astar):
        """
        """
        # Calculate squared distance from center for each vector
        distances_squared = np.sum((astar - self.center_metres_xyz) ** 2, axis=1)
        
        # Compare distances to the squared radius
        return (distances_squared <= self.radius_metres ** 2).astype(int)

    def line_sphere_intersection_two_points(
            self,
            astar_in_sphere,
            astar_out_sphere
    ):
        # Extract variables
        # Convert inputs to numpy arrays for easier vector operations
        P1 = astar_in_sphere
        P2 = astar_out_sphere
        C = self.center_metres_xyz
        R = self.radius_metres
        
        # Compute the quadratic coefficients
        a = np.dot(P2 - P1, P2 - P1)
        b = 2 * np.dot(P2 - P1, P1 - C)
        c = np.dot(P1 - C, P1 - C) - R**2
        
        # Solve the quadratic equation: at^2 + bt + c = 0
        discriminant = b**2 - 4 * a * c
        
        if discriminant < 0:
            # No real solutions, no intersection
            return None
        
        # Two solutions for t
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        # Calculate the intersection points
        intersection1 = P1 + t1 * (P2 - P1) if 0 <= t1 <= 1 else None
        intersection2 = P1 + t2 * (P2 - P1) if 0 <= t2 <= 1 else None
        
        # Choose the intersection point closer to P2
        if intersection1 is not None and intersection2 is not None:
            # Calculate distances to P2
            dist1 = np.linalg.norm(intersection1 - P2)
            dist2 = np.linalg.norm(intersection2 - P2)
            # Return the point closer to P2
            return intersection1 if dist1 < dist2 else intersection2
        elif intersection1 is not None:
            return intersection1
        elif intersection2 is not None:
            return intersection2
        else:
            return None
    
    def get_next_sdf_center(self,astar:np.array):
        """
        """
        # Check what elements of Astar path are within the sphere
        astar_in_sphere = self.astar_within_sphere(astar)
        print(astar_in_sphere)

        # Check if last position is in sphere
        # If it is we done!
        if astar_in_sphere[-1] == 1:
            return np.zeros(3), True
        
        # Get last switch from 1 to 0
        last_in = np.zeros(3)
        last_out = np.zeros(3)
        for _i in range(len(astar_in_sphere) - 1, 0, -1):
            if astar_in_sphere[_i] == 0 and astar_in_sphere[_i-1] == 1:
                last_in = np.copy(astar[_i-1])
                last_out = np.copy(astar[_i])

        # Get intersection between sphere and Astar line
        new_sdf_center = self.line_sphere_intersection_two_points(last_in,last_out)

        # TODO add some randomness

        return new_sdf_center, False

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
    #state_initial, state_goal = Environment.get_two_states_separated_by_distance(map_, min_distance=26)
    state_initial = np.zeros(12)
    state_initial[:3] = 10
    state_goal = np.zeros(12)
    state_goal[:3] = 26

    # # Generate a path from the initial state to the goal state
    xyz_initial = state_initial[0:3]
    xyz_goal = state_goal[0:3]
    path_xyz = np.array([xyz_initial, xyz_goal])
    path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*4) # Ultra safe
    path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz)

    # Create a list to hold centers and radii
    sdfs = [ SDF.get_optimal_sdf(xyz_initial, map_) ]
    max_spheres = 10
    for _i in range(max_spheres):
        # Get new center
        new_start_point, search_complete = sdfs[-1].get_next_sdf_center(path_xyz_smooth)

        # Check if new sdf is needed
        if search_complete:
            break

        # Build new sdf
        sdfs.append(SDF.get_optimal_sdf(new_start_point, map_))
        print(sdfs[-1].center_metres_xyz)
        print(sdfs[-1].radius_metres)

    print(len(sdfs))

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

    # No simulation

    # ----------------------------------------------------------------

    # Log everything of interest
    environment.log(log_folder)

    # Render visuals
    visual = Visual(log_folder)
    #visual.plot_histories()
    #visual.render_video(desired_fps=25)
    visual.plot_environment()
