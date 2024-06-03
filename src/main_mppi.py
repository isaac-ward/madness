import numpy as np
from tqdm import tqdm
import os
import pickle

from map import Map 
import path
import control 
import dynamics
import visuals2 as visuals
import utils 
import globals
from mppi import MPPI

if __name__ == "__main__":
    
    # Will log everything to here
    log_folder = utils.make_log_folder(name="run")

    # Get the map info for the map that we'll be using
    map_config = globals.MAP_CONFIGS["downup-o"]
    metres_per_pixel    = map_config["metres_per_pixel"]
    filename            = map_config["filename"]
    start_coord_metres  = map_config["start_coord_metres"]
    finish_coord_metres = map_config["finish_coord_metres"]

    # Other configuration variables which we will use later
    fudge_factor = 1.6
    nominal_downsample_factor = 0.05
    fitted_downsample_factor = 0.1
    map_scale_factor = 0.2
    dt = 1/25 # This makes rendering easier because playback fps is 25

    # Compute a unique hash for the current configuration
    config_hash = utils.compute_hash(
        filename, 
        metres_per_pixel, 
        start_coord_metres, 
        finish_coord_metres
    )

    # Path to cache file - if the config is the same, we can just load the cache
    cache_filepath = f"{utils.get_cache_dir()}/cache_{config_hash}.pkl"
    cache = {}

    # --------------------------------

    # Need a map representation
    map = Map(
        map_filepath=f"{utils.get_assets_dir()}/{filename}",
        metres_per_pixel=metres_per_pixel,
        scale_factor=map_scale_factor # Anything less than 0.2 doesn't resolve obstacles well
    )

    # And a dynamics model
    dyn = dynamics.Quadrotor2D(dt=dt)
    
    # --------------------------------
    # there 

    # Check if we have a cache
    if os.path.exists(cache_filepath):
        # Take from cache if it exists
        print(f"Loading cache from {cache_filepath}")
        with open(cache_filepath, "rb") as f:
            cache = pickle.load(f)
        nominal_xy = cache["nominal_xy"]
        fitted_nominal_xy = cache["fitted_nominal_xy"]

    else:
        # No cache - recompute

        # Generate a path from start to finish
        nominal_xy = map.path_a_to_b_metres(
            a_coord_metres=start_coord_metres,
            b_coord_metres=finish_coord_metres,
            fudge_factor=fudge_factor,
        )
        nominal_xy = nominal_xy.downsample_to_average_adjacent_distance_metres(nominal_downsample_factor)

        # Smooth it with respect to the boundaries by first getting 
        # obstacle/freespace boxes
        boxes = map.path_box(nominal_xy)
        state_vector, control_vector = dyn.bezier_trajectory_fitting(nominal_xy.path_metres, boxes)
        fitted_path_metres = state_vector[:,[0,2]]
        fitted_nominal_xy = path.Path(fitted_path_metres)
        fitted_nominal_xy = fitted_nominal_xy.downsample_to_average_adjacent_distance_metres(fitted_downsample_factor)

        # Save to cache
        cache["nominal_xy"] = nominal_xy
        cache["fitted_nominal_xy"] = fitted_nominal_xy
        print(f"Saving cache to {cache_filepath}")
        with open(cache_filepath, "wb") as f:
            pickle.dump(cache, f)

    # --------------------------------

    # Create an MPPI controller
    controller = MPPI(
        dynamics_fn=dyn.dynamics_true_no_disturbances,
        # Allow reverse thrust
        control_bounds_lower=np.array([-globals.MAX_THRUST_PER_PROP, -globals.MAX_THRUST_PER_PROP]),
        control_bounds_upper=np.array([+globals.MAX_THRUST_PER_PROP, +globals.MAX_THRUST_PER_PROP]),
        K=256,
        H=5,
        lambda_=1000000, # Take the best would be lambda_ -> inf
        nominal_xy_path=fitted_nominal_xy,
        map=map
    )
    print(f"Created MPPI controller with {controller.K} samples and horizon {controller.H} ({controller.H * dt} seconds lookahead)")

    # Enter a loop where we apply the controller, get a control, roll forward
    # the dynamics and repeat
    N = 1000
    # Initial state
    x = np.zeros(dyn.n_dim)
    x[0] = start_coord_metres[0]
    x[1] = 0.25 # a gentle push
    x[2] = start_coord_metres[1]
    sample_obtained = False
    # Logging
    state_trajectory    = np.zeros((N+1, dyn.n_dim))
    state_trajectory[0] = x
    control_trajectory  = np.zeros((N, dyn.m_dim))
    scored_rollouts_per_step = []
    # This has to be ran forward, we can't parallelize
    pbar = tqdm(total=N, desc="Approaching sample region")
    for i in range(N):

        # What have we done so far
        prev_X = state_trajectory[:i+1]
        prev_U = control_trajectory[:i]

        # Apply the controller to get a control sequence
        U, (samples, scores) = controller.optimal_control_sequence(prev_X, prev_U, return_scored_rollouts=True)
        scored_rollouts_per_step.append((samples, scores))
        u = U[0]
        # Roll forward the dynamics
        x = dyn.dynamics_true_no_disturbances(x, u)

        # The game is over if we hit a wall, or if we go out of bounds
        if map.does_point_hit_boundary(x[0], x[2]) or map.out_of_bounds(x[0], x[2]):
            pbar.set_description("Out of bounds or hit a wall, stopping simulation")
            pbar.close()
            state_trajectory   = state_trajectory[:i]
            control_trajectory = control_trajectory[:i]
            break

        # If we reach the finish point, then then we reverse the MPPI path
        if np.linalg.norm(x[[0,2]] - finish_coord_metres) < globals.REACHED_GOAL_THRESHOLD and not sample_obtained:
            controller.update_nominal_xy_path(fitted_nominal_xy.reversed())
            sample_obtained = True
            pbar.set_description("Sample taken, returning to base")
            # TODO edit the dynamics too - heavier now!

        # If we reach the start with the sample then we're done
        if np.linalg.norm(x[[0,2]] - start_coord_metres) < globals.REACHED_GOAL_THRESHOLD and sample_obtained:
            pbar.set_description("Sample retrieved!")
            pbar.close()
            state_trajectory   = state_trajectory[:i]
            control_trajectory = control_trajectory[:i]
            break

        # Log the state and control
        state_trajectory[i+1] = x
        control_trajectory[i] = u

        # Update the progress bar
        pbar.set_postfix({
            "m_to_goal": f"{np.linalg.norm(x[[0,2]] - finish_coord_metres):.2f}",
            "x": f"{x[0]:.2f}",
            "y": f"{x[2]:.2f}",
            "v": f"{np.linalg.norm(x[[1,3]]):.2f}",
        })
        pbar.update(1)
    
    # Oh sick I got a sample!
    
    # --------------------------------
    # and back again
    # TODO
        
        
    # --------------------------------

    # Visualize the run
    output_filepath = f"{log_folder}/experiment.mp4"
    visuals.plot_experiment_video(
        output_filepath,
        map,
        start_coord_metres,
        finish_coord_metres,
        [ # paths to render
            {
                "path": nominal_xy,
                "color": "lightgrey",
            },
            {
                "path": fitted_nominal_xy,
                "color": "grey",
            }
        ], 
        simulation_dt=dt,
        state_trajectory=state_trajectory,
        control_trajectory=control_trajectory,
        scored_rollouts_per_step=scored_rollouts_per_step,
    )