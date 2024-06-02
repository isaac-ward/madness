import numpy as np
from tqdm import tqdm

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

    # Need a map representation
    map = Map(
        map_filepath=f"{utils.get_assets_dir()}/{filename}",
        metres_per_pixel=metres_per_pixel,
        scale_factor=0.2 # Anything less than 0.2 doesn't resolve obstacles well
    )

    # And a dynamics model
    dt = 1/25 # This makes rendering easier because playback fps is 25
    dyn = dynamics.Quadrotor2D(dt=dt)
    
    # --------------------------------
    # there 

    # Generate a path from start to finish
    nominal_xy = map.path_a_to_b_metres(
        a_coord_metres=start_coord_metres,
        b_coord_metres=finish_coord_metres,
        fudge_factor=1.6,
    )
    nominal_xy = nominal_xy.downsample_to_average_adjacent_distance_metres(0.05)

    # Smooth it with respect to the boundaries by first getting 
    # obstacle/freespace boxes
    boxes = map.path_box(nominal_xy)
    state_vector, control_vector = dyn.bezier_trajectory_fitting(nominal_xy.path_metres, boxes)
    fitted_path_metres = state_vector[:,[0,2]]
    fitted_nominal_xy = path.Path(fitted_path_metres)
    fitted_nominal_xy = fitted_nominal_xy.downsample_to_average_adjacent_distance_metres(0.1)

    # Create an MPPI controller
    controller = MPPI(
        dynamics_fn=dyn.dynamics_true_no_disturbances,
        # Allow reverse thrust
        control_bounds_lower=np.array([-globals.MAX_THRUST_PER_PROP, -globals.MAX_THRUST_PER_PROP]),
        control_bounds_upper=np.array([+globals.MAX_THRUST_PER_PROP, +globals.MAX_THRUST_PER_PROP]),
        K=4096,
        H=12,
        lambda_=1000000, # Take the best
        nominal_xy_positions=fitted_nominal_xy.path_metres,
        map=map
    )
    print(f"Created MPPI controller with {controller.K} samples and horizon {controller.H} ({controller.H * dt} seconds lookahead)")

    # Enter a loop where we apply the controller, get a control, roll forward
    # the dynamics and repeat
    N = 64*10
    # Initial state
    x = np.zeros(dyn.n_dim)
    x[0] = start_coord_metres[0]
    x[2] = start_coord_metres[1]
    # Logging
    state_trajectory    = np.zeros((N+1, dyn.n_dim))
    state_trajectory[0] = x.flatten()
    control_trajectory  = np.zeros((N, dyn.m_dim))
    scored_rollouts_per_step = []
    # This has to be ran forward, we can't parallelize
    for i in tqdm(range(N), desc="Running simulation"):
        # Apply the controller to get a control sequence
        U, (samples, scores) = controller.optimal_control_sequence(x, return_scored_rollouts=True)
        scored_rollouts_per_step.append((samples, scores))
        u = U[0]
        # Roll forward the dynamics
        x = dyn.dynamics_true_no_disturbances(x, u)
        # The game is over if we hit a wall, or if we go out of bounds
        if map.does_point_hit_boundary(x[0], x[2]) or map.out_of_bounds(x[0], x[2]):
            print("Ending simulation early.")
            state_trajectory   = state_trajectory[:i]
            control_trajectory = control_trajectory[:i]
            break
        # Log the state and control
        state_trajectory[i+1] = x.flatten()
        control_trajectory[i] = u.flatten()
    
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