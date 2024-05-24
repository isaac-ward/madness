import numpy as np

from map import Map 
import path
import control 
import dynamics
import visuals
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
        scale_factor=0.4 # Anything less doesn't resolve obstacles
    )
    
    # --------------------------------
    # there 

    # Generate a path from start to finish
    path_metres = map.path_a_to_b_metres(
        a_coord_metres=start_coord_metres,
        b_coord_metres=finish_coord_metres
    )

    # Create an MPPI controller
    controller = MPPI(
        dynamics_fn=dynamics.Quadrotor2D().dynamics_true,
        control_bounds_lower=np.array([0, 0]),
        control_bounds_upper=np.array([globals.MAX_THRUST_PER_PROP, globals.MAX_THRUST_PER_PROP]),
        K=32,
        H=32,
        lambda_=0,
    )

    # Enter a loop where we apply the controller, get an action, roll forward
    # the dynamics and repeat
    N = 100
    x0 = np.zeros((dynamics.Quadrotor2D().n, 1))
    x0[0] = start_coord_metres[0]
    x0[2] = start_coord_metres[1]
    for i in range(N):
        # Apply the controller to get an action
        action = controller.optimal_control_sequence(x0)
        # Roll forward the dynamics
        x0 = dynamics.Quadrotor2D().dynamics_true(x0, action)

    # Oh sick I got a sample!
    
    # --------------------------------
    # and back again

    # Visualize the run
    output_filepath = f"{log_folder}/experiment.png"
    visuals.plot_experiment(
        output_filepath,
        map,
        start_coord_metres,
        finish_coord_metres,
        [ # paths to render
            {
                "path": path_metres,
                "color": "blue",
            }
        ], 
        state_trajectory=[],
        control_trajectory=[],
    )