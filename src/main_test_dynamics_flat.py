import numpy as np
from tqdm import tqdm
import scipy

from map import Map
import control 
import dynamics
import visuals
import utils 
import globals


if __name__ == "__main__":

    # Will log everything to here
    log_folder = utils.make_log_folder(name="run")

    # Get the map info for the map that we'll be using
    map_config = globals.MAP_CONFIGS["3x28"]
    metres_per_pixel    = map_config["metres_per_pixel"]
    filename            = map_config["filename"]
    start_coord_metres  = map_config["start_coord_metres"]
    finish_coord_metres = map_config["finish_coord_metres"]

    # Load the map file as an occupancy grid
    map1 = Map(
        map_filepath=f"{utils.get_assets_dir()}/{filename}",
        metres_per_pixel=metres_per_pixel,
        scale_factor=0.2
    )
    
    # --------------------------------
    # there 

    # Generate a path from start to finish
    path1 = map1.path_a_to_b_metres(
        a_coord_metres=start_coord_metres,
        b_coord_metres=finish_coord_metres
    )

    # Which must be downsampled aggressively
    path1.downsample_every_n(5)
    # And a smoothed version
    

    # We need to know the locations of boundaries (parts of the occupancy grid)
    # that touch both an occupied cell and an unoccupied cell
    obstacles_metres = map1.boundary_positions

    # Visualize the occupancy grid with a few points marked
    visuals.vis_occupancy_grid(
        filepath=f"{log_folder}/occupancy_grid.png",
        occupancy_grid=map1.occupancy_grid,
        metres_per_pixel=map1.metres_per_pixel,
        # start and finish points (in metres)
        points_metres=[
            start_coord_metres,
            finish_coord_metres
        ],
        path_metres=path1.path_metres,
        #path2_metres=fitted_path_metres,
        plot_coordinates=True
    )

    # Generate a sample trajectory
    xtrue = np.array(path1.path_metres[:,0])
    ytrue = np.array(path1.path_metres[:,1])
    
    # Verify dynamics code
    quad = dynamics.Quadrotor2D(0.1)
    state_trajectory, action_trajectory = quad.dynamics_test(
        log_folder, 
        xtrue, 
        ytrue, 
        obstacles_metres, 
        v_desired=1, 
        spline_alpha=0
    )

    visuals.plot_trajectory(
        filepath=f"{log_folder}/dynamicstest.mp4",
        state_trajectory=state_trajectory,
        state_element_labels=[],
        action_trajectory=action_trajectory,
        action_element_labels=[],
        dt=dynamics.dt
    )