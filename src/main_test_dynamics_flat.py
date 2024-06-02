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
    map_config = globals.MAP_CONFIGS["downup-o"]
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
    path1 = path1.downsample_every_n(5)
    # And a smoothed version
    

    # We need to know the locations of boundaries (parts of the occupancy grid)
    # that touch both an occupied cell and an unoccupied cell
    obstacles_metres = map1.boundary_positions

    # Get Boxes
    boxes = map1.path_box(path1)

    # Verify dynamics code
    quad = dynamics.Quadrotor2D(0.1)

    # Test bezier fit
    state_vector,control_vector = quad.bezier_trajectory_fitting(path1.path_metres,boxes)
    fitted_path_metres = state_vector[:,[0,2]]
    print(control_vector)
    print(state_vector)

    # Test SCP path planning
    """state_vector,control_vector = quad.SCP_nominal_trajectory(astar_path=path1.path_metres,
                                                              boxes=boxes,
                                                              R=np.eye(2),
                                                              Q=np.eye(6),
                                                              P=10*np.eye(6))
    fitted_path_metres = state_vector[:,[0,2]]"""

    # Test controls on dynamic model
    T = np.shape(state_vector)[0]
    truestate = np.copy(state_vector[0])
    x_next = np.copy(state_vector[0])
    for i in range(1,T):
        x_next = quad.dynamics_true_no_disturbances(x_next, control_vector[i-1])
        truestate = np.vstack([truestate,x_next])
    follow_path_metres = truestate[:,[0,2]]

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
        path2_metres=fitted_path_metres,
        #path3_metres=follow_path_metres,
        plot_coordinates=True,
        path_boxes=boxes
    )

    # Generate a sample trajectory
    xtrue = np.array(path1.path_metres[:,0])
    ytrue = np.array(path1.path_metres[:,1])

    """state_trajectory, action_trajectory = quad.dynamics_test(
        log_folder, 
        xtrue, 
        ytrue, 
        obstacles_metres[::100], 
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
    """