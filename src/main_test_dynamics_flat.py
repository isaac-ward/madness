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
    #path1 = path1.downsample_to_average_adjacent_distance_metres(0.2)
    path1 = path1.downsample_every_n(5)

    # Need bounding boxes that encapsulate traversable space
    boxes = map1.path_box(path1,percent_overlap=60)

    # Create a 2D Quadrotor
    quad = dynamics.Quadrotor2D(0.1)

    # Run Bezier-curve path planning
    print("Generating Bezier-Curve path")
    state_vector,control_vector = quad.bezier_trajectory_fitting(path1.path_metres,boxes)
    fitted_path_metres = state_vector[:,[0,2]]

    # Test SCP path planning
    """state_vector,control_vector = quad.SCP_nominal_trajectory(astar_path=path1.path_metres,
                                                              boxes=boxes,
                                                              R=np.eye(2),
                                                              Q=np.eye(6),
                                                              P=10*np.eye(6))
    fitted_path_metres = state_vector[:,[0,2]]"""

    # Test OL controls on dynamic model
    print("Implementing Open-Loop control model")
    N = np.shape(state_vector)[0]
    OL_states = np.zeros(state_vector.shape)
    OL_states[0] = state_vector[0]
    dt = np.zeros(control_vector.shape[0])
    for i in range(1,N):
        dt[i-1] = np.linalg.norm(state_vector[i,[0,2]]-state_vector[i-1,[0,2]])/np.linalg.norm(state_vector[i-1,[1,3]])
        OL_states[i] = quad.dynamics_true_no_disturbances(OL_states[i-1], control_vector[i-1],dt=dt[i-1])
    follow_path_metres = OL_states[:,[0,2]]
    #print(state_vector-OL_states)

    # Visualize the occupancy grid with OL trajectory
    visuals.vis_occupancy_grid(
        filepath=f"{log_folder}/OL_Control.png",
        occupancy_grid=map1.occupancy_grid,
        metres_per_pixel=map1.metres_per_pixel,
        # start and finish points (in metres)
        points_metres=[
            start_coord_metres,
            finish_coord_metres
        ],
        path_metres=path1.path_metres,
        path2_metres=fitted_path_metres,
        path3_metres=follow_path_metres,
        plot_coordinates=True,
        path_boxes=boxes
    )

    # Test ILQR controls on dynamic model
    print("Implementing iLQR control")
    x_bar, u_bar, Y, y = control.ilqr(state_vector, 
                                      control_vector,
                                      dt,
                                      N-1, 
                                      quadrotor=quad, 
                                      Q=100*np.eye(quad.n_dim), 
                                      R=np.eye(quad.m_dim), 
                                      QN=np.diag([1000, 10, 1000, 10, 10, 10]),#1000*np.eye(quad.n_dim), 
                                      eps=1e-5, 
                                      max_iters=1000)
    
    ilqr_states = np.zeros(state_vector.shape)
    ilqr_control = np.zeros(control_vector.shape)
    ilqr_states[0] = np.copy(x_bar[0])
    dt = np.zeros(control_vector.shape[0])
    for k in range(N-1):
        dt[k] = np.linalg.norm(x_bar[k+1,[0,2]]-x_bar[k,[0,2]])/np.linalg.norm(x_bar[k,[1,3]])
        #print(ilqr_states[k]-x_bar[k])
        ilqr_control[k] = u_bar[k] + y[k] + Y[k]@(ilqr_states[k]-x_bar[k])
        ilqr_states[k+1] = quad.dynamics_true_no_disturbances(ilqr_states[k], ilqr_control[k], dt=dt[k])
    iLQR_follow_path_metres = ilqr_states[:,[0,2]]
    iLQR_fitted_path_metres = x_bar[:,[0,2]]


    # Visualize the occupancy grid with OL trajectory
    visuals.vis_occupancy_grid(
        filepath=f"{log_folder}/iLQR_Control.png",
        occupancy_grid=map1.occupancy_grid,
        metres_per_pixel=map1.metres_per_pixel,
        # start and finish points (in metres)
        points_metres=[
            start_coord_metres,
            finish_coord_metres
        ],
        path_metres=path1.path_metres,
        path2_metres=iLQR_fitted_path_metres,
        path3_metres=iLQR_follow_path_metres,
        plot_coordinates=True,
        path_boxes=boxes
    )

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