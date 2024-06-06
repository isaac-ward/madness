import numpy as np
from tqdm import tqdm
import scipy
import os
import pickle
import csv
import time

from map import Map
from path import Path
import control 
import dynamics
import visuals
import utils 
import globals
import visuals2


if __name__ == "__main__":

    # Will log everything to here
    log_folder = utils.make_log_folder(name="run_ilqr")

    # Get the map info for the map that we'll be using
    map_config = globals.MAP_CONFIGS["downup-o"]
    metres_per_pixel    = map_config["metres_per_pixel"]
    filename            = map_config["filename"]
    start_coord_metres  = map_config["start_coord_metres"]
    finish_coord_metres = map_config["finish_coord_metres"]
    fudge_factor        = map_config["fudge_factor"]

    # Compute a unique hash for the current configuration
    config_hash = utils.compute_hash(
        "ilqr",
        filename, 
        metres_per_pixel, 
        start_coord_metres, 
        finish_coord_metres
    )

    # Path to cache file - if the config is the same, we can just load the cache
    cache_filepath = f"{utils.get_cache_dir()}/cache_{config_hash}.pkl"
    cache = {}

    # Load the map file as an occupancy grid
    map1 = Map(
        map_filepath=f"{utils.get_assets_dir()}/{filename}",
        metres_per_pixel=metres_per_pixel,
        scale_factor=0.2
    )
    
    # --------------------------------
    # Let's go THERE!

    # Check if we have a cache
    if os.path.exists(cache_filepath):
        # Take from cache if it exists
        print(f"Loading cache from {cache_filepath}")
        with open(cache_filepath, "rb") as f:
            cache = pickle.load(f)
        path_there = cache["path_there"]
        boxes = cache["boxes"]
        state_vector = cache["state_vector"]
        control_vector = cache["control_vector"]

        # Create a 2D Quadrotor
        quad = dynamics.Quadrotor2D(0.1)
    else:
        # No cache - recompute

        # Generate a path from start to finish
        # Generate a path from start to finish
        path_there = map1.path_a_to_b_metres(
            a_coord_metres=start_coord_metres,
            b_coord_metres=finish_coord_metres,
            fudge_factor=fudge_factor,
        )
        path_there = path_there.downsample_every_n(5)

        visuals.vis_occupancy_grid(
            filepath=f"{log_folder}/occupancy_grid.png",
            occupancy_grid=map1.occupancy_grid,
            metres_per_pixel=map1.metres_per_pixel,
            # start and finish points (in metres)
            points_metres=[start_coord_metres, finish_coord_metres],
            path_metres=path_there.path_metres,
            #path_boxes=boxes
        )

        # Need bounding boxes that encapsulate traversable space
        boxes = map1.path_box(path_there,percent_overlap=70)

        # Create a 2D Quadrotor
        quad = dynamics.Quadrotor2D(0.1)

        # Run Bezier-curve path planning for THERE
        state_vector,control_vector = quad.bezier_trajectory_fitting(path_there.path_metres,boxes)

        # Save to cache
        cache["path_there"] = path_there
        cache["boxes"] = boxes
        cache["state_vector"] = state_vector
        cache["control_vector"] = control_vector

        print(f"Saving cache to {cache_filepath}")
        with open(cache_filepath, "wb") as f:
            pickle.dump(cache, f)

    # Run Bezier-curve path planning
    fitted_path_metres = state_vector[:,[0,2]]

    # Start your timers
    start_time = time.time()

    # Test ILQR controls on dynamic model
    print("Implementing iLQR control")
    N = np.shape(state_vector)[0]
    x_bar, u_bar, Y, y = control.ilqr(state_vector, 
                                    control_vector,
                                    N-1, 
                                    quadrotor=quad, 
                                    Q=100*np.eye(quad.n_dim), 
                                    R=np.eye(quad.m_dim), 
                                    QN=np.diag([1000, 10, 1000, 10, 10, 10]),
                                    eps=1e-5, 
                                    max_iters=1000)
    
    ilqr_states = np.zeros(state_vector.shape)
    ilqr_control = np.zeros(control_vector.shape)
    ilqr_states[0] = np.copy(x_bar[0])
    dt = np.zeros(control_vector.shape[0])
    for k in range(N-1):
        dt[k] = np.linalg.norm(x_bar[k+1,[0,2]]-x_bar[k,[0,2]])/np.linalg.norm(x_bar[k,[1,3]])
        ilqr_control[k] = u_bar[k] + y[k] + Y[k]@(ilqr_states[k]-x_bar[k])
        ilqr_states[k+1] = quad.dynamics_true(ilqr_states[k], ilqr_control[k], dt=dt[k])
        if map1.out_of_bounds(ilqr_states[k,[0]],ilqr_states[k,[2]]) or map1.does_point_hit_boundary(ilqr_states[k,[0]],ilqr_states[k,[2]]):
            ilqr_states = ilqr_states[:k]
            break
    iLQR_follow_path_metres = ilqr_states[:,[0,2]]
    iLQR_fitted_path_metres = x_bar[:,[0,2]]

    # Visualize the occupancy grid with OL trajectory
    visuals.vis_occupancy_grid(
        filepath=f"{log_folder}/iLQR_Control_There.png",
        occupancy_grid=map1.occupancy_grid,
        metres_per_pixel=map1.metres_per_pixel,
        # start and finish points (in metres)
        points_metres=[
            start_coord_metres,
            finish_coord_metres
        ],
        path_metres=path_there.path_metres,
        path2_metres=iLQR_fitted_path_metres,
        path3_metres=iLQR_follow_path_metres,
        plot_coordinates=True,
        path_boxes=boxes
    )

    # --------------------------------
    # Let's go BACK AGAIN!

    # Generate a path from start to finish
    finish_coord_metres = start_coord_metres
    start_coord_metres = np.copy(ilqr_states[-1,[0,2]])
    path_back = map1.path_a_to_b_metres(
        a_coord_metres=start_coord_metres,
        b_coord_metres=finish_coord_metres,
        fudge_factor=fudge_factor,
    )

    # Which must be downsampled aggressively
    #path1 = path1.downsample_to_average_adjacent_distance_metres(0.2)
    path_back = path_back.downsample_every_n(5)

    # Run Bezier-curve path planning
    print("Generating Bezier-Curve path")
    state_vector,control_vector = quad.bezier_trajectory_fitting(path_back.path_metres,boxes[::-1])
    fitted_path_metres = state_vector[:,[0,2]]

    # Test ILQR controls on dynamic model
    print("Implementing iLQR control")
    N = np.shape(state_vector)[0]
    x_bar, u_bar, Y, y = control.ilqr(state_vector, 
                                    control_vector,
                                    N-1, 
                                    quadrotor=quad, 
                                    Q=100*np.eye(quad.n_dim), 
                                    R=np.eye(quad.m_dim), 
                                    QN=np.diag([1000, 10, 1000, 10, 10, 10]),
                                    eps=1e-5, 
                                    max_iters=1000)
    
    ilqr_states_back = np.zeros(state_vector.shape)
    ilqr_control_back = np.zeros(control_vector.shape)
    ilqr_states_back[0] = np.copy(x_bar[0])
    dt_back = np.zeros(control_vector.shape[0])
    for k in range(N-1):
        dt_back[k] = np.linalg.norm(x_bar[k+1,[0,2]]-x_bar[k,[0,2]])/np.linalg.norm(x_bar[k,[1,3]])
        ilqr_control_back[k] = u_bar[k] + y[k] + Y[k]@(ilqr_states_back[k]-x_bar[k])
        ilqr_states_back[k+1] = quad.dynamics_true(ilqr_states_back[k], ilqr_control_back[k], dt=dt_back[k])
        if map1.out_of_bounds(ilqr_states_back[k,[0]],ilqr_states_back[k,[2]]) or map1.does_point_hit_boundary(ilqr_states_back[k,[0]],ilqr_states_back[k,[2]]):
            ilqr_states_back = ilqr_states_back[:k]
            break
    iLQR_follow_path_metres = ilqr_states_back[:,[0,2]]
    iLQR_fitted_path_metres_back = x_bar[:,[0,2]]

    # report time and control trajectories
    print(f"Time to compute: {time.time()-start_time}")
    print(f"Time in simulation: {np.sum(dt)+np.sum(dt_back)}")
    visuals2.state_trajectory_analysis(
        filepath=f"{log_folder}/iLQR_Control_Back_Analysis.png",
        state_trajectory=np.vstack([ilqr_states,ilqr_states_back]),
    )
    visuals2.control_trajectory_analysis(
        filepath=f"{log_folder}/iLQR_Control_Back_Analysis.png",
        control_trajectory=np.vstack([ilqr_control,ilqr_control_back]),
    )

    # Visualize the occupancy grid with OL trajectory
    visuals.vis_occupancy_grid(
        filepath=f"{log_folder}/iLQR_Control_Back.png",
        occupancy_grid=map1.occupancy_grid,
        metres_per_pixel=map1.metres_per_pixel,
        # start and finish points (in metres)
        points_metres=[
            start_coord_metres,
            finish_coord_metres
        ],
        path_metres=path_there.path_metres,
        path2_metres=iLQR_fitted_path_metres_back,
        path3_metres=iLQR_follow_path_metres,
        plot_coordinates=True,
        path_boxes=boxes
    )

    # Control Effort:
    control_effort = np.vstack([ilqr_control,ilqr_control_back])
    with open(f"{log_folder}/OL_Control_Effort.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in control_effort:
            csvwriter.writerow(row)

    visuals2.plot_experiment_video(filepath=f"{log_folder}/iLQR_Control.mp4",
                                    map=map1,
                                    start_point=start_coord_metres,
                                    finish_point=finish_coord_metres,
                                    paths=[
                                        {"path":Path(iLQR_fitted_path_metres),"color":"green"},
                                        {"path":Path(iLQR_fitted_path_metres_back),"color":"purple"},
                                        {"path":Path(state_vector[:,[0,2]]),"color":"red"}
                                    ],
                                    simulation_dts=np.hstack([dt,dt_back]),
                                    state_trajectory=np.vstack([ilqr_states,ilqr_states_back]),
                                    control_trajectory=np.vstack([ilqr_control,ilqr_control_back]))