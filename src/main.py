import mapping 
import planning
import control 
import dynamics
import visuals
import utils 

if __name__ == "__main__":

    # Get the map that we'll be using
    metres_per_pixel = 0.01
    occupancy_grid = mapping.load_map_file_as_occupancy_grid(
        filepath=f"{utils.get_assets_dir()}/3x7.png",
        metres_per_pixel=metres_per_pixel
    )
    # Downscale to allow for easier path planning computation
    # 20% of the original resolution
    scale_factor = 0.2
    occupancy_grid = mapping.reduce_occupancy_grid_resolution(
        occupancy_grid, 
        scale_factor=scale_factor
    )
    # Note that the metres per pixel has now changed
    metres_per_pixel /= scale_factor

    # Visualize the occupancy grid with a few points marked
    visuals.vis_occupancy_grid(
        occupancy_grid=occupancy_grid,
        metres_per_pixel=metres_per_pixel,
        # start and finish points (in metres)
        points_metres=[(0, 0), (1, 2), (6, 2)],
        plot_coordinates=True
    )
    
    # --------------------------------
    # there 

    # Generate a path from start to finish
    path = planning.compute_path_over_occupancy_grid(
        occupancy_grid=occupancy_grid,
        start_metres=(0, 0),
        finish_metres=(0, 7),
        agent_radius_metres=0.5
    )

    # TODO
    # Some sort of control loop 
    control.optimal_control(path)
    
    # --------------------------------
    # and back again

    # TODO