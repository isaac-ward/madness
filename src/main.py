import mapping 
import planning
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
    occupancy_grid = mapping.load_map_file_as_occupancy_grid(
        filepath=f"{utils.get_assets_dir()}/{filename}",
        metres_per_pixel=metres_per_pixel
    )
    # Downscale to allow for easier path planning computation
    scale_factor = 0.2
    occupancy_grid = mapping.reduce_occupancy_grid_resolution(
        occupancy_grid, 
        scale_factor=scale_factor
    )
    # Note that the metres per pixel has now changed
    metres_per_pixel /= scale_factor
    
    # --------------------------------
    # there 

    # Generate a path from start to finish
    path_metres = planning.compute_path_over_occupancy_grid(
        occupancy_grid=occupancy_grid,
        metres_per_pixel=metres_per_pixel,
        start_coord_metres=start_coord_metres,
        finish_coord_metres=finish_coord_metres,
        agent_radius_metres=globals.AGENT_RADIUS_METRES
    )

    # Visualize the occupancy grid with a few points marked
    visuals.vis_occupancy_grid(
        filepath=f"{log_folder}/occupancy_grid.png",
        occupancy_grid=occupancy_grid,
        metres_per_pixel=metres_per_pixel,
        # start and finish points (in metres)
        points_metres=[
            start_coord_metres,
            finish_coord_metres
        ],
        path_metres=path_metres,
        plot_coordinates=True
    )

    # TODO
    # Some sort of control loop 
    control.optimal_control(path_metres)

    # Oh sick I got a sample!
    
    # --------------------------------
    # and back again

    # TODO