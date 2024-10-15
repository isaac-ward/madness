import numpy as np
from mapping import Map
from dynamics_jax import DynamicsQuadcopter3D

class SDF_Types:
    sphere=0
    cube=1

class Environment_SDF:
    """
    Class to hold all SDFs that make up environment representation
    """
    def __init__(
            self,
            dynamics:DynamicsQuadcopter3D,
    ):
        # Class variables
        self.sdf_list = []          # List of SDF objects
        self.dynamics = dynamics    # Quadrotor dynamics class
    
    def add_sdf(
            self,
            sdf
    ):
        """
        New SDF object to add to environment representation

        Parameters
        ----------
        sdf: _SDF
            Some type of _SDF class object describing an SDF
        """
        self.sdf_list.append(sdf)

    def characterize_env_with_spheres_perturbations(
            self,
            start_point_meters:np.ndarray,
            end_point_meters:np.ndarray,
            path_xyz: np.ndarray,
            map_env: Map,
            max_spheres=200,
            randomness_deg=15
    ):
        """
        Characterize an environment with sphere SDFs. Sphere expansion is performed using a
        perturbation technique accounting for the direction of collisions, xyzpath, and randomness.

        Parameters
        ----------
        start_point_meters: numpy.ndarray
            The starting position in the environment
        end_point_meters: numpy.ndarray
            The goal position in the environment
        path_xyz: numpy.ndarray
            The xyz path from the starting to goal position
        map_env: Map
            The map resentation of the environment
        max_spheres: int
            The maximum number of spheres allowed to characterize the environment. Default 200
        randomness_deg: int
            The range of randomness desired in the perturbed direction (degrees)
        """
        # Make collision radius
        collision_radius_metres = self.dynamics.diameter/2

        # Create first SDF about start point
        self.add_sdf(Sphere_SDF.get_optimal_sdf(
            center_metres_xyz=start_point_meters, 
            collision_radius_metres=collision_radius_metres, 
            mapping=map_env
        ))

        # Search and add more SDFs
        new_start_point = np.zeros(3)
        for _i in range(max_spheres-1):
            print(_i)
            print(new_start_point)
            # Check if new sdf is needed
            search_complete = self.sdf_list[-1].points_within_sphere(end_point_meters)
            if search_complete:
                # If search complete, add one final sphere to end point
                final_sphere = Sphere_SDF.get_optimal_sdf(end_point_meters,collision_radius_metres,map_env)
                self.add_sdf(final_sphere)
                break

            # Get new center
            new_start_point = self.sdf_list[-1].get_next_sdf_center_perturb(path_xyz,randomness_deg)
            
            # Build next sdf
            building = True
            back_up = 1
            while(building):
                # Build next sphere
                next_sphere = Sphere_SDF.get_optimal_sdf(new_start_point,collision_radius_metres,map_env)

                # Try to refine sphere further
                next_sphere = next_sphere.sphere_refinement(
                    collision_radius_metres=collision_radius_metres,
                    mapping=map_env
                )

                # Check if new sdf has volume
                if next_sphere.radius_voxels <= 1:
                    # Try new point
                    new_start_point = self.sdf_list[-back_up].get_next_sdf_center_perturb(path_xyz,randomness_deg)
                    # If this doesn't work, let's go backward to a previous SDF
                    if back_up <= len(self.sdf_list):
                        back_up += 1
                    else:
                        raise Exception("Valid SDF representation not found")
                else:
                    # Add sphere to SDF list
                    self.add_sdf(next_sphere)
                    building=False
    
    def characterize_env_with_spheres_xyzpath(
            self,
            start_point_meters:np.ndarray,
            end_point_meters:np.ndarray,
            path_xyz: np.ndarray,
            map_env: Map,
            max_spheres=200,
    ):
        """
        Characterize an environment with sphere SDFs. Sphere SDFs will track xyzpath
        exactly.

        Parameters
        ----------
        start_point_meters: numpy.ndarray
            The starting position in the environment
        end_point_meters: numpy.ndarray
            The goal position in the environment
        path_xyz: numpy.ndarray
            The xyz path from the starting to goal position
        map_env: Map
            The map resentation of the environment
        max_spheres: int
            The maximum number of spheres allowed to characterize the environment. Default 200
        """
        # Make collision radius
        collision_radius_metres = self.dynamics.diameter/2

        # Create first SDF about start point
        self.add_sdf(Sphere_SDF.get_optimal_sdf(
            center_metres_xyz=start_point_meters, 
            collision_radius_metres=collision_radius_metres, 
            mapping=map_env
        ))

        # Search and add more SDFs
        new_start_point = np.zeros(3)
        for _i in range(max_spheres-1):
            print(_i)
            print(new_start_point)
            
            # Check if new sdf is needed
            search_complete = self.sdf_list[-1].points_within_sphere(end_point_meters)
            if search_complete:
                # If search complete, add one final sphere to end point
                final_sphere = Sphere_SDF.get_optimal_sdf(end_point_meters,collision_radius_metres,map_env)

                # Attempt to refine final sphere SDF
                final_sphere = final_sphere.sphere_refinement(
                        collision_radius_metres=collision_radius_metres,
                        mapping=map_env
                    )
                
                # Add and end operations
                self.add_sdf(final_sphere)
                break

            # Get new center
            new_start_point = self.sdf_list[-1].get_next_sdf_center_from_xyzpath(path_xyz)
            
            # Build next sphere
            next_sphere = Sphere_SDF.get_optimal_sdf(new_start_point,collision_radius_metres,map_env)

            # Try to refine sphere further TODO Debug stuck loop
            next_sphere = next_sphere.sphere_refinement(
                collision_radius_metres=collision_radius_metres,
                mapping=map_env
            )

            # Add sphere to SDF list
            self.add_sdf(next_sphere)

class Sphere_SDF:
    """
    Class to hold a sphere Signed Distance Function (SDF) object
    """
    def __init__(
        self,
        center_metres_xyz,
        radius_metres,
        center_voxel_coords,
        radius_voxels,
        interior_voxel_coords,
        interior_metre_coords,
        nearest_collisions_voxel,
        nearest_collisions_metres,
        voxel_per_x_metres,
    ):
        # Class variables
        self.sdf_type = SDF_Types.sphere                            # Type of SDF
        self.center_metres_xyz = center_metres_xyz                  # Sphere center in meters
        self.radius_metres = radius_metres                          # Sphere radius in meters
        self.center_voxel_coords = center_voxel_coords              # Sphere center in voxel coords
        self.radius_voxels = radius_voxels                          # Sphere radius in voxel coords
        self.interior_voxel_coords = interior_voxel_coords          # Voxel coords encompassed by sphere
        self.interior_metre_coords = interior_metre_coords          # Meter coords encompassed by sphere
        self.nearest_collisions_voxel = nearest_collisions_voxel    # Array of closest collisions in voxel coords
        self.nearest_collisions_metres = nearest_collisions_metres  # Array of closest collisions in meters
        self.voxel_per_x_metres = voxel_per_x_metres                # voxel per meters

    @staticmethod
    def find_max_non_collision_radius(
        center_metres_xyz:np.ndarray,
        collision_radius_metres:float,
        maximum_radius_metres:float,
        mapping:Map,
    ):
        """
        Finds the maximum non-collision sphere radius about a point given the environment

        Parameters
        ----------
        center_metres_xyz: numpy.ndarray
            The location of the center of the sphere in meters
        collision_radius_metres: float
            The collision radius of the quadrotor in meters
        maximum_radius_metres: float
            The maximum allowable radius of the sphere SDF in meters
        mapping: Map
            The map resentation of the environment

        Returns
        -------
        radius_metres: float
            The maximum non-collision sphere radius
        voxel_collisions: numpy.ndarray
            The voxel coordinates corresponding to those that would be in collision with the 
            sphere if its radius was expanded any further
        """
        # Begin by converting from metres to voxel
        center_voxel_xyz = mapping.metres_to_voxel_coords(center_metres_xyz)

        # Start with a zero radius (just the start point voxel)
        radius_voxels = 0
        in_collision = False
        while radius_voxels < maximum_radius_metres and not in_collision:
            # Increase the radius by one voxel
            radius_voxels += 1

            # Get all voxels within the current radius
            voxels_to_check = mapping.get_voxels_within_radius(center_voxel_xyz,radius_voxels)

            # Check if any of the voxels are occupied
            collision_bools = mapping.batch_is_collision_voxel_coords(voxels_to_check,collision_radius_metres)
            in_collision = any(collision_bools)
        
        # Get all voxel collisions
        voxel_collisions = voxels_to_check[collision_bools]

        # Reduce the radius by one to get the maximum non-collision radius
        radius_voxels -= 1

        # Convert the radius back to metres
        radius_metres = radius_voxels * mapping.voxel_per_x_metres

        return radius_metres,voxel_collisions
    
    @staticmethod
    def get_optimal_sdf(
        center_metres_xyz:np.ndarray,
        collision_radius_metres:float,
        mapping:Map,
        maximum_radius_metres=100.
    ):
        """
        Build a sphere SDF object given a center point and an environment

        Parameters
        ----------
        center_metres_xyz: numpy.ndarray
            The location of the center of the sphere in meters
        collision_radius_metres: float
            The collision radius of the quadrotor in meters
        mapping: Map
            The map resentation of the environment
        maximum_radius_metres: float
            The maximum allowable radius of the sphere SDF in meters. Default to 100m

        Returns
        -------
        sphere_sdf: Sphere_SDF
            The largest sphere SDF at this center given as a Sphere_SDF class object
        """
        # Get the maximum radius (in metres)
        radius_metres,voxel_collisions = Sphere_SDF.find_max_non_collision_radius(
            center_metres_xyz=center_metres_xyz,
            collision_radius_metres=collision_radius_metres,
            maximum_radius_metres=maximum_radius_metres,
            mapping=mapping
        )
        metre_collisions = np.apply_along_axis(mapping.voxel_coords_to_metres, 1, voxel_collisions)


        # To get the interior voxels, create a meshgrid at the correct resolution,
        # the same size as the map, and then check if each point is within the sphere
        center_voxel_coords = mapping.metres_to_voxel_coords(center_metres_xyz)
        radius_voxels = radius_metres / mapping.voxel_per_x_metres
        interior_voxel_coords = mapping.get_voxels_within_radius(
            center_voxel_coords,
            radius_voxels,
        )
        interior_metre_coords = np.zeros(np.shape(interior_voxel_coords))
        interior_metre_coords = np.apply_along_axis(mapping.voxel_coords_to_metres, 1, interior_voxel_coords)
        
        # Create the SDF object
        sphere_sdf = Sphere_SDF(
            center_metres_xyz=center_metres_xyz,
            radius_metres=radius_metres,
            center_voxel_coords=center_voxel_coords,
            radius_voxels=radius_voxels,
            interior_voxel_coords=interior_voxel_coords,
            interior_metre_coords=interior_metre_coords,
            nearest_collisions_voxel=voxel_collisions,
            nearest_collisions_metres=metre_collisions,
            voxel_per_x_metres=mapping.voxel_per_x_metres,
        )

        return sphere_sdf
    
    def __str__(
        self
    ):
        s = "SDF:\n"
        s += f"  center_metres_xyz: {self.center_metres_xyz}\n"
        s += f"  radius_metres: {self.radius_metres}\n"
        s += f"  center_voxel_coords: {self.center_voxel_coords}\n"
        s += f"  radius_voxels: {self.radius_voxels}\n"
        s += f"  num_interior_voxels: {len(self.interior_voxel_coords)}\n"
        s += f"  voxel_per_x_metres: {self.voxel_per_x_metres}\n"
        return s

    def points_within_sphere(
            self,
            points:np.ndarray
    ):
        """
        Given a list of points, return a binary array describing if the point is within (1) or outside (0) 
        the sphere SDF

        Parameters
        ----------
        points: numpy.ndarray
            Array containing all points to check
        
        Returns
        -------
        within_sphere: numpy.ndarray
            Array of 1s and 0s describing if the corresponding point is within the sphere SDF volume
        """
        try:
            # Calculate squared distance from center for each vector
            distances_squared = np.sum((points - self.center_metres_xyz) ** 2, axis=1)
            
            # Compare distances to the squared radius
            within_sphere = (distances_squared <= self.radius_metres ** 2).astype(int)
        
        except:
            # Calculate squared distance from center for each vector
            distances_squared = np.sum((points - self.center_metres_xyz) ** 2)
            
            # Compare distances to the squared radius
            within_sphere = (distances_squared <= self.radius_metres ** 2).astype(int)
        
        return within_sphere

    def lineseg_sphere_intersect(
            self,
            point1:np.ndarray,
            point2:np.ndarray
    ):
        """
        Find the location where a line segment intersects with a sphere. If intersects twice,
        returns point closer to second input line segment point

        Parameters
        ----------
        point1: numpy.ndarray
            First point describing the line segment
        point2: numpy.ndarray
            Second point describing the line segment
        
        Returns
        -------
        intersection: numpy.ndarray
            The point of intersection between the sphere and the line segment. If no intersection, 
            is None. If 2 intersections, pick intersection closer to point2
        """
        # Get sphere variables
        sphere_center = self.center_metres_xyz
        sphere_radius = self.radius_metres
        
        # Compute the quadratic coefficients
        a = np.dot(point2 - point1, point2 - point1)
        b = 2 * np.dot(point2 - point1, point1 - sphere_center)
        c = np.dot(point1 - sphere_center, point1 - sphere_center) - sphere_radius**2
        
        # Solve the quadratic equation: at^2 + bt + c = 0
        discriminant = b**2 - 4 * a * c

        # Create variable to hold intersection value
        intersection = None
        
        # If no intersections, return None
        if discriminant < 0:
            return intersection
        
        # Two solutions for t
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        # Calculate the intersection points
        if 0 <= t1 <= 1:
            intersection1 = point1 + t1 * (point2 - point1)
        else:
            intersection1 = None
        if 0 <= t2 <= 1:
            intersection2 = point1 + t2 * (point2 - point1)
        else:
            intersection2 = None
        
        # Choose the intersection point closer to P2
        if intersection1 is not None and intersection2 is not None:
            # Calculate distances to P2
            dist1 = np.linalg.norm(intersection1 - point2)
            dist2 = np.linalg.norm(intersection2 - point2)
            
            # Return the point closer to P2
            if dist1 < dist2:
                intersection = intersection1
            else:
                intersection = intersection2
        # or choose the intersection that exists
        elif intersection1 is not None:
            intersection = intersection1
        elif intersection2 is not None:
            intersection = intersection2
        
        return intersection
    
    def get_next_sdf_center_from_xyzpath(
            self,
            astar:np.ndarray
    ):
        """
        Given the A* path, pick a point to build a new sphere SDF intersecting with current sphere SDF

        Parameters
        ----------
        astar: numpy.ndarray
            Array containing A* path
        
        Returns
        -------
        new_sdf_center: numpy.ndarray
            Array containing suggested location (meters) for next sphere SDF center
        search_complete: Boolean
            True if current SDF contains endpoint of A*, False if otherwise
        """
        # Return variables
        new_sdf_center = np.zeros(3)

        # Check what elements of A* path are within the sphere
        astar_in_sphere = self.points_within_sphere(astar)
        print(astar_in_sphere)
        
        # Get last switch from 1 to 0
        last_in = np.zeros(3)
        last_out = np.zeros(3)
        for _i in range(len(astar_in_sphere) - 1, 0, -1):
            if astar_in_sphere[_i] == 0 and astar_in_sphere[_i-1] == 1:
                last_in = np.copy(astar[_i-1])
                last_out = np.copy(astar[_i])

        # Get intersection between sphere and A* line
        new_sdf_center = self.lineseg_sphere_intersect(last_in,last_out)

        return new_sdf_center
    
    def get_average_collision_dir(
            self,
    ):
        """
        Get the average direction of the nearest collisions

        Returns
        -------
        average_direction: numpy.ndarray
            Unit vector in average direction of collisions
        """
        # Turn the collision points into vectors
        vectors = self.nearest_collisions_voxel - self.center_voxel_coords

        # Average the collision vectors
        average_vector = np.mean(vectors, axis=0)

        # Normalize average vector to get direction (unit vector)
        average_direction = average_vector / np.linalg.norm(average_vector)

        return average_direction
    
    def get_next_sdf_center_perturb(
            self,
            xyzpath:np.ndarray,
            randomness_deg:float,
    ):
        """
        Get the next recommended sphere SDF center based off of perturbed directions accounting for
        - Nearest environment collisions
        - An xyz Path from start to goal points
        - Randomness

        Parameters
        ----------
        path_xyz: numpy.ndarray
            The xyz path from the starting to goal position
        randomness_deg: float
            The range of randomness desired in the perturbed direction (degrees)

        Returns
        -------
        new_center_metres: numpy.ndarray
            The new center point for a sphere SDF to be build (meters)
        """
        ## Collision contribution ------------------------------------------------
        # Get average direction of collision
        average_direction = self.get_average_collision_dir()

        # Get opposite direction
        opposite_direction = -average_direction

        ## Randomness contribution ------------------------------------------------
        # Generate a random perturbation
        random_perturb = np.random.uniform(-randomness_deg,randomness_deg,3)
        random_perturb_rad = np.radians(random_perturb)

        # Normalize perturbation vector
        normalize_perturbation_vect = random_perturb_rad
        if randomness_deg != 0.:
            normalize_perturbation_vect /= np.linalg.norm(random_perturb_rad)

        # Add the random perturbation to the opposite direction
        perturbed_direction = opposite_direction + normalize_perturbation_vect

        ## xyzpath contribution ------------------------------------------------
        # Get closest point to sphere center
        distances = np.linalg.norm(xyzpath - self.center_metres_xyz, axis=1)
        closest_index = np.argmin(distances)

        try:
            # Get direction to next position
            xyzdir = xyzpath[closest_index+1] - xyzpath[closest_index]
            normalize_xyzdir = xyzdir / np.linalg.norm(xyzdir)
        except:
            # Get direction toward end
            xyzdir = xyzpath[closest_index] - self.center_metres_xyz
            normalize_xyzdir = xyzdir / np.linalg.norm(xyzdir)
        
        # Add xyzpath perturbation to perturbed direction
        perturbed_direction += normalize_xyzdir

        # Normalize the final vector to ensure it points in the correct direction
        perturbed_direction /= np.linalg.norm(perturbed_direction)

        # Scale by the radius and calculate the opposite point
        new_center_voxel_approx = self.center_voxel_coords + perturbed_direction * self.radius_voxels

        # Get the closest point we know to be unoccupied
        distances = np.linalg.norm(self.interior_voxel_coords - new_center_voxel_approx, axis=1)
        closest_index = np.argmin(distances)
        new_center_voxel = self.interior_voxel_coords[closest_index]

        # Convert to meters
        new_center_metres = new_center_voxel * self.voxel_per_x_metres

        return new_center_metres

    def sphere_refinement(
            self,
            collision_radius_metres:float,
            mapping:Map,
            max_refinements=10
    ):
        """
        Improve sphere SDF representation by attempting to locate a larger sphere that encompasses
        the current SDF

        Parameters
        ----------
        collision_radius_metres: float
            The collision radius of the quadrotor in meters
        mapping: Map
            The map resentation of the environment
        max_refinements: int
            The maximum allowable improvement iterations. Default to 10
        """
        # Initial sdf as current sdf
        sdf = self

        for _i in range(max_refinements):
            # Get average direction of collision
            average_direction = self.get_average_collision_dir()

            # Search for improvement in opposite direction
            opposite_direction = -average_direction

            # Scale by the radius and calculate the opposite point
            new_center_voxel_approx = sdf.center_voxel_coords + opposite_direction * sdf.radius_voxels

            # Get the closest point we know to be unoccupied
            distances = np.linalg.norm(sdf.interior_voxel_coords - new_center_voxel_approx, axis=1)
            closest_index = np.argmin(distances)
            new_center_voxel = sdf.interior_voxel_coords[closest_index]

            # Build new SDF
            new_sdf = sdf.get_optimal_sdf(
                center_metres_xyz=new_center_voxel * sdf.voxel_per_x_metres,
                collision_radius_metres=collision_radius_metres,
                mapping=mapping
            )

            # New SDF bigger than current?
            more_volume = np.shape(new_sdf.interior_voxel_coords)[0] > np.shape(sdf.interior_voxel_coords)[0]

            # New SDF contains current SDF?
            sdf_contained = np.all(np.all(np.isin(sdf.interior_voxel_coords, new_sdf.interior_voxel_coords), axis=1))

            # If not larger or containing original SDF, done
            if not more_volume or not sdf_contained:
                break
            else:
                sdf = new_sdf
        
        return sdf