from dynamics import DynamicsQuadcopter3D
from map import Map

def get_standard_dynamics_quadcopter_3d():
    return DynamicsQuadcopter3D(
        diameter=0.2,
        mass=2.5,
        Ix=0.5,
        Iy=0.5,
        Iz=0.3,
        # +z is down
        g=+9.81, 
        # higher makes it easier to roll and pitch
        thrust_coef=5,      
        # higher makes it easier to yaw
        drag_yaw_coef=5,   
        # higher values lower the max velocity
        drag_force_coef=5,   
        dt=0.025,
    )

def get_standard_map():
    return Map(
        map_filepath="maps/empty.csv",
        voxel_per_x_metres=0.25,
        extents_metres_xyz=[
            [0, 30], 
            [0, 30], 
            [0, 30]
        ],
    )