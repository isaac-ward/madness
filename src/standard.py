from dynamics import DynamicsQuadcopter3D 
from dynamics_linear import DynamicsQuadcopter3DLinear
from mapping import Map

def _dynamics_arguments():
    return {
        "diameter": 0.5,
        "mass": 2.5,
        "Ix": 0.5,
        "Iy": 0.5,
        "Iz": 0.3,
        # +z is down
        "g": +9.81,
        # higher makes it easier to roll and pitch
        "thrust_coef": 5,
        # higher makes it easier to yaw
        "drag_yaw_coef": 5,
        # higher values lower the max velocity
        "drag_force_coef": 5,
        "dt": 0.025,
    }

def get_standard_dynamics_linear():
    return DynamicsQuadcopter3DLinear(**_dynamics_arguments())

def get_standard_dynamics():
    return DynamicsQuadcopter3D(**_dynamics_arguments())

def get_standard_map():
    return Map(
        map_filepath=None,
        voxel_per_x_metres=0.25,
        extents_metres_xyz=[
            [0, 30], 
            [0, 30], 
            [0, 30]
        ],
    )

def get_28x28x28_at_111():
    return Map(
        map_filepath="/workspace/assets/maps/28x28x28_at_1-1-1.obj",
        voxel_per_x_metres=1,
        extents_metres_xyz=[
            [0, 30], 
            [0, 30], 
            [0, 30]
        ],
    )

def get_28x28x28_at_111_with_obstacles():
    return Map(
        map_filepath="/workspace/assets/maps/28x28x28_at_1-1-1_with_obstacles.obj",
        voxel_per_x_metres=1, # Any lower and point cloud is to sparse for connectivity
        extents_metres_xyz=[
            [0, 30], 
            [0, 30], 
            [0, 30]
        ],
    )

def get_tunnels():
    return Map(
        map_filepath="/workspace/assets/maps/tunnels.obj",
        voxel_per_x_metres=0.2,
        extents_metres_xyz=[
            [0, 60], 
            [0, 35], 
            [0, 8]
        ],
    )

def get_chamber():
    return Map(
        map_filepath="/workspace/assets/maps/chamber.obj",
        voxel_per_x_metres=1,
        extents_metres_xyz=[
            [0, 70], 
            [0, 45], 
            [0, 12]
        ],
    )   

def get_standard_flow_action_dist_policy_arguments():
    dyn = get_standard_dynamics()
    map_ = get_standard_map()
    return {
        "dynamics": dyn,
        "K": 512,
        "H": 50,
        "lambda_": None,
        "map_": map_,
        "context_input_size": 2*dyn.state_size(),
        "context_output_size": 256,
        "num_flow_layers": 8,
        "learning_rate": 1e-6,
    }