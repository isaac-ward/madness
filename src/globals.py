METRES_PER_PIXEL  = 0.01

DRONE_HALF_LENGTH = 0.2 # m
DRONE_MASS = 2.5 # kg
g = 9.81 # m/s^2
#MAX_THRUST_PER_PROP = 0.75 * DRONE_MASS * g  # total thrust-to-weight ratio = 1.5
MAX_THRUST_PER_PROP = 3 * DRONE_MASS * g # Ike's tasty edit
REACHED_SAMPLE_REGION_THRESHOLD = 0.5 # m
REACHED_ENTRY_POINT_THRESHOLD   = 0.25 # m

# For testing disturbances
DISTURBANCE_VARIANCE_ROTORS = MAX_THRUST_PER_PROP * 0 #0.33

# TODO should define goal states (i.e. set velocities and angles)
MAP_CONFIGS = {
    "3x7": {
        "filename": "3x7.png",
        "metres_per_pixel": METRES_PER_PIXEL,
        "start_coord_metres": (1.5, 2),   # x, y
        "finish_coord_metres": (6.5, 2),
    },
    "3x28": {
        "filename": "3x28.png",
        "metres_per_pixel": METRES_PER_PIXEL,
        "start_coord_metres": (1.5, 2),
        "finish_coord_metres": (27.5, 2),
    },
    "downup": {
        "filename": "downup.png",
        "metres_per_pixel": METRES_PER_PIXEL,
        "start_coord_metres": (1.5, 8),
        "finish_coord_metres": (27.5, 8),
    },
    "downup-o": {
        "filename": "downup-o.png",
        "metres_per_pixel": METRES_PER_PIXEL,
        "start_coord_metres": (1.5, 8),
        "finish_coord_metres": (27.5, 8),
    },
}