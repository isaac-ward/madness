import numpy as np
import random

np.random.seed(0)
random.seed(0)

METRES_PER_PIXEL  = 0.01

DRONE_HALF_LENGTH = 0.2 # m
DRONE_MASS = 2.5 # kg
g = 9.81 # m/s^2
#MAX_THRUST_PER_PROP = 0.75 * DRONE_MASS * g  # total thrust-to-weight ratio = 1.5
MAX_THRUST_PER_PROP = 3 * DRONE_MASS * g # Ike's tasty edit
REACHED_SAMPLE_REGION_THRESHOLD = 0.5 # m
REACHED_ENTRY_POINT_THRESHOLD   = 0.25 # m

# For testing disturbances
disturbance_config = "both-minor"
if disturbance_config == "both-major":
    DISTURBANCE_VARIANCE_ROTORS = MAX_THRUST_PER_PROP * 0.33
    DISTURBANCE_VELOCITY_VARIANCE_WIND = 0.0003 # m/s
    DISTURBANCE_ANGULAR_VELOCITY_VARIANCE_WIND = 0.0001 # rad/s
if disturbance_config == "both-minor":
    DISTURBANCE_VARIANCE_ROTORS = MAX_THRUST_PER_PROP * 1
    DISTURBANCE_VELOCITY_VARIANCE_WIND = 0.00005 # m/s
    DISTURBANCE_ANGULAR_VELOCITY_VARIANCE_WIND = 0.00005 # rad/s
elif disturbance_config == "wind":
    DISTURBANCE_VARIANCE_ROTORS = MAX_THRUST_PER_PROP * 0.
    DISTURBANCE_VELOCITY_VARIANCE_WIND = 0.00005 # m/s
    DISTURBANCE_ANGULAR_VELOCITY_VARIANCE_WIND = 0.00005 # rad/s
elif disturbance_config == "rotor":
    DISTURBANCE_VARIANCE_ROTORS = MAX_THRUST_PER_PROP * 1
    DISTURBANCE_VELOCITY_VARIANCE_WIND = 0. # m/s
    DISTURBANCE_ANGULAR_VELOCITY_VARIANCE_WIND = 0. # rad/s
elif disturbance_config == "none":
    DISTURBANCE_VARIANCE_ROTORS = MAX_THRUST_PER_PROP * 0.
    DISTURBANCE_VELOCITY_VARIANCE_WIND = 0. # m/s
    DISTURBANCE_ANGULAR_VELOCITY_VARIANCE_WIND = 0. # rad/s
else:
    raise ValueError(f"Unknown disturbance configuration: {disturbance_config}")

print(f"Disturbance variances:")
print(f"\tRotor: {DISTURBANCE_VARIANCE_ROTORS}")
print(f"\tWind velocity: {DISTURBANCE_VELOCITY_VARIANCE_WIND}")
print(f"\tWind angular velocity: {DISTURBANCE_ANGULAR_VELOCITY_VARIANCE_WIND}")

print(f"Thrust allowed per prop: {MAX_THRUST_PER_PROP}")

# TODO should define goal states (i.e. set velocities and angles)
MAP_CONFIGS = {
    "3x7": {
        "filename": "3x7.png",
        "metres_per_pixel": METRES_PER_PIXEL,
        "start_coord_metres": (1.5, 2),   # x, y
        "finish_coord_metres": (6.5, 2),
        "fudge_factor": 1.6,
    },
    "3x28": {
        "filename": "3x28.png",
        "metres_per_pixel": METRES_PER_PIXEL,
        "start_coord_metres": (1.5, 2),
        "finish_coord_metres": (27.5, 2),
        "fudge_factor": 1.6,
    },
    "downup": {
        "filename": "downup.png",
        "metres_per_pixel": METRES_PER_PIXEL,
        "start_coord_metres": (1.5, 8),
        "finish_coord_metres": (27.5, 8),
        "fudge_factor": 1.6,
    },
    "downup-o": {
        "filename": "downup-o.png",
        "metres_per_pixel": METRES_PER_PIXEL,
        "start_coord_metres": (1.5, 8),
        "finish_coord_metres": (27.5, 8),
        "fudge_factor": 2.2,
    },
    "grapevine": {
        "filename": "grapevine2x.png",
        "metres_per_pixel": METRES_PER_PIXEL,
        # If you have a map with pixels, this is how you can do the start end points,
        # note that x value is right from the left of the image, and the y value is 
        # up from the bottom of the image
        "start_coord_metres": 2*np.array((300, 3000 - 300)) * METRES_PER_PIXEL,
        "finish_coord_metres": 2*np.array((1500, 3000 - 2100)) * METRES_PER_PIXEL,
        "fudge_factor": 2.6,
    },
}