import os 

def dynamics_true(x, u):
    """
    Compute the true next state (with disturbances)
    """

    return next_u

def dynamics_model(x, u):
    """
    Compute next state given current state and action
    based on our model of the world (no disturbances)
    """

    # TODO

    return next_u

def opt_control(path):

    # TODO: get nominal controls and trajectories through state space

    # TODO: do (i)LQR to produce control sequence with dynamics model

    # TODO: execute control (on actual simuation) with true dynamics

    pass


if __name__ == "__main__":
    # There()

    # IRW - load image map and return as occupancy grid (2d)
    occ_grid = get_occ_grid()

    done = False
    while not done:

        # MWP - A*, RRT, RRT* w/ bounds
        path = pathPlanning(start, finish, occ_grid)

        opt_control(path)

        # TODO: some done condition