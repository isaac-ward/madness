import jax.numpy as jnp

class SignedDistanceField:
    """
    The signed distance field (SDF) class produces environment representations
    of the SDF type, that is: a voxel map where at each cube we have the shortest
    point to an obstacle. We also represent a notion of inside and outside
    a given boundary - shortest distance values are + inside, - outside

    See Figure 27A https://arxiv.org/pdf/2106.09125
    """

    def __init__(
        self,
        occupancy_grid,
    ):
        """
        Compute the SDF from the occupancy grid
        """
        
        # Naive approach is to do Dijkstra's algorithm from each point

        pass


    def get(
        self, 
    ):
        """
        Return the SDF
        """
        return 

