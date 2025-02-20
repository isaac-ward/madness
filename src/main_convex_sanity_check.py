import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import heapq
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d
from utils.general import gradient_log_softmax, log_softmax
import cvxpy as cp
from dynamics_tiny import DynamicsTiny

class World:
    def __init__(self, side_length, middle_obstacle_side_length, agent_radius):
        self.side_length = side_length
        self.middle_obstacle_side_length = middle_obstacle_side_length
        self.agent_radius = agent_radius

        # Create a numpy array of zeros
        self.world = np.zeros((side_length, side_length))

        # 1s around the edge
        self.world[0, :] = 1
        self.world[-1, :] = 1
        self.world[:, 0] = 1
        self.world[:, -1] = 1

        # 1s in the middle
        middle_start = int((side_length - middle_obstacle_side_length) / 2)
        middle_end = middle_start + middle_obstacle_side_length
        self.world[middle_start:middle_end, middle_start:middle_end] = 1

        # 0s are the free space, 1s are the obstacles, now we want
        # to set up 2s for the exclusion radius (the agent has a radius)
        # so we'll set up a dilation using scipy
        exclusion_space = binary_dilation(self.world, iterations=agent_radius)
        # Find where the exclusion space is and the world=1 isn't
        self.world[exclusion_space & (self.world == 0)] = 2

        # Now we want to set up the start and goal positions
        attempts = 1000
        # Need to bnoth be free and separated by some distance
        separation_distance = side_length // 2
        for _ in range(attempts):
            start = np.random.randint(0, side_length, size=2)
            goal = np.random.randint(0, side_length, size=2)
            if self.is_free_and_fits_agent(*start) and self.is_free_and_fits_agent(*goal) and np.linalg.norm(start - goal) > separation_distance:
                self.start = start
                self.goal = goal
                break
        else:
            raise ValueError("Could not find start and goal positions")
    
    def is_free(self, x, y):
        in_bounds = 0 <= x < self.side_length and 0 <= y < self.side_length
        if not in_bounds:
            return False
        else:
            return self.world[x, y] != 1
        
    def is_free_and_fits_agent(self, x, y):
        in_bounds = 0 <= x < self.side_length and 0 <= y < self.side_length
        if not in_bounds:
            return False
        else:
            return self.world[x, y] != 1 and self.world[x, y] != 2
        
    def a_star(self, start, goal):
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_set = []
        heapq.heappush(open_set, (0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == tuple(goal):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(tuple(start))
                return np.array(path[::-1])

            neighbors = [(current[0] + dx, current[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            for neighbor in neighbors:
                if not self.is_free_and_fits_agent(*neighbor):
                    continue
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
class Circle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    @staticmethod
    def get_largest_possible_circle(world, x, y):
        # Start at the largest possible circle and work our way down
        reverse_size_list = list(range(world.side_length))[::-1][:-1]
        #print(reverse_size_list)
        pbar = tqdm(reverse_size_list)
        for radius in pbar:
            pbar.set_description(f"Trying radius {radius} about ({x}, {y})")
            circle = Circle(x, y, radius)
            if circle.is_valid(world):
                return circle
        # Throw an error if we can't find a circle
        raise ValueError(f"Could not find a circle for center ({x}, {y}) with radius > 0")
            
    def is_valid(self, world):
        # Go through all intertnal points and check if they are free
        for x in range(self.x - self.radius, self.x + self.radius + 1):
            for y in range(self.y - self.radius, self.y + self.radius + 1):
                # Don't want every point in the square but in the radius
                if np.linalg.norm([x - self.x, y - self.y]) <= self.radius:
                    if not world.is_free(x, y):
                        return False
        return True
                    
    def is_point_inside(self, x, y):
        return np.linalg.norm([x - self.x, y - self.y]) <= self.radius
                    
    def get_furthest_circle_along_path(self, world, path):
        # A path is a list of points

        # Look at the path iterate througha ll points. Check if the point is in the circle,
        # if we go from it is to it isn't then that is the furthest point
        for i in range(len(path)):
            if self.is_point_inside(*path[i]):
                first_inside_index = i
                break
        for i in range(first_inside_index, len(path)):
            if not self.is_point_inside(*path[i]):
                # Now find the largest circle at the previous point
                return Circle.get_largest_possible_circle(world, *path[i - 1])
    
    def sdf_value(self, x, y):
        # Inside is positive, outside negative
        # 1 is right in the middle,
        # 0 is the edge (at the radius)
        return 1 - np.linalg.norm([x - self.x, y - self.y]) / self.radius

    def sdf_value_cp(self, x, y):
        # Inside is positive, outside negative
        # 1 is right in the middle,
        # 0 is the edge (at the radius)
        return 1 - cp.norm2([x - self.x, y - self.y]) / self.radius

    def __str__(self):
        return f"Circle c=({self.x}, {self.y}) r={self.radius}"

class Plotter:
    @staticmethod
    def plot_world(world, path, path_smooth, circles):
        # Plot the grid world, zeros are white, ones are black, twos are grey - explicitly
        white = np.array([1, 1, 1])
        black = np.array([0, 0, 0])
        grey = np.array([0.5, 0.5, 0.5])
        matrix = np.zeros((world.side_length, world.side_length, 3))
        matrix[world.world == 0] = white
        matrix[world.world == 2] = grey
        matrix[world.world == 1] = black
        # tranpose spatial
        matrix = matrix.transpose(1, 0, 2)
        # Extent => lrbt
        # extent=[0, world.side_length, 0, world.side_length]
        plt.imshow(matrix, interpolation="nearest", zorder=0) 
        
        # Grid lines with axlines
        for i in range(world.side_length + 1):
            plt.axvline(i - 0.5, color="black", linestyle="--", linewidth=0.5)
            plt.axhline(i - 0.5, color="black", linestyle="--", linewidth=0.5)

        # Plot the start and goal positions
        plt.scatter(*world.start, color="green", label=f"Start {world.start}", marker="o", s=100)
        plt.scatter(*world.goal, color="red", label=f"Goal {world.goal}", marker="x", s=100)

        # Plot the path an orange line
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], color="orange", label="Path", alpha=0.8)

        # Plot the smoothed path a blue line
        path_smooth = np.array(path_smooth)
        plt.plot(path_smooth[:, 0], path_smooth[:, 1], color="blue", label="Path CVX", alpha=0.5)
        plt.scatter(path_smooth[:, 0], path_smooth[:, 1], color="blue", marker="x", s=30, alpha=0.5)

        # Plot the circles in purple (centers are xs, and otuline)
        for i, circle in enumerate(circles):
            plt.scatter(circle.x, circle.y, color="purple", marker=".")
            if i == 0:
                circle_plot = plt.Circle((circle.x, circle.y), circle.radius, color="purple", fill=False, label="Circle", alpha=0.5, linestyle="-")
            else:
                circle_plot = plt.Circle((circle.x, circle.y), circle.radius, color="purple", fill=False, alpha=0.5, linestyle="-")
            plt.gca().add_artist(circle_plot)
        plt.legend()

        # Cut it off at the world size
        plt.xlim(-0.5, world.side_length - 0.5)
        plt.ylim(-0.5, world.side_length - 0.5)

        # Remove ticks
        plt.xticks([])
        plt.yticks([])

        # Flip axes y and x by transposing
        plt.gca().invert_yaxis()

        # Save the plot , save it big
        plt.savefig("world.png", dpi=600, bbox_inches="tight")
                
# Create a world, and get the path of circles from the start to the end
world = World(side_length=40, middle_obstacle_side_length=20, agent_radius=2)
path = world.a_star(world.start, world.goal)
# Create a circle at the start
circles = [Circle.get_largest_possible_circle(world, *world.start)]
# Get the furthest circle along the path until we contain the goal
attempts = 1000
for _ in range(attempts):
    circle = circles[-1].get_furthest_circle_along_path(world, path)
    circles.append(circle)
    if circle.is_point_inside(*world.goal):
        break
# Print them all out
for circle in circles:
    print(circle)

# Interpolate subsample the path
num_points_original = len(path)
path_length = np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))
actual_points_per_unit = num_points_original / path_length
desired_points_per_unit = 8
sample_factor = desired_points_per_unit / actual_points_per_unit
# TODO use scipy
path = interp1d(np.arange(num_points_original), path, axis=0)(np.linspace(0, num_points_original - 1, int(num_points_original * sample_factor)))

# ----------------------------------------------------------------
    
# This is where we get convex with it

# Lets have some dynamics
dynamics = DynamicsTiny(dt=0.1)

def sdf_values(path, circles):
    # If it's 1 point long make it 2d
    if len(path.shape) == 1:
        path = path[np.newaxis, :]
    # (num points in path, num circles)
    d = np.zeros((path.shape[0], len(circles)))
    # Each row is the sdf value of the circle at that point 
    # ('how in each circle is the ith path point')
    for i in range(len(path)):
        for j in range(len(circles)):
            d[i,j] = circles[j].sdf_value(*path[i])
    return d

def scp_smooth_path(world, path, circles, sig=50, eps_ss=1e-4, max_iters=15, tol=1e-3):
    """
    Smooths a given path using Sequential Convex Programming (SCP) while ensuring each point 
    remains inside at least one circle.

    world: World object
    path: List of (x, y) tuples from A*
    circles: List of Circle objects
    lambda_smooth: Weight for the smoothness penalty
    lambda_inside: Weight for inside-circle constraint
    max_iters: Maximum number of SCP iterations
    tol: Convergence tolerance
    """
    path = np.array(path)
    num_points = len(path)
    num_circles = len(circles)

    # Initialize the path
    X_prev = np.copy(path)  # Initial guess (previous iteration's solution)

    for iter in tqdm(range(max_iters), desc="SCP Iteration"):

        # Define optimization variables (x, y coordinates for each path point)
        X = cp.Variable((num_points, 2))
        slack_sdf = cp.Variable((num_points,num_circles))

        # ----------------------------------------------------------------

        # Constraints
        constraints = []

        # ~~~~ Start and end constraints ~~~~

        # Start and end
        constraints.append(X[0] == path[0])  # Fix start
        constraints.append(X[-1] == path[-1])  # Fix goal

        # ~~~~ SDF constraints ~~~~

        # slack sdf prev is going to be a matrix (num_timesteps, num_sdfs)
        slack_sdf_prev = sdf_values(X_prev[:,:2],circles)

        # G's shape is the same
        G = gradient_log_softmax(sig, slack_sdf_prev)

        # # affine part of the assembled matrix form of the constraints
        L0 = log_softmax(sig, slack_sdf_prev)
        
        # The SDF constraints
        # This set of constraints ensures that the slack continues to minimize
        constraints += [cp.diag( G @ (slack_sdf - slack_sdf_prev).T) + L0 >= 0]
        # This set of constraints ensures that the slack is always less than the true sdf values
        for i in range(num_circles):
            constraints += [slack_sdf[k,i] <= 1 - (1/circles[i].radius)*cp.norm2(X[k] - np.array([circles[i].x,circles[i].y])) for k in range(num_points)]

        # ~~~~ Dynamics constraints ~~~~
            
        # # Get affinized dynamics about the current point. We do this
        # # in a batch so we actually have multiple A's and B's and C's\
        # # here
        # A, B, C = dynamics.affinize(X_prev[:-1], U_prev)
        # A, B, C = np.array(A),np.array(B),np.array(C)

        # # Create virtual control term (recommended to be identity matrix)
        # E = np.eye(2)

        # # From the prior solution
        # nu_max = 1

        # # Construct the dynamic feasibility constraint
        # constraints += [x[k+1] == A[k] @ x[k] + B[k] @ u[k] + C[k] + E @ nu[k] for k in range(self.N - 1)]

        # constraints += [cp.max(cp.abs(nu)) <= nu_max]

        # ----------------------------------------------------------------

        # Formulate the objective
        objective = 0

        # The distance objective is the sum of the squared distances between consecutive points (normalized)
        distance_max = (cp.norm2(path[0] - path[-1]))**2 * (num_points - 1)
        distance_objective = cp.sum([(cp.norm2(X[_k] - path[-1]))**2 for _k in range(num_points - 1)]) / distance_max
        objective += distance_objective

        # The SDF objective attempts to minimize the slack
        sdf_objective = -eps_ss * cp.sum(slack_sdf)
        objective += sdf_objective

        # ----------------------------------------------------------------

        # Solve it
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        # Check convergence
        if np.max(np.linalg.norm(X.value - X_prev, axis=0)) < tol:
            break

        # Update previous solution
        X_prev = X.value

    return X.value


# Try to get convex path
path_smooth = scp_smooth_path(world,path,circles).tolist()

# ----------------------------------------------------------------

# Plot everything
Plotter.plot_world(world, path, path_smooth, circles)


