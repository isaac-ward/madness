import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import heapq
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d
from utils.general import gradient_log_softmax, log_softmax
import utils.geometric
import utils.general
import utils.logging
import cvxpy as cp
from dynamics_tiny import DynamicsTiny
import standard
from environment import Environment
from visual import Visual
import os
import time
import glob

utils.general.random_seed(42)

# Here's where we'll save everything
log_folder = utils.logging.make_log_folder(name="cvx")

# Load up a map, get two points, get the path joining them, get the
# spheres along the path
map_ = standard.get_28x28x28_at_111_with_obstacles()
#dyn = standard.get_standard_dynamics()
dyn = standard.get_standard_dynamics_linear()
state_initial, state_goal = Environment.get_two_states_separated_by_distance(
    map_, 
    template=dyn.state_randomization_template(),
    min_distance=26,
)
#state_initial = np.array([5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#state_goal = np.array([10, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
avoid_radius = 2*dyn.diameter
path = map_.plan_path(state_initial[:3], state_goal[:3], avoid_radius)
path = utils.geometric.resample_path_to_some_points_per_meter(path, desired_points_per_meter=2)
spheres = map_.compute_spheres_along_path(path, avoid_radius)

# ----------------------------------------------------------------
    
# This is where we get convex with it

def sdf_matrix(path, spheres):
    # This matrix is going to be (num points in path, num spheres),
    # and each element is the sdf value of the sphere at that point, i.e.,
    # how in the sphere is the ith path point
    # 1 -> the center of the sphere
    # 0 -> the boundary of the sphere
    # -1 and more -ve -> outside the sphere
    d = np.zeros((len(path), len(spheres)))
    for i in tqdm(range(len(path)), desc="Computing SDF matrix", leave=False):
        for j in range(len(spheres)):
            d[i,j] = spheres[j].sdf_value(path[i][:3])
    return d

def plan_trajectory_with_scp(
    path, 
    spheres, 
    dynamics, 
    num_iters, 
    softmax_sigma, 
    w_dist_to_goal,
    w_nav_slack, 
    w_ctrl, 
    w_dyn_slack, 
    tol
):
    """
    We're going to use sequential convex programming to plan a trajectory
    and return the states and actions that define that trajectory wrt
    the dynamics
    """

    num_spheres = len(spheres)
    num_states = len(path)
    num_actions = num_states - 1

    # Store this for iterative improvement
    states_prev = np.zeros((num_states, dynamics.state_size()))
    states_prev[:, :3] = path # Initial guess is just the path
    actions_prev = np.zeros((num_actions, dynamics.action_size())) # Initial guess is zero
    objective_prev = np.inf

    # Trust region constraint
    trust_region_radius = 1
    trust_decay = 0.9

    # Dynamics relaxation
    # Start with enough slack to get an initial solution
    # This is the max any given state element can be off by in the dynamics constraint
    nu_max = 0
    # Set a minimum threshold to avoid infeasibility
    nu_min = 1e-4 
    # Reduce nu_max by this factor per iteration (multiply)
    nu_decay = 0.95

    print(f"Planning trajectory with {num_iters} iterations, over {num_states} states and {num_actions} actions, with {num_spheres} spheres")

    results_per_iteration = []

    pbar = tqdm(range(num_iters), desc="SCP")
    for i in pbar:

        # Need to iteratively improve the states and actions solutions
        # wrt some cost
        states = cp.Variable((num_states, dynamics.state_size()), name="states")    
        actions = cp.Variable((num_actions, dynamics.action_size()), name="actions")
        # the following pertains to dynamics feasibility, it slackens it
        # slightly to allow for more feasible solutions
        nu = cp.Variable((num_actions, dynamics.state_size()), name="nu")
        slack = cp.Variable((num_states, num_spheres))

        # ----------------------------------------------------------------

        # Constraints
        constraints = []

        # ~~~~ Start and end constraints ~~~~

        desired_start_state = np.zeros((dynamics.state_size()))
        desired_start_state[:3] = path[0]
        desired_goal_state = np.zeros((dynamics.state_size()))
        desired_goal_state[:3] = path[-1]

        # Start and end
        constraints.append(states[0]  == desired_start_state) # Fix start
        constraints.append(states[-1] == desired_goal_state)  # Fix goal

        # ~~~~ Trust region constraint ~~~~

        # Sum of absolute values of the difference between the states must be less than the radius
        #constraints += [cp.norm2(states[k] - states_prev[k]) <= trust_region_radius for k in range(num_states)]

        # ~~~~ 'Stay in the spheres' constraints ~~~~

        # slack sdf prev is going to be a matrix (num_timesteps, num_sdfs)
        # TODO start and end may not need to be here - they're fixed
        # What were the slack values wrt to the previous state trajectory solution?
        slack_prev = sdf_matrix(states_prev[:,:3], spheres)
        assert slack_prev.shape == (num_states, num_spheres)

        # the ith row of slack_prev tells you the sdf value of the ith state
        # wrt each sphere

        # the ith row of G is a probability distribution over the spheres,
        # where the probability of each sphere is proportional to how significant
        # the sdf value of that sphere is wrt the ith state. So if there are 2
        # spheres and the ith state has an sdf value of -10 for sphere 1 and 0
        # for sphere 2, then the probability distribution would be [1, 0]. So
        # small probabilities are assigned to spheres we are MORE inside of
        # and larger probabilities are assigned to spheres we are LESS inside of

        # every row of slack_prev gets converted into a number in L0. This number
        # is the log-sum-exp of the slack_prev row, so each element is exponentiated
        # and then summed, and then the natural log is taken. This convexly approximates
        # the max of the row. So the ith element of L0 is about the max
        # of the ith row of slack_prev, meaning that the ith elment of L0 is the
        # sdf value of the sphere that we are the most inside of 
        # (larger numbers = more positive = closer to zero = more inside)

        # When we do G @ (delta slack).T, we're getting a matrix where the ith diagonal
        # element is a probability weighted sum of the ith column of delta slack. Recall that in
        # the slack matrix, the columns go across all spheres, the ith diagonal element
        # of this product is the probability weighted sum of the CHANGE IN sdf values across all spheres
        # for the ith state

        # Let's say state 3 is very far outside sphere 1, very close but just outside sphere 2, and
        # in this iteration, has moved further away from sphere 1, but just inside sphere 2
        # Originally, the sdf values for state 3 were [-10, -0.5], and now they are [-12, 0].
        # The changes are [-2, 0.5]. The G matrix for this state's row would be
        # [0.99, 0.01], in this latest iteration, noting that we are more outside
        # sphere 1 than sphere 2
        # When we multiply through G @ (slack - slack_prev).T, the 3rd diagonal element
        # corresponding to this state would be 0.99 * -2 + 0.01 * 0.5 = -1.99. Which is
        # closer to the change in sdf value of the sphere we're MOST outside of
        # We add to this L0 (which approximates the max of the slack_prev row) and
        # get -1.99 + 0 = -1.99. We're saying that this should be greater than zero

        # For this number to be greater than zero, the sphere that we're LEAST inside of
        # should have an sdf value that is becoming more positive, i.e. we're moving
        # further towards it

        # All points that we're optimizing over need to meet this criteria - they need
        # to be moving towards the sphere that they're the most OUTSIDE of

        # G's shape is the same
        G = gradient_log_softmax(softmax_sigma, slack_prev)
        assert G.shape == (num_states, num_spheres)

        # affine part of the assembled matrix form of the constraints
        L0 = log_softmax(softmax_sigma, slack_prev)
        assert L0.shape == (num_states,)
        
        # The 'stay in the spheres' constraints
        # This set of constraints ensures that the slack continues to minimize
        constraints += [cp.diag( G @ (slack - slack_prev).T) + L0 >= 0] 
        # This set of constraints ensures that the path is respecting the slack
        # bounds, which are improving at each iteration
        for s, sphere in enumerate(spheres):
            # i.e. what we're about to solve for (slack) should improve upon the sdf
            # value for the original path
            constraints += [slack[k,s] <= sphere.sdf_value_cvx(states[k][:3]) for k in range(num_states)]

        # ~~~~ Dynamics constraints ~~~~
            
        # Get affinized dynamics about the current point. We do this
        # in a batch so we actually have multiple A's and B's and C's
        # here
        # Have one more state than action
        A, B, C = dynamics.affinize(states_prev[:-1], actions_prev)
        A, B, C = np.array(A), np.array(B), np.array(C)
        
        # What are the shapes?
        # A -> (num_actions, state_size, state_size)
        # B -> (num_actions, state_size, action_size)
        # C -> (num_actions, state_size)
        assert A.shape == (num_actions, dynamics.state_size(), dynamics.state_size())
        assert B.shape == (num_actions, dynamics.state_size(), dynamics.action_size())
        assert C.shape == (num_actions, dynamics.state_size())

        # Construct the dynamic feasibility constraint. Note the nu term allows
        # for some relaxation of the dynamics constraints
        constraints += [ states[k+1] == A[k] @ states[k] + B[k] @ actions[k] + C[k] + nu[k] for k in range(num_actions)]

        # Don't allow for too much relaxation. This is the 'budget' of 'dynamics
        # rule breaking' that we allow. If this is zero, then the propagated 
        # path should exactly equal the solution path
        constraints += [cp.max(cp.abs(nu)) <= nu_max]

        # Constrain action inputs to be in the allowed range - note that it needs to be scaled by dynamics
        constraints += [actions[k] * dynamics.dt >= dynamics.action_ranges()[:,0] for k in range(num_actions)]
        constraints += [actions[k] * dynamics.dt <= dynamics.action_ranges()[:,1] for k in range(num_actions)]

        # ----------------------------------------------------------------

        # Formulate the objective
        objective = 0

        # This is a multiobjective optimization problem balancing:
        # 1. distance along path to the goal (normalized) should be minimized
        #max_distance = (cp.norm2(path[0] - path[-1]))**2 * (num_actions)
        obj_normalized_path_distance = w_dist_to_goal * cp.sum([(cp.norm2(states[_k][:3] - path[-1]))**2 for _k in range(num_actions)])
        objective += obj_normalized_path_distance
        # 2. the sum of the sphere-inside slack variables should be minimized
        obj_slack_sum = -1 * w_nav_slack * cp.sum(slack)
        objective += obj_slack_sum
        # 3. the sum of the control inputs should be minimized
        obj_control_sum = w_ctrl * cp.sum(actions**2)
        objective += obj_control_sum
        # 4. the sum of the dynamics slack variables should be minimized 
        # (want to minimize the amount of dynamics rule breaking)
        obj_dynamics_slack = w_dyn_slack * cp.sum(nu**2)
        objective += obj_dynamics_slack

        # ----------------------------------------------------------------

        # Solve it this iteration
        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            problem.solve(
                verbose=False, 
                solver='MOSEK' #'MOSEK' 'CLARABEL' 'SCS' 'ECOS'
            )
        except Exception as e:
            print(f"Solver failure: {e}")
            break

        # If infeasible, break
        if problem.status != cp.OPTIMAL:
            print(f"Optimization problem infeasible at iteration {i}")
            break
        
        # Store the results
        results_per_iteration.append({
            "states": states.value,
            "actions": actions.value,
            "slack": slack.value,
            # objective breakdown
            "obj_normalized_path_distance": obj_normalized_path_distance.value,
            "obj_slack_sum": obj_slack_sum.value,
            "obj_control_sum": obj_control_sum.value,
            "obj_dynamics_slack": obj_dynamics_slack.value,     
            "obj_total": problem.value,
        })

        # Report via tqdm
        change = problem.value - objective_prev # -ve is good
        pbar.set_postfix({
            "stat.": problem.status,
            "obj": problem.value,
            "o_path_d": obj_normalized_path_distance.value,
            "o_slack_sph": obj_slack_sum.value,
            "o_ctrl": obj_control_sum.value,
            "o_slack_dyn": obj_dynamics_slack.value,
            "o_change": change,
        })

        # Check convergence criteria - have we improved from last time?
        if abs(change) < tol:
            print(f"Converged after {i} iterations, abs(improvement)={abs(change):.6f} < tol={tol}")
            break

        # Reduce constraints 
        # Trust region must get smaller
        trust_region_radius *= trust_decay
        # Gradually reduce the allowance for dynamics slack
        nu_max = max(nu_max * nu_decay, nu_min)  

        # Plot the solution
        v = Visual(run_folder=log_folder)
        # Propagate the states and actions with the dynamics
        r = results_per_iteration[-1]
        states_propagated = np.zeros((num_states, dyn.state_size()))
        states_propagated[0] = r["states"][0]
        for k in range(num_actions):
            states_propagated[k+1] = dyn.step(states_propagated[k], r["actions"][k])
        v.plot_environment_from_objects(
            map_=map_,
            sdfs=spheres,
            path_xyz=path,
            path_xyz_smooth=None,
            path_xyz_cvx=states.value[:,:3],
            path_propagated=states_propagated[:,:3],
            path_al_ilqr=None,
            save_filename=os.path.join("cvx", f"sol_{i}.png"),
        )
        #time.sleep(1)

        # Update previous solution
        states_prev = states.value
        actions_prev = actions.value
        slack_prev = slack.value
        objective_prev = problem.value

    return results_per_iteration

# Try to get convex path
num_iters = 10
results_per_iteration = plan_trajectory_with_scp(
    path, 
    spheres, 
    dynamics=dyn, 
    num_iters=num_iters,
    softmax_sigma=50,  # higher is sharper, used for navigation slack
    w_dist_to_goal=1,
    w_nav_slack=0.1, 
    w_ctrl=0.01, 
    w_dyn_slack=1, 
    tol=1e-3
)

# last_results = results_per_iteration[-1]
# print(last_results["states"])
# print(last_results["actions"])

# ----------------------------------------------------------------


# Pickle the results dictionary
utils.logging.pickle_to_filepath(
    os.path.join(log_folder, "results_per_iteration.pkl"),
    results_per_iteration,
)

# Plot the results over time
def plot_objective_values(results_per_iteration, log_folder, filename="cvx_objective_values.png"):
    def extract_values(results, keys):
        return {key: [r[key] for r in results] for key in keys}

    objective_keys = {
        "obj_total": "Total",
        "obj_normalized_path_distance": "Normalized Path Distance",
        "obj_slack_sum": "Navigable Space Slack",
        "obj_control_sum": "Control Sum",
        "obj_dynamics_slack": "Dynamics Slack"
    }

    obj_values = extract_values(results_per_iteration, objective_keys.keys())

    num_subplots = len(objective_keys)
    
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True)

    if num_subplots == 1:
        axes = [axes]  # Ensure axes is always iterable

    for ax, (key, label) in zip(axes, objective_keys.items()):
        values = obj_values[key]
        ax.plot(values, marker="x", label=label, linestyle="-")

        for x, y in enumerate(values):
            # TypeError: unsupported format string passed to NoneType.__format__
            if x is not None and y is not None:
                ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Objective Value Breakdown Over Iterations", fontsize=14)

    os.makedirs(log_folder, exist_ok=True)
    plt.savefig(os.path.join(log_folder, filename), bbox_inches="tight")
    plt.close()
plot_objective_values(results_per_iteration, log_folder)

# Convert the images in log_folder/visuals/cvx to a video 
image_filepaths = glob.glob(os.path.join(log_folder, "visuals", "cvx", "*.png"))
# Sory by the iteration number sol_x.png
image_filepaths = sorted(image_filepaths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
utils.logging.save_video_from_images(
    os.path.join(log_folder, "visuals", "cvx_video.mp4"), 
    image_filepaths,
    fps=5
)

