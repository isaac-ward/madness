import numpy as np
from tqdm import tqdm
import time
from visual import Visual
import policies
import copy
from policies.mppi import PolicyMPPI
from policies.ilqr import PolicyALiLQR
from agent import Agent
import utils
import concurrent
from concurrent.futures import ProcessPoolExecutor
import os
import warnings

from utils.general import Timer

class Benchmarker:
    def __init__(
        self,
        environment,
        num_episodes,
        log_folder,
    ):
        # Seed it up
        utils.general.random_seed(42)

        # We have only one environment and many agents
        self.environment = environment
        # How many to benchmark on?
        self.num_episodes = num_episodes
        # There will be sub folders for the agents and runs,
        # but we'll print out aggregated statistics in the 
        # main folder
        self.log_folder = log_folder

    @staticmethod
    def get_mppi_agent(environment, log_folder):
        dyn = environment.dynamics
        map_ = environment.map
        state_initial = environment.state_history_tracker.get_first_item()
        state_goal = environment.state_goal

        # Create the agent, which has an initial state and a policy
        K = 500
        H = 50 #int(0.5/dynamics.dt), # X second horizon
        action_sampler = policies.samplers.RolloverGaussianActionSampler(K, H, dyn.action_ranges())
        policy = PolicyMPPI(
            dynamics=copy.deepcopy(dyn),
            action_sampler=action_sampler,
            K=K,
            H=H,
            lambda_=100,
            map_=map_,
            use_gpu_if_available=False,
        )
        policy.enable_logging(log_folder)
        policy.update_state_goal(state_goal) 

        # Can now create an agent
        agent = Agent(
            state_initial=state_initial,
            policy=policy,
            state_size=dyn.state_size(),
            action_ranges=dyn.action_ranges(),
            zero_pad_state=dyn.zero_state(),
        ) 

        return agent

    @staticmethod
    def get_alilqr_agent(environment, log_folder):
        dyn = environment.dynamics
        n = dyn.state_size()
        m = dyn.action_size()
        safety_factor = 1

        # Create a map representation
        map_ = environment.map

        # Start and goal states
        state_initial = environment.state_history_tracker.get_first_item()
        state_goal = environment.state_goal

        # Generate a path from the initial state to the goal state
        xyz_initial = state_initial[0:3]
        xyz_goal = state_goal[0:3]
        path_xyz = np.array([xyz_initial, xyz_goal])
        path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*safety_factor)
        try:
            path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz, desired_points_per_meter=20)
        except Exception as e:
            print("Error: " + e)
            path_xyz_smooth = path_xyz

        # Duplicate last entry
        last_entry_duplicate = 100
        last_entry_duplicated = np.tile(path_xyz_smooth[-1], (last_entry_duplicate, 1))

        # Append the duplicated rows to the original array
        path_xyz_smooth = np.vstack([path_xyz_smooth, last_entry_duplicated])

        # Get length of A* path
        K = path_xyz_smooth.shape[0] - 1

        # Create AL-iLQR policy
        Q = np.eye(n) * 1       # Cost for state error along trajectory
        Q[:3] = Q[:3] * 5
        R_cost = np.eye(m) * 1  # Cost for control input
        QN = np.eye(n) * 10     # Cost for final state error
        QN[:3] = QN[:3] * 10
        W = np.eye(m) * 0       # Control continuity cost

        # Create trajectory to track (A* path with 0s at other states)
        basic_state_traj = np.zeros((K+1,n))
        basic_state_traj[:,:3] = path_xyz_smooth

        # Create initial control inputs
        basic_hover_action = np.ones((K,m)) * np.sqrt(dyn.mass*dyn.g/(4*dyn.thrust_coef))

        # Segment trajectory so that each path is ~120 time steps
        segs = round(np.shape(basic_state_traj)[0] / 120.)
        if not segs:
            segs = 1

        # Solve AL-iLQR policy
        policy = PolicyALiLQR(
            dynamics=copy.deepcopy(dyn),
            Q=Q,
            R=R_cost,
            QN=QN,
            W=W,
            x_track=basic_state_traj,
            u_track=basic_hover_action,
            segments=segs,
            eps=1e-1,
            max_iters=300,
            verbose=True,
            run_folder=log_folder,
        )

        # Create an agent
        agent = Agent(
            state_initial=state_initial,
            policy=policy,
            state_size=dyn.state_size(),
            action_ranges=dyn.action_ranges(),
            zero_pad_state=None
        ) 

        # Attaching useful objects to agent
        agent.path_xyz_smooth = path_xyz_smooth
        agent.path_xyz = path_xyz

        return agent

    @staticmethod
    def get_flowmppi_agent(environment, log_folder):
        return Benchmarker.get_mppi_agent(environment, log_folder)

    @staticmethod
    def get_flowmppi_alilqr_agent(environment, log_folder):
        return Benchmarker.get_alilqr_agent(environment, log_folder)

    @staticmethod
    def run_single_agent_single_environment(
        agent, 
        environment,
        state_initial,
        state_goal,
        log_folder,
        instantiation_time_s=0,
        suffix="",
        render_videos=False,
    ):
        # Shorthand
        dyn = environment.dynamics
        dt = dyn.dt

        # Compute the number of steps
        num_steps = environment.episode_length
        num_seconds = num_steps * dt

        xyz_initial = state_initial[0:3]
        xyz_goal = state_goal[0:3]
        print(f"Task is to move from {np.round(xyz_initial,2)} to {np.round(xyz_goal,2)}")

        # Run the simulation for some number of steps
        continue_after_done_secs = 2
        continue_after_done_steps = int(continue_after_done_secs / dt)   
        done_task_flag = False 
        print(f"Will run for {num_seconds} seconds ({num_steps} steps) and continue for {continue_after_done_secs} seconds ({continue_after_done_steps} steps) after reaching goal (within {environment.close_enough_position} m of goal)")
        
        # We want to track the time to compute stuff and the time to run the simulation
        timer_agent = Timer()
        timer_environment = Timer()
        steps_to_done = 0

        # iLQR trajectory
        ilqr_traj = [state_initial[:3]]

        pbar = tqdm(total=num_steps, desc="Running simulation")
        for i in range(num_steps):
            # Take an action (this is based on previous observations)
            timer_agent.start()
            action = agent.act()
            timer_agent.stop()
            
            timer_environment.start()
            state, done_flag, _ = environment.step(action)
            timer_environment.stop()
            ilqr_traj.append(state[:3])

            if done_flag and not done_task_flag:
                done_message = _
            pbar.update(1)

            # If we're done exit the loop in X timesteps
            if done_flag or done_task_flag:
                done_task_flag = True
                continue_after_done_steps -= 1
            if continue_after_done_steps == 0:
                break

            # Make new observations
            agent.observe(state)

            # Update the pbar with the current state and action
            p_string = ", ".join([f"{x:<5.2f}" for x in state[0:3]])
            v_string = f"{np.linalg.norm(state[6:9]):<4.1f}"
            w_string = f"{np.linalg.norm(state[9:12]):<4.1f}"
            a_string = ", ".join([f"{x:<4.1f}" for x in action])
            dist_to_goal_string = f"{np.linalg.norm(state[0:3] - state_goal[0:3]):<4.1f}"
            steps_remaining_string = f" ({continue_after_done_steps} rem)"
            pbar.set_description(
                f"t={(i+1)*dt:.2f}/{num_seconds:.2f} | done={'y' if done_task_flag else 'n'}{steps_remaining_string if done_task_flag else ''} | d={dist_to_goal_string} | p=[{p_string}] | v={v_string} | w={w_string} | a=[{a_string}]")
        
            if not done_task_flag:
                steps_to_done += 1
        
        # Close the bar
        pbar.close()
        time.sleep(0.5)

        # Announce the done message
        print(f"{done_message}")

        print(f"Logging to folder: {log_folder}")

        # Log everything of interest
        agent.log(log_folder)
        environment.log(log_folder)

        # We want to return 
        # success/failure. If the done message has 'reached' in it we succeeded
        success = 1 if ("reached" in done_message.lower()) else 0
        # timer results
        agent_time = timer_agent.elapsed_time()
        environment_time = steps_to_done #timer_environment.elapsed_time()
        # velocity average
        state_trajectory = agent.state_history_tracker.get_history()
        action_trajectory = agent.action_history_tracker.get_history()
        vels = np.linalg.norm(state_trajectory[:, 6:9], axis=1)
        avg_vel = np.mean(vels)
        # distance traveled along path
        positions = state_trajectory[:, 0:3]
        position_deltas = positions[1:] - positions[:-1]
        path_lengths = np.linalg.norm(position_deltas, axis=1)
        path_length = np.sum(path_lengths)
        # control effort average
        control_efforts = np.linalg.norm(action_trajectory, axis=1)
        avg_control_effort = np.mean(control_efforts)
        # We'll write it to a file called 'benchmark.csv'
        with open(f"{log_folder}/benchmark{suffix}.csv", "a") as f:
            # Header
            f.write("Policy,Success,Agent Time Precompute,Agent Time,Environment Time,Avg Vel,Path Length,Avg Control Effort\n")
            f.write(f"{agent.policy.__class__.__name__},{success},{instantiation_time_s},{agent_time},{environment_time},{avg_vel},{path_length},{avg_control_effort}\n")
        
        # Create visual
        visual = Visual(log_folder)

        # If AL-iLQR is agent, special plotting
        if isinstance(agent.policy,PolicyALiLQR):
            # Log the A* path
            utils.logging.save_to_npz(
                os.path.join(log_folder, "a_star", "start_to_goal.npz"),
                agent.path_xyz,
            )

            # Log the smooth A* path
            utils.logging.save_to_npz(
                os.path.join(log_folder, "a_star", "start_to_goal_smooth.npz"),
                agent.path_xyz_smooth,
            )

            # Log the iLQR path
            utils.logging.save_to_npz(
                os.path.join(log_folder, "al_ilqr", "al_ilqr.npz"),
                ilqr_traj,
            )
            visual.plot_environment()

        # Render visuals
        visual.plot_histories()
        if render_videos:
            visual.render_video(desired_fps=25)

    @staticmethod
    def run_episode(i, environment, log_folder, agent_functions, render_videos):
        """Runs a single episode for all agents."""
        print(f"Running episode {i}")
        try:

            # Get the initial state and goal
            state_initial, state_goal = environment.get_two_states_separated_by_distance(
                environment.map,
                template=environment.dynamics.state_randomization_template(),
                min_distance=26,
            )
            # Reset the environment
            environment.reset(state_initial, state_goal)

            # Loop through all agents and benchmark
            for af in agent_functions:
                # Each agent gets its own envirnoment copy
                environment_copy = copy.deepcopy(environment)

                # Extract agent name for logging
                agent_function_name = af.__name__.replace("get_", "")
                log_folder_inner = f"{log_folder}/{agent_function_name}/ep{i}"
                os.makedirs(log_folder_inner, exist_ok=True)

                # Instantiate the agent (and time it)
                timer = Timer()
                timer.start()
                agent = af(environment_copy, log_folder_inner)  # Includes solving time
                timer.stop()

                print(f"Running agent {agent_function_name} on episode {i}")

                Benchmarker.run_single_agent_single_environment(
                    agent=agent,
                    environment=environment_copy,
                    state_initial=state_initial,
                    state_goal=state_goal,
                    instantiation_time_s=timer.elapsed_time(),
                    log_folder=log_folder_inner,
                    render_videos=render_videos,
                )

            print(f"Completed episode {i}")

        except Exception as e:
            print(f"Error in episode {i}: {e}")

    def benchmark(self, agent_functions, render_videos=False):
        """Benchmark all the policies on the environment in parallel over episodes."""

        print("\nStarting benchmarking!!!\n")

        if render_videos:
            warnings.warn("Rendering videos will slow down the benchmarking process")

        print(f"Benchmarking the following agents consistently over {self.num_episodes} episodes:")
        for af in agent_functions:
            print(f"  {af.__name__}")

        # Determine number of parallel workers (half of available CPU cores)
        max_workers = os.cpu_count() // 2 if os.cpu_count() else 2  # Fallback to 2 if unknown
        max_workers = min(self.num_episodes,max_workers)
        print(f"Using {max_workers} workers for benchmarking")

        # Essentially doing this num_episode times, in batches of max_workers in parallel
        # Benchmarker.run_episode(i, self.environment, self.log_folder, agent_functions, render_videos)

        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(Benchmarker.run_episode, i, self.environment, self.log_folder, agent_functions, render_videos)
                for i in range(self.num_episodes)
            ]

            # Explicitly wait for all tasks
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Raises exceptions if any occur
                except Exception as e:
                    print(f"Worker failed: {e}")