import numpy as np
from tqdm import tqdm

class Benchmarker:
    def __init__(
        self,
        environment,
        agents,
        num_episodes,
        log_folder,
    ):
        # We have only one environment and many agents
        self.environment = environment
        self.agents = agents
        self.num_episodes = num_episodes
        self.log_folder = log_folder

    def benchmark(self):
        """
        Benchmark all the policies on the environment 
        """
        
        # Run the simulation for some number of steps
        pbar = tqdm(total=num_steps, desc="Running simulation")
        for i in range(num_steps):
            # Take an action (this is based on previous observations)
            action = agent.act()
            state, done_flag, done_message = environment.step(action)
            pbar.update(1)

            # If we're done exit the loop
            if done_flag:
                pbar.set_description(done_message)
                break

            # Make new observations
            agent.observe(state)

            # Update the pbar with the current state and action
            p_string = ", ".join([f"{x:<5.1f}" for x in state[0:3]])
            v_string = f"{np.linalg.norm(state[6:9]):<4.1f}"
            w_string = ", ".join([f"{x:<4.1f}" for x in state[9:12]])
            a_string = ", ".join([f"{x:<4.1f}" for x in action])
            dist_to_goal_string = f"{np.linalg.norm(state[0:3] - state_goal[0:3]):<4.1f}"
            pbar.set_description(
                f"t={(i+1)*dyn.dt:.2f}/{num_seconds:.2f} | d={dist_to_goal_string} | p=[{p_string}] | v={v_string} | w=[{w_string}] | a=[{a_string}] | gpu={'yes' if use_gpu_if_available else 'no'}")
        # Close the bar
        pbar.close()

        # ----------------------------------------------------------------

        # Log everything of interest
        agent.log(log_folder)
        environment.log(log_folder)

        # Render visuals
        visual = Visual(log_folder)
        visual.plot_histories()
        visual.render_video(desired_fps=25)

        # Clean up stored data 
        try:
            if not keep_policy_logs:
                policy.delete_logs()
        except:
            pass