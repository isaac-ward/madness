import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
import warnings
import utils.logging
import utils.general
from tqdm import tqdm
import numpy as np

from environment import Environment

class EnvironmentDataset(Dataset):
    """
    Wraps a sequential decision-making environment into a
    torch dataset. It works by returning items with respect
    to some agent's action decision. If we exceed an 
    episode length, or are otherwise done, then we reset
    the environment and start again

    To avoid spikes in the learning curves, and variability 
    in performance across episodes, we can: 
        1) use parallel environments,
        2) use experience replay 
        3) learn on batches of experiences before updating
           the policy
    """

    def __init__(self, environment, agent, stage, log_folder):
        super().__init__()
        self.environment = environment
        self.agent = agent
        self.stage = stage
        self.log_folder = log_folder

        # Generate all the task configurations ahead of time
        self.task_configurations = []
        for i in tqdm(range(100), desc="Generating task configurations"):
            state_initial, state_goal = Environment.get_two_states_separated_by_distance(
                self.environment.map,
                min_distance=26
            )
            self.task_configurations.append({
                "state_initial": state_initial,
                "state_goal": state_goal,
            })      
        self.current_task_configuration_idx = np.random.randint(0, len(self.task_configurations))
        
        # Reset everything
        self.reset()

    def reset(self):

        # If the log folder is set, we log the state and action trajectories
        # Note that this overwrites the old
        if self.log_folder is not None:
            self.environment.log(self.log_folder)
            self.agent.log(self.log_folder)

        # Get a new task
        state_initial = self.task_configurations[self.current_task_configuration_idx]["state_initial"]
        state_goal    = self.task_configurations[self.current_task_configuration_idx]["state_goal"]
        self.current_task_configuration_idx = (self.current_task_configuration_idx + 1) % len(self.task_configurations)

        # Report what the new task is
        if self.stage == "val":
            print(f"New val task: idx={self.current_task_configuration_idx}, {state_initial[:3]} -> {state_goal[:3]}")

        # Update the environment and agent
        self.environment.reset(state_initial, state_goal)
        self.agent.reset(state_initial)
        self.agent.policy.update_state_goal(state_goal)

    def __len__(self):
        return self.environment.episode_length

    def __getitem__(self, idx):
        """
        An on-policy approach to getting the next learning instance
        """

        # Without gradients, we determine the next action fm the policy/agent
        with torch.no_grad():
            action = self.agent.act()

        # We step the environment with the action
        state, done_flag, done_message = self.environment.step(action)

        # Note what goal was used to make this inference
        state_goal_used = self.environment.state_goal

        # and make new observations for future decision
        self.agent.observe(state)

        # We reset the environment and agent as needed
        if done_flag:
            #warnings.warn(f"Environment is done, flag: {done_message}")
            self.reset()

        # TODO should our paradigm be adjusted to traditional RL where the environment
        # returns the reward?

        # TODO fairly sure that we're doing everything twice - agent acts without
        # grad and then with grad during a training step
            
        # Get the history of states and actions (as observed by the AGENT, 
        # not the ENVIRONMENT (ground truth)) so that they can be used for inference
        state_history, action_history = self.agent.get_histories()

        # We return the information required for learning
        data = {
            "state_history": torch.tensor(state_history, dtype=torch.float32),
            "action_history": torch.tensor(action_history, dtype=torch.float32),
            "state_goal": torch.tensor(state_goal_used, dtype=torch.float32),
            "done_flag": done_flag,
            "done_message": done_message,
        }

        return data
    
