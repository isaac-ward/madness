import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

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

    def __init__(self, environment, agent):
        super().__init__()
        self.environment = environment
        self.agent = agent

        # Ensure the environment is reset
        state_initial, state_goal = self.environment.get_two_states_separated_by_distance()
        self.environment.reset(
            state_initial=state_initial,
            state_goal=state_goal,
        )

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
        agent.observe(state)

        # We reset the environment as needed
        if done_flag:
            state_initial, state_goal = self.environment.get_two_states_separated_by_distance()
            self.environment.reset(
                state_initial=state_initial,
                state_goal=state_goal,
            )

        # TODO should our paradigm be adjusted to traditional RL where the environment
        # returns the reward?

        # We return the information required for learning
        data = {
            "state": torch.tensor(state, dtype=torch.float32),
            "state_goal": torch.tensor(state_goal_used, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "done_flag": done_flag,
            "done_message": done_message,
        }

        return data
    
