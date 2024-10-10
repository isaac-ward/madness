import numpy as np

class PolicyNothing:
    def __init__(
        self,
        state_size,
        action_size,
    ):
        """
        A policy that does nothing
        """
        self.state_size = state_size
        self.action_size = action_size

    def act(
        self,
        state_history,
        action_history,
    ):
        """
        Do nothing
        """
        return np.zeros(self.action_size)

class PolicyRandom:
    def __init__(
        self,
        state_size,
        action_size,
    ):
        """
        A policy that acts randomly
        """
        self.state_size = state_size
        self.action_size = action_size

    def act(
        self,
        state_history,
        action_history,
    ):
        """
        Return a random action
        """
        return np.random.randn(self.action_size)

class PolicyConstant:
    def __init__(
        self,
        state_size,
        action_size,
        constant_action,
        perturb=False,
    ):
        """
        A policy that tries to hover
        """
        self.state_size = state_size
        self.action_size = action_size
        self.constant_action = constant_action
        self.perturb = perturb

    def act(
        self,
        state_history,
        action_history,
    ):
        """
        Return a constant action
        """
        if self.perturb:
            return self.constant_action + np.random.randn(self.action_size) * 0.1
        else:
            return self.constant_action
        
# Example usage
# input_none  = [0, 0, 0, 0]
# input_hover = [1, 1, 1, 1]
# # These have been tested to be correct for the given dynamics
# # (positive meaning the rotating direction given by right hand
# # rule with thumb pointing in the positive direction of each axis
# # in the NED frame)
# input_positive_roll  = [1.25, 0, 0.75, 0]
# input_positive_pitch = [0, 0.75, 0, 1.25]
# input_positive_yaw   = [0.25, 1.25, 0.25, 1.25]
# policy = PolicyConstant(
#     state_size=dyn.state_size(),
#     action_size=dyn.action_size(),
#     constant_action=np.array(input_positive_pitch),
#     perturb=False,
# )
