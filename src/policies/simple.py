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