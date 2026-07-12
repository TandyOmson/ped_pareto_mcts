from typing import Literal
from abc import ABC, abstractmethod

from typing import Dict, Hashable

Objective = Hashable
ObjectiveVector = Dict[Objective, float]

class ObjectiveFunc(ABC):
    """ Reward functions will always return floats
    """
    def __init__(self, name):
        self.name = name

    def update_context(self, *, archive=None):
        """ Used to inject pareto front context to certain rewards
        """
        pass

    @abstractmethod
    def evaluate(self, sample) -> float:
        raise NotImplementedError
    
    def scale_reward(self, reward: float) -> float:
        """ Use to smooth reward
        """
        if self.hasattr(self, "reward_mean") and self.hasattr(self, "reward_std"):
            reward = (reward - self.reward_mean) / self.reward_std

            reward_norm  = 1 / (1 + abs(reward))
            return reward_norm
        else:
            return reward

    @property
    def optimisation_direction(self) -> Literal["max", "min"]:
        """ Use this to invert to maximisation at reward compute if neede
        """
        return "max"
    
    @property
    def failure_val(self) -> float:
        return -1.0
    
class BackpropPayload(ABC):
    """ Determines how objective function results are converted into rewards
        and how rewards update node stats
    """
    @abstractmethod
    def update(self):
        """ Combines old stats with new reward
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_reward(objective_vector: ObjectiveVector):
        """ Produces a sequence reward from objective vector
            (reward has no fixed type, could be vector or float)
        """
        raise NotImplementedError
