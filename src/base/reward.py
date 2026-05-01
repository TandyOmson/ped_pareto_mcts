from typing import Literal
from abc import ABC, abstractmethod

from typing import Dict, Hashable

Objective = Hashable
ObjecitiveVector = Dict[Objective, float]

class ObjectiveFunc(ABC):
    """ Reward functions will always return floats
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self, sample) -> float:
        raise NotImplementedError

    @property
    def optimisation_direction(self) -> Literal["max", "min"]:
        """ Use this to invert to maximisation at reward compute if neede
        """
        return "max"
    
class NodeStats(ABC):
    """ Determines how objective function results are converted into rewards
        and how rewards update node stats
    """
    @abstractmethod
    def update(self, reward):
        """ Combines old stats with new reward
        """
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def get_reward(objective_vector: ObjecitiveVector):
        """ Produces a sequence reward from objective vector
            (reward has no fixed type, could be vector or float)
        """
        raise NotImplementedError