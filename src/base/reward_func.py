from typing import Literal
from abc import ABC, abstractmethod

from typing import Dict, Hashable

Objective = Hashable
RewardVector = Dict[Objective, float]

class RewardFunc(ABC):
    """ Reward functions will always return floats
    """
    @abstractmethod
    def evaluate(self, sample) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def optimisation_direction(self) -> Literal["max", "min"]:
        return "max"
    
class NodeStats(ABC):
    """ Determines how rewards are transformed into node statistics
    """
    @abstractmethod
    def get_backprop_payload(self, rewards: RewardVector):
        """ Produce a backprop-ready payload from reward vector
            (subclasses may involve other arguments like pareto archive)
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, update_stats):
        """ Combines old update stats with update stats, returns new ones
        """
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def init_stats(node):
        """ Initialises archive stats 
        """