""" Reward functions will always return floats
    They'll be aggregated and collated for pareto elsewhere
"""

from typing import Literal
from abc import ABC, abstractmethod

class RewardFunc(ABC):
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