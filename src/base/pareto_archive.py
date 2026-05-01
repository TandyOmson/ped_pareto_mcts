from abc import ABC, abstractmethod
from typing import Dict, Hashable

from  base.policies import NodeStats

Objective = Hashable
RewardVector = Dict[Objective, float]

class ParetoArchive(ABC):
    """ Maintains a set of non-dominated reward vectors
        Produces pareto-derived metrics for MCTS functions
    """

    @abstractmethod
    def update_archive(self, rewards: RewardVector) -> None:
        """ Try to insert a reward vector into the archive
        """
        raise NotImplementedError
    
    @abstractmethod
    def is_non_dominated(self, rewards: RewardVector) -> bool:
        """ Test wheter a reward vector is non-dominated w.r.t. the current archive
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_backprop_payload(self, rewards: RewardVector) -> NodeStats:
        """ Produce a backprop-ready payload of pareto derived metrics using 
            a reward vector and current archive state
        """
        raise NotImplementedError