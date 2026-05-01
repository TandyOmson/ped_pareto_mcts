from typing import List, Dict
from dataclasses import dataclass
from base.reward import NodeStats, ObjectiveFunc

@dataclass(frozen=True)
class Molecule:
    sequence: str
    reward: Dict

class ParetoArchive:
    """ Maintains a set of sequences with non-dominated reward vectors
    """
    def __init__(self):
        self.front = []

    def update(self, seq):
        """ Try to insert a completed sequence (Molecule) into the archive
        """
        dominated_seqs = self.is_non_dominated(seq.reward)
        if dominated_seqs is not False: 
            for s in list(self.front):
                if s.sequence in dominated_seqs:
                    self.front.remove(s)

            self.front.append(seq)

    @staticmethod
    def dominates(a, b):
        return all(a[k] >= b[k] for k in a) and any(a[k] > b[k] for k in a)

    def is_non_dominated(self, reward):
        """ Test wheter an objective vector is non-dominated w.r.t. the current archive
        """            
        dominated_seqs = []

        for s in self.front:
            if self.dominates(s.reward, reward):
                return False
            if self.dominates(reward, s.reward):
                dominated_seqs.append(s.sequence)

        return dominated_seqs
    
class ParetoStats(NodeStats):
    """ Pareto statistics for nodes
        Objective functions are converted to a reward vector using input from archive
        Node stats are updated by adding reward vector to node stats
    """
    def __init__(self, obj_funcs: List[ObjectiveFunc]):
        """ Node stats are a vector (dict) of pareto-aggregated objectives
        """
        self.stats = {obj.name: 0.0 for obj in obj_funcs}
    
    def update(self, reward):
        for key, val in reward.items():
            self.stats[key] += val

    @staticmethod
    def get_reward(objective_vector, pareto_archive):
        reward = {}

        if not pareto_archive.front:
            return {k: 0.0 for k in objective_vector}

        for key, val in objective_vector.items():
            # loop over all pareto optimal molecles in global pool
            pareto_sum = 0
            for s in pareto_archive.front:
                if val >= s.reward[key]:
                    pareto_sum += 1 
            
            reward[key] = pareto_sum / len(pareto_archive.front)
        
        return reward