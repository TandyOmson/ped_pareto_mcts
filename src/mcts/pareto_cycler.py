
from base.tree import Tree, Node
from base.reward import ObjectiveFunc, NodeStats
from utils.utils import load_class, filter_class_config

from typing import List

# idea for later (change updated in ParetoArchive to updated and feedback, generating this payload)
# from dataclasses import dataclass
# @dataclass(frozen=True)
# class ParetoPayload:
#     is_pareto: bool         # did this rollout go to pareto front
#     dominance_gain: int     # 0 or 1
#     hv_gain: float = 0.0    # hypervolume gain

class ParetoArchive:
    """ Maintains a set of sequences with non-dominated reward vectors
    """
    def __init__(self):
        # sequencues are dicts consisting of {sequence: reward}
        self.front = []

    def update(self, seq):
        """ Try to insert a reward vector into the archive
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

        for seq in self.front:
            if self.dominates(seq.reward, reward):
                return False
            if self.dominates(reward, seq.reward):
                dominated_seqs.append(seq.sequence)

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
            for seq in pareto_archive.front:
                if val >= seq.reward[key]:
                    pareto_sum += 1 
            
            reward[key] = pareto_sum / len(pareto_archive.front)
        
        return reward

class ParetoMCTSCycler:
    def __init__(self, config):
        # Load policy classes and configs
        selectionClass = load_class(config["selection_policy"]["class_path"])
        selection_args = filter_class_config(selectionClass, **config["selection_policy"]["kwargs"])
        
        expansionClass = load_class(config["expansion_policy"]["class_path"])
        expansion_args = filter_class_config(expansionClass, **config["expansion_policy"]["kwargs"])
        
        rolloutClass = load_class(config["rollout_policy"]["class_path"])
        rollout_args = filter_class_config(rolloutClass, **config["rollout_policy"]["kwargs"])

        self.selection = selectionClass(**selection_args)
        self.expansion = expansionClass(**expansion_args)
        self.rollout = rolloutClass(**rollout_args)
        
        # Load reward function classes and config for evaluator

        # Define Node Stats

        # Initialise tree object and pareto archive
        root_node = Node(token='&', parent=None)
        tree = Tree(root_node)

    def step(self):
        leaf = self.selection.select(self.tree, self.archive)
        child = self.expansion.expand(leaf)
        rollouts = self.rollout.simulate(child)
        results = self.evaluator(rollouts)
        
        self.archive.update(rollouts, results)
        self.tree.backpropagate(child, results, self.archive)

        return # logging materials? leaf, child, results, archive size etc.
