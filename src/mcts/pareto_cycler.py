
from base.tree import Tree, Node
from base.reward import ObjectiveFunc, NodeStats
from policies.global_pareto_archive import ParetoArchive, ParetoStats, Molecule
from utils.utils import load_class, filter_class_config

# idea for later (change updated in ParetoArchive to updated and feedback, generating this payload)
# from dataclasses import dataclass
# @dataclass(frozen=True)
# class ParetoPayload:
#     is_pareto: bool         # did this rollout go to pareto front
#     dominance_gain: int     # 0 or 1
#     hv_gain: float = 0.0    # hypervolume gain

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
        objective_functions = []
        for objective in config["objective_functions"]:
            objective_config = config["objective_functions"]["objective"]
            objectiveClass = load_class(objective_config["class_path"])
            objective_args = objective_config["kwargs"]
            
            objective_functions.append(objectiveClass(objective_config["name"], **objective_args))

        self.objective_functions = objective_functions

        # Initialise tree object and pareto archive
        # Node stats are defined in ParetoStats above
        root_node = Node(token='&', parent=None)
        self.tree = Tree(root_node)
        self.archive = ParetoArchive()

    def step(self):
        leaf = self.selection.select(self.tree, self.archive)
        child = self.expansion.expand(leaf)
        rollouts = self.rollout.simulate(child)
        
        results = []
        for node, sequence in rollouts:
            objective_vector = {}
            for obj in self.objective_functions:
                objective_vector[obj.name] = obj.evaluate(sequence)
            results.append(Molecule(sequence=sequence, reward=objective_vector))
        
            backprop_payload = ParetoStats.get_reward(objective_vector, self.archive)
            self.tree.backpropagate(node, backprop_payload)

        for i in results:
            self.archive.update(i)

        return # logging materials? leaf, child, results, archive size etc.
