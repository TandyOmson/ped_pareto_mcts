
from base.tree import Tree, Node
from policies.puct import ParetoPUCT
from policies.generative_expansion import GenerativeExpansion
from policies.generative_rollout import GenerativeRollout
from policies.global_pareto_archive import ParetoArchive, ParetoStats, Molecule
from utils.utils import load_class

# idea for later (change updated in ParetoArchive to updated and feedback, generating this payload)
# from dataclasses import dataclass
# @dataclass(frozen=True)
# class ParetoPayload:
#     is_pareto: bool         # did this rollout go to pareto front
#     dominance_gain: int     # 0 or 1
#     hv_gain: float = 0.0    # hypervolume gain

class ParetoMCTSCycler:
    def __init__(self, gen_model, config):
        # Load policy classes and configs
        self.selection = ParetoPUCT(config["MCTS"]["exploration_const"], config["MCTS"]["maxlen"], gen_model)
        self.expansion = GenerativeExpansion(self.tree, gen_model)
        self.rollout = GenerativeRollout(self.tree, config["MCTS"]["maxlen"], gen_model)
        
        # Load reward function classes and config for evaluator
        objective_functions = []
        for objective in config["objective_functions"]:
            objective_config = config["objective_functions"][objective]
            objectiveClass = load_class(objective_config["class_path"])
            objective_args = objective_config["kwargs"]
            
            objective_functions.append(objectiveClass(objective, **objective_args))

        self.objective_functions = objective_functions

        # Initialise tree object and pareto archive
        # Node stats are defined in ParetoStats above
        root_node = Node(token='&', parent=None)
        self.tree = Tree(root_node)
        self.archive = ParetoArchive()

    def step(self):
        leaf = self.selection.select(self.tree, self.archive)
        possible_child_nodes = self.expansion.expand(leaf)
        sequence = self.rollout.simulate(leaf)
        
        objective_vector = {}
        for obj in self.objective_functions:
            if hasattr(obj, "extra_reward_info"):
                extra_reward_info = obj.extra_reward_info()
            objective_vector[obj.name] = obj.evaluate(sequence)
        newmol = Molecule(sequence=sequence, reward=objective_vector)
        
        backprop_payload = ParetoStats.get_reward(objective_vector, self.archive)
        self.tree.backpropagate(leaf, backprop_payload)

        self.archive.update(newmol)

        # return logging info
        step_info = {}
        step_info["leaf"] = leaf
        step_info["possible_child_nodes"] = possible_child_nodes
        step_info["path"] = sequence
        step_info["pareto_front"] = self.archive.front  
        step_info["reward_info"] = extra_reward_info
        
        return step_info