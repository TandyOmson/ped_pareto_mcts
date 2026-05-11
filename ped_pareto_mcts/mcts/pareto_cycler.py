
import logging

from ped_pareto_mcts.base.tree import Tree, Node
from ped_pareto_mcts.policies.puct import ParetoPUCT
from ped_pareto_mcts.policies.generative_expansion import GenerativeExpansion
from ped_pareto_mcts.policies.generative_rollout import GenerativeRollout
from ped_pareto_mcts.policies.global_pareto_archive import ParetoArchive, ParetoBackprop, Molecule
from ped_pareto_mcts.utils.utils import load_class
from ped_pareto_mcts.utils.rdmol_utils import smiles_to_mol

# idea for later (change updated in ParetoArchive to updated and feedback, generating this payload)
# from dataclasses import dataclass
# @dataclass(frozen=True)
# class ParetoPayload:
#     is_pareto: bool         # did this rollout go to pareto front
#     dominance_gain: int     # 0 or 1
#     hv_gain: float = 0.0    # hypervolume gain

log = logging.getLogger()

class ParetoMCTSCycler:
    def __init__(self, gen_model, config):
        # Initialise tree object and pareto archive
        # Node stats are defined in ParetoStats above
        root_node = Node(token='<bos>', parent=None)
        self.tree = Tree(root_node)
        self.archive = ParetoArchive()

        # Load policy classes and configs
        self.selection = ParetoPUCT(config["MCTS"]["exploration_const"], len(config["objective_functions"]), config["MCTS"]["maxlen"], gen_model)
        self.expansion = GenerativeExpansion(self.tree, gen_model)
        self.rollout = GenerativeRollout(config["MCTS"]["maxlen"], gen_model)
        
        # Load reward function classes and config for evaluator
        objective_functions = []
        for objective in config["objective_functions"]:
            log.info(f"Loading objective: {objective}")
            objective_config = config["objective_functions"][objective]
            objectiveClass = load_class(objective_config["class_path"])
            objective_args = objective_config["kwargs"]
            
            objective_functions.append(objectiveClass(objective, **objective_args))

        self.objective_functions = objective_functions

    def step(self):
        leaf, root_to_leaf = self.selection.traverse(self.tree.root)        
        possible_child_nodes = self.expansion.expand(leaf)
        sequence = self.rollout.simulate(leaf)

        # validate generated sequence(s) (add configs to this)
        try:
            smiles_to_mol("".join(sequence[1:-1]), allow_charges=False)
        except:
            return None
        
        objective_vector = {}
        extra_reward_info = {}
        for obj in self.objective_functions:
            obj.update_context(archive=self.archive)
            if hasattr(obj, "extra_reward_info"):
                extra_reward_info[obj.name] = obj.extra_reward_info()
            objective_vector[obj.name] = obj.evaluate("".join(sequence[1:-1]))
        newmol = Molecule(sequence="".join(sequence[1:-1]), reward=objective_vector)
        
        backprop_payload = ParetoBackprop(objective_vector, self.archive)
        self.tree.backpropagate(leaf, backprop_payload)

        self.archive.update(newmol)

        # return logging info
        step_info = {}
        step_info["leaf"] = leaf
        #step_info["possible_child_nodes"] = possible_child_nodes
        step_info["root_to_leaf"] = root_to_leaf
        
        step_info["molecule"] = "".join(sequence[1:-1])
        step_info["reward"] = objective_vector
        step_info["reward_info"] = extra_reward_info

        step_info["pareto_front"] = self.archive.front  
        
        return step_info
