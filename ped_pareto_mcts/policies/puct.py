""" ParetoPUCT Selection policy
"""
import random
import numpy as np
from ped_pareto_mcts.base.policies import SelectionPolicy

class ParetoPUCT(SelectionPolicy):
    def __init__(self, exploration_const, num_objectives, maxlen, model):
        self.c_val = exploration_const
        self.num_objectives = num_objectives
        self.maxlen = maxlen
        self.model = model

        # SCALE INVARIANT EXPLORATION TERM
        # Exploitation Q is scaled for objective (raw objective values are not scaled)
        self.global_min = np.full(num_objectives, np.inf)
        self.global_max = np.full(num_objectives, -np.inf)
        # Keep running standard deviation of each objective for tanh scaling (reduces the effect of outliers massively changing min and max)
        self.running_scale = np.full(num_objectives, 0.5)
        self.alpha = 0.1

    def traverse(self, root_node):
        node = root_node
        root_to_leaf = [node.token]
        # while non-terminal valid leaf node
        while len(node.children) != 0 and node.token != '<eos>':
            node = self.select(node)
            root_to_leaf.append(node.token)

        return node, root_to_leaf

    def select(self, node):
        puct = []
        if node.visit_count == 0:
            return random.choice(list(node.children.values()))
        
        for childnode in node.children.values():
            puct.append(self.calc_puct(childnode))
        
        puct_idxs = self.get_pareto_front_idxs(puct)
        idx = random.choice(puct_idxs)
        
        return list(node.children.values())[idx]

    def calc_puct(self, node):
        # first term stats/visits ensures the MCTS exploits the nodes with mulltiple high stat metrics
        if node.visit_count == 0:
            exploitation = np.zeros(self.num_objectives)
        else:
            q_vec = np.array([i for i in node.stats.values()]) / node.visit_count

            # MIN MAX SCALING
            # # update bounds on Q, not raw stats
            # self.global_min = np.minimum(self.global_min, q_vec)
            # self.global_max = np.maximum(self.global_max, q_vec)

            # # safe scaling
            # valid = self.global_max > self.global_min
            # scaling = np.where(valid, self.global_max - self.global_min, 1.0)

            # exploitation = (q_vec - self.global_min) / scaling

            # # only apply symmetric scaling for valid bounds
            # if np.any(valid):
            #     exploitation = 2 * exploitation - 1
            # else:
            #     exploitation = np.zeros_like(q_vec)

            # TANH SCALING
            if node.visit_count == 1:            
                self.running_scale = np.maximum(
                        (1 - self.alpha) * self.running_scale + self.alpha * np.abs(q_vec),
                        np.abs(q_vec)
                    )
            # exploitation
            scale = np.maximum(self.running_scale, 0.05) # minmum floor to scale
            exploitation = np.tanh(q_vec / scale)
         
        # model probability (that this is the next token in sequence)
        path = []
        current_node = node
        while current_node.parent is not None:
            path.insert(0, current_node.token)
            current_node = current_node.parent

        # model probability multiplied by second term constant guides MCTS to initially prefer nodes with low vist count
        not_visited_const = np.sqrt(node.parent.visit_count) / (1 + node.visit_count)
        
        prob = self.model.get_all_prob_next_symbol(path)[node.token]

        # This is an array
        return exploitation + (self.c_val * prob * not_visited_const)

    @staticmethod
    def get_pareto_front_idxs(puct_vectors):
        indices = []
        if len(puct_vectors) == 1:
            indices.append(0)
            return indices
        for i in range(len(puct_vectors)):
            puct_vector = np.array(puct_vectors[i])
            # [0.5, 1, 0.5] should be dominated by [0.6, 1, 0.6]
            if max((puct_vector <= np.array(puct_vectors[:i] + puct_vectors[i + 1:])).mean(axis=1)) < 1:  # non-dominated
                indices.append(i)
        return indices
