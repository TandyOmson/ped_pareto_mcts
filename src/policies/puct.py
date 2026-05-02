""" ParetoPUCT Selection policy
"""
import random
import numpy as np
from base.policies import SelectionPolicy

class ParetoPUCT(SelectionPolicy):
    def __init__(self, exploration_const, num_objectives, maxlen, model):
        self.c_val = exploration_const
        self.num_objectives = num_objectives
        self.maxlen = maxlen
        self.model = model

    def traverse(self, root_node):
        node = root_node
        root_to_leaf = [node.token]
        # while non-terminal valid leaf node
        while len(node.children) != 0 and node.token != '$':
            node = self.select(node)
            root_to_leaf.append(node.token)

        return node, root_to_leaf

    def select(self, node):
        puct = []
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
            exploitation = np.array([i for i in node.stats.values()])/node.visit_count
         
        # model probability (that this is the next token in sequence)
        path = []
        current_node = node
        while current_node.parent is not None:
            path.insert(0, current_node.token)
            current_node = current_node.parent

        # model probability multiplied by second term constant guides MCTS to initially prefer nodes with low vist count
        not_visited_const = np.sqrt(current_node.visit_count) / (1 + node.visit_count) 
        
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