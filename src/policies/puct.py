""" ParetoPUCT Selection policy
"""
import random
import numpy as np
from base.policies import SelectionPolicy

class ParetoPUCT(SelectionPolicy):
    def __init__(self, exploration_const, maxlen, model):
        self.c_val = exploration_const
        self.maxlen = maxlen
        self.model = model

    def traverse(self, root_node):
        node = root_node
        # while non-terminal valid leaf node
        while len(node.children.values()) != 0 and node.token != '$':
            node = self.select(node)

        return node

    def select(self, node):
        puct = []
        for childnode in node.children.values():
            puct.append(self.calc_puct(childnode, self.model))
        
        puct_idxs = self.get_pareto_front_idxs(puct)
        idx = random.choice(puct_idxs)
        
        return list(node.children.values())[idx]

    def calc_puct(self, node):
        if node.visit_count == 0:
            return np.full(len(node.stats), np.inf)

        # first term stats/visits ensures the MCTS exploits the nodes with mulltiple high stat metrics
        node_stats = np.array([i for i in node.stats.values()])/node.visit_count
        
        # second term constant guides MCTS to initially prefer nodes with low vist count
        visit_const = np.sqrt(node.parent.visit_count) / (1 + node.visit_count) 
        
        # multiplied by model probability (that this is the next token in sequence)
        path = []
        current_node = node
        while current_node.parent is not None:
            path.insert(0, current_node.token)
            current_node = current_node.parent
        
        prob = self.model.get_prob_next_symbol(path, node.token)

        # This is an array
        return node_stats + (self.c_val * prob * visit_const)

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