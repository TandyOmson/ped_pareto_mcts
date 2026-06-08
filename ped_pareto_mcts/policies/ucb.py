""" UCB (upper confidence bound) selection policy for single objective
"""

import numpy as np
import random
from ped_pareto_mcts.base.policies import SelectionPolicy

class UCB(SelectionPolicy):
    def __init__(self, exploration_const):
        self.c_val = exploration_const

    def traverse(self, root_node):
        node = root_node
        root_to_leaf = [node.token]
        # while non-terminal valid leaf node
        while len(node.children) != 0 and node.token != '<eos>':
            node = self.select(node)
            root_to_leaf.append(node.token)

        return node, root_to_leaf

    def select(self, node):
        ucb = []
        for childnode in node.children.values():
            ucb.append(self.calc_ucb(childnode))

        max_ucb_idxs = np.argwhere(ucb == np.max(ucb)).flatten()
        idx = random.choice(list(max_ucb_idxs))

        return list(node.children.values())[idx]

    def calc_ucb(self, node):
        if node.visit_count == 0:
            exploitation = 0
        else:
            exploitation = list(node.stats.values())[0]/node.visit_count
        
        not_visited_const = np.sqrt(2 * np.log(node.parent.visit_count) / (0.001 + node.visit_count))
        
        return exploitation + (self.c_val * not_visited_const)
