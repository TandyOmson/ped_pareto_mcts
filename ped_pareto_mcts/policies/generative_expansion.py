""" Applies an autoregressive generative model to execute expansion and rollouts 
"""
from ped_pareto_mcts.base.policies import ExpansionPolicy

class GenerativeExpansion(ExpansionPolicy):
    """ Controls expansion using next-token probabilities from an autoreggresive generative model
    """
    def __init__(self, tree, model):
        self.tree = tree
        self.model = model

    def expand(self, node):
        path = []
        current_node = node
        while current_node.parent is not None:
            path.insert(0, current_node.token)
            current_node = current_node.parent
        path.insert(0, '<bos>')
        
        prob_dict = self.model.get_all_prob_next_symbol(path)
        # filter out impossible or really unlikely nodes
        prob_dict_filtered = {k: v for k, v in prob_dict.items() if k != '<pad>' and k != '<bos>' and k != '<unk>' and v > 1e-5}

        # initialize child nodes
        for token in prob_dict_filtered.keys():
            self.tree.get_or_create_child(parent=node, token=token)
        
        return prob_dict_filtered
