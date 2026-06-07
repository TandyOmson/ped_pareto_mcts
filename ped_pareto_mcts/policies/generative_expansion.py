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
        prob_dict = {k: v for k, v in prob_dict.items() if k != '<pad>' and k != '<bos>' and k != '<unk>'}

        # renormalize probabilities after filtering out unwanted tokens
        total_prob = sum(prob_dict.values())
        prob_dict = {k: v / total_prob for k, v in prob_dict.items()}

        # Use cumulative probabilities to filter out unlikely tokens (i.e. sort by most likely and include until threshold is reached)
        expansion_threshold = 0.995
        sorted_dict = sorted(prob_dict, key=prob_dict.get, reverse=True)
        
        cumulative_prob = 0.0
        prob_dict_filtered = {}
        while cumulative_prob < expansion_threshold and len(sorted_dict) > 0:
            token = sorted_dict.pop(0)
            prob = prob_dict[token]
            cumulative_prob += prob
            prob_dict_filtered[token] = prob

        # check not initial node (prevents nothing being generated)
        if node.parent is None and "<eos>" in prob_dict_filtered.keys():
            prob_dict_filtered.pop("<eos>")

        # initialize child nodes
        for token in prob_dict_filtered.keys():
            self.tree.get_or_create_child(parent=node, token=token)
        
        return prob_dict_filtered
