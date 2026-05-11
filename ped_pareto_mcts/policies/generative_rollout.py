from ped_pareto_mcts.base.policies import RolloutPolicy
from numpy.random import default_rng

class GenerativeRollout(RolloutPolicy):
    """ Rollout conisists of selecting each symbol greedily (according to max probability from model)
    """
    def __init__(self, maxlen, model):
        self.maxlen = maxlen
        self.model = model

    def simulate(self, node):
        """ Get completed sequence
        """
        path = [node.token]
        current_node = node
        while current_node.parent is not None:
            current_node = current_node.parent
            path.insert(0, current_node.token)
        
        while len(path) < self.maxlen+1 and path[-1] != '<eos>':
            next_symbol_probs = self.model.get_all_prob_next_symbol(path)
            next_symbol_probs = {k: v for k, v in next_symbol_probs.items() if k != '<pad>' and k != '<bos>' and k != '<unk>'}

            # renormalize probabilities (in case of numerical issues)
            total_prob = sum(next_symbol_probs.values())
            next_symbol_probs = {k: v / total_prob for k, v in next_symbol_probs.items()}
            
            # weight choice using probabilities
            next_symbol = default_rng().choice(list(next_symbol_probs.keys()), p=list(next_symbol_probs.values()))
            path.append(next_symbol)
        
        if path[-1] != '<eos>':
            path.append('<eos>')

        return path