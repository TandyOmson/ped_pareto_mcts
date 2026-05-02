from base.policies import RolloutPolicy

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
        
        while len(path) < self.maxlen and path[-1] != '$':
            next_symbol_probs = self.model.get_all_prob_next_symbol(path)
            next_symbol = max(next_symbol_probs, key=next_symbol_probs.get)
            path.append(next_symbol)

        return path