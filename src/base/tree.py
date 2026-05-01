from typing import Hashable

Token = Hashable
from base.policies import NodeStats

class Node:
    """ Node containing a sequence token
    """
    def __init__(self, token=None, parent=None):
        self.token = token
        if parent is not None:
            assert isinstance(parent, Node)
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.stats = None

class Tree:
    """ Owns the MCTS node linkage and updates node statistics
    """
    def __init__(self, root: Node):
        self.root = root
    
    def get_or_create_child(self, parent: Node, token: Token):
        """ Return the child corresponding to token, if it doesnt exist, create and attach
        """
        if token not in parent.children.keys():
            parent.children[token] = Node(token=token, parent=parent)
        return parent.children[token]
    
    def backpropagate(self, node: Node, backprop_payload: NodeStats) -> None:
        """ Update the node and its ancestors visit count and archive stats (pareto dervied metrics)
            ArchiveStats is defined in policies 
        """
        while node is not None:
            node.visit_count += 1            
            # update node statistics
            if node.stats is not None:
                node.stats.update(backprop_payload)
            else:
                NodeStats.init_stats(node)
                node.stats.update(backprop_payload)

            node = node.parent
