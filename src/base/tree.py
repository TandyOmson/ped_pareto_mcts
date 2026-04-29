from abc import abstractmethod, ABC
from typing import Hashable

Token = Hashable
from base.pareto_archive import NodeArchiveStats

class Node(ABC):
    """ Node containing a sequence token
    """
    def __init__(self, token, parent=None):
        self.token = token
        if parent is not None:
            assert isinstance(parent, Node)
        
        self.visit_count = 0
        self.archive_stats = None

class Tree(ABC):
    """ Owns the MCTS node linkage and updates node statistics
    """
    def __init__(self, root: Node):
        self.root = root
    
    @abstractmethod
    def get_or_create_child(self, parent: Node, token: Token):
        """ Return the child corresponding to token, if it doesnt exist, create and attach
        """
        raise NotImplementedError
    
    def backpropagate(self, node: Node, backprop_payload: NodeArchiveStats) -> None:
        """ Update the node and its ancestors visit count and archive stats (pareto dervied metrics)
            ArchiveStats is defined in policies 
        """
        current = node
        while current is not None:
            current.visit_count += 1
            
            # update node statistics
            stats = node.archive_stats
            if hasattr(stats, "update_node"):
                stats.update(backprop_payload)

            current = current.parent
        