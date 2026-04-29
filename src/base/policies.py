""" MCTS policies
    - Selection: Read-only, looks at the tree linkage and node stats  
    - Expansion: The only thing that mutates the tree structure
    - Rollout: Apply simulations
"""

from abc import ABC, abstractmethod

from base.tree import Tree, Node
from base.pareto_archive import RewardVector

class SelectionPolicy(ABC):
    """ Read-only, looks at tree linkage and node stats
        Decides which child node to traverse
    """
    @abstractmethod
    def select(self, tree: Tree, node: Node) -> Node:
        raise NotImplementedError
    
class ExpansionPolicy(ABC):
    """ The only thing that mutates the tree structure
        Controls when and how nodes are expanded
    """
    @abstractmethod
    def can_expand(self, tree: Tree, node: Node) -> bool:
        """ Check elegibility of a node of expansion
        """
        raise NotImplementedError
    
    @abstractmethod
    def expand(self, tree: Tree, node: Node) -> Node:
        """ Attach one new child to node and return the new node
            Initialises stats for the new node
        """
        return NotImplementedError
    
class RolloutPolicy(ABC):
    """ Executes a rollout from a given node
    """
    @abstractmethod
    def rollout(self, node: Node) -> RewardVector:
        """ Perform a rollout starting from node, return a RewardVector
            (this vecctor will be pareto-processed later)
        """
        raise NotImplementedError
