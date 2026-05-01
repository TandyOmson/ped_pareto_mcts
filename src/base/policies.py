""" MCTS policies
    - Selection: Read-only, looks at the tree linkage and node stats  
    - Expansion: The only thing that mutates the tree structure
    - Rollout: Apply simulations
"""

from abc import ABC, abstractmethod
from typing import List
from base.tree import Tree, Node

class SelectionPolicy(ABC):
    """ Read-only 
    """
    @abstractmethod
    def traverse(self, root_node: Node) -> tuple[List, Node]:
        """ Calls select from current root node until a Node is selected
            Returns the sequence up to that point and the node
        """
        raise NotImplementedError

    @abstractmethod
    def select(self, node: Node) -> Node:
        """looks at a node and decides which child node to traverse
        """
        raise NotImplementedError
    
class ExpansionPolicy(ABC):
    """ The only thing that mutates the tree structure
        Controls when and how nodes are expanded
    """
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
    def simulate(self, node: Node) -> List:
        """ Perform a rollout starting from node, return a sequence objective vector
            (this vector will be processed later for backprop, as defined in NodeStas)
        """
        raise NotImplementedError
