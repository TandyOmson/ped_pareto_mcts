
from base.tree import Tree, Node
from utils.utils import load_class, filter_class_config

class ParetoArchive:
    """ Maintains a set of non-dominated reward vectors
        Produces pareto-derived metrics for MCTS functions
    """

    @abstractmethod
    def update_archive(self, rewards: RewardVector) -> None:
        """ Try to insert a reward vector into the archive
        """
        raise NotImplementedError
    
    @abstractmethod
    def is_non_dominated(self, rewards: RewardVector) -> bool:
        """ Test wheter a reward vector is non-dominated w.r.t. the current archive
        """
        raise NotImplementedError

class ParetoMCTSCycler:
    def __init__(self, config):
        # Initialise tree object and pareto archive
        root_node = Node(token='&', parent=None)
        tree = Tree(root_node)

        # Load policy classes and configs
        selectionClass = load_class(config["selection_policy"]["class_path"])
        selection_args = filter_class_config(selectionClass, **config["selection_policy"]["kwargs"])
        
        expansionClass = load_class(config["expansion_policy"]["class_path"])
        expansion_args = filter_class_config(expansionClass, **config["expansion_policy"]["kwargs"])
        
        rolloutClass = load_class(config["rollout_policy"]["class_path"])
        rollout_args = filter_class_config(rolloutClass, **config["rollout_policy"]["kwargs"])

        self.selection = selectionClass(**selection_args)
        self.expansion = expansionClass(**expansion_args)
        self.rollout = rolloutClass(**rollout_args)
        
        # Load reward function classes and config for evaluator  

    def step(self):
        leaf = self.selection.select(self.tree, self.archive)
        child = self.expansion.expand(leaf)
        rollouts = self.rollout.simulate(child)
        results = self.evaluator(rollouts)
        
        self.archive.update(rollouts, results)
        self.tree.backpropagate(child, results, self.archive)

        return # logging materials? leaf, child, results, archive size etc.
