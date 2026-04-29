class MCTSCycler:
    def __init__(self):
        pass

    def step(self):
        leaf = self.selection.select(self.tree, self.archive)
        child = self.expansion.expand(leaf)
        rollouts = self.rollout.simulate(child)
        results = self.evaluator(rollouts)
        
        self.archive.update(rollouts, results)
        self.tree.backpropagate(child, results, self.archive)

        return # logging materials? leaf, child, results, archive size etc.
