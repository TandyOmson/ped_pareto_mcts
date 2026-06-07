""" Dummy score function for testing. Returns a random score between 0 and 10 (similar to vina score range)
"""

from ped_pareto_mcts.base.reward import ObjectiveFunc
import random

class DummyScore(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        return random.uniform(0, 10)