from ped_pareto_mcts.base.reward import ObjectiveFunc
from utils.utils import load_class


def get_model(modelfile, map_location):
    global _model
    if _model is None:
        _model = load_pretrained_encoder(modelfile, map_location)
        _model.eval()
    return _model

class PairwiseEmbeddingDistance(ObjectiveFunc):
    def __init__(self, name, modelfile, map_location="cpu"):
        super().__init__(name)
        self.model = get_model(modelfile, map_location)

    def evaluate(self, smi):
        with torch.no_grad():
            ped = 
        return ped