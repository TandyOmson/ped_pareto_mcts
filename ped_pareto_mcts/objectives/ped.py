""" Loads a trained graph encoder model that converts molecular graphs to embeddings 
    Assigns a reward based on the euclidian embedding distance between the generated molecule and the current pareto front
    Metric used will be minimum euclidian pairwise distance to pareto front
"""

from ssl_graph_encoder.utils.smiles_to_graph import SmilesToGraph
from ssl_graph_encoder.utils.model_io import load_pretrained_encoder

from ped_pareto_mcts.base.reward import ObjectiveFunc

import torch
from pathlib import Path
import numpy as np

_model = None
_stg = None

def get_model(encoderfile, stgfile, map_location):
    global _model
    global _stg
    if _model is None:
        # modelfile includes encoder class path
        _model, _ = load_pretrained_encoder(Path(encoderfile), map_location)
        _model.eval()
    if _stg is None:
        _stg = SmilesToGraph.from_config(Path(stgfile))
    return _model, _stg

class PairwiseEmbeddingDistance(ObjectiveFunc):
    def __init__(self, name, encoder_file, smiles_to_graph_file, map_location="cpu"):
        super().__init__(name)

        self.encoder, self.stg = get_model(encoder_file, smiles_to_graph_file, map_location)

    def update_context(self, *, archive):       
        # convert pareto front members to smiles
        archive_smiles = [i.sequence for i in archive]
        self._embed_ref = [self.get_embeddings(smi) for smi in archive_smiles]

    def evaluate(self, smi):
        with torch.no_grad():
            z = self.get_embeddings(smi)
            return min(np.linalg.norm(z - z_i) for z_i in self._embed_ref)
        
    def get_embeddings(self, smi):
        # convert smiles to graph based on spec from ssl_graph_encoder
        z, _ = _stg.smiles_to_graphs(smi, return_all_confs=False)
        # embed graphs using model
        emb = self.encoder(z)
        
        return emb
