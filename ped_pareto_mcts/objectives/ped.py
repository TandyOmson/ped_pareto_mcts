""" Loads a trained graph encoder model that converts molecular graphs to embeddings 
    Assigns a reward based on the euclidian embedding distance between the generated molecule and the current pareto front
    Metric used will be minimum euclidian pairwise distance to pareto front
"""

from ped_pareto_mcts.utils.get_ssl_embeddings import sslEmbeddings
from ped_pareto_mcts.base.reward import ObjectiveFunc

import torch
import numpy as np

class PairwiseEmbeddingDistance(ObjectiveFunc):
    def __init__(self, name, encoder_file, smiles_to_graph_file, map_location="cpu"):
        super().__init__(name)
        self.embedder = sslEmbeddings(encoder_file, smiles_to_graph_file, map_location="cpu")
        self._embed_ref = {}

    def update_context(self, *, archive):       
        # convert pareto front members to smiles
        for i in archive.front:
            if i.sequence not in self._embed_ref:
                self._embed_ref[i.sequence] = self.embedder.get_embeddings(i.sequence)

    def evaluate(self, smi):
        with torch.no_grad():
            if not self._embed_ref.values():
                return 0.0
            else:
                z = self.embedder.get_embeddings(smi)
                return float(np.mean(np.linalg.norm(z - z_i) for z_i in self._embed_ref.values()))
