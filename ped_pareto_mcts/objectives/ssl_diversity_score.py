""" Diversity score using embddings from SSL graph encoder
    Currently I am using what I call the "fast frechet distance" which doesnt use the sqrtm covariance product
    Could change that later
"""
import torch
import numpy as np
from pathlib import Path

from ped_pareto_mcts.utils.get_ssl_embeddings import sslEmbeddings
from ped_pareto_mcts.base.reward import ObjectiveFunc

class sslDiversityScore(ObjectiveFunc):
    def __init__(
        self,
        name,
        encoder_file,
        smiles_to_graph_file,
        map_location="cpu",
        ref_threshold=5000,
        gen_threshold=1000,
    ):
        super().__init__(name)

        self.embedder = sslEmbeddings(
            encoder_file,
            smiles_to_graph_file,
            map_location=map_location,
        )

        self.ref_threshold = ref_threshold
        self.gen_threshold = gen_threshold
        self.n_seen = 0

        z = self.embedder.get_embeddings("C")[0]
        dim = len(z)
        self.ref_embs = np.zeros((ref_threshold, dim))
        self.gen_embs = np.zeros((gen_threshold, dim))

        self.mu_g = None
        self.mu_r = None
        self.tr_cov_g = None
        self.tr_cov_r = None
        self.sumsq_g = None
        self.n_g = None
        self.base_fd = None

    def evaluate(self, smi):
        self.n_seen += 1

        with torch.no_grad():
            z = self.embedder.get_embeddings(smi)[0]

        # --- build reference set ---
        if self.n_seen < self.ref_threshold:
            self.ref_embs[self.n_seen - 1] = z
            return 0.0

        # --- build initial generated set ---
        elif self.n_seen < self.ref_threshold + self.gen_threshold:
            idx = self.n_seen - self.ref_threshold - 1
            self.gen_embs[idx] = z
            return 0.0

        # --- initialise statistics ---
        elif self.n_seen == self.ref_threshold + self.gen_threshold:
            self.init_mus_covs()
            return 0.0

        # --- normal operation ---
        else:
            score = self.delta_fd(z)
            self.update_mus_covs(z)
            return score
        
    def init_mus_covs(self):
        # --- gen stats ---
        self.n_g = len(self.gen_embs)

        self.mu_g = self.gen_embs.mean(axis=0)
        Xc = self.gen_embs - self.mu_g

        # trace covariance via sum of squared deviations
        self.sumsq_g = np.sum(Xc**2)
        self.tr_cov_g = self.sumsq_g / (self.n_g - 1)

        # --- ref stats ---
        m = len(self.ref_embs)
        self.mu_r = self.ref_embs.mean(axis=0)
        Yc = self.ref_embs - self.mu_r
        tr_cov_r = np.sum(Yc**2) / (m - 1)

        self.tr_cov_r = tr_cov_r

        # --- baseline FD ---
        diff = self.mu_g - self.mu_r
        self.base_fd = diff @ diff + self.tr_cov_g + self.tr_cov_r

    def delta_fd(self, z):
        n = self.n_g
        n_new = n + 1

        mu = self.mu_g
        sumsq = self.sumsq_g

        # simulate add-one
        delta = z - mu
        mu_new = mu + delta / n_new
        delta2 = z - mu_new

        sumsq_new = sumsq + np.dot(delta, delta2)
        tr_cov_new = sumsq_new / (n_new - 1)

        diff_new = mu_new - self.mu_r
        fd_new = diff_new @ diff_new + tr_cov_new + self.tr_cov_r

        return fd_new - self.base_fd

class sslDiversityScoreToRef(ObjectiveFunc):
    def __init__(
        self,
        name,
        encoder_file,
        smiles_to_graph_file,
        reference_smiles_file,
        map_location="cpu",
        gen_threshold=1000,
    ):
        super().__init__(name)

        self.embedder = sslEmbeddings(
            encoder_file,
            smiles_to_graph_file,
            map_location=map_location,
        )

        self.gen_threshold = gen_threshold
        self.n_seen = 0

        z = self.embedder.get_embeddings("C")[0]
        dim = len(z)
        self.ref_embs = [self.embedder.get_embeddings(smi.strip()) for smi in open(Path(reference_smiles_file), "r").readlines()]
        self.gen_embs = np.zeros((gen_threshold, dim))

        self.mu_g = None
        self.mu_r = None
        self.tr_cov_g = None
        self.tr_cov_r = None
        self.sumsq_g = None
        self.n_g = None
        self.base_fd = None

    def evaluate(self, smi):
        self.n_seen += 1

        with torch.no_grad():
            z = self.embedder.get_embeddings(smi)[0]

        # --- build initial generated set ---
        if self.n_seen < self.gen_threshold:
            idx = self.n_seen - 1
            self.gen_embs[idx] = z
            return 0.0

        # --- initialise statistics ---
        elif self.n_seen == self.gen_threshold:
            self.init_mus_covs()
            return 0.0

        # --- normal operation ---
        else:
            score = self.delta_fd(z)
            self.update_mus_covs(z)
            return score
        
    def init_mus_covs(self):
        # --- gen stats ---
        self.n_g = len(self.gen_embs)

        self.mu_g = self.gen_embs.mean(axis=0)
        Xc = self.gen_embs - self.mu_g

        # trace covariance via sum of squared deviations
        self.sumsq_g = np.sum(Xc**2)
        self.tr_cov_g = self.sumsq_g / (self.n_g - 1)

        # --- ref stats ---
        m = len(self.ref_embs)
        self.mu_r = self.ref_embs.mean(axis=0)
        Yc = self.ref_embs - self.mu_r
        tr_cov_r = np.sum(Yc**2) / (m - 1)

        self.tr_cov_r = tr_cov_r

        # --- baseline FD ---
        diff = self.mu_g - self.mu_r
        self.base_fd = diff @ diff + self.tr_cov_g + self.tr_cov_r

    def delta_fd(self, z):
        n = self.n_g
        n_new = n + 1

        mu = self.mu_g
        sumsq = self.sumsq_g

        # simulate add-one
        delta = z - mu
        mu_new = mu + delta / n_new
        delta2 = z - mu_new

        sumsq_new = sumsq + np.dot(delta, delta2)
        tr_cov_new = sumsq_new / (n_new - 1)

        diff_new = mu_new - self.mu_r
        fd_new = diff_new @ diff_new + tr_cov_new + self.tr_cov_r

        return fd_new - self.base_fd
