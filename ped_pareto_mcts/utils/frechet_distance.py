import numpy as np
import scipy as sp

class FrechetDistance:
    def __init__(self):
        pass

    def evaluate(self, gen_embs, ref_embs):
        gen_embs = np.vstack(gen_embs)
        ref_embs = np.vstack(ref_embs)

        gen_mu = np.mean(gen_embs, axis=0)
        ref_mu = np.mean(ref_embs, axis=0)

        gen_cov = np.cov(gen_embs.T)
        ref_cov = np.cov(ref_embs.T)

        fd = self.frechet_distance(gen_mu, gen_cov, ref_mu, ref_cov)

        return fd
    
    def frechet_distance(self, mu1, cov1, mu2, cov2, eps=1e-6):
        diff = mu1 - mu2

        covmean, _ = sp.linalg.sqrtm(cov1.dot(cov2), disp=False)
        is_real = np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)
        if not is_real:
            offset = np.eye(cov1.shape[0]) * eps
            covmean = sp.linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * tr_covmean)
