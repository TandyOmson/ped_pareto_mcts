from ssl_graph_encoder.utils.smiles_to_graph import SmilesToGraph
from ssl_graph_encoder.utils.model_io import load_pretrained_encoder

import argparse
import traceback

import torch
from pathlib import Path
import numpy as np
import scipy as sp

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

class FrechetDistance:
    def __init__(self, encoder_file, smiles_to_graph_file, map_location="cpu"):
        self.encoder, self.stg = get_model(encoder_file, smiles_to_graph_file, map_location)

    def evaluate(self, gen_embs, ref_embs):
        gen_embs = np.vstack(gen_embs)
        ref_embs = np.vstack(ref_embs)

        gen_mu = np.mean(gen_embs, axis=0)
        ref_mu = np.mean(ref_embs, axis=0)

        gen_cov = np.cov(gen_embs.T)
        ref_cov = np.cov(ref_embs.T)

        fd = self.frechet_distance(gen_mu, gen_cov, ref_mu, ref_cov)

        return fd

    def get_embeddings(self, smi):
        z, _ = _stg.smiles_to_graphs(smi, return_all_confs=False)
        emb = self.encoder(z)

        return emb.to("cpu").numpy().astype(np.float64)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_file", type=str, required=True)
    parser.add_argument("--smiles_to_graph_file", type=str, required=True)
    parser.add_argument("--gen_smis", type=str, required=True)
    parser.add_argument("--ref_smis", type=str, required=True)

    args = parser.parse_args()

    print("loading models...")
    calculator = FrechetDistance(args.encoder_file, args.smiles_to_graph_file)

    gen_smis = [i.strip() for i in open(args.gen_smis, "r").readlines()]
    ref_smis = [i.strip() for i in open(args.ref_smis, "r").readlines()]
    
    print(f"loaded {len(gen_smis)} generated and {len(ref_smis)} reference sample smiles")

    print("generating embeddings for generated molecules")
    gen_embs = []
    for count, smi in enumerate(gen_smis):
        print(f"gen mols {count} of {len(gen_smis)}", end="\r")
        try:
            with torch.no_grad():
                emb = calculator.get_embeddings(smi)
            gen_embs.append(emb)
        except:
            print("embedding failed for", count, smi)
            #traceback.print_exc()

    print("generating embeddings for reference molecules")
    ref_embs = []
    for count, smi in enumerate(ref_smis):
        print(f"ref mols {count} of {len(ref_smis)}", end="\r")
        try:
            with torch.no_grad():
                emb = calculator.get_embeddings(smi)
            ref_embs.append(emb)
        except:
            print("embedding failed for", count, smi)

    FD = calculator.evaluate(gen_embs, ref_embs)
    score = np.exp(-0.2 * FD)
    
    print(f"Frechet Distance: {FD}")
    print(f"score: {score}")

    with open("frechet.out", "w") as fw:
        fw.write(f"FD:{FD}, score:{score}")
