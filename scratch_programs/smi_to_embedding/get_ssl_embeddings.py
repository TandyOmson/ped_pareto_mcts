""" Retrieve SSL embeddings for a given model for a given set of SMILES
"""

from ped_pareto_mcts.utils.get_ssl_embeddings import sslEmbeddings
import argparse
from rdkit import Chem
import torch
import traceback
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 


def safe_embed(smi, embedder, emb_dim):
    try:
        with torch.no_grad():
            return embedder.get_embeddings(smi)[0]
    except:
        return np.array([np.nan]*emb_dim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_file", type=str, required=True)
    parser.add_argument("--smiles_to_graph_file", type=str, required=True)
    parser.add_argument("--smis", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=4)

    args = parser.parse_args()

    print("loading and canonicalising smiles")
    smis = [i.rstrip() for i in open(args.smis, "r").readlines()]
    canon_smis = []
    for i in smis:
        try:
            smi = Chem.CanonSmiles(i)
            canon_smis.append(smi)
        except:
            continue

    smis = canon_smis
    
    print(f"loaded {len(smis)} SMILES")
    print("loading model")
    embedder = sslEmbeddings(args.encoder_file, args.smiles_to_graph_file)
    
    print("generating embeddings")
    emb_dim = len(embedder.get_embeddings("C")[0])
    embs = Parallel(
        n_jobs=args.n_workers,
        backend="threading",
    )(
        delayed(safe_embed)(smi, embedder, emb_dim)
        for smi in tqdm(smis)
    )

    np.save(args.outfile, embs)
