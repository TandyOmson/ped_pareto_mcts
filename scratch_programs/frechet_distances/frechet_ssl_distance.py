""" Generates the frechet distance between two sets of SMILES
"""

from ped_pareto_mcts.utils.get_ssl_embeddings import sslEmbeddings
from ped_pareto_mcts.utils.frechet_distance import FrechetDistance

import argparse
import numpy as np
import torch
from rdkit import Chem
from joblib import Parallel, delayed
from tqdm import tqdm

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
    parser.add_argument("--gen_smis", type=str, default=None)
    parser.add_argument("--gen_embs", type=str, default=None)
    parser.add_argument("--ref_smis", type=str, required=True)
    parser.add_argument("--outfile", type=str, default=None)

    args = parser.parse_args()

    embedder = sslEmbeddings(args.encoder_file, args.smiles_to_graph_file)
    emb_dim = len(embedder.get_embeddings("C")[0])

    ref_smis = [i.strip() for i in open(args.ref_smis, "r").readlines()]

    if args.gen_embs is not None:
        gen_embs = np.load(args.gen_embs)
    elif args.gen_smis is not None and args.gen_embs is None:
        gen_smis = [i.strip() for i in open(args.gen_smis, "r").readlines()]

        canon_gen_smis = []
        for i in gen_smis:
            try:
                smi = Chem.CanonSmiles(i)
                canon_gen_smis.append(smi)
            except:
                continue
        gen_smis = canon_gen_smis

        print("generating embeddings for generated molecules")
        gen_embs = Parallel(
            n_jobs=args.n_workers,
            backend="threading",
        )(
            delayed(safe_embed)(smi, embedder, emb_dim)
            for smi in tqdm(gen_smis)
        )
    else:
        raise ValueError("must provide either gen_embs or gen_smis")

    canon_ref_smis = []
    for i in ref_smis:
        try:
            smi = Chem.CanonSmiles(i)
            canon_ref_smis.append(smi)
        except:
            continue
    ref_smis = canon_ref_smis
    
    print("generating embeddings for reference molecules")
    ref_embs = Parallel(
        n_jobs=args.n_workers,
        backend="threading",
    )(
        delayed(safe_embed)(smi, embedder, emb_dim)
        for smi in tqdm(ref_smis)
    )

    # remove nan values from the embeddings
    gen_embs = np.array([i for i in gen_embs if not np.isnan(i).any()])
    ref_embs = np.array([i for i in ref_embs if not np.isnan(i).any()])

    print(f"loaded {len(gen_embs)} generated and {len(ref_embs)} reference sample embeddings")

    calculator = FrechetDistance()
    FD = calculator.evaluate(gen_embs, ref_embs)
    score = np.exp(-0.2 * FD)
    
    print(f"Frechet Distance: {FD}")
    print(f"score: {score}")

    if args.outfile is not None:
        with open(args.outfile, "w") as fw:
            fw.write(f"FD:{FD}, score:{score}")
