""" Generates the frechet distance between two sets of SMILES
"""

from ped_pareto_mcts.utils.get_ssl_embeddings import sslEmbeddings
from ped_pareto_mcts.utils.frechet_distance import FrechetDistance

import argparse
import traceback

import numpy as np
import torch
from pathlib import Path
from rdkit import Chem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_file", type=str, required=True)
    parser.add_argument("--smiles_to_graph_file", type=str, required=True)
    parser.add_argument("--gen_smis", type=str, required=True)
    parser.add_argument("--ref_smis", type=str, required=True)
    parser.add_argument("--outfile", type=str, default=None)

    args = parser.parse_args()

    print("loading models...")
    gen_smis = [i.strip() for i in open(args.gen_smis, "r").readlines()]
    ref_smis = [i.strip() for i in open(args.ref_smis, "r").readlines()]

    print("canonicalising smiles")
    canon_gen_smis = []
    for i in gen_smis:
        try:
            smi = Chem.CanonSmiles(i)
            canon_gen_smis.append(smi)
        except:
            continue
    gen_smis = canon_gen_smis

    canon_ref_smis = []
    for i in ref_smis:
        try:
            smi = Chem.CanonSmiles(i)
            canon_ref_smis.append(smi)
        except:
            continue
    ref_smis = canon_ref_smis
    
    print(f"loaded {len(gen_smis)} generated and {len(ref_smis)} reference sample smiles")
    embedder = sslEmbeddings(args.encoder_file, args.smiles_to_graph_file)

    print("generating embeddings for generated molecules")
    gen_embs = []
    for count, smi in enumerate(gen_smis):
        print(f"gen mols {count} of {len(gen_smis)}", end="\r")
        try:
            with torch.no_grad():
                emb = embedder.get_embeddings(smi)
            gen_embs.append(emb)
        except:
            print("embedding failed for", count, smi)
            traceback.print_exc()

    print("generating embeddings for reference molecules")
    ref_embs = []
    for count, smi in enumerate(ref_smis):
        print(f"ref mols {count} of {len(ref_smis)}", end="\r")
        try:
            with torch.no_grad():
                emb = embedder.get_embeddings(smi)
            ref_embs.append(emb)
        except:
            print("embedding failed for", count, smi)

    calculator = FrechetDistance()
    FD = calculator.evaluate(gen_embs, ref_embs)
    score = np.exp(-0.2 * FD)
    
    print(f"Frechet Distance: {FD}")
    print(f"score: {score}")

    if args.outfile is not None:
        with open(args.outfile, "w") as fw:
            fw.write(f"FD:{FD}, score:{score}")
