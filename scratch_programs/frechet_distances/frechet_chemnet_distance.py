from ped_pareto_mcts.utils.frechet_distance import FrechetDistance

import argparse
from rdkit import Chem
from fcd import load_ref_model, get_predictions, canonical_smiles
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_smis", type=str, required=True)
    parser.add_argument("--ref_smis", type=str, required=True)
    parser.add_argument("--outfile", type=str, default=None)

    args = parser.parse_args()

    print("loading ChemNet model...")
    model = load_ref_model()

    gen_smis = [i.strip() for i in open(args.gen_smis, "r").readlines()]
    ref_smis = [i.strip() for i in open(args.ref_smis, "r").readlines()]

    print("canonicalising smiles")
    gen_smis = [i for i in canonical_smiles(gen_smis) if i is not None]
    ref_smis = [i for i in canonical_smiles(ref_smis) if i is not None]
  
    print(f"loaded {len(gen_smis)} generated and {len(ref_smis)} reference sample smiles")
    
    print("generating embeddings for generated molecules")
    gen_embs = get_predictions(model, gen_smis)
    print("generating embeddings for reference molecules")
    ref_embs = get_predictions(model, ref_smis)

    calculator = FrechetDistance()
    FD = calculator.evaluate(gen_embs, ref_embs)
    score = np.exp(-0.2 * FD)
    
    print(f"Frechet Distance: {FD}")
    print(f"score: {score}")

    if args.outfile is not None:
        with open(args.outfile, "w") as fw:
            fw.write(f"FD:{FD}, score:{score}")