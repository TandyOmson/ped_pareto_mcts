from ped_pareto_mcts.utils.frechet_distance import FrechetDistance
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import argparse
import numpy as np

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

def get_ECFP4(smi, nbits=2048):
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(smi))
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)

    return fpgen.GetFingerprint(mol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_smis", type=str, required=True)
    parser.add_argument("--ref_smis", type=str, required=True)
    parser.add_argument("--outfile", type=str, default=None)

    args = parser.parse_args()

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
    
    print("generating embeddings for generated molecules")
    gen_embs = [get_ECFP4(s) for s in gen_smis]
    print("generating embeddings for reference molecules")
    ref_embs = [get_ECFP4(s) for s in ref_smis]

    calculator = FrechetDistance()
    FD = calculator.evaluate(gen_embs, ref_embs)
    score = np.exp(-0.2 * FD)
    
    print(f"Frechet Distance: {FD}")
    print(f"score: {score}")

    if args.outfile is not None:
        with open(args.outfile, "w") as fw:
            fw.write(f"FD:{FD}, score:{score}")
