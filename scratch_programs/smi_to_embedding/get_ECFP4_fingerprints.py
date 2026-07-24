""" Retrieves ECFP4 fingerprints
"""
import argparse
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)    

    print("generating embeddings")
    #embs = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smis]

    embs = Parallel(
        n_jobs=args.n_workers,
        backend="threading",
    )(
        delayed(lambda smi: fpgen.GetFingerprint(Chem.MolFromSmiles(smi)))(smi)
        for smi in tqdm(smis)
    )


    np.save(args.outfile, embs)
