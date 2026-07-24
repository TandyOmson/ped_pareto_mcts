import argparse

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolSurf

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

def calc_descriptors(smi):
    try:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            return {
                "smiles": smi,
                "logP": np.nan,
                "ASA": np.nan,
                "MolWt": np.nan,
            }

        return {
            "smiles": smi,
            "logP": Descriptors.MolLogP(mol),
            "ASA": MolSurf.LabuteASA(mol)
            "MolWt": Descriptors.MolWt(mol),
        }

    except Exception:
        return {
            "smiles": smi,
            "logP": np.nan,
            "ASA": np.nan,
            "MolWt": np.nan,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smifile",
        required=True,
        type=str,
        help="Text file containing one SMILES per line",
    )
    parser.add_argument(
        "--outfile",
        required=True,
        type=str,
        help="Output CSV file",
    )
    parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
    )

    args = parser.parse_args()

    print("Loading SMILES...")
    with open(args.smifile) as f:
        smis = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(smis):,} SMILES")

    print("Calculating descriptors...")

    results = Parallel(
        n_jobs=args.n_workers,
        backend="threading",
    )(
        delayed(calc_descriptors)(smi)
        for smi in tqdm(smis)
    )

    df = pd.DataFrame(results)

    print(f"Writing {args.outfile}")
    df.index = list(range(1, len(df)+1))
    df.to_csv(args.outfile)

    print("Done")


if __name__ == "__main__":
    main()
