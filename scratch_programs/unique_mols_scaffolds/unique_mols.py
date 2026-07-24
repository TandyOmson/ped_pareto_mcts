import argparse

from joblib import Parallel, delayed
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

def process_smiles(smi):
    """
    Returns:
        canonical_smiles, scaffold_smiles
    """
    try:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            return None

        canon = Chem.MolToSmiles(mol, canonical=True)

        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )

        return canon, scaffold

    except Exception:
        return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smifile",
        required=True,
        type=str,
        help="File containing one SMILES per line",
    )

    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    print("Loading SMILES...")

    with open(args.smifile) as f:
        smis = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(smis):,} SMILES")

    print("Processing molecules...")

    results = Parallel(
        n_jobs=args.n_workers,
        backend="threading",
    )(
        delayed(process_smiles)(smi)
        for smi in tqdm(smis)
    )

    results = [r for r in results if r is not None]

    unique_molecules = {
        canon
        for canon, scaffold in results
    }

    unique_scaffolds = {
        scaffold
        for canon, scaffold in results
        if scaffold
    }

    n_unique_molecules = len(unique_molecules)
    n_unique_scaffolds = len(unique_scaffolds)

    scaffold_ratio = (
        n_unique_scaffolds / n_unique_molecules
        if n_unique_molecules > 0
        else 0.0
    )

    print()
    print("=" * 50)
    print(f"Unique molecules : {n_unique_molecules:,}")
    print(f"Unique scaffolds : {n_unique_scaffolds:,}")
    print(f"Scaffold ratio   : {scaffold_ratio:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
