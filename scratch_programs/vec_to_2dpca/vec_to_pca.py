import argparse
import numpy as np
from sklearn.decomposition import PCA
import pickle

def main():
    parser = argparse.ArgumentParser(
        description="Perform 2D PCA on a NumPy array."
    )
    parser.add_argument(
        "--infile",
        type=str,
        required=True,
        help="Input .npy file containing an (N, D) array",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output .npy file for PCA coordinates",
    )
    parser.add_argument(
        "--pcafile",
        type=str,
        default=None,
        help="Output .pkl file for fitted PCA",
    )

    args = parser.parse_args()

    print(f"Loading {args.infile}")
    X = np.load(args.infile)
    X = np.array([i for i in X if not np.isnan(i).any()])

    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, got shape {X.shape}"
        )

    print(f"Input shape: {X.shape}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print(f"Output shape: {X_pca.shape}")
    print(
        f"Explained variance ratio: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    np.save(args.outfile, X_pca)
    print(f"Saved to {args.outfile}")

    if args.pcafile is not None:
        with open(args.pcafile, "wb") as fw:
            pickle.dump(pca, fw)

if __name__ == "__main__":
    main()
