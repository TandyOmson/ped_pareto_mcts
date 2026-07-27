#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle

def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D PCA coordinates stored in a .npy file."
    )

    parser.add_argument(
        "--infile",
        default=None,
        type=str,
        help="Input .npy file containing an (N, 2) array",
    )

    parser.add_argument(
        "--pcafile",
        default=None,
        type=str,
        help="Input .pkl file containing fitted PCA (optional)",
    )

    parser.add_argument(
        "--embfile",
        default=None,
        type=str,
        help="Input .npy file containing embeddings (optional)",
    )

    parser.add_argument(
        "--outfile",
        required=True,
        type=str,
        help="Output image file (e.g. pca.png)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Point transparency",
    )

    parser.add_argument(
        "--size",
        type=float,
        default=5,
        help="Marker size",
    )

    args = parser.parse_args()

    print(f"Loading {args.infile}")

    if args.infile is not None:
        X = np.load(args.infile)

        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D array, got shape {X.shape}"
            )

        if X.shape[1] != 2:
            raise ValueError(
                f"Expected shape (N, 2), got {X.shape}"
            )

        print(f"Loaded {len(X):,} points")
    elif args.pcafile is not None:
        with open(args.pcafile, "rb") as fr:
            pca = pickle.load(fr)

        if args.embfile is None:
            raise ValueError(
                "if using --pcafile, must specify --embfile"
            )
        X = np.load(args.embfile)
        X = pca.transform(X)
    else:
        raise ValueError(
            "must specify either --infile or both --pcafile and --embfile"
        )

    plt.figure(figsize=(8, 8))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        s=args.size,
        alpha=args.alpha,
        linewidths=0,
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D PCA")
    plt.tight_layout()

    plt.savefig(
        args.outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot to {args.outfile}")


if __name__ == "__main__":
    main()
