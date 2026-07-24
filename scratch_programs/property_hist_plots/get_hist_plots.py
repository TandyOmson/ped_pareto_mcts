import argparse

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="One or more CSV files",
    )

    parser.add_argument(
        "--plotfile",
        required=True,
        help="Output plot image (e.g. histograms.png)",
    )

    parser.add_argument(
        "--combined_csv",
        default=None,
        help="Optional output combined CSV",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    dfs = [pd.read_csv(f, index_col=0) for f in args.csvs]
    dfs = [df.drop(columns=["smiles"]) for df in dfs if "smiles" in df.columns]

    lengths = [len(df) for df in dfs]

    if len(set(lengths)) != 1:
        raise ValueError(
            f"Input CSVs have different lengths: {lengths}"
        )

    print(f"All CSVs contain {lengths,} rows")

    # ------------------------------------------------------------------
    # Combine column-wise
    # ------------------------------------------------------------------

    combined_df = pd.concat(dfs, axis=1)

    if args.combined_csv is not None:
        combined_df.to_csv(args.combined_csv, index=False)
        print(f"Saved combined dataframe to {args.combined_csv}")

    # ------------------------------------------------------------------
    # Determine plotting columns
    # ------------------------------------------------------------------

    all_columns = sorted(
        set(
            col
            for df in dfs
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        )
    )

    n_cols = min(3, len(all_columns))
    n_rows = (len(all_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
    )

    if len(all_columns) == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    colours = plt.cm.tab10.colors

    for ax, column in zip(axes, all_columns):

        found = False

        for i, (csv_file, df) in enumerate(zip(args.csvs, dfs)):

            if column not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df[column]):
                continue

            found = True

            ax.hist(
                df[column].dropna(),
                bins=args.bins,
                alpha=0.5,
                density=False,
                label=Path(csv_file).stem,
                color=colours[i % len(colours)],
            )

        if found:
            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    for i in range(len(all_columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(args.plotfile, dpi=300)
    plt.close()

    print(f"Saved plot to {args.plotfile}")

if __name__ == "__main__":
    main()
