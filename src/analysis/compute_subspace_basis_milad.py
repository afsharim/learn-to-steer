import argparse
import os

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD


def main():
    parser = argparse.ArgumentParser(
        description="Fit a truncated SVD on per-sample shift vectors and cache the top-k orthonormal basis U for subspace-constrained Reparo."
    )
    parser.add_argument(
        "--shifts_path",
        type=str,
        required=True,
        help="Path to .pth produced by compute_contrastive_vectors (non-_mean), with key 'steering_vector' of shape (N, D).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Where to save the basis dict {'U': (D, k), 'mean_shift': (D,), 'k': int, 'singular_values': (k,), 'explained_variance_ratio': (k,)}.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of components.",
    )
    parser.add_argument(
        "--shift_key",
        type=str,
        default="steering_vector",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    blob = torch.load(args.shifts_path, map_location="cpu")
    shifts = blob[args.shift_key]
    if shifts.dim() == 3:
        shifts = shifts.squeeze(1)
    assert shifts.dim() == 2, f"Expected (N, D); got {tuple(shifts.shape)}"
    N, D = shifts.shape
    assert args.k <= min(N, D), f"k={args.k} must be <= min(N, D)={min(N, D)}"

    X = shifts.float().numpy()
    mean_shift = X.mean(axis=0)
    X_centered = X - mean_shift

    svd = TruncatedSVD(n_components=args.k, random_state=args.seed)
    svd.fit(X_centered)

    U = torch.from_numpy(svd.components_.T.copy()).float()  # (D, k)
    cumulative = float(svd.explained_variance_ratio_.sum())

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(
        {
            "U": U,
            "mean_shift": torch.from_numpy(mean_shift).float(),
            "k": args.k,
            "singular_values": torch.from_numpy(svd.singular_values_).float(),
            "explained_variance_ratio": torch.from_numpy(
                svd.explained_variance_ratio_
            ).float(),
            "source_shifts_path": args.shifts_path,
            "n_samples": N,
            "hidden_dim": D,
        },
        args.save_path,
    )

    print(f"Saved basis to {args.save_path}")
    print(f"  N={N}, D={D}, k={args.k}")
    print(f"  cumulative explained variance @ k={args.k}: {cumulative:.4f}")


if __name__ == "__main__":
    main()
