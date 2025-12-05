#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_latent(folder: Path, step: int, pattern: str) -> torch.Tensor:
    path = folder / pattern.format(step=step)
    if not path.exists():
        raise FileNotFoundError(f"Missing latent file: {path}")
    latent = torch.load(path, map_location='cpu').float()
    if latent.dim() != 4:
        raise ValueError(
            f"Latent tensor must have shape [C, T, H, W], got {tuple(latent.shape)} for {path}"
        )
    return latent


def build_mask_from_two(z_a: torch.Tensor, z_b: torch.Tensor,
                        percentile: float):
    diff = (z_a - z_b).abs().mean(dim=0)  # [T, H, W]
    flat = diff.flatten().numpy()
    thresh = np.percentile(flat, percentile)
    mask = (diff.numpy() > thresh).astype(np.uint8)
    return diff, mask, float(thresh)


def save_quicklooks(step: int, diff: torch.Tensor, mask: np.ndarray,
                    outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Histogram of pixel diffs (log scale)
    plt.figure(figsize=(6, 4))
    plt.hist(diff.flatten().numpy(), bins=100, log=True)
    plt.title(f"Latent abs diff histogram (step {step})")
    plt.xlabel("abs diff")
    plt.ylabel("count (log)")
    plt.tight_layout()
    plt.savefig(outdir / f"hist_step{step:03d}.png")
    plt.close()

    # Heatmap (avg over time)
    heatmap = diff.mean(dim=0).numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="viridis")
    plt.colorbar(label="mean abs diff")
    plt.title(f"Mean abs diff heatmap (step {step})")
    plt.tight_layout()
    plt.savefig(outdir / f"heatmap_step{step:03d}.png")
    plt.close()

    # Middle frame of the binary mask
    tmid = diff.shape[0] // 2
    plt.figure(figsize=(6, 6))
    plt.imshow(mask[tmid], cmap="hot")
    plt.colorbar(label="mask (1=changed)")
    plt.title(f"Mask (step {step}, frame {tmid})")
    plt.tight_layout()
    plt.savefig(outdir / f"mask_step{step:03d}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build masks by comparing two sets of saved latents.")
    parser.add_argument("--latent_dir_a", required=True, type=Path)
    parser.add_argument("--latent_dir_b", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--steps", nargs="+", type=int, required=True,
        help="List of step indices (integers) to process, e.g. --steps 10 15 20")
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile of abs diff used as threshold (default: 90).")
    parser.add_argument(
        "--pattern",
        type=str,
        default="latent_step{step:03d}.pt",
        help="Filename pattern for latent steps (default: latent_stepXYZ.pt)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] Build masks from {args.latent_dir_a} vs {args.latent_dir_b}, "
        f"percentile={args.percentile}")

    for step in args.steps:
        print(f"[BUILD] step {step:03d}")
        z_a = load_latent(args.latent_dir_a, step, args.pattern)
        z_b = load_latent(args.latent_dir_b, step, args.pattern)
        diff, mask, thresh = build_mask_from_two(z_a, z_b, args.percentile)

        masked_frac = float(mask.mean())
        unmasked_frac = float(1.0 - masked_frac)

        mask_path = args.output_dir / f"mask_step{step:03d}.pt"
        diff_path = args.output_dir / f"diff_step{step:03d}.pt"
        stats_path = args.output_dir / f"stats_step{step:03d}.txt"

        torch.save(torch.from_numpy(mask.astype(np.uint8)), mask_path)
        torch.save(diff, diff_path)
        with open(stats_path, "w") as fh:
            fh.write(f"mean_abs_diff={float(diff.mean()):.6f}\n")
            fh.write(f"max_abs_diff={float(diff.max()):.6f}\n")
            fh.write(f"masked_frac={masked_frac:.6f}\n")
            fh.write(f"unmasked_frac={unmasked_frac:.6f}\n")
            fh.write(f"threshold_percentile={thresh:.6f}\n")

        save_quicklooks(step, diff, mask, args.output_dir)

    print(f"[DONE] Masks saved to {args.output_dir}")


if __name__ == "__main__":
    main()
