"""Predict properties for arbitrary SMILES.

    python -m src.predict --checkpoint runs/base/best.pt --smiles CCO c1ccccc1
    python -m src.predict --checkpoint runs/base/best.pt --csv molecules.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .chem import SmilesError, to_graph
from .data import descriptors
from .model import MolecularGNN, collate_graphs


def load_model(checkpoint: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]
    model = MolecularGNN(
        hidden_dim=m["hidden_dim"], n_layers=m["n_layers"], n_tasks=1,
        dropout=m["dropout"], readout=m["readout"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt["target_mean"], ckpt["target_std"]


@torch.no_grad()
def predict_smiles(model, smiles_list, device, mean, std, batch_size=64):
    """Predict for a list of SMILES; unparseable entries come back as nan."""
    predictions = np.full(len(smiles_list), np.nan)
    valid_graphs, valid_positions = [], []

    for position, smiles in enumerate(smiles_list):
        try:
            valid_graphs.append(to_graph(smiles))
            valid_positions.append(position)
        except SmilesError as error:
            print(f"  skipping {smiles!r}: {error}")

    for start in range(0, len(valid_graphs), batch_size):
        chunk = valid_graphs[start : start + batch_size]
        batch = collate_graphs(chunk)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        out = model(batch).squeeze(-1).cpu().numpy() * std + mean
        for offset, value in enumerate(out):
            predictions[valid_positions[start + offset]] = value
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--smiles", nargs="+", default=None)
    parser.add_argument("--csv", default=None,
                        help="CSV with a 'smiles' column.")
    parser.add_argument("--show-descriptors", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.smiles and not args.csv:
        parser.error("provide --smiles or --csv")

    device = torch.device(args.device)
    model, cfg, mean, std = load_model(args.checkpoint, device)

    if args.csv:
        import csv
        with open(args.csv, newline="") as handle:
            smiles_list = [row["smiles"] for row in csv.DictReader(handle)]
    else:
        smiles_list = args.smiles

    print(f"predicting for {len(smiles_list)} molecules")
    predictions = predict_smiles(model, smiles_list, device, mean, std)

    print(f"\n{'smiles':<40}{'prediction':>12}")
    for smiles, value in zip(smiles_list, predictions):
        shown = smiles if len(smiles) <= 38 else smiles[:35] + "..."
        print(f"{shown:<40}{value:>12.4f}"
              if np.isfinite(value) else f"{shown:<40}{'invalid':>12}")
        if args.show_descriptors and np.isfinite(value):
            for name, descriptor in descriptors(smiles).items():
                print(f"    {name:<20} {descriptor:.3f}")

    if args.out:
        lines = ["smiles,prediction"]
        lines += [f"{s},{v:.6f}" if np.isfinite(v) else f"{s},"
                  for s, v in zip(smiles_list, predictions)]
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
