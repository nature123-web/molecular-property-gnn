"""Train the molecular GNN and compare against descriptor baselines.

    python -m src.train --config configs/base.yaml
    python -m src.train --config configs/base.yaml --split random   # the trap
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from .data import (
    DESCRIPTOR_NAMES,
    build_graphs,
    descriptor_matrix,
    load_csv,
    make_dataset,
    random_split,
    scaffold_split,
)
from .metrics import evaluate, format_report
from .model import MolecularGNN, collate_graphs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batches(indices: np.ndarray, batch_size: int, shuffle: bool,
            rng: np.random.Generator):
    order = rng.permutation(indices) if shuffle else indices
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def move(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def predict(model, graphs, indices, targets, device, batch_size=64):
    model.eval()
    preds, trues = [], []
    rng = np.random.default_rng(0)
    for chunk in batches(indices, batch_size, False, rng):
        batch = move(collate_graphs([graphs[i] for i in chunk]), device)
        preds.append(model(batch).squeeze(-1).cpu().numpy())
        trues.append(targets[chunk])
    return np.concatenate(trues), np.concatenate(preds)


def descriptor_baselines(X, y, train_idx, test_idx, seed):
    """Ridge and random forest on classical descriptors.

    A GNN that cannot beat a random forest on twelve hand-computed descriptors
    has not learned anything the descriptors did not already contain -- which,
    for many property-prediction tasks, is the honest outcome.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X[train_idx])
    X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])

    results = {}
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X_train, y[train_idx])
    results["ridge_descriptors"] = evaluate(y[test_idx], ridge.predict(X_test))

    forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=seed, n_jobs=-1)
    forest.fit(X_train, y[train_idx])
    results["random_forest_descriptors"] = evaluate(
        y[test_idx], forest.predict(X_test)
    )
    importances = dict(sorted(
        zip(DESCRIPTOR_NAMES, forest.feature_importances_),
        key=lambda kv: -kv[1],
    ))
    return results, importances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--split", default=None, choices=["scaffold", "random"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.split:
        cfg["data"]["split"] = args.split
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  split={cfg['data']['split']}  out_dir={out_dir}")

    if args.csv:
        smiles, targets = load_csv(args.csv, cfg["data"]["smiles_column"],
                                   cfg["data"]["target_column"])
    else:
        smiles, targets = make_dataset(cfg["data"]["n_molecules"], cfg["seed"])
    print(f"{len(smiles)} molecules")

    if cfg["data"]["split"] == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split(
            smiles, cfg["data"]["val_fraction"], cfg["data"]["test_fraction"],
            cfg["seed"],
        )
    else:
        train_idx, val_idx, test_idx = random_split(
            len(smiles), cfg["data"]["val_fraction"],
            cfg["data"]["test_fraction"], cfg["seed"],
        )
    print(f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    graphs = build_graphs(smiles)
    sizes = [g["n_atoms"] for g in graphs]
    print(f"graph sizes: {min(sizes)}-{max(sizes)} atoms, "
          f"mean {np.mean(sizes):.1f}")

    # Standardise the target on the training split only.
    mean, std = targets[train_idx].mean(), targets[train_idx].std()
    scaled = (targets - mean) / max(std, 1e-8)

    m = cfg["model"]
    model = MolecularGNN(
        hidden_dim=m["hidden_dim"], n_layers=m["n_layers"], n_tasks=1,
        dropout=m["dropout"], readout=m["readout"],
    ).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e3:.1f}k")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.HuberLoss(delta=1.0)
    rng = np.random.default_rng(cfg["seed"])

    history, best_rmse, patience, best_state = [], float("inf"), 0, None
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total, seen = 0.0, 0
        for chunk in tqdm(
            list(batches(train_idx, cfg["train"]["batch_size"], True, rng)),
            desc=f"epoch {epoch}", leave=False,
        ):
            batch = move(collate_graphs([graphs[i] for i in chunk]), device)
            y = torch.from_numpy(scaled[chunk]).float().to(device)
            loss = criterion(model(batch).squeeze(-1), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["train"]["grad_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(chunk)
            seen += len(chunk)

        y_true, y_pred = predict(model, graphs, val_idx, scaled, device)
        val = evaluate(y_true * std + mean, y_pred * std + mean)
        scheduler.step(val["rmse"])

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {total/seen:.4f}  "
                  f"val_RMSE {val['rmse']:.4f}  val_R2 {val['r2']:.4f}")
        history.append({"epoch": epoch, "loss": total / seen, "val": val})

        if val["rmse"] < best_rmse - 1e-5:
            best_rmse, patience = val["rmse"], 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                print(f"early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = predict(model, graphs, test_idx, scaled, device)
    gnn_results = evaluate(y_true * std + mean, y_pred * std + mean)
    print("\n" + format_report(gnn_results, "GNN"))

    X = descriptor_matrix(smiles)
    baselines, importances = descriptor_baselines(
        X, targets, train_idx, test_idx, cfg["seed"]
    )
    for name, results in baselines.items():
        print("\n" + format_report(results, name))

    best_baseline = min(baselines.values(), key=lambda r: r["rmse"])
    delta = best_baseline["rmse"] - gnn_results["rmse"]
    print(f"\nGNN vs best descriptor baseline RMSE: {delta:+.4f} "
          f"({'GNN wins' if delta > 0 else 'baseline wins'})")

    print("\ntop descriptors by random-forest importance:")
    for name, value in list(importances.items())[:6]:
        print(f"  {name:<20} {value:.4f}")

    torch.save({"model": model.state_dict(), "config": cfg,
                "target_mean": float(mean), "target_std": float(std)},
               out_dir / "best.pt")
    (out_dir / "results.json").write_text(json.dumps(
        {"gnn": gnn_results, "baselines": baselines,
         "descriptor_importance": importances,
         "split": cfg["data"]["split"]}, indent=2, default=float,
    ))
    print(f"\nsaved {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
