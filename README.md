# Molecular Property Prediction with a Graph Neural Network

A message-passing GNN for molecular property prediction, with the **SMILES
parser, the graph featuriser, the descriptors, the scaffold split and the
batching all written from scratch**. No RDKit, no PyTorch Geometric — so every
step from text to prediction is inspectable and tested.

```
"CC(=O)Oc1ccccc1C(=O)O"
   │  SMILES parser        atoms, bonds, rings, aromaticity, charges
   ▼
molecular graph            atom features (29-d), bond features (5-d)
   │  message passing ×N   edge-conditioned messages + GRU update
   ▼
atom embeddings
   │  mean+max readout
   ▼
predicted property   +   descriptor baselines for comparison
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train --config configs/base.yaml
python -m src.train --config configs/base.yaml --split random   # see the trap
python -m src.predict --checkpoint runs/base/best.pt --smiles "CCO" "c1ccccc1"
python -m src.train --config configs/base.yaml --csv data/esol.csv
```

## Scaffold splitting is the whole ballgame

The single most consequential decision in molecular property prediction is how
you split the data. A **random split** puts close analogues of your test
molecules into training — the model interpolates within a chemical series and
reports a score that evaporates on genuinely new chemistry.

A **scaffold split** groups molecules by their Bemis-Murcko core and keeps each
scaffold entirely within one set, approximating the real question: *does this
generalise to a scaffold it has never seen?*

Two tests state the difference directly:

```python
def test_scaffold_split_keeps_scaffolds_in_one_set():
    assert not (train_scaffolds & test_scaffolds)

def test_random_split_does_leak_scaffolds():
    assert train_scaffolds & test_scaffolds     # exactly why the above exists
```

Run both splits and compare. The gap between them is the size of the lie a
random split tells you.

## The baseline is twelve numbers

`src/train.py` fits ridge regression and a random forest on twelve classical
descriptors — molecular weight, logP, ring count, rotatable bonds, H-bond
donors/acceptors, fraction sp3, halogen count — and prints them alongside the
GNN, plus the random forest's feature importances.

This matters because on many property-prediction tasks, **descriptors win**.
A GNN that cannot beat a random forest on twelve hand-computed numbers has not
learned anything the descriptors did not already contain. The run says which
won, either way.

## Implementation notes worth reading

**The parser** handles the organic subset, bracket atoms with charges and
explicit hydrogens, branches, ring-closure digits including `%nn`, explicit bond
symbols, aromatic lowercase atoms, and disconnected components. Stereochemistry
(`@`, `/`, `\`) is parsed and *discarded* — a 2D graph cannot represent it, and
pretending otherwise would be worse than ignoring it.

Ring perception is **bridge detection**: a bond is in a ring exactly when
removing it leaves its endpoints connected. Implemented iteratively, because a
recursive DFS overflows on long chains — there is a test with a 900-atom chain.

**Batching** merges molecules into one disconnected graph with offset node
indices and a `batch` vector. This avoids padding to the largest molecule and
lets one scatter-add do the whole readout. The classic bug — forgetting the
offset, silently wiring molecules together — is caught by a test that verifies
every edge stays inside its own molecule. Two further tests assert that a
molecule's prediction does not depend on what it was batched with, or on
ordering.

**Message passing** conditions each message on the bond features, so a double
bond and a single bond between the same atom types send different messages. The
update is a `GRUCell` rather than an MLP: the gate lets each node decide how
much of the incoming message to absorb, which is what prevents the
over-smoothing that collapses all node states after a few layers.

**Readout** defaults to concatenated mean+max. The choice is chemistry, not
convenience: `sum` is right for *extensive* properties that scale with molecule
size (molecular weight, total polarisability), `mean` for *intensive* ones
(logP per atom, density). Getting it backwards costs real accuracy.

## The synthetic target is deliberately not linear

The generated property follows the shape of the classic ESOL solubility
equation — logP, molecular weight, aromatic fraction, hydrogen bonding — but with
**interaction terms** (aromatic rings hurt solubility more in heavy molecules)
and a saturating term. A purely additive target would be solved exactly by
linear regression on the descriptors and the GNN could never do better than tie.
A test asserts linear regression stays below R² 0.98, keeping the task honest as
the generator evolves.

## Using real data

```bash
python -m src.train --config configs/base.yaml --csv data/delaney.csv
```

Expects a CSV with `smiles` and `target` columns (configurable). Unparseable
rows are skipped and counted rather than crashing the run. Good public sets:

| Dataset | Property |
| --- | --- |
| [ESOL / Delaney](https://moleculenet.org/) | aqueous solubility |
| [FreeSolv](https://moleculenet.org/) | hydration free energy |
| [Lipophilicity](https://moleculenet.org/) | octanol/water logD |
| [BACE, BBBP, Tox21](https://moleculenet.org/) | classification tasks |

For production work on real chemistry, install RDKit and swap `src/chem.py`'s
parser for it — the featurisation interface is a dict of arrays, so nothing
downstream changes.

## Layout

```
src/
  chem.py      SMILES parser, ring perception, atom/bond featurisation
  data.py      descriptors, synthetic library, scaffold split, CSV loading
  model.py     message passing, scatter ops, graph batching
  metrics.py   RMSE, MAE, R², Pearson, Spearman
  train.py     training loop + descriptor baselines
  predict.py   inference on arbitrary SMILES
tests/         pytest suite
```

## License

MIT
