"""Molecular datasets, scaffold splitting, and descriptor baselines."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .chem import (
    BOND_ORDER_VALUE,
    Molecule,
    SmilesError,
    implicit_hydrogens,
    parse_smiles,
    to_graph,
)

# Approximate atomic masses, for the molecular-weight descriptor.
ATOMIC_MASS = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Si": 28.086, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "Se": 78.97, "Br": 79.904, "I": 126.90,
}

# Crippen-style atomic logP contributions, heavily simplified. Enough to make
# the synthetic target chemically meaningful rather than arbitrary.
LOGP_CONTRIBUTION = {
    "C": 0.30, "N": -0.55, "O": -0.55, "S": 0.30, "F": 0.35,
    "Cl": 0.60, "Br": 0.75, "I": 0.90, "P": -0.20, "B": 0.10,
}


def molecular_weight(molecule: Molecule) -> float:
    order_sums = _order_sums(molecule)
    total = 0.0
    for index, atom in enumerate(molecule.atoms):
        total += ATOMIC_MASS.get(atom.symbol, 12.0)
        total += ATOMIC_MASS["H"] * implicit_hydrogens(atom, order_sums[index])
    return total


def _order_sums(molecule: Molecule) -> np.ndarray:
    sums = np.zeros(molecule.n_atoms)
    for bond in molecule.bonds:
        value = BOND_ORDER_VALUE[bond.order]
        sums[bond.begin] += value
        sums[bond.end] += value
    return sums


def count_rings(molecule: Molecule) -> int:
    """Ring count via the cyclomatic number: edges - nodes + components."""
    n_components = _count_components(molecule)
    return len(molecule.bonds) - molecule.n_atoms + n_components


def _count_components(molecule: Molecule) -> int:
    neighbors = molecule.neighbors()
    seen, components = set(), 0
    for start in range(molecule.n_atoms):
        if start in seen:
            continue
        components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(neighbors[node])
    return components


def count_rotatable_bonds(molecule: Molecule) -> int:
    """Single, acyclic bonds between two non-terminal heavy atoms."""
    total = 0
    for bond in molecule.bonds:
        if bond.order != "single" or bond.in_ring:
            continue
        if (molecule.atoms[bond.begin].degree > 1
                and molecule.atoms[bond.end].degree > 1):
            total += 1
    return total


def hydrogen_bond_donors(molecule: Molecule) -> int:
    """N or O carrying at least one hydrogen."""
    order_sums = _order_sums(molecule)
    return sum(
        1 for i, atom in enumerate(molecule.atoms)
        if atom.symbol in {"N", "O"} and implicit_hydrogens(atom, order_sums[i]) > 0
    )


def hydrogen_bond_acceptors(molecule: Molecule) -> int:
    return sum(1 for atom in molecule.atoms if atom.symbol in {"N", "O"})


def estimate_logp(molecule: Molecule) -> float:
    """Crude additive logP with an aromaticity bonus and a polarity penalty."""
    total = sum(LOGP_CONTRIBUTION.get(a.symbol, 0.0) for a in molecule.atoms)
    total += 0.10 * sum(1 for a in molecule.atoms if a.aromatic)
    total -= 0.35 * hydrogen_bond_donors(molecule)
    return total


def descriptors(smiles: str) -> Dict[str, float]:
    """Classical descriptor vector -- the baseline a GNN must beat."""
    molecule = parse_smiles(smiles)
    order_sums = _order_sums(molecule)
    return {
        "molecular_weight": molecular_weight(molecule),
        "n_atoms": float(molecule.n_atoms),
        "n_bonds": float(len(molecule.bonds)),
        "n_rings": float(count_rings(molecule)),
        "n_aromatic": float(sum(a.aromatic for a in molecule.atoms)),
        "n_rotatable": float(count_rotatable_bonds(molecule)),
        "hbd": float(hydrogen_bond_donors(molecule)),
        "hba": float(hydrogen_bond_acceptors(molecule)),
        "logp": estimate_logp(molecule),
        "fraction_sp3": float(
            sum(1 for i, a in enumerate(molecule.atoms)
                if not a.aromatic and order_sums[i] <= a.degree)
            / max(1, molecule.n_atoms)
        ),
        "n_halogen": float(sum(a.symbol in {"F", "Cl", "Br", "I"}
                               for a in molecule.atoms)),
        "formal_charge": float(sum(a.charge for a in molecule.atoms)),
    }


DESCRIPTOR_NAMES = list(descriptors("CCO").keys())


# --------------------------------------------------------------------------- #
# Synthetic library
# --------------------------------------------------------------------------- #

FRAGMENTS = [
    "C", "CC", "CCC", "CCCC", "C(C)C", "CC(C)C",
    "c1ccccc1", "c1ccncc1", "c1ccsc1", "c1cco1", "c1ccc2ccccc2c1",
    "C1CCCCC1", "C1CCNCC1", "C1CCOCC1",
    "O", "N", "S", "F", "Cl", "Br",
    "C=O", "C(=O)O", "C(=O)N", "S(=O)(=O)N", "C#N", "[N+](=O)[O-]",
]
LINKERS = ["", "C", "CC", "O", "N", "C(=O)", "S", "CO", "CN"]


def make_molecule_library(n_molecules: int = 3000, seed: int = 0
                          ) -> List[str]:
    """Assemble SMILES by concatenating fragments through linkers.

    Deliberately simple string composition: it produces valid, varied,
    drug-like-ish SMILES without needing a chemistry toolkit, and every string
    is validated by the parser before being kept.
    """
    rng = np.random.default_rng(seed)
    molecules: List[str] = []
    seen = set()
    attempts = 0

    while len(molecules) < n_molecules and attempts < n_molecules * 50:
        attempts += 1
        n_fragments = int(rng.integers(1, 4))
        parts = []
        for k in range(n_fragments):
            if k > 0:
                parts.append(str(rng.choice(LINKERS)))
            parts.append(str(rng.choice(FRAGMENTS)))
        smiles = "".join(parts)

        if smiles in seen:
            continue
        try:
            molecule = parse_smiles(smiles)
        except SmilesError:
            continue
        if molecule.n_atoms < 2 or molecule.n_atoms > 60:
            continue
        seen.add(smiles)
        molecules.append(smiles)
    return molecules


def topological_motif_count(molecule: Molecule) -> int:
    """Branch points bearing a heteroatom neighbour.

    Deliberately **not** exposed as a descriptor. It is a local connectivity
    pattern -- "an atom with three or more bonds, one of which goes to N or O"
    -- which is exactly the kind of thing message passing discovers and which
    no scalar in ``descriptors()`` can express.

    Its purpose is to keep the benchmark honest. Without a term like this the
    synthetic target is a smooth function of the very descriptors the baseline
    is given, so the baseline has privileged access to the generating process
    and the GNN can only ever tie. That would be a rigged comparison in the
    opposite direction from the usual one.
    """
    neighbors = molecule.neighbors()
    count = 0
    for index, atom in enumerate(molecule.atoms):
        if len(neighbors[index]) < 3:
            continue
        if any(molecule.atoms[j].symbol in {"N", "O"} for j in neighbors[index]):
            count += 1
    return count


def synthetic_target(smiles: str, rng: np.random.Generator) -> float:
    """A solubility-like target: a nonlinear function of structure.

    Follows the shape of the classic ESOL equation -- logP, molecular weight,
    aromaticity, hydrogen bonding -- plus two ingredients that make it a fair
    test rather than a rigged one:

    * **Interaction terms**, so a purely linear model on descriptors cannot fit
      it exactly.
    * **A topological term** (:func:`topological_motif_count`) that is not in the
      descriptor set at all, so there is signal only a structural model can
      reach.
    """
    molecule = parse_smiles(smiles)
    d = descriptors(smiles)
    value = (
        0.16
        - 0.42 * d["logp"]
        - 0.0062 * d["molecular_weight"]
        + 0.066 * d["n_rotatable"]
        - 0.74 * (d["n_aromatic"] / max(1.0, d["n_atoms"]))
        + 0.35 * d["hbd"]
        # Interaction: aromatic rings hurt solubility more in heavy molecules.
        - 0.004 * d["n_aromatic"] * d["molecular_weight"] / 10.0
        # Saturation: extra donors help less and less.
        + 0.5 * np.sqrt(max(0.0, d["hba"]))
        # Structural term, invisible to the descriptor vector.
        + 0.55 * topological_motif_count(molecule)
    )
    return float(value + rng.normal(0, 0.35))


def make_dataset(n_molecules: int = 3000, seed: int = 0
                 ) -> Tuple[List[str], np.ndarray]:
    smiles = make_molecule_library(n_molecules, seed)
    rng = np.random.default_rng(seed + 1)
    targets = np.array([synthetic_target(s, rng) for s in smiles],
                       dtype=np.float32)
    return smiles, targets


def load_csv(path: str | Path, smiles_column: str = "smiles",
             target_column: str = "target") -> Tuple[List[str], np.ndarray]:
    """Read a CSV of SMILES and targets, skipping unparseable rows."""
    import csv

    smiles, targets, skipped = [], [], 0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                parse_smiles(row[smiles_column])
            except (SmilesError, KeyError):
                skipped += 1
                continue
            smiles.append(row[smiles_column])
            targets.append(float(row[target_column]))
    if skipped:
        print(f"skipped {skipped} unparseable rows")
    return smiles, np.array(targets, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def bemis_murcko_scaffold(smiles: str) -> str:
    """Approximate Bemis-Murcko scaffold: the ring systems and their linkers.

    Iteratively strips terminal atoms until only ring atoms and the paths
    between them remain, then serialises the result as a canonical-ish
    signature. It is not a true canonical SMILES -- that needs a full
    graph-canonicalisation -- but it groups molecules sharing a core, which is
    all the split requires.
    """
    molecule = parse_smiles(smiles)
    neighbors = molecule.neighbors()
    keep = {i for i, atom in enumerate(molecule.atoms) if atom.in_ring}

    if not keep:
        return ""       # acyclic molecules share the empty scaffold

    # Add atoms lying on a path between two ring systems.
    changed = True
    while changed:
        changed = False
        for i in range(molecule.n_atoms):
            if i in keep:
                continue
            # An atom bridging two kept atoms is part of a linker.
            if sum(1 for j in neighbors[i] if j in keep) >= 2:
                keep.add(i)
                changed = True

    symbols = sorted(
        f"{molecule.atoms[i].symbol}{'a' if molecule.atoms[i].aromatic else ''}"
        f"{len(neighbors[i])}"
        for i in keep
    )
    ring_bonds = sum(
        1 for b in molecule.bonds
        if b.begin in keep and b.end in keep and b.in_ring
    )
    return f"{'-'.join(symbols)}|r{ring_bonds}"


def scaffold_split(
    smiles: Sequence[str],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split so that no scaffold appears in more than one set.

    This is the standard split for molecular property prediction, and it matters
    far more than it might seem. A random split puts close analogues of test
    molecules in the training set, so the model can interpolate within a
    chemical series and reports a score that collapses on genuinely new
    chemistry. Scaffold splitting approximates the real question: does this
    model generalise to a scaffold it has never seen?

    Largest scaffold groups go to training, which is the convention -- it keeps
    the rare, structurally unusual molecules in the evaluation sets.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, molecule in enumerate(smiles):
        try:
            groups[bemis_murcko_scaffold(molecule)].append(index)
        except SmilesError:
            groups[""].append(index)

    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n_total = len(smiles)
    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for group in ordered:
        if len(test_idx) + len(group) <= n_test:
            test_idx.extend(group)
        elif len(val_idx) + len(group) <= n_val:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    rng = np.random.default_rng(seed)
    return (
        rng.permutation(train_idx), rng.permutation(val_idx),
        rng.permutation(test_idx),
    )


def random_split(n: int, val_fraction: float = 0.1, test_fraction: float = 0.1,
                 seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random split, provided for the explicit comparison against scaffold."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test, n_val = int(n * test_fraction), int(n * val_fraction)
    return (order[n_test + n_val:], order[n_test : n_test + n_val],
            order[:n_test])


def build_graphs(smiles: Sequence[str]) -> List[dict]:
    return [to_graph(s) for s in smiles]


def descriptor_matrix(smiles: Sequence[str]) -> np.ndarray:
    return np.array([[descriptors(s)[name] for name in DESCRIPTOR_NAMES]
                     for s in smiles], dtype=np.float32)
