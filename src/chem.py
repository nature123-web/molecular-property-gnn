"""A minimal SMILES parser and molecular graph featuriser.

RDKit is the right tool for this in production and is a heavy dependency. This
module implements enough of SMILES to build graphs for property prediction --
atoms, bonds, branches, ring closures, aromaticity, charges and isotopes -- so
the repo runs anywhere with only numpy, and so the featurisation is explicit
rather than a black box.

What is deliberately *not* implemented: stereochemistry (``@``/``@@``, ``/``,
``\\``) is parsed and discarded, because the graph representation used here
cannot express it anyway. Anything predicting a stereochemistry-dependent
property needs a 3D representation, not a 2D graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Atoms writable without brackets in SMILES, and their standard valences.
ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
DEFAULT_VALENCE = {
    "B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2,
    "F": 1, "Cl": 1, "Br": 1, "I": 1,
}
ATOM_VOCAB = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si",
              "Se", "H", "other"]
BOND_TYPES = {"single": 0, "double": 1, "triple": 2, "aromatic": 3}

# Approximate Pauling electronegativity, used as a numeric atom feature.
ELECTRONEGATIVITY = {
    "C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58, "F": 3.98, "Cl": 3.16,
    "Br": 2.96, "I": 2.66, "P": 2.19, "B": 2.04, "Si": 1.90, "Se": 2.55,
    "H": 2.20,
}


class SmilesError(ValueError):
    """Raised for SMILES that cannot be parsed."""


@dataclass
class Atom:
    symbol: str
    aromatic: bool = False
    charge: int = 0
    explicit_hydrogens: Optional[int] = None
    in_ring: bool = False
    degree: int = 0


@dataclass
class Bond:
    begin: int
    end: int
    order: str = "single"
    in_ring: bool = False


@dataclass
class Molecule:
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    smiles: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def neighbors(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {i: [] for i in range(self.n_atoms)}
        for bond in self.bonds:
            out[bond.begin].append(bond.end)
            out[bond.end].append(bond.begin)
        return out


_BRACKET_RE = re.compile(
    r"\[(?P<isotope>\d+)?(?P<symbol>[A-Za-z][a-z]?)(?P<chiral>@{1,2})?"
    r"(?:H(?P<hcount>\d*))?(?P<charge>(?:\+{1,3}|-{1,3}|\+\d+|-\d+))?"
    r"(?::\d+)?\]"
)
_BOND_CHARS = {"-": "single", "=": "double", "#": "triple", ":": "aromatic"}


def parse_smiles(smiles: str) -> Molecule:
    """Parse a SMILES string into a :class:`Molecule`.

    Handles the organic subset, bracketed atoms, branches, ring-closure digits
    (including ``%nn``), explicit bond symbols and aromatic lowercase atoms.
    """
    if not smiles or not smiles.strip():
        raise SmilesError("empty SMILES")

    molecule = Molecule(smiles=smiles)
    # Stack of atom indices for branch open/close.
    branch_stack: List[int] = []
    # Pending ring closures: digit -> (atom index, bond order or None).
    ring_bonds: Dict[str, Tuple[int, Optional[str]]] = {}
    previous: Optional[int] = None
    pending_bond: Optional[str] = None
    i = 0

    while i < len(smiles):
        char = smiles[i]

        if char == "(":
            if previous is None:
                raise SmilesError(f"branch opens before any atom at position {i}")
            branch_stack.append(previous)
            i += 1
            continue

        if char == ")":
            if not branch_stack:
                raise SmilesError(f"unbalanced ')' at position {i}")
            previous = branch_stack.pop()
            i += 1
            continue

        if char in _BOND_CHARS:
            pending_bond = _BOND_CHARS[char]
            i += 1
            continue

        if char in "/\\":
            # Directional bonds carry stereochemistry only; ignored by design.
            i += 1
            continue

        if char == ".":
            # Disconnected component: break the chain rather than bond across it.
            previous = None
            i += 1
            continue

        if char.isdigit() or char == "%":
            if char == "%":
                label, i = smiles[i + 1 : i + 3], i + 3
            else:
                label, i = char, i + 1
            if previous is None:
                raise SmilesError(f"ring closure {label} before any atom")
            if label in ring_bonds:
                start, order = ring_bonds.pop(label)
                molecule.bonds.append(Bond(
                    start, previous, pending_bond or order or _implicit_order(
                        molecule.atoms[start], molecule.atoms[previous]
                    ),
                ))
            else:
                ring_bonds[label] = (previous, pending_bond)
            pending_bond = None
            continue

        if char == "[":
            end = smiles.find("]", i)
            if end == -1:
                raise SmilesError(f"unclosed '[' at position {i}")
            match = _BRACKET_RE.match(smiles, i)
            if not match:
                raise SmilesError(f"cannot parse bracket atom {smiles[i:end+1]!r}")
            symbol = match.group("symbol")
            atom = Atom(
                symbol=symbol.capitalize() if symbol.islower() else symbol,
                aromatic=symbol[0].islower(),
                charge=_parse_charge(match.group("charge")),
                explicit_hydrogens=_parse_hcount(match.group("hcount")),
            )
            i = match.end()
        else:
            symbol, length = _match_organic(smiles, i)
            if symbol is None:
                raise SmilesError(
                    f"unexpected character {char!r} at position {i} in {smiles!r}"
                )
            atom = Atom(symbol=symbol.capitalize() if symbol.islower() else symbol,
                        aromatic=symbol[0].islower())
            i += length

        molecule.atoms.append(atom)
        index = molecule.n_atoms - 1
        if previous is not None:
            order = pending_bond or _implicit_order(
                molecule.atoms[previous], atom
            )
            molecule.bonds.append(Bond(previous, index, order))
        previous = index
        pending_bond = None

    if branch_stack:
        raise SmilesError(f"unbalanced '(' in {smiles!r}")
    if ring_bonds:
        raise SmilesError(
            f"unclosed ring bond(s) {sorted(ring_bonds)} in {smiles!r}"
        )

    _annotate_rings(molecule)
    _annotate_degrees(molecule)
    return molecule


def _match_organic(smiles: str, i: int) -> Tuple[Optional[str], int]:
    """Match a two-letter element first, so 'Cl' is not read as 'C' then 'l'."""
    two = smiles[i : i + 2]
    if two in {"Cl", "Br"}:
        return two, 2
    one = smiles[i]
    if one.upper() in ORGANIC_SUBSET or one in "bcnops":
        return one, 1
    return None, 0


def _parse_charge(token: Optional[str]) -> int:
    if not token:
        return 0
    sign = 1 if token[0] == "+" else -1
    digits = token[1:]
    if digits.isdigit():
        return sign * int(digits)
    return sign * len(token)          # '++' means +2


def _parse_hcount(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    return int(token) if token else 1     # bare 'H' means one hydrogen


def _implicit_order(a: Atom, b: Atom) -> str:
    """Two adjacent aromatic atoms are joined by an aromatic bond."""
    return "aromatic" if a.aromatic and b.aromatic else "single"


def _annotate_rings(molecule: Molecule) -> None:
    """Mark atoms and bonds that lie on a cycle.

    A bond is in a ring exactly when removing it leaves its endpoints still
    connected, so this is a bridge-detection problem. Rings matter as a feature
    because ring membership changes an atom's chemistry substantially.
    """
    adjacency: Dict[int, List[Tuple[int, int]]] = {
        i: [] for i in range(molecule.n_atoms)
    }
    for index, bond in enumerate(molecule.bonds):
        adjacency[bond.begin].append((bond.end, index))
        adjacency[bond.end].append((bond.begin, index))

    # Iterative DFS bridge-finding (Tarjan); recursion would overflow on long
    # chains, which occur in real molecules.
    discovery = [-1] * molecule.n_atoms
    low = [0] * molecule.n_atoms
    bridges = set()
    timer = 0

    for root in range(molecule.n_atoms):
        if discovery[root] != -1:
            continue
        stack: List[Tuple[int, int, int]] = [(root, -1, 0)]
        discovery[root] = low[root] = timer
        timer += 1
        while stack:
            node, parent_edge, child_index = stack[-1]
            if child_index < len(adjacency[node]):
                stack[-1] = (node, parent_edge, child_index + 1)
                neighbor, edge = adjacency[node][child_index]
                if edge == parent_edge:
                    continue
                if discovery[neighbor] == -1:
                    discovery[neighbor] = low[neighbor] = timer
                    timer += 1
                    stack.append((neighbor, edge, 0))
                else:
                    low[node] = min(low[node], discovery[neighbor])
            else:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[node])
                    if low[node] > discovery[parent]:
                        bridges.add(parent_edge)

    for index, bond in enumerate(molecule.bonds):
        bond.in_ring = index not in bridges
        if bond.in_ring:
            molecule.atoms[bond.begin].in_ring = True
            molecule.atoms[bond.end].in_ring = True


def _annotate_degrees(molecule: Molecule) -> None:
    for bond in molecule.bonds:
        molecule.atoms[bond.begin].degree += 1
        molecule.atoms[bond.end].degree += 1


def implicit_hydrogens(atom: Atom, bond_order_sum: float) -> int:
    """Hydrogens needed to fill the standard valence.

    Needed because SMILES omits hydrogens on organic-subset atoms, yet hydrogen
    count is one of the more predictive atom features (it distinguishes a
    methyl from a methylene from a quaternary carbon).
    """
    if atom.explicit_hydrogens is not None:
        return atom.explicit_hydrogens
    valence = DEFAULT_VALENCE.get(atom.symbol)
    if valence is None:
        return 0
    # Aromatic carbons contribute 1.5 per aromatic bond; round the total down.
    used = int(round(bond_order_sum))
    return max(0, valence + atom.charge - used)


BOND_ORDER_VALUE = {"single": 1.0, "double": 2.0, "triple": 3.0,
                    "aromatic": 1.5}


def atom_features(molecule: Molecule) -> np.ndarray:
    """Per-atom feature matrix, ``(n_atoms, n_features)``."""
    order_sums = np.zeros(molecule.n_atoms)
    for bond in molecule.bonds:
        value = BOND_ORDER_VALUE[bond.order]
        order_sums[bond.begin] += value
        order_sums[bond.end] += value

    features = []
    for index, atom in enumerate(molecule.atoms):
        symbol_onehot = [0.0] * len(ATOM_VOCAB)
        position = (ATOM_VOCAB.index(atom.symbol)
                    if atom.symbol in ATOM_VOCAB else ATOM_VOCAB.index("other"))
        symbol_onehot[position] = 1.0

        degree_onehot = [0.0] * 5
        degree_onehot[min(atom.degree, 4)] = 1.0

        hydrogens = implicit_hydrogens(atom, order_sums[index])
        hydrogen_onehot = [0.0] * 5
        hydrogen_onehot[min(hydrogens, 4)] = 1.0

        features.append(symbol_onehot + degree_onehot + hydrogen_onehot + [
            float(atom.aromatic),
            float(atom.in_ring),
            float(atom.charge),
            ELECTRONEGATIVITY.get(atom.symbol, 2.5) / 4.0,
            order_sums[index] / 4.0,
        ])
    return np.asarray(features, dtype=np.float32)


def bond_features(molecule: Molecule) -> np.ndarray:
    """Per-bond feature matrix, ``(n_bonds, n_features)``."""
    features = []
    for bond in molecule.bonds:
        onehot = [0.0] * len(BOND_TYPES)
        onehot[BOND_TYPES[bond.order]] = 1.0
        features.append(onehot + [float(bond.in_ring)])
    return np.asarray(features, dtype=np.float32).reshape(
        len(molecule.bonds), len(BOND_TYPES) + 1
    )


ATOM_FEATURE_DIM = len(ATOM_VOCAB) + 5 + 5 + 5
BOND_FEATURE_DIM = len(BOND_TYPES) + 1


def to_graph(smiles: str) -> dict:
    """SMILES to the arrays a GNN consumes.

    Edges are emitted in **both** directions. Message passing on a chemical
    graph must be symmetric -- a bond is not directional -- and storing only one
    direction silently makes information flow one way along every bond.
    """
    molecule = parse_smiles(smiles)
    if molecule.n_atoms == 0:
        raise SmilesError(f"no atoms parsed from {smiles!r}")

    nodes = atom_features(molecule)
    edge_feature_list = bond_features(molecule)

    sources, targets, edge_features = [], [], []
    for index, bond in enumerate(molecule.bonds):
        for a, b in ((bond.begin, bond.end), (bond.end, bond.begin)):
            sources.append(a)
            targets.append(b)
            edge_features.append(edge_feature_list[index])

    return {
        "nodes": nodes,
        "edge_index": np.array([sources, targets], dtype=np.int64).reshape(2, -1),
        "edge_features": (np.asarray(edge_features, dtype=np.float32)
                          if edge_features
                          else np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)),
        "n_atoms": molecule.n_atoms,
        "smiles": smiles,
        "molecule": molecule,
    }
