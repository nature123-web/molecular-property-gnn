"""Tests for the SMILES parser, featurisation, batching, GNN and splitting."""

import numpy as np
import pytest
import torch

from src.chem import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    SmilesError,
    atom_features,
    bond_features,
    implicit_hydrogens,
    parse_smiles,
    to_graph,
)
from src.data import (
    bemis_murcko_scaffold,
    count_rings,
    count_rotatable_bonds,
    descriptor_matrix,
    descriptors,
    hydrogen_bond_donors,
    make_dataset,
    make_molecule_library,
    molecular_weight,
    random_split,
    scaffold_split,
)
from src.metrics import evaluate, r2, rmse
from src.model import MolecularGNN, collate_graphs, scatter_max, scatter_sum


# --------------------------------------------------------------------------- #
# SMILES parsing
# --------------------------------------------------------------------------- #

def test_parses_simple_chain():
    molecule = parse_smiles("CCO")
    assert molecule.n_atoms == 3
    assert [a.symbol for a in molecule.atoms] == ["C", "C", "O"]
    assert len(molecule.bonds) == 2


def test_two_letter_element_is_not_split():
    """'Cl' must not be read as carbon followed by a stray 'l'."""
    molecule = parse_smiles("CCCl")
    assert [a.symbol for a in molecule.atoms] == ["C", "C", "Cl"]


def test_parses_branches():
    molecule = parse_smiles("CC(C)C")
    assert molecule.n_atoms == 4
    neighbors = molecule.neighbors()
    # The second atom is the branch point and has three neighbours.
    assert len(neighbors[1]) == 3


def test_parses_nested_branches():
    molecule = parse_smiles("CC(C(C)C)CC")
    assert molecule.n_atoms == 7


def test_parses_ring_closure():
    molecule = parse_smiles("C1CCCCC1")
    assert molecule.n_atoms == 6
    assert len(molecule.bonds) == 6          # the ring bond closes the cycle
    assert all(a.in_ring for a in molecule.atoms)


def test_parses_aromatic_ring():
    molecule = parse_smiles("c1ccccc1")
    assert molecule.n_atoms == 6
    assert all(a.aromatic for a in molecule.atoms)
    assert all(b.order == "aromatic" for b in molecule.bonds)


def test_parses_double_and_triple_bonds():
    assert parse_smiles("C=C").bonds[0].order == "double"
    assert parse_smiles("C#N").bonds[0].order == "triple"


def test_parses_bracket_atom_with_charge_and_hydrogens():
    molecule = parse_smiles("[NH4+]")
    atom = molecule.atoms[0]
    assert atom.symbol == "N"
    assert atom.charge == 1
    assert atom.explicit_hydrogens == 4


def test_parses_multi_charge():
    assert parse_smiles("[Mg++]").atoms[0].charge == 2
    assert parse_smiles("[O-2]").atoms[0].charge == -2


def test_parses_two_digit_ring_closure():
    molecule = parse_smiles("C%10CCCCC%10")
    assert len(molecule.bonds) == 6


def test_disconnected_components_are_not_bonded():
    """A '.' separates components; bonding across it would be wrong."""
    molecule = parse_smiles("CC.OO")
    assert molecule.n_atoms == 4
    assert len(molecule.bonds) == 2


def test_stereochemistry_is_accepted_and_ignored():
    """Parsed without error, but the 2D graph cannot represent it."""
    plain = parse_smiles("CC(N)C(=O)O")
    chiral = parse_smiles("C[C@H](N)C(=O)O")
    assert chiral.n_atoms == plain.n_atoms
    assert len(chiral.bonds) == len(plain.bonds)


def test_directional_bonds_are_ignored():
    assert parse_smiles("C/C=C/C").n_atoms == 4


@pytest.mark.parametrize("bad", ["", "   ", "C(", "C)", "C1CC", "CX2Y"])
def test_invalid_smiles_raise(bad):
    with pytest.raises(SmilesError):
        parse_smiles(bad)


def test_error_message_names_the_problem():
    with pytest.raises(SmilesError, match="unclosed ring"):
        parse_smiles("C1CCC")


def test_ring_detection_marks_only_ring_atoms():
    """Toluene: the ring carbons are cyclic, the methyl is not."""
    molecule = parse_smiles("Cc1ccccc1")
    assert not molecule.atoms[0].in_ring
    assert all(a.in_ring for a in molecule.atoms[1:])


def test_ring_detection_handles_fused_rings():
    molecule = parse_smiles("c1ccc2ccccc2c1")       # naphthalene
    assert all(a.in_ring for a in molecule.atoms)
    assert count_rings(molecule) == 2


def test_ring_detection_on_long_chain_does_not_recurse():
    """Bridge-finding is iterative; a recursive version overflows here."""
    molecule = parse_smiles("C" * 900)
    assert molecule.n_atoms == 900
    assert not any(a.in_ring for a in molecule.atoms)


# --------------------------------------------------------------------------- #
# Chemistry helpers
# --------------------------------------------------------------------------- #

def test_implicit_hydrogens_on_methane_and_ethanol():
    methane = parse_smiles("C")
    assert implicit_hydrogens(methane.atoms[0], 0.0) == 4
    ethanol = parse_smiles("CCO")
    # Terminal carbon has one bond, so three hydrogens.
    assert implicit_hydrogens(ethanol.atoms[0], 1.0) == 3
    # Oxygen has one bond, so one hydrogen.
    assert implicit_hydrogens(ethanol.atoms[2], 1.0) == 1


def test_explicit_hydrogens_override_the_valence_rule():
    molecule = parse_smiles("[CH3-]")
    assert implicit_hydrogens(molecule.atoms[0], 0.0) == 3


def test_molecular_weight_of_ethanol():
    """C2H6O = 46.07 g/mol."""
    assert molecular_weight(parse_smiles("CCO")) == pytest.approx(46.07, abs=0.1)


def test_molecular_weight_of_benzene():
    """C6H6 = 78.11 g/mol."""
    assert molecular_weight(parse_smiles("c1ccccc1")) == pytest.approx(78.11,
                                                                      abs=0.5)


def test_ring_count_by_cyclomatic_number():
    assert count_rings(parse_smiles("CCO")) == 0
    assert count_rings(parse_smiles("C1CCCCC1")) == 1
    assert count_rings(parse_smiles("c1ccc2ccccc2c1")) == 2


def test_rotatable_bonds_exclude_rings_and_terminals():
    # Butane has one rotatable bond: the central C-C.
    assert count_rotatable_bonds(parse_smiles("CCCC")) == 1
    # Cyclohexane has none -- all its single bonds are in the ring.
    assert count_rotatable_bonds(parse_smiles("C1CCCCC1")) == 0


def test_hydrogen_bond_donors():
    assert hydrogen_bond_donors(parse_smiles("CCO")) == 1       # the -OH
    assert hydrogen_bond_donors(parse_smiles("CCOC")) == 0      # ether, no H


def test_descriptors_are_finite_across_the_library():
    for smiles in make_molecule_library(200, seed=0):
        values = descriptors(smiles)
        assert all(np.isfinite(v) for v in values.values()), smiles


# --------------------------------------------------------------------------- #
# Featurisation
# --------------------------------------------------------------------------- #

def test_feature_dimensions_match_the_declared_constants():
    molecule = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")     # aspirin
    assert atom_features(molecule).shape == (molecule.n_atoms, ATOM_FEATURE_DIM)
    assert bond_features(molecule).shape == (len(molecule.bonds),
                                             BOND_FEATURE_DIM)


def test_graph_edges_are_bidirectional():
    """Message passing on a chemical graph must be symmetric."""
    graph = to_graph("CCO")
    assert graph["edge_index"].shape[1] == 2 * len(parse_smiles("CCO").bonds)
    edges = set(zip(*graph["edge_index"].tolist()))
    for a, b in list(edges):
        assert (b, a) in edges


def test_graph_of_single_atom_has_no_edges():
    graph = to_graph("C")
    assert graph["n_atoms"] == 1
    assert graph["edge_index"].shape == (2, 0)
    assert graph["edge_features"].shape == (0, BOND_FEATURE_DIM)


def test_atom_features_are_finite_and_bounded():
    features = atom_features(parse_smiles("CC(=O)Oc1ccccc1C(=O)O"))
    assert np.isfinite(features).all()
    assert np.abs(features).max() < 10


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #

def test_scatter_sum_groups_correctly():
    source = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    index = torch.tensor([0, 0, 1, 1])
    assert torch.equal(scatter_sum(source, index, 2), torch.tensor([[3.0], [7.0]]))


def test_scatter_max_handles_empty_groups():
    """A group with no members must give 0, not -inf propagating to nan."""
    source = torch.tensor([[1.0], [5.0]])
    index = torch.tensor([0, 0])
    out = scatter_max(source, index, 3)
    assert torch.isfinite(out).all()
    assert out[0].item() == 5.0 and out[2].item() == 0.0


def test_collate_offsets_node_indices():
    """The classic GNN batching bug: edges must never cross molecules."""
    graphs = [to_graph("CCO"), to_graph("CCC"), to_graph("c1ccccc1")]
    batch = collate_graphs(graphs)

    sizes = [g["n_atoms"] for g in graphs]
    assert batch["n_nodes"] == sum(sizes)
    assert batch["nodes"].shape[0] == sum(sizes)

    # Every edge must stay inside the molecule it came from.
    boundaries = np.cumsum([0] + sizes)
    for a, b in zip(*batch["edge_index"].tolist()):
        molecule_a = np.searchsorted(boundaries, a, side="right") - 1
        molecule_b = np.searchsorted(boundaries, b, side="right") - 1
        assert molecule_a == molecule_b


def test_collate_batch_vector_labels_each_node():
    graphs = [to_graph("CCO"), to_graph("C")]
    batch = collate_graphs(graphs)
    assert batch["batch"].tolist() == [0, 0, 0, 1]
    assert batch["n_graphs"] == 2


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def make_model(**kwargs):
    defaults = dict(hidden_dim=32, n_layers=2, n_tasks=1, dropout=0.0)
    defaults.update(kwargs)
    return MolecularGNN(**defaults)


def test_forward_output_shape():
    model = make_model().eval()
    batch = collate_graphs([to_graph(s) for s in ("CCO", "c1ccccc1", "CCN")])
    assert model(batch).shape == (3, 1)


@pytest.mark.parametrize("readout", ["sum", "mean", "max", "mean_max"])
def test_all_readouts_work(readout):
    model = make_model(readout=readout).eval()
    batch = collate_graphs([to_graph("CCO"), to_graph("CC")])
    assert model(batch).shape == (2, 1)


def test_unknown_readout_raises():
    model = make_model(readout="nonsense").eval()
    with pytest.raises(ValueError, match="unknown readout"):
        model(collate_graphs([to_graph("CCO")]))


def test_single_atom_molecule_does_not_produce_nan():
    """No bonds means no messages; the model must survive it."""
    model = make_model().eval()
    with torch.no_grad():
        out = model(collate_graphs([to_graph("C")]))
    assert torch.isfinite(out).all()


def test_prediction_is_independent_of_batch_composition():
    """A molecule's prediction must not depend on what it was batched with."""
    torch.manual_seed(0)
    model = make_model().eval()
    target = to_graph("CC(=O)Oc1ccccc1C(=O)O")

    with torch.no_grad():
        alone = model(collate_graphs([target]))
        with_others = model(collate_graphs(
            [to_graph("CCO"), target, to_graph("c1ccncc1")]
        ))[1]
    assert torch.allclose(alone[0], with_others, atol=1e-5)


def test_prediction_is_permutation_invariant():
    """Graph output must not depend on atom ordering within the batch."""
    torch.manual_seed(0)
    model = make_model().eval()
    a, b = to_graph("CCO"), to_graph("c1ccccc1")
    with torch.no_grad():
        forward = model(collate_graphs([a, b]))
        reverse = model(collate_graphs([b, a]))
    assert torch.allclose(forward[0], reverse[1], atol=1e-5)
    assert torch.allclose(forward[1], reverse[0], atol=1e-5)


def test_gradients_flow():
    model = make_model()
    batch = collate_graphs([to_graph("CCO"), to_graph("c1ccccc1")])
    model(batch).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())


def test_model_can_overfit_a_tiny_set():
    """Sanity check that the architecture can learn at all."""
    torch.manual_seed(0)
    smiles = ["CCO", "c1ccccc1", "CCCCCC", "CC(=O)O", "c1ccncc1"]
    graphs = [to_graph(s) for s in smiles]
    y = torch.tensor([-1.0, 1.0, 2.0, -2.0, 0.5])

    model = make_model(hidden_dim=64, n_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    batch = collate_graphs(graphs)
    for _ in range(300):
        loss = torch.nn.functional.mse_loss(model(batch).squeeze(-1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss) < 0.05


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def test_scaffold_of_benzene_ring_is_shared():
    """Molecules sharing a core must share a scaffold."""
    a = bemis_murcko_scaffold("Cc1ccccc1")
    b = bemis_murcko_scaffold("CCc1ccccc1")
    assert a == b


def test_different_cores_give_different_scaffolds():
    assert bemis_murcko_scaffold("Cc1ccccc1") != \
        bemis_murcko_scaffold("CC1CCCCC1")


def test_acyclic_molecules_share_the_empty_scaffold():
    assert bemis_murcko_scaffold("CCCC") == ""
    assert bemis_murcko_scaffold("CCO") == ""


def test_scaffold_split_covers_everything_without_overlap():
    smiles = make_molecule_library(400, seed=0)
    train, val, test = scaffold_split(smiles, 0.1, 0.1, seed=0)
    all_idx = np.concatenate([train, val, test])
    assert len(all_idx) == len(smiles)
    assert len(set(all_idx.tolist())) == len(smiles)


def test_scaffold_split_keeps_scaffolds_in_one_set():
    """The whole point: no scaffold may appear in two sets."""
    smiles = make_molecule_library(500, seed=1)
    train, val, test = scaffold_split(smiles, 0.15, 0.15, seed=0)

    def scaffolds(indices):
        return {bemis_murcko_scaffold(smiles[i]) for i in indices}

    train_s, val_s, test_s = scaffolds(train), scaffolds(val), scaffolds(test)
    assert not (train_s & test_s)
    assert not (train_s & val_s)
    assert not (val_s & test_s)


def test_random_split_does_leak_scaffolds():
    """Contrast test: this is exactly why scaffold splitting exists."""
    smiles = make_molecule_library(500, seed=1)
    train, _, test = random_split(len(smiles), 0.15, 0.15, seed=0)
    train_s = {bemis_murcko_scaffold(smiles[i]) for i in train}
    test_s = {bemis_murcko_scaffold(smiles[i]) for i in test}
    assert train_s & test_s, "random split unexpectedly kept scaffolds apart"


def test_scaffold_split_sizes_are_reasonable():
    smiles = make_molecule_library(600, seed=2)
    train, val, test = scaffold_split(smiles, 0.1, 0.1, seed=0)
    assert len(train) > len(val) and len(train) > len(test)


# --------------------------------------------------------------------------- #
# Dataset and metrics
# --------------------------------------------------------------------------- #

def test_library_molecules_all_parse():
    for smiles in make_molecule_library(300, seed=0):
        assert parse_smiles(smiles).n_atoms >= 2


def test_library_is_deduplicated():
    library = make_molecule_library(300, seed=0)
    assert len(set(library)) == len(library)


def test_dataset_targets_vary():
    _, targets = make_dataset(300, seed=0)
    assert np.isfinite(targets).all()
    assert targets.std() > 0.5


def test_target_is_not_perfectly_linear_in_descriptors():
    """Interaction terms must leave headroom above a linear descriptor model."""
    from sklearn.linear_model import LinearRegression

    smiles, targets = make_dataset(800, seed=0)
    X = descriptor_matrix(smiles)
    predicted = LinearRegression().fit(X, targets).predict(X)
    assert r2(targets, predicted) < 0.98, "target is trivially linear"


def test_target_contains_signal_outside_the_descriptor_set():
    """Otherwise the descriptor baseline has privileged access to the truth.

    A random forest given every descriptor should still leave meaningful
    residual variance, because the topological motif term is not among them.
    """
    from sklearn.ensemble import RandomForestRegressor

    smiles, targets = make_dataset(800, seed=0)
    X = descriptor_matrix(smiles)
    forest = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    # Out-of-bag style check via a simple holdout.
    forest.fit(X[:600], targets[:600])
    assert r2(targets[600:], forest.predict(X[600:])) < 0.97


def test_topological_motif_is_not_a_descriptor():
    """If it leaked into the descriptor vector the test above would be void."""
    from src.data import DESCRIPTOR_NAMES
    assert not any("motif" in name for name in DESCRIPTOR_NAMES)


def test_topological_motif_counts_branch_points_with_heteroatoms():
    from src.data import topological_motif_count

    # Isobutane: a branch point, but no heteroatom neighbour.
    assert topological_motif_count(parse_smiles("CC(C)C")) == 0
    # Same skeleton with an oxygen on the branch point.
    assert topological_motif_count(parse_smiles("CC(O)C")) == 1
    # Linear chain: no branch point at all.
    assert topological_motif_count(parse_smiles("CCO")) == 0


def test_descriptor_matrix_shape():
    smiles = make_molecule_library(50, seed=0)
    assert descriptor_matrix(smiles).shape[0] == 50


def test_metrics_on_a_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    results = evaluate(y, y)
    assert results["rmse"] == 0.0 and results["r2"] == 1.0


def test_r2_is_negative_for_a_bad_model():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, np.full(4, 100.0)) < 0


def test_rmse_penalises_large_errors():
    y = np.zeros(10)
    small = np.full(10, 0.1)
    one_big = np.zeros(10)
    one_big[0] = 1.0
    assert rmse(y, one_big) > rmse(y, small)
