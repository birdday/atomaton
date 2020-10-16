import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import numpy as np
import pytest
from pytest import approx
from collections import OrderedDict

from lammps_tools.helper import (
    mod,
    get_unique_items,
    get_center_of_positions,
    get_center_of_cell,
    convert_to_fractional,
    convert_to_cartesian,
    )
from lammps_tools.analyze_structure import (
    calculate_distance,
    create_extended_cell,
    guess_bonds,
    guess_angles,
    guess_dihedrals_and_impropers,
    get_number_of_bonds_on_atom
    )


def test_ethane_atoms_properly_parsed():
    atoms = ase.io.read('tests/ethane.xyz')

    symbols = atoms.get_chemical_symbols()
    expected_symbols = ['H', 'C', 'H', 'H', 'C', 'H', 'H', 'H']
    assert symbols == expected_symbols

    positions = atoms.get_positions()
    expected_positions = [
        [ 1.185080, -0.003838,  0.987524],
        [ 0.751621, -0.022441, -0.020839],
        [ 1.166929,  0.833015, -0.569312],
        [ 1.115519, -0.932892, -0.514525],
        [-0.751587,  0.022496,  0.020891],
        [-1.166882, -0.833372,  0.568699],
        [-1.115691,  0.932608,  0.515082],
        [-1.184988,  0.004424, -0.987522]
    ]
    assert np.testing.assert_array_equal(positions, expected_positions) == None


def test_ethane_bonds_properly_predicted():
    atoms = ase.io.read('tests/ethane.xyz')
    bonds, bond_types, _, _ = guess_bonds(atoms, np.ones(len(atoms)), [20,20,20], [90,90,90], degrees=True, fractional_in=False, cutoff=1.6, periodic=None)
    unique_bond_types, _ = get_unique_items(bond_types)
    expected_bonds = [[0, 1], [1, 2], [1, 3], [1, 4], [4, 5], [4, 6], [4, 7]]
    expected_bond_types = [['C', 'H'], ['C', 'H'], ['C', 'H'], ['C', 'C'], ['C', 'H'], ['C', 'H'], ['C', 'H']]
    expected_unique_bond_types = [['C', 'H'], ['C', 'C']]

    assert np.testing.assert_array_equal(bonds, expected_bonds) == None
    assert np.testing.assert_array_equal(bond_types, expected_bond_types) == None
    assert np.testing.assert_array_equal(unique_bond_types, expected_unique_bond_types) == None


def test_ethane_bond_orders_correct():
    atoms = ase.io.read('tests/ethane.xyz')
    bonds, bond_types, _, _ = guess_bonds(atoms, np.ones(len(atoms)), [20,20,20], [90,90,90], degrees=True, fractional_in=False, cutoff=1.6, periodic=None)
    bond_count = get_number_of_bonds_on_atom(atoms, bonds)
    expected_bond_count = OrderedDict({'0': 1, '1': 4, '2': 1, '3': 1, '4': 4, '5': 1, '6': 1, '7': 1})

    assert bond_count == expected_bond_count


def test_periodic_bonding():
    atoms = Atoms('CCCC', positions=[(0.1, 5, 5), (1.1, 5, 5), (2.1, 5, 5), (3.1, 5, 5)])
    bonds, _, _, _ = guess_bonds(atoms, np.ones(len(atoms)), [4,10,10], [90,90,90], degrees=True, fractional_in=False, cutoff=1.5, periodic='xyz')
    bond_count = get_number_of_bonds_on_atom(atoms, bonds)
    expected_bonds = [[0, 1], [0, 3], [1, 2], [2, 3]]
    expected_bond_count = OrderedDict({'0': 2, '1': 2, '2': 2, '3': 2})

    assert np.testing.assert_array_equal(bonds, expected_bonds) == None
    assert bond_count == expected_bond_count
