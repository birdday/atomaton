import ase as ase
import copy
import numpy as np


from atomaton.helper import (
    get_center_of_positions,
    get_center_of_cell,
)


def build_supercell(crystal, num_cells, filename=None):
    ao, bo, co = [num_cells[0], 0, 0], [0, num_cells[1], 0], [0, 0, num_cells[2]]
    crystal_new = ase.build.cut(
        crystal, a=ao, b=bo, c=co, origo=(0, 0, 0), tolerance=0.001
    )

    if filename != None:
        ase.io.write(crystal_new)

    return crystal_new


def shift_bond_indicies(bonds, shift_by):
    return [[i+shift_by, j+shift_by] for i,j in bonds]


def insert_molecule(
    crystal,
    molecule,
    mol_shift=[0, 0, 0],
    filename=None,
):
    # Load Crystal and Molecule files, if needed
    if type(crystal) == "str":
        crystal = ase.io.read(crystal)
    if type(molecule) == "str":
        molecule = ase.io.read(molecule)
    if type(crystal) != ase.atoms.Atoms and type(molecule) != ase.atoms.Atoms:
        return "Invalid structures!"

    # Create MOF cell
    crystal_copy = copy.deepcopy(crystal)
    a, b, c, alpha, beta, gamma = crystal_copy.get_cell_lengths_and_angles()

    # Align molecule to center of crystal
    molecule_copy = copy.deepcopy(molecule)
    crystal_center = get_center_of_cell([a, b, c], [alpha, beta, gamma])
    molecule_center = get_center_of_positions(molecule_copy)
    molecule_copy.translate(crystal_center - molecule_center + mol_shift)

    # Update molecule ids
    molecule_ids = np.zeros(len(crystal_copy))
    molecule_ids = np.append(molecule_ids, np.ones(len(molecule_copy)))

    # Create config and write to file if desired
    final_config = crystal_copy + molecule_copy
    if filename != None:
        ase.io.write(filename, final_config)

    return final_config, molecule_ids
