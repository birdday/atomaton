import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import copy
import numpy as np


from lammps_tools.helper import (
    mod,
    get_unique_items,
    get_center_of_positions,
    get_center_of_cell,
    convert_to_fractional,
    convert_to_cartesian
    )


def build_supercell(crystal, num_cells, filename=None):
    ao, bo, co = [num_cells[0], 0, 0], [0, num_cells[1], 0], [0, 0, num_cells[2]]
    crystal_new = ase.build.cut(crystal, a=ao, b=bo, c=co, origo=(0,0,0), tolerance=0.001)

    if filename != None:
        ase.io.write(crystal_new)

    return crystal_new


def insert_molecule(crystal, molecule, num_cells=[1,1,1], spacegroup='P1', mol_shift=[0,0,0], filename=None):
    # Load Crystal and Molecule files, if needed
    if type(crystal) ==  'str':
        crystal = ase.io.read(crystal)
    if type(molecule) == 'str':
        molecule = ase.io.read(molecule)
    if type(crystal) != ase.atoms.Atoms and type(molecule) != ase.atoms.Atoms:
        return('Invalid structures!')

    # Create MOF cell
    crystal_new = build_supercell(crystal, num_cells)
    a, b, c, alpha, beta, gamma = crystal_new.get_cell_lengths_and_angles()

    # Align molecule to center of crystal
    molecule_new = copy.deepcopy(molecule)
    crystal_cop = get_center_of_cell([a,b,c], [alpha,beta,gamma])
    molecule_cop = get_center_of_positions(molecule_new)
    molecule_new.translate(crystal_cop-molecule_cop+mol_shift)

    # Update molecule ids
    molecule_ids = np.zeros(len(crystal_new))
    molecule_ids = np.append(molecule_ids, np.ones(len(molecule_new)))

    # Create config and write to file if desired
    final_config = crystal_new + molecule_new
    if filename != None:
        ase.io.write(filename, final_config)

    return final_config, molecule_ids
