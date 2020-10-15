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


def insert_molecule(mof, molecule, num_cells=[1,1,1], spacegroup='P1', mol_shift=[0,0,0], filename=None):

    # Load MOF and Molecule files, if needed
    if type(mof) ==  'str':
        mof = ase.io.read(mof)
    if type(molecule) == 'str':
        molecule = ase.io.read(molecule_file)
    if type(mof) != ase.atoms.Atoms and type(mol) != ase.atoms.Atoms:
        return('Invalid structures!')

    # Load MOF parameters, and convert to ASE crystal
    mof_cell_params = np.round(1E8*mof.get_cell_lengths_and_angles())*1E-8
    [a, b, c, alpha, beta, gamma] = mof_cell_params
    cell_lengths = [a, b, c]
    cell_angles_deg = [alpha, beta, gamma]
    cell_angles_rad = np.deg2rad(cell_angles_deg)
    mof_atomtypes = mof.get_chemical_symbols()
    mof_cart_coords = mof.get_positions()
    mof_frac_coords = mof.get_scaled_positions()
    mof_crystal = ase.spacegroup.crystal(mof_atomtypes, mof_frac_coords, spacegroup=spacegroup, cellpar=mof_cell_params)

    # Tile MOF
    ao, bo, co = [num_cells[0], 0, 0], [0, num_cells[1], 0], [0, 0, num_cells[2]]
    n_layers = num_cells[2]
    mof_new = ase.build.cut(mof_crystal, a=ao, b=bo, origo=(0,0,0), nlayers=n_layers, tolerance=0.001)
    mol_id = np.zeros(len(mof_new))

    # Align MOF and Molecule
    mol_new = copy.deepcopy(molecule)
    mof_cop = get_center_of_positions(mof_new)
    mol_cop = get_center_of_positions(mol_new)
    mol_new.translate(mof_cop-mol_cop+mol_shift)
    mol_id = np.append(mol_id, np.ones(len(mol_new)))

    # Create config and write to file if desired
    final_config = mol_new + mof_new
    if filename != None:
        ase.io.write(filename, final_config)

    return final_config, mol_id


def create_new_unit_cell(atom_config, mol_ids, cell_lengths, cell_angles, num_cells=[1,1,1], filename=None):

    cell_lengths = [a*b for a,b in zip(cell_lengths, num_cells)]
    cell_cop = get_center_of_cell(cell_lengths, cell_angles)
    atom_config_cop = get_center_of_positions(atom_config)
    atom_config_new = copy.deepcopy(atom_config)
    atom_config_new.translate(cell_cop-atom_config_cop)
    convert_to_fractional(atom_config_new, cell_lengths, cell_angles, degrees=True)

    atoms_to_keep = []
    atoms_to_remove = []
    all_xyz = atom_config_new.get_positions()
    mol_ids_new = []
    for i in range(len(all_xyz)):
        pos = all_xyz[i]
        if np.min(pos) < 0 or np.max(pos) > 1:
            atoms_to_remove.extend([i])
        else:
            atoms_to_keep.extend([i])
            mol_ids_new.extend([mol_ids[i]])

    atom_config_new = Atoms([atom for atom in atom_config_new if atom.index in atoms_to_keep])
    config_as_crystal = ase.spacegroup.crystal(atom_config_new.get_chemical_symbols(), atom_config_new.get_positions(), spacegroup='P1', cellpar=cell_lengths+cell_angles)

    if filename != None:
        ase.io.write(filename, config_as_crystal)

    return atom_config_new, mol_ids_new
