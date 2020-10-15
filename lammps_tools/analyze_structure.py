import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import copy
import numpy as np
import glob
from collections import OrderedDict

from helper import (
    mod,
    convert_to_fractional,
    convert_to_cartesian
    )

def calculate_distance(p1, p2):
    dist = np.sum([(p1[i]-p2[i])**2 for i in range(len(p1))])**0.5

    return dist


def create_extended_cell(atoms, mol_ids, cell_lengths, cell_angles, degrees=True, fractional_in=False, fractional_out=True, periodic='xyz', filename=None):

    atoms_copy = copy.deepcopy(atoms)
    if fractional_in == False:
        atoms_copy = convert_to_fractional(atoms_copy, cell_lengths, cell_angles, degrees=degrees)
    all_atomtypes = atoms_copy.get_chemical_symbols()
    all_xyz_frac = atoms_copy.get_positions().transpose()
    all_indicies = [atom.index for atom in atoms_copy]

    atom_types_extended = []
    atom_positions_extended = []
    pseudo_indicies = []
    mol_ids_extended = []

    trans = [0, -1, 1]
    # Develop better strategy for this section, else is unecessarily long
    if periodic == 'xyz':
        for xt in trans:
            for yt in trans:
                for zt in trans:
                    x_trans = all_xyz_frac[0] + xt
                    y_trans = all_xyz_frac[1] + yt
                    z_trans = all_xyz_frac[2] + zt
                    atom_positions_translated = np.array([x_trans, y_trans, z_trans]).transpose()

                    atom_types_extended.extend(all_atomtypes)
                    atom_positions_extended.extend(atom_positions_translated)
                    pseudo_indicies.extend(all_indicies)
                    mol_ids_extended.extend(mol_ids)

    zt = 0
    if periodic == 'xy':
        for xt in trans:
            for yt in trans:
                x_trans = all_xyz_frac[0] + xt
                y_trans = all_xyz_frac[1] + yt
                z_trans = all_xyz_frac[2] + zt
                atom_positions_translated = np.array([x_trans, y_trans, z_trans]).transpose()

                atom_types_extended.extend(all_atomtypes)
                atom_positions_extended.extend(atom_positions_translated)
                pseudo_indicies.extend(all_indicies)
                mol_ids_extended.extend(mol_ids)

    if periodic == None:
        atom_types_extended = all_atomtypes
        atom_positions_extended = all_xyz_frac.transpose()
        pseudo_indicies = all_indicies
        mol_ids_extended = mol_ids

    atoms_extended = Atoms(atom_types_extended, atom_positions_extended)
    if filename != None:
        ase.io.write(filename, convert_to_cartesian(atoms_extended, cell_lengths, cell_angles, degrees=True))
    if fractional_out == False:
        atoms_extended = convert_to_cartesian(atoms_extended, cell_lengths, cell_angles, degrees=True)

    return atoms_extended, mol_ids_extended, pseudo_indicies


def guess_bonds(atoms, mol_ids, cell_lengths, cell_angles, degrees=True, fractional_in=False, cutoff=1.5, periodic='xyz'):

    # add deepcopy atoms
    if fractional_in == True:
        atoms = convert_to_cartesian(atoms, cell_lengths, cell_angles, degrees=True)
    atoms_xyz = atoms.get_positions()
    atom_types = atoms.get_chemical_symbols()

    atoms_ext, mol_ids_ext, pseudo_indicies = create_extended_cell(atoms, mol_ids, cell_lengths, cell_angles, degrees=True, fractional_in=False, fractional_out=False, periodic=periodic)
    print(len(atoms))
    print(len(atoms_ext))
    atoms_ext_xyz = atoms_ext.get_positions()
    atoms_ext_types = atoms_ext.get_chemical_symbols()

    all_bonds = []
    all_bond_types = []
    all_bond_lengths = []
    num_across_boundary = 0

    for i in range(len(atoms_xyz)):
        # if mod(i, 100) == 0:
        #     print(i, '  Still running...')
        p1, type1 = atoms_xyz[i], atom_types[i]

        for j in range(i+1,len(atoms_ext)):
            p2, type2 = atoms_ext_xyz[j], atoms_ext_types[j]

            r = calculate_distance(p1,p2)
            bondtype = '-'.join(sorted([type1, type2]))

            if type(cutoff) == dict:
                if bondtype in cutoff.keys():
                    if r <= cutoff[bondtype] and mol_ids[i] == mol_ids_ext[j]:
                        bond = sorted(set((i,pseudo_indicies[j])))
                        if bond not in all_bonds:
                            all_bonds.extend([bond])
                            all_bond_types.extend([sorted([type1, type2])])
                            all_bond_lengths.extend([r])
                            if j > len(atoms_xyz):
                                num_across_boundary += 1
                else:
                    if r <= cutoff['default'] and mol_ids[i] == mol_ids_ext[j]:
                        bond = sorted(set((i,pseudo_indicies[j])))
                        if bond not in all_bonds:
                            all_bonds.extend([bond])
                            all_bond_types.extend([sorted([type1, type2])])
                            all_bond_lengths.extend([r])
                            if j > len(atoms_xyz):
                                num_across_boundary += 1

            else:
                if r <= cutoff and mol_ids[i] == mol_ids_ext[j]:
                    bond = sorted(set((i,pseudo_indicies[j])))
                    if bond not in all_bonds:
                        all_bonds.extend([bond])
                        all_bond_types.extend([sorted([type1, type2])])
                        all_bond_lengths.extend([r])
                        if j > len(atoms_xyz):
                            num_across_boundary += 1


    return all_bonds, all_bond_types, all_bond_lengths, num_across_boundary


def guess_angles(atoms, bonds):
    all_angles = []
    all_angle_types = []

    for i in range(len(bonds)):
        bond1 = bonds[i]

        for j in range(i+1,len(bonds)):
            bond2 = bonds[j]

            atoms_in_angle = sorted(set(bond1+bond2))
            if len(atoms_in_angle) == 3:
                center_atom = sorted(set(bond1).intersection(bond2))
                end_atoms = sorted(set(atoms_in_angle).difference(center_atom))

                ordered_atoms_in_angle = copy.deepcopy(end_atoms)
                ordered_atoms_in_angle.insert(1, *center_atom)
                ordered_atom_types_in_angle = [atoms[index].symbol for index in end_atoms]
                ordered_atom_types_in_angle.insert(1, *[atoms[index].symbol for index in center_atom])

                all_angles.extend([[*center_atom, atoms_in_angle]])
                all_angle_types.extend([ordered_atom_types_in_angle])

    all_angles = sorted(all_angles)

    return all_angles, all_angle_types


def guess_dihedrals_and_impropers(atoms, bonds, angles):
    all_dihedrals = []
    all_dihedral_types = []
    all_impropers = []
    all_improper_types = []

    for i in range(len(angles)):
        center_atom = angles[i][0]
        angle = angles[i][1]

        for j in range(len(bonds)):
            bond = bonds[j]
            atoms_in_group = sorted(list(set(angle+bond)))
            shared_atom = sorted(set(angle).intersection(bond))

            if len(atoms_in_group) == 4 and shared_atom != [center_atom]:
                all_dihedrals.extend([atoms_in_group])

                ordered_atoms = list(set(angle).difference([center_atom]).difference(shared_atom))
                ordered_atoms.insert(1, center_atom)
                ordered_atoms.insert(2, *shared_atom)
                ordered_atoms.insert(3, *list(set(bond).difference(shared_atom)))
                ordered_atom_types = [atoms[index].symbol for index in ordered_atoms]
                all_dihedral_types.extend([ordered_atom_types])

            if len(atoms_in_group) == 4 and shared_atom == [center_atom]:
                all_impropers.extend([[center_atom, atoms_in_group]])

                ordered_atoms = set(copy.deepcopy(atoms_in_group))
                ordered_atoms = ordered_atoms.difference([center_atom])
                ordered_atom_types = sorted(atoms[index].symbol for index in ordered_atoms)
                ordered_atom_types.insert(0, atoms[center_atom].symbol)
                all_improper_types.extend([ordered_atom_types])

    return all_dihedrals, all_dihedral_types, all_impropers, all_improper_types


def get_number_of_bonds_on_atom(atoms, bonds):
    bond_count = {str(i): 0 for i in range(len(atoms))}
    for bond in bonds:
        for index in bond:
            bond_count[str(index)] += 1

    return OrderedDict(bond_count)
