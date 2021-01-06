import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import copy
import numpy as np
import glob
from collections import OrderedDict


from lammps_tools.helper import (
    mod,
    convert_to_fractional,
    convert_to_cartesian
    )


def calculate_distance(p1, p2):
    dist = np.sum([(p1[i]-p2[i])**2 for i in range(len(p1))])**0.5

    return dist


def create_extended_cell(atoms, mol_ids, cell_lengths, cell_angles, degrees=True, fractional_in=False, fractional_out=False, periodic='xyz', filename=None):

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


def guess_bonds(atoms_in, mol_ids, cell_lengths, cell_angles, degrees=True, fractional_in=False, cutoff={'default': 1.5}, periodic='xyz'):

    # Prepare Atoms Object
    atoms = copy.deepcopy(atoms_in)
    if fractional_in == True:
        atoms = convert_to_cartesian(atoms, cell_lengths, cell_angles, degrees=True)
    atoms_out = copy.deepcopy(atoms)
    atoms_ext, mol_ids_ext, pseudo_indicies = create_extended_cell(atoms, mol_ids, cell_lengths, cell_angles, degrees=True, fractional_in=False, fractional_out=False, periodic=periodic)

    # Check Cutoff Dictionary
    if 'default' not in cutoff.keys():
        cutoff['default'] = 1.5

    all_bonds = []
    all_bonds_alt = []
    all_bond_types = []

    bonds_across_boundary = []
    bonds_across_boundary_count = 0

    extra_atoms_for_plot = Atoms()
    extra_bonds_for_plot = []

    for i in range(len(atoms)):
        p1 = atoms[i].position
        type1 = atoms[i].symbol

        for j in range(i+1,len(atoms_ext)):
            p2 = atoms_ext[j].position
            type2 = atoms_ext[j].symbol

            r = calculate_distance(p1,p2)
            bondtype = '-'.join(sorted([type1, type2]))

            if ((bondtype in cutoff.keys() and r <= cutoff[bondtype]) or r <= cutoff['default']) and mol_ids[i] == mol_ids_ext[j]:
                bond = sorted(set((i,pseudo_indicies[j])))
                bond_alt = bond
                if bond not in all_bonds:
                    all_bonds.extend([bond])
                    all_bond_types.extend([sorted([type1, type2])])
                    if j > len(atoms):
                        bond_alt = sorted(set((i,len(atoms)+bonds_across_boundary_count)))
                        bonds_across_boundary.extend([bond])
                        bonds_across_boundary_count += 1
                        atoms_out += atoms_ext[j]
                    all_bonds_alt.extend([bond_alt])
                if j > len(atoms):
                    extra_atoms_for_plot += atoms_ext[j]
                    extra_bonds_for_plot.extend([[i,len(atoms)+len(extra_atoms_for_plot)]])

    return atoms_out, all_bonds, all_bonds_alt, all_bond_types, bonds_across_boundary, extra_atoms_for_plot, extra_bonds_for_plot


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

                all_angles.extend([[*center_atom, ordered_atoms_in_angle]])
                all_angle_types.extend([ordered_atom_types_in_angle])

    all_angles = sorted(all_angles)

    return all_angles, all_angle_types


def guess_dihedrals_and_impropers(atoms_in, bonds, angles, improper_tol=0.1):
    atoms = copy.deepcopy(atoms_in)
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
                # all_dihedrals.extend([atoms_in_group])

                ordered_atoms = list(set(angle).difference([center_atom]).difference(shared_atom))
                ordered_atoms.insert(1, center_atom)
                ordered_atoms.insert(2, *shared_atom)
                ordered_atoms.insert(3, *list(set(bond).difference(shared_atom)))

                if ordered_atoms[0] > ordered_atoms[-1]:
                    ordered_atoms.reverse()
                all_dihedrals.extend([ordered_atoms])

                ordered_atom_types = [atoms[index].symbol for index in ordered_atoms]
                all_dihedral_types.extend([ordered_atom_types])

            if len(atoms_in_group) == 4 and shared_atom == [center_atom]:
                # Impropers should lie approxiamtely in the same plane
                ordered_atoms = set(copy.deepcopy(atoms_in_group))
                ordered_atoms = ordered_atoms.difference([center_atom])
                ordered_atoms = list(ordered_atoms)

                # Create two vectors from non-central points
                pc = atoms[center_atom].position
                p0  = atoms[ordered_atoms[0]].position
                v01 = atoms[ordered_atoms[1]].position - atoms[ordered_atoms[0]].position
                v02 = atoms[ordered_atoms[2]].position = atoms[ordered_atoms[0]].position
                a,b,c = np.cross(v01, v02)
                d = -1*(a*p0[0] + b*p0[1] + c*p0[2])
                num, den = a*pc[0]+b*pc[1]+c*pc[2]+d, (a**2 + b**2 +c**2)**0.5
                if den != 0:
                    dmin = abs(num/den)
                else:
                    dmin = 0
                # print(a,b,c,d,dmin)

                if dmin <= improper_tol:
                    all_impropers.extend([[center_atom, [center_atom, *sorted(ordered_atoms)]]])
                    ordered_atom_types = sorted(atoms[index].symbol for index in ordered_atoms)
                    ordered_atom_types.insert(0, atoms[center_atom].symbol)
                    all_improper_types.extend([ordered_atom_types])

    return all_dihedrals, all_dihedral_types, all_impropers, all_improper_types


def get_bonds_on_atom(atoms, bonds):
    bond_count = {str(i): 0 for i in range(len(atoms))}
    bonds_present = {str(i): [] for i in range(len(atoms))}
    bonds_with = {str(i): [] for i in range(len(atoms))}
    for bond in bonds:
        for index in bond:
            bond_count[str(index)] += 1
            bonds_present[str(index)].extend([bond])
            bonds_with[str(index)].extend([atoms[ai].symbol for ai in bond if ai != index])

    return OrderedDict(bond_count), OrderedDict(bonds_present), OrderedDict(bonds_with)


def update_bond_or_dihedral_types(all_bonds, ff_atom_types):
    bond_types = [[ff_atom_types[index] for index in bond] for bond in all_bonds]
    return bond_types


def update_angle_or_improper_types(all_angles, ff_atom_types):
    angle_types = [[ff_atom_types[index] for index in angle[-1]] for angle in all_angles]
    return angle_types


def sort_bond_angle_dihedral_type_list(all_types):
    if len(all_types[0]) == 2:
        for i in range(len(all_types)):
            all_types[i] = sorted(all_types[i])

    else:
        for i in range(len(all_types)):
            if [all_types[i][0], all_types[i][-1]] != sorted([all_types[i][0], all_types[i][-1]]):
                all_types[i] = all_types[i][::-1]

    return all_types


def sort_improper_type_list(all_types):
    for i in range(len(all_types)):
        all_types[i][1::] = sorted(all_types[i][1::])

    return all_types
