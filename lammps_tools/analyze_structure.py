import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import copy
import numpy as np
import glob
from collections import OrderedDict


from lammps_tools.helper import (
    mod,
    column,
    get_unique_items,
    convert_to_fractional,
    convert_to_cartesian,
    atom_in_atoms
    )


def calculate_distance(p1, p2):
    dist = np.sum([(p1[i]-p2[i])**2 for i in range(len(p1))])**0.5

    return dist


def create_extended_cell(atoms, mol_ids, periodic='xyz'):

    atoms_copy = copy.deepcopy(atoms)
    all_atomtypes = atoms_copy.get_chemical_symbols()
    all_xyz_frac = atoms_copy.get_scaled_positions().transpose()
    all_indicies = [atom.index for atom in atoms_copy]

    atom_types_extended = []
    atom_positions_extended = []
    pseudo_indicies = []
    mol_ids_extended = []

    trans = [0, -1, 1]
    # Develop better strategy for this section, else is unecessarily long - recursion??
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

    elif periodic == 'xy':
        zt = 0
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

    elif periodic == None:
        atom_types_extended = all_atomtypes
        atom_positions_extended = all_xyz_frac.transpose()
        pseudo_indicies = all_indicies
        mol_ids_extended = mol_ids

    else:
        raise NameError('Unsupported or invalid periodicity.')

    cell_params = atoms_copy.get_cell_lengths_and_angles()
    cell_lengths, cell_angles = cell_params[0:3], cell_params[3::]
    atom_positions_extended = convert_to_cartesian(atom_positions_extended, cell_lengths, cell_angles, degrees=True)
    atoms_extended = Atoms(atom_types_extended, atom_positions_extended)
    atoms_extended.set_cell([*cell_lengths, *cell_angles])

    return atoms_extended, mol_ids_extended, pseudo_indicies


def guess_bonds(atoms_in, mol_ids, cutoff={'default': 1.5}, periodic='xyz'):

    # Prepare Atoms Object
    atoms = copy.deepcopy(atoms_in)
    atoms_out = copy.deepcopy(atoms)
    atoms_ext, mol_ids_ext, pseudo_indicies = create_extended_cell(atoms, mol_ids, periodic=periodic)

    # Check Cutoff Dictionary
    for key in cutoff:
        if len(cutoff[key]) == 1:
            cutoff[key] = [0, cutoff[key]]
        elif len(cutoff[key]) > 2:
            raise NameError('Invalid cutoff!')

    if 'default' not in cutoff.keys():
        cutoff['default'] = [0, 1.5]

    all_bonds = []
    all_bonds_alt = []
    all_bond_types = []

    bonds_across_boundary = []

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

            if ((bondtype in cutoff.keys() and r >= cutoff[bondtype][0] and r <= cutoff[bondtype][1]) \
            or (bondtype not in cutoff.keys() and r >= cutoff['default'][0] and r <= cutoff['default'][1])) \
            and mol_ids[i] == mol_ids_ext[j]:
                bond = sorted(set((i,pseudo_indicies[j])))
                bond_alt = bond
                if bond not in all_bonds:
                    all_bonds.extend([bond])
                    all_bond_types.extend([sorted([type1, type2])])
                    if j > len(atoms):
                        bonds_across_boundary.extend([bond])
                        truth_val, atom_to_use = atom_in_atoms(atoms_ext[j], atoms_out)
                        if truth_val == True:
                            bond_alt = sorted(set((i,atom_to_use.index)))
                        else:
                            bond_alt = sorted(set((i,len(atoms_out))))
                            atoms_out += atoms_ext[j]
                    all_bonds_alt.extend([bond_alt])

                if j > len(atoms):
                    extra_bonds_for_plot.extend([[i,len(atoms)+len(extra_atoms_for_plot)]])
                    extra_atoms_for_plot += atoms_ext[j]

    return atoms_out, all_bonds, all_bonds_alt, all_bond_types, bonds_across_boundary, extra_atoms_for_plot, extra_bonds_for_plot


def guess_angles(atoms, bonds, bonds_alt):
    all_angles = []
    all_angles_alt = []
    all_angle_types = []

    for i in range(len(bonds)):
        bond1 = bonds[i]
        bond1_alt = bonds_alt[i]

        for j in range(i+1,len(bonds)):
            bond2 = bonds[j]
            bond2_alt = bonds_alt[j]

            atoms_in_angle = sorted(set(bond1+bond2))
            atoms_in_angle_alt = sorted(set(bond1_alt+bond2_alt))
            if len(atoms_in_angle) == 3:

                # Angle defined in by core atom numbers, used for calcuating dihedrals and impropers and writing lammps files
                center_atom = sorted(set(bond1).intersection(bond2))
                end_atoms = sorted(set(atoms_in_angle).difference(center_atom))
                ordered_atoms_in_angle = copy.deepcopy(end_atoms)
                ordered_atoms_in_angle.insert(1, *center_atom)
                ordered_atom_types_in_angle = [atoms[index].symbol for index in end_atoms]
                ordered_atom_types_in_angle.insert(1, *[atoms[index].symbol for index in center_atom])

                all_angles.extend([[*center_atom, ordered_atoms_in_angle]])
                all_angle_types.extend([ordered_atom_types_in_angle])

                # Angle defined by extended atom numbers, used for calculating angle properties
                # Is this missing a condition for center atom not in either bond??
                if center_atom[0] in bond1_alt and center_atom[0] in bond2_alt:
                    center_atoms = center_atom
                elif center_atom[0] in bond1_alt:
                    bond2_center_atom = [i for i in bond2_alt if i >= len(atoms)]
                    center_atoms = center_atom + bond2_center_atom
                elif center_atom[0] in bond2_alt:
                    bond1_center_atom = [i for i in bond1_alt if i >= len(atoms)]
                    center_atoms = center_atom + bond1_center_atom
                else:
                    center_atoms = [i for i in bond1_alt if i >= len(atoms)] + [i for i in bond2_alt if i >= len(atoms)]
                all_angles_alt.extend([[center_atoms, [bond1_alt, bond2_alt]]])

    sorted_indicies = np.argsort(column(all_angles,0))
    all_angles_sorted = [all_angles[index] for index in sorted_indicies]
    all_angles_alt_sorted = [all_angles_alt[index] for index in sorted_indicies]
    all_angle_types_sorted = [all_angle_types[index] for index in sorted_indicies]

    return all_angles_sorted, all_angles_alt_sorted, all_angle_types_sorted


def guess_dihedrals_and_impropers(atoms_in, bonds, bonds_alt, angles, angles_alt, improper_tol=0.1):
    atoms = copy.deepcopy(atoms_in)
    all_dihedrals = []
    all_dihedrals_alt = []
    all_dihedral_types = []
    all_impropers = []
    all_impropers_alt = []
    all_improper_types = []

    for i in range(len(angles)):
        center_atom = angles[i][0]
        angle = angles[i][1]
        angle_alt = angles_alt[i][1]

        for j in range(len(bonds)):
            bond = bonds[j]
            bond_alt = bonds_alt[j]
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
                all_dihedrals_alt.extend([angle_alt+[bond_alt]])

                ordered_atom_types = [atoms[index].symbol for index in ordered_atoms]
                all_dihedral_types.extend([ordered_atom_types])

            if len(atoms_in_group) == 4 and shared_atom == [center_atom]:
                # Impropers should lie approxiamtely in the same plane
                ordered_atoms = set(copy.deepcopy(atoms_in_group))
                ordered_atoms = ordered_atoms.difference([center_atom])
                ordered_atoms = list(ordered_atoms)

                # Create two vectors from non-central points
                # This method likely needs to use alt atom positions...
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

                if dmin <= improper_tol:
                    all_impropers.extend([[center_atom, [center_atom, *sorted(ordered_atoms)]]])
                    ordered_atom_types = sorted(atoms[index].symbol for index in ordered_atoms)
                    ordered_atom_types.insert(0, atoms[center_atom].symbol)
                    all_improper_types.extend([ordered_atom_types])

                    if center_atom in angle_alt:
                        bond_center_atom = [i for i in bond_alt if i >= len(atoms)]
                        all_impropers_alt.extend([ [[center_atom]+bond_center_atom, angle_alt+[bond_alt]] ])
                    elif center_atom in bond_alt:
                        angle_center_atom = [i for i in angle_alt[0] if i >= len(atoms)]
                        all_impropers_alt.extend([ [[center_atom]+angle_center_atom, angle_alt+[bond_alt]] ])

    return all_dihedrals, all_dihedrals_alt, all_dihedral_types, all_impropers, all_impropers_alt, all_improper_types


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


def get_bond_properties(atoms, bonds, bond_types):
    unique_bond_types, _ = get_unique_items(bond_types)
    keys = ['-'.join(bond_type) for bond_type in unique_bond_types]
    bond_type_indicies = {key:[] for key in keys}

    bond_lengths = []
    for i in range(len(bonds)):
        bond_type = bond_types[i]
        key = '-'.join(bond_type)
        bond_type_indicies[key].extend([i])

        bond = bonds[i]
        p1 = atoms[bond[0]].position
        p2 = atoms[bond[1]].position
        d = calculate_distance(p1, p2)
        bond_lengths.extend([d])

    return bond_type_indicies, bond_lengths


def get_angle_properties(atoms, all_angles, all_angle_types):
    unique_angle_types, _ = get_unique_items(all_angle_types)
    keys = ['-'.join(angle_type) for angle_type in unique_angle_types]

    angle_type_indicies = {key:[] for key in keys}
    angle_type_angles = []
    angle_type_mag_ij = []
    angle_type_mag_jk = []
    angle_type_mag_ik = []

    for i in range(len(all_angles)):
        angle_type = all_angle_types[i]
        key = '-'.join(angle_type)

        angle = all_angles[i]
        if len(angle[0]) == 1:
            id_i = [index for index in angle[1][0] if index not in angle[0]]
            id_k = [index for index in angle[1][1] if index not in angle[0]]
            if len(id_i) != 1 or len(id_k) != 1:
                print('index: ', i)
                print('angle: ', angle)
                print('ids: ', id_i, id_k)
                raise NameError('Invalid ID length')
            atom_i = atoms[id_i[0]]
            atom_j = atoms[angle[0][0]]
            atom_k = atoms[id_k[0]]

            # v = vector, u = unit vector, mag = magnitude
            v_ji = atom_j.position-atom_i.position
            mag_ji = np.sqrt(v_ji.dot(v_ji))
            u_ji = v_ji/mag_ji

            v_jk = atom_j.position-atom_k.position
            mag_jk = np.sqrt(v_jk.dot(v_jk))
            u_jk = v_jk/mag_jk

            v_ik = atom_i.position-atom_k.position
            mag_ik = np.sqrt(v_ik.dot(v_ik))
            u_ik = v_ik/mag_ik

        elif len(angle[0]) == 2:
            id_i = [index for index in angle[1][0] if index not in angle[0]]
            id_j1 = [index for index in angle[0] if index in angle[1][0]]
            id_j2 = [index for index in angle[0] if index in angle[1][1]]
            id_k = [index for index in angle[1][1] if index not in angle[0]]
            if len(id_i) != 1 or len(id_j1) != 1 or len(id_j2) != 1 or len(id_k) != 1:
                print(id_i, id_j1, id_j2, id_k)
                raise NameError('Invalid ID length')
            atom_i = atoms[id_i[0]]
            atom_j1 = atoms[id_j1[0]]
            atom_j2 = atoms[id_j2[0]]
            atom_k = atoms[id_k[0]]

            # v = vector, u = unit vector, mag = magnitude
            v_ji = atom_j1.position-atom_i.position
            mag_ji = np.sqrt(v_ji.dot(v_ji))
            u_ji = v_ji/mag_ji

            v_jk = atom_j2.position-atom_k.position
            mag_jk = np.sqrt(v_jk.dot(v_jk))
            u_jk = v_jk/mag_jk

            v_ik = atom_i.position-atom_k.position
            mag_ik = np.sqrt(v_ik.dot(v_ik))
            u_ik = v_ik/mag_ik

        theta_ijk = np.rad2deg(np.arccos(np.clip(np.dot(u_ji, u_jk), -1.0, 1.0)))

        angle_type_indicies[key].extend([i])
        angle_type_angles.extend([theta_ijk])
        angle_type_mag_ij.extend([mag_ji])
        angle_type_mag_jk.extend([mag_jk])
        angle_type_mag_ik.extend([mag_ik])

    return angle_type_indicies, angle_type_angles, angle_type_mag_ij, angle_type_mag_jk, angle_type_mag_ik,


def split_by_property(type_indicies, values, tol=5):
    types_final = []
    properties_final = {}
    indicies_final = {}

    for key in type_indicies.keys():
        data = [values[i] for i in type_indicies[key]]
        di, dv = bin_data(data, std_dev_tol=tol)

        if len(di) == 1:
            key_new = key
            types_final.extend([key_new])
            indicies_final[key_new] = [type_indicies[key][i] for i in di['0']]
            properties_final[key_new] = [np.mean(dv['0']), np.std(dv['0'])]

        else:
            for key2 in di:
                key_new = key+'---'+key2
                types_final.extend([key_new])
                properties_final[key_new] = [np.mean(dv[key2]), np.std(dv[key2])]
                indicies_final[key_new] = [type_indicies[key][i] for i in di[key2]]

    return types_final, properties_final, indicies_final


def bin_data(data, std_dev_tol=1, max_bins=20):
    min_val, max_val = np.min(data), np.max(data)
    for n_bins in range(1,max_bins+1):

        # Create bins
        bin_size = (max_val-min_val)/n_bins
        bins =[[min_val+bin_size*(i), min_val+bin_size*(i+1)] for i in range(n_bins)]
        binned_indicies_dict = {str(val):[] for val in range(n_bins)}
        binned_values_dict = {str(val):[] for val in range(n_bins)}

        # Bin data points
        num_binned_points = 0
        for i in range(len(bins)):
            min, max = bins[i]
            for j in range(len(data)):
                val = data[j]
                if val >= min and val <= max:
                    binned_indicies_dict[str(i)].extend([j])
                    binned_values_dict[str(i)].extend([val])
                    num_binned_points += 1

        # Check that all points were binned, and that no points were binned twice.
        if num_binned_points != len(data):
            print('Warning: Number of points in bin (' + str(num_binned_points) + ') not equal to the number of points in data (' + str(len(data)) + ').' )

        # Filter out empty bins
        final_keys = [key for key in binned_indicies_dict.keys() if binned_values_dict[key] != []]
        binned_indicies_dict = {str(i):binned_indicies_dict[final_keys[i]] for i in range(len(final_keys))}
        binned_values_dict = {str(i):binned_values_dict[final_keys[i]] for i in range(len(final_keys))}
        n_bins_temp = len(final_keys)

        # Calculate the standard deviation of each bin
        std_devs = []
        for i in range(n_bins_temp):
            data_in_bin = binned_values_dict[str(i)]
            if len(data_in_bin) != 0:
                std_dev = np.std(data_in_bin)
            else:
                std_dev = 0
            std_devs.extend([std_dev])

        # Check if converged
        convergence_status = np.all([std_dev <= std_dev_tol for std_dev in std_devs])
        if convergence_status == True:
            return binned_indicies_dict, binned_values_dict
        elif n_bins >= max_bins:
            print('Warning: Did not successfully parition data.')
            return binned_indicies_dict, binned_values_dict


def remove_duplicate_atoms(atoms, tol=0.1):

    atoms_copy = copy.deepcopy(atoms)
    dup_atom_indicies = []
    for i in range(len(atoms_copy)):
        for j in range(i+1,len(atoms_copy)):
            p1 = atoms_copy[i].position
            p2 = atoms_copy[j].position
            d = calculate_distance(p1, p2)
            if d < tol:
                dup_atom_indicies.extend([j])

    del atoms_copy[[i for i in dup_atom_indicies]]

    return atoms_copy


def remove_atoms_outside_cell(atoms, cell_lengths):

    atoms_copy = copy.deepcopy(atoms)
    atom_indicies_to_del = []
    for i in range(len(atoms_copy)):
        p = atoms_copy[i].position
        if np.all( [p[i]>=0 and p[i]<=cell_lengths[i] for i in range(3)] ) == False:
            atom_indicies_to_del.extend([i])

    del atoms_copy[[i for i in atom_indicies_to_del]]

    return atoms_copy
