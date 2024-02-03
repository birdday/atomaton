import ase as ase
from ase import Atom, Atoms
import copy
import numpy as np

from atomaton.helper import (
    column,
    get_unique_items,
    convert_to_cartesian,
)


def calculate_distance(p1, p2):
    """Function which calculates the distance between two positions in N-dimensional space. 

    Args:
        p1 (list[float...]): position in N-dimensional space
        p2 (list[float...]): position in N-dimensional space

    Returns:
        float: distance between two positions
    """

    dist = np.sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))]) ** 0.5

    return dist


def _translate_and_extend(
        fractional_shifts,
        atomtpyes,
        atomtypes_extended,
        fractional_positions,
        fractional_positions_extended,
        indicies,
        pseudo_indicies,
    ):
    """Function which shifts the positions of atoms in a unit cell by some given amount and extended those positions to the list of all atoms.

    Args:
        fractional_shifts (list[float, float, float]): Amount by which to shift the fractional coordinates of the atoms
        atomtpyes (list[str]): list of original atom types.
        atomtypes_extended (list[str]): list of extended atom types
        fractional_positions (list[list[float, float, float]]): list of original fractional positions
        fractional_positions_extended (list[list[float, float, float]]): list of all fractional positions
        indicies (list[int]): list of original atom indicies
        pseudo_indicies (list[int]): list of all atom indicies, corresponding to atom in original cell

    Returns:
        list[str]: extended list of all atom types
        list[list[float, float, float]]: extended list of all fractional coordinates
        list[int]: extended list of all indicies, corresponding to index of original atom
    """

    xt, yt, zt = fractional_shifts
    x_trans = fractional_positions[0] + xt
    y_trans = fractional_positions[1] + yt
    z_trans = fractional_positions[2] + zt
    fractional_positions_translated = np.array(
        [x_trans, y_trans, z_trans]
    ).transpose()

    atomtypes_extended.extend(atomtpyes)
    fractional_positions_extended.extend(fractional_positions_translated)
    pseudo_indicies.extend(indicies)

    return atomtypes_extended, fractional_positions_extended, pseudo_indicies


def _extended_params_to_cell(atoms, atomtypes_extended, fractional_positions_extended):
    """Takes parameters for extended cell and concerts it to an Atoms object with appropriate cell parameters and positions.

    Args:
        atoms (Atoms): Atoms object with original cell parameters
        atomtypes_extended (list[str]): array of all the new atom types
        fractional_positions_extended (list[list[float, float, float]]): fractional coordinates of all atoms in the extended cell

    Returns:
        Atoms: Atoms object containing the original cell parameters, but including the extended cell atoms
    """

    cell_params = atoms.cell.cellpar()
    cell_lengths, cell_angles = cell_params[0:3], cell_params[3::]
    cartesian_positions_extended = convert_to_cartesian(
        fractional_positions_extended, cell_lengths, cell_angles, degrees=True
    )
    extended_cell = Atoms(atomtypes_extended, cartesian_positions_extended)
    extended_cell.set_cell([*cell_lengths, *cell_angles])

    return extended_cell


def create_extended_cell(atoms):
    """Creates a 3x3x3 supercell of atoms, along with pseudo indicies of the new atoms which matches the index of the original atom.

    N.B. this can be done natively by ase using ase.build.cut or ase.build.make_supercell, and I have a wrapper for the first function
    in build_structure.py. Would just need to make the pseudo_indicies list, which is trivial. Still, this was fun to write and is fast enough, so I'm leaving it,
    but could get a perfromance boost out of the (complied in C?) ase functions.
    Also, the built in functions modify the cell parameters, so these would need to be reset.

    Args:
        atoms (Atoms): ase Atoms object with cell parameters.

    Returns:
        Atoms: ase Atoms object with the atoms of the 3x3x3 cell, but the original cell parameters
        list[int]: index of the atom corresponding to the original unit cell
    """

    # Get atoms information
    atomtpyes = atoms.get_chemical_symbols()
    fractional_positions = atoms.get_scaled_positions().transpose()
    indicies = [atom.index for atom in atoms]

    # Initialize extended cell arrays
    atomtypes_extended = []
    fractional_positions_extended = []
    pseudo_indicies = []

    # Get extended cell parameters
    trans = [0, -1, 1]
    for xt in trans:
        for yt in trans:
            for zt in trans:
                _translate_and_extend(
                    [xt, yt, zt],
                    atomtpyes,
                    atomtypes_extended,
                    fractional_positions,
                    fractional_positions_extended,
                    indicies,
                    pseudo_indicies,
                    )

    # Convert params into Atoms object
    extended_cell = _extended_params_to_cell(atoms, atomtypes_extended, fractional_positions_extended)

    return extended_cell, pseudo_indicies


def create_extended_cell_minimal(atoms, max_bond_length=5.0):
    """Creates a minimally extended cell to speed up O(N^2) bond check. This functions is O(N).

    Args:
        atoms (Atoms): Atoms object with cell parameters
        max_bond_length (float, optional): Maximum possible bond length, used to determine degree to which cell is extended. Defaults to 5.0.

    Raises:
        TypeError: max_bond_length must be a single value, or dictionary of bond cutoffs.
        ValueError: max_bond_length must be less than half the length of the shortes unit cell dimension.

    Returns:
        Atoms: minimally extended cell
    """

    cell_x, cell_y, cell_z = atoms.cell.cellpar()[0:3]
    atoms_extended = Atoms()
    pseudo_indicies = [i for i, _ in enumerate(atoms)]

    if type(max_bond_length) == dict:
        max_bond_length = max([max(max_bond_length[key]) for key in max_bond_length])
    elif type(max_bond_length) != int and type(max_bond_length) != float:
        raise TypeError("Invalid max_bond_length type.")

    if (
        max_bond_length >= 0.5 * cell_x
        or max_bond_length >= 0.5 * cell_y
        or max_bond_length >= 0.5 * cell_z
    ):
        raise ValueError("max_bond_length greater than half the cell length.")

    for i, atom in enumerate(atoms):
        px, py, pz = atom.position
        perx, pery, perz = None, None, None
        pseudo_indicies.extend([i])

        # Check X
        if px >= 0 and px <= max_bond_length:
            perx = "+x"
            ext_x = cell_x
            atoms_extended += Atom(atom.symbol, [px + ext_x, py, pz])
            pseudo_indicies.extend([i])
        if px >= cell_x - max_bond_length and px <= cell_x:
            perx = "-x"
            ext_x = -cell_x
            atoms_extended += Atom(atom.symbol, [px + ext_x, py, pz])
            pseudo_indicies.extend([i])

        # Check Y
        if py >= 0 and py <= max_bond_length:
            pery = "+y"
            ext_y = cell_y
            atoms_extended += Atom(atom.symbol, [px, py + ext_y, pz])
            pseudo_indicies.extend([i])
        if py >= cell_y - max_bond_length and py <= cell_y:
            pery = "-y"
            ext_y = -cell_y
            atoms_extended += Atom(atom.symbol, [px, py + ext_y, pz])
            pseudo_indicies.extend([i])

        # Check Z
        if pz >= 0 and pz <= max_bond_length:
            perz = "+z"
            ext_z = cell_z
            atoms_extended += Atom(atom.symbol, [px, py, pz + ext_z])
            pseudo_indicies.extend([i])
        if pz >= cell_z - max_bond_length and pz <= cell_z:
            perz = "-z"
            ext_z = -cell_z
            atoms_extended += Atom(atom.symbol, [px, py, pz + ext_z])
            pseudo_indicies.extend([i])

        # Check XY
        if perx != None and pery != None:
            atoms_extended += Atom(atom.symbol, [px + ext_x, py + ext_y, pz])
            pseudo_indicies.extend([i])

        # Check XZ
        if perx != None and perz != None:
            atoms_extended += Atom(atom.symbol, [px + ext_x, py, pz + ext_z])
            pseudo_indicies.extend([i])

        # Check YZ
        if pery != None and perz != None:
            atoms_extended += Atom(atom.symbol, [px, py + ext_y, pz + ext_z])
            pseudo_indicies.extend([i])

        # Check XYZ
        if perx != None and pery != None and perz != None:
            atoms_extended += Atom(atom.symbol, [px + ext_x, py + ext_y, pz + ext_z])
            pseudo_indicies.extend([i])

    print(pseudo_indicies)

    return atoms + atoms_extended, pseudo_indicies


def _resolve_bond_cutoffs_dict(cutoffs):
    """Function which validates the dictionary of bond types. If "default" not included, will be added with value of [0, 1.5].

    TODO: Add check that key is valid.

    Args:
        cutoffs (dict): Dictionary of atomType-atomType: [min, max] bond distances in angstrom.

    Raises:
        ValueError: Cutoff must be a single value or an array of two numbers.

    Returns:
        dict: dictionary of bond cutoffs, with single values updated to [0, value] and default added.
    """

    for key, value in cutoffs.items():
        if len(value) == 1:
            cutoffs[key] = [0, value]
        if len(value) > 2:
            raise ValueError("Invalid cutoff!")

    if "default" not in cutoffs.keys():
        cutoffs["default"] = [0, 1.5]

    return cutoffs


def guess_bonds(atoms, cutoffs={"default": [0, 1.5]}):
    """Finds the bonds in a cell of atoms, as defined by the cutoffs dict.

    Args:
        atoms (Atoms): Atoms object with cell parameters.
        cutoffs (dict, optional): Dictionary of bond cutoffs. Keys must match atom types, separated by a -, in alphabetically order. Defaults to {"default": [0, 1.5]}.

    Returns:
        list[list[int, int]]: list of bonds, by sorted atom index.
        list[list[str, str]]: list of bond types, sorted alphabetically by atom type.
        list[list[int, int]]: list of bonds which cross the cell boundary, by sorted orignal atom index.
        Atoms: list of atoms outside of the cell which are part of bonds crossing the cell.
        list[list[int, int]]: list of bonds which cross the cell boundary, using the extended atoms index.
    """

    # Prepare Atoms Object
    num_atoms = len(atoms)
    atoms_ext, pseudo_indicies = create_extended_cell_minimal(atoms)
    cutoff = _resolve_bond_cutoffs_dict(cutoffs)

    bonds = []
    bond_types = []

    bonds_across_boundary = []
    extra_atoms_for_plot = Atoms()
    extra_bonds_for_plot = []

    for i in range(len(atoms)):
        p1 = atoms[i].position
        type1 = atoms[i].symbol

        for j in range(i + 1, len(atoms_ext)):
            p2 = atoms_ext[j].position
            type2 = atoms_ext[j].symbol

            r = calculate_distance(p1, p2)
            bondtype = "-".join(sorted([type1, type2]))

            if (
                (
                    bondtype in cutoff.keys()
                    and r >= cutoff[bondtype][0]
                    and r <= cutoff[bondtype][1]
                )
                or (
                    bondtype not in cutoff.keys()
                    and r >= cutoff["default"][0]
                    and r <= cutoff["default"][1]
                )
            ):
                bond = sorted(set((i, pseudo_indicies[j])))

                # Check if bond already found (i.e., the second time we encounter a bond crossing the cell boundary)
                if bond not in bonds:
                    bonds.extend([bond])
                    bond_types.extend([sorted([type1, type2])])

                    # Check if bond crosses boundary and if so, record bond to special category.
                    if j > num_atoms:
                        bonds_across_boundary.extend([bond])

                # If bond crosses boundary, record extra atom and extra bond.
                # Note that we do not care if the bond has already been found, since this will be a different instance of that bond with respect to the boundary it crosses.
                # Also, we must index the extra bond slightly differently since we only record atoms as needed.
                if j > num_atoms:
                    extra_atoms_for_plot += atoms_ext[j]
                    # We must also subtract 1 since we extend atoms first and python uses 0 indexing. 
                    extra_bonds_for_plot.extend([[i, num_atoms+len(extra_atoms_for_plot)-1]])

    return bonds, bond_types, bonds_across_boundary, extra_atoms_for_plot, extra_bonds_for_plot


def get_bonds_on_atom(atoms, bonds):
    """Gets a list of number of bonds, bond indicies, and bond types for all atoms.

    Args:
        atoms (Atoms): Collection of atoms
        bonds (list[list[int, int]]): List of bonds for that atom set

    Returns:
        Dict[int: int]: _description_
        Dict[int: List[int, int]]: _description_
        Dict[int: List[str, str]]: _description_
    """

    bond_count = {i: 0 for i in range(len(atoms))}
    bonds_present = {i: [] for i in range(len(atoms))}
    bonds_with = {i: [] for i in range(len(atoms))}
    for bond in bonds:
        for index in bond:
            bond_count[str(index)] += 1
            bonds_present[str(index)].extend([bond])
            bonds_with[str(index)].extend(
                [atoms[ai].symbol for ai in bond if ai != index]
            )

    return bond_count, bonds_present, bonds_with

# TODO: Continue updating this code... left off here.
def guess_angles(atoms, bonds, bonds_alt):
    # TODO: Need to check how angle terms which cross unit cells are defined in lammps and adjust code accordingly. 
    all_angles = []
    all_angles_alt = []
    all_angle_types = []

    for i in range(len(bonds)):
        bond1 = bonds[i]
        bond1_alt = bonds_alt[i]

        for j in range(i + 1, len(bonds)):
            bond2 = bonds[j]
            bond2_alt = bonds_alt[j]

            atoms_in_angle = sorted(set(bond1 + bond2))
            if len(atoms_in_angle) == 3:
                # Angle defined in by core atom numbers, used for calcuating dihedrals and impropers and writing lammps files
                center_atom = sorted(set(bond1).intersection(bond2))
                end_atoms = sorted(set(atoms_in_angle).difference(center_atom))
                ordered_atoms_in_angle = copy.deepcopy(end_atoms)
                ordered_atoms_in_angle.insert(1, *center_atom)
                ordered_atom_types_in_angle = [
                    atoms[index].symbol for index in end_atoms
                ]
                ordered_atom_types_in_angle.insert(
                    1, *[atoms[index].symbol for index in center_atom]
                )

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
                    center_atoms = [i for i in bond1_alt if i >= len(atoms)] + [
                        i for i in bond2_alt if i >= len(atoms)
                    ]
                all_angles_alt.extend([[center_atoms, [bond1_alt, bond2_alt]]])

    sorted_indicies = np.argsort(column(all_angles, 0))
    all_angles_sorted = [all_angles[index] for index in sorted_indicies]
    all_angles_alt_sorted = [all_angles_alt[index] for index in sorted_indicies]
    all_angle_types_sorted = [all_angle_types[index] for index in sorted_indicies]

    return all_angles_sorted, all_angles_alt_sorted, all_angle_types_sorted


def guess_dihedrals_and_impropers(
    atoms_in, bonds, bonds_alt, angles, angles_alt, improper_tol=0.1
):
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
            atoms_in_group = sorted(list(set(angle + bond)))
            shared_atom = sorted(set(angle).intersection(bond))

            if len(atoms_in_group) == 4 and shared_atom != [center_atom]:
                # all_dihedrals.extend([atoms_in_group])

                ordered_atoms = list(
                    set(angle).difference([center_atom]).difference(shared_atom)
                )
                ordered_atoms.insert(1, center_atom)
                ordered_atoms.insert(2, *shared_atom)
                ordered_atoms.insert(3, *list(set(bond).difference(shared_atom)))

                if ordered_atoms[0] > ordered_atoms[-1]:
                    ordered_atoms.reverse()
                all_dihedrals.extend([ordered_atoms])
                all_dihedrals_alt.extend([angle_alt + [bond_alt]])

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
                p0 = atoms[ordered_atoms[0]].position
                v01 = (
                    atoms[ordered_atoms[1]].position - atoms[ordered_atoms[0]].position
                )
                v02 = atoms[ordered_atoms[2]].position = atoms[
                    ordered_atoms[0]
                ].position
                a, b, c = np.cross(v01, v02)
                d = -1 * (a * p0[0] + b * p0[1] + c * p0[2])
                num, den = (
                    a * pc[0] + b * pc[1] + c * pc[2] + d,
                    (a**2 + b**2 + c**2) ** 0.5,
                )
                if den != 0:
                    dmin = abs(num / den)
                else:
                    dmin = 0

                if dmin <= improper_tol:
                    all_impropers.extend(
                        [[center_atom, [center_atom, *sorted(ordered_atoms)]]]
                    )
                    ordered_atom_types = sorted(
                        atoms[index].symbol for index in ordered_atoms
                    )
                    ordered_atom_types.insert(0, atoms[center_atom].symbol)
                    all_improper_types.extend([ordered_atom_types])

                    if center_atom in angle_alt:
                        bond_center_atom = [i for i in bond_alt if i >= len(atoms)]
                        all_impropers_alt.extend(
                            [[[center_atom] + bond_center_atom, angle_alt + [bond_alt]]]
                        )
                    elif center_atom in bond_alt:
                        angle_center_atom = [i for i in angle_alt[0] if i >= len(atoms)]
                        all_impropers_alt.extend(
                            [
                                [
                                    [center_atom] + angle_center_atom,
                                    angle_alt + [bond_alt],
                                ]
                            ]
                        )

    return (
        all_dihedrals,
        all_dihedrals_alt,
        all_dihedral_types,
        all_impropers,
        all_impropers_alt,
        all_improper_types,
    )


def get_bond_properties(atoms, bonds, bond_types):
    unique_bond_types, _ = get_unique_items(bond_types)
    keys = ["-".join(bond_type) for bond_type in unique_bond_types]
    bond_type_indicies = {key: [] for key in keys}

    bond_lengths = []
    for i in range(len(bonds)):
        bond_type = bond_types[i]
        key = "-".join(bond_type)
        bond_type_indicies[key].extend([i])

        bond = bonds[i]
        p1 = atoms[bond[0]].position
        p2 = atoms[bond[1]].position
        d = calculate_distance(p1, p2)
        bond_lengths.extend([d])

    return bond_type_indicies, bond_lengths


def get_angle_properties(atoms, all_angles, all_angle_types):
    unique_angle_types, _ = get_unique_items(all_angle_types)
    keys = ["-".join(angle_type) for angle_type in unique_angle_types]

    angle_type_indicies = {key: [] for key in keys}
    angle_type_angles = []
    angle_type_mag_ij = []
    angle_type_mag_jk = []
    angle_type_mag_ik = []

    for i in range(len(all_angles)):
        angle_type = all_angle_types[i]
        key = "-".join(angle_type)

        angle = all_angles[i]
        if len(angle[0]) == 1:
            id_i = [index for index in angle[1][0] if index not in angle[0]]
            id_k = [index for index in angle[1][1] if index not in angle[0]]
            if len(id_i) != 1 or len(id_k) != 1:
                print("index: ", i)
                print("angle: ", angle)
                print("ids: ", id_i, id_k)
                raise NameError("Invalid ID length")
            atom_i = atoms[id_i[0]]
            atom_j = atoms[angle[0][0]]
            atom_k = atoms[id_k[0]]

            # v = vector, u = unit vector, mag = magnitude
            v_ji = atom_j.position - atom_i.position
            mag_ji = np.sqrt(v_ji.dot(v_ji))
            u_ji = v_ji / mag_ji

            v_jk = atom_j.position - atom_k.position
            mag_jk = np.sqrt(v_jk.dot(v_jk))
            u_jk = v_jk / mag_jk

            v_ik = atom_i.position - atom_k.position
            mag_ik = np.sqrt(v_ik.dot(v_ik))
            u_ik = v_ik / mag_ik

        elif len(angle[0]) == 2:
            id_i = [index for index in angle[1][0] if index not in angle[0]]
            id_j1 = [index for index in angle[0] if index in angle[1][0]]
            id_j2 = [index for index in angle[0] if index in angle[1][1]]
            id_k = [index for index in angle[1][1] if index not in angle[0]]
            if len(id_i) != 1 or len(id_j1) != 1 or len(id_j2) != 1 or len(id_k) != 1:
                print(id_i, id_j1, id_j2, id_k)
                raise NameError("Invalid ID length")
            atom_i = atoms[id_i[0]]
            atom_j1 = atoms[id_j1[0]]
            atom_j2 = atoms[id_j2[0]]
            atom_k = atoms[id_k[0]]

            # v = vector, u = unit vector, mag = magnitude
            v_ji = atom_j1.position - atom_i.position
            mag_ji = np.sqrt(v_ji.dot(v_ji))
            u_ji = v_ji / mag_ji

            v_jk = atom_j2.position - atom_k.position
            mag_jk = np.sqrt(v_jk.dot(v_jk))
            u_jk = v_jk / mag_jk

            v_ik = atom_i.position - atom_k.position
            mag_ik = np.sqrt(v_ik.dot(v_ik))
            u_ik = v_ik / mag_ik

        theta_ijk = np.rad2deg(np.arccos(np.clip(np.dot(u_ji, u_jk), -1.0, 1.0)))

        angle_type_indicies[key].extend([i])
        angle_type_angles.extend([theta_ijk])
        angle_type_mag_ij.extend([mag_ji])
        angle_type_mag_jk.extend([mag_jk])
        angle_type_mag_ik.extend([mag_ik])

    return (
        angle_type_indicies,
        angle_type_angles,
        angle_type_mag_ij,
        angle_type_mag_jk,
        angle_type_mag_ik,
    )


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
            indicies_final[key_new] = [type_indicies[key][i] for i in di["0"]]
            properties_final[key_new] = [np.mean(dv["0"]), np.std(dv["0"])]

        else:
            for key2 in di:
                key_new = key + "---" + key2
                types_final.extend([key_new])
                properties_final[key_new] = [np.mean(dv[key2]), np.std(dv[key2])]
                indicies_final[key_new] = [type_indicies[key][i] for i in di[key2]]

    return types_final, properties_final, indicies_final


def bin_data(data, std_dev_tol=1, max_bins=20):
    min_val, max_val = np.min(data), np.max(data)
    for n_bins in range(1, max_bins + 1):
        # Create bins
        bin_size = (max_val - min_val) / n_bins
        bins = [
            [min_val + bin_size * i, min_val + bin_size * (i + 1)]
            for i in range(n_bins)
        ]
        binned_indicies_dict = {str(val): [] for val in range(n_bins)}
        binned_values_dict = {str(val): [] for val in range(n_bins)}

        # Bin data points
        num_binned_points = 0
        for i in range(len(bins)):
            min_bin, max_bin = bins[i]
            for j in range(len(data)):
                val = data[j]
                if min_bin <= val <= max_bin:
                    binned_indicies_dict[str(i)].extend([j])
                    binned_values_dict[str(i)].extend([val])
                    num_binned_points += 1

        # Check that all points were binned, and that no points were binned twice.
        if num_binned_points != len(data):
            print(
                "Warning: Number of points in bin ("
                + str(num_binned_points)
                + ") not equal to the number of points in data ("
                + str(len(data))
                + ")."
            )

        # Filter out empty bins
        final_keys = [
            key for key in binned_indicies_dict.keys() if binned_values_dict[key] != []
        ]
        binned_indicies_dict = {
            str(i): binned_indicies_dict[final_keys[i]] for i in range(len(final_keys))
        }
        binned_values_dict = {
            str(i): binned_values_dict[final_keys[i]] for i in range(len(final_keys))
        }
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
        if convergence_status:
            return binned_indicies_dict, binned_values_dict
        elif n_bins >= max_bins:
            print("Warning: Did not successfully partition data.")
            return binned_indicies_dict, binned_values_dict


def remove_duplicate_atoms(atoms, tol=0.1):
    atoms_copy = copy.deepcopy(atoms)
    dup_atom_indicies = []
    for i in range(len(atoms_copy)):
        for j in range(i + 1, len(atoms_copy)):
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
        if not np.all([0 <= p[i] <= cell_lengths[i] for i in range(3)]):
            atom_indicies_to_del.extend([i])

    del atoms_copy[[i for i in atom_indicies_to_del]]

    return atoms_copy


def wrap_atoms_outside_cell(atoms, cell_lengths):
    atoms_copy = copy.deepcopy(atoms)
    cell_x, cell_y, cell_z = cell_lengths

    for i in range(len(atoms_copy)):
        px, py, pz = atoms_copy[i].position

        if px < 0:
            atoms_copy[i].position[0] += cell_x
        if px >= cell_x:
            atoms_copy[i].position[0] += -cell_x

        if py < 0:
            atoms_copy[i].position[1] += cell_y
        if py >= cell_y:
            atoms_copy[i].position[1] += -cell_y

        if pz < 0:
            atoms_copy[i].position[2] += cell_z
        if pz >= cell_z:
            atoms_copy[i].position[2] += -cell_z

    return atoms_copy
