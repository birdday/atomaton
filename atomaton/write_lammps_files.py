import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import copy
import numpy as np
import glob
from collections import OrderedDict


from atomaton.helper import get_unique_items


def get_masses():
    from atomaton.forcefields.atoms import MASS

    return MASS


def get_lammps_box_parameters(cell_lengths, cell_angles, degrees=True):
    a, b, c = cell_lengths
    if degrees == True:
        alpha, beta, gamma = np.deg2rad(cell_angles)
    else:
        alpha, beta, gamma = cell_angles

    lx = a
    xy = b * np.cos(gamma)
    xz = c * np.cos(beta)
    ly = np.abs(np.sqrt(b**2 - xy**2))
    yz = (b * c * np.cos(alpha) - xy * xz) / ly
    lz = np.abs(np.sqrt(c**2 - xz**2 - yz**2))

    return lx, ly, lz, xy, xz, yz


def write_lammps_data_file(
    filename,
    atoms,
    ff_atom_types,
    atom_type_params,
    mol_ids,
    cell_lengths,
    cell_angles,
    all_bonds,
    all_bond_types,
    bond_type_params,
    all_angles,
    all_angle_types,
    angle_type_params,
    all_dihedrals,
    all_dihedral_types,
    all_impropers,
    all_improper_types,
    degrees=True,
):
    f = open(filename, "w")

    xhi, yhi, zhi, xy, xz, yz = get_lammps_box_parameters(
        cell_lengths, cell_angles, degrees=degrees
    )
    unique_atom_types, _ = get_unique_items(ff_atom_types)
    unique_bond_types, _ = get_unique_items(all_bond_types)
    unique_angle_types, _ = get_unique_items(all_angle_types)
    # unique_dihedral_types, _ = get_unique_items(all_dihedral_types)
    # unique_improper_types, _ = get_unique_items(all_improper_types)

    # Comment Line
    f.write("#COMMENT LINE 1 \n")
    f.write("#COMMENT LINE 2 \n")

    # Header - Specify num_atoms, atom_types, box_dimensions
    f.write("{} atoms\n".format(len(atoms)))
    f.write("{} bonds\n".format(len(all_bonds)))
    f.write("{} angles\n".format(len(all_angles)))
    # f.write('{} dihedrals\n'.format(len(all_dihedrals)))
    # f.write('{} impropers\n'.format(len(all_impropers)))

    f.write("{} atom types\n".format(len(unique_atom_types)))
    f.write("{} bond types\n".format(len(unique_bond_types)))
    f.write("{} angle types\n".format(len(unique_angle_types)))
    # f.write('{} dihedral types\n'.format(len(unique_dihedral_types)))
    # f.write('{} improper types\n'.format(len(unique_improper_types)))

    f.write("{} {} xlo xhi\n".format(0.0, np.round(xhi, 6)))
    f.write("{} {} ylo yhi\n".format(0.0, np.round(yhi, 6)))
    f.write("{} {} zlo zhi\n".format(0.0, np.round(zhi, 6)))
    f.write("{} {} {} xy xz yz\n".format(0.0, *np.round([xy, xz, yz], 6)))

    # Note the different atom, bond, angle, dihedral, and improper types, along with index
    f.write("\nPair Coeffs\n \n")
    pair_coeff = {}
    counter = 1
    for atom_type in unique_atom_types:
        pair_coeff[atom_type] = counter
        params = " ".join(str(val) for val in atom_type_params[atom_type])
        f.write("{} {} # {} \n".format(counter, params, atom_type))
        counter += 1

    f.write("\nBond Coeffs\n \n")
    bond_coeff = {}
    counter = 1
    for bond_type in bond_type_params:
        bond_coeff[bond_type] = counter
        params = " ".join(str(val) for val in bond_type_params[bond_type])
        f.write("{} {} # {} \n".format(counter, params, bond_type))
        counter += 1

    f.write("\n# Angle Coeffs\n#\n")
    angle_coeff = {}
    counter = 1
    for angle_type in angle_type_params:
        angle_coeff[angle_type] = counter
        params = " ".join(str(val) for val in angle_type_params[angle_type])
        f.write("{} {} # {} \n".format(counter, params, angle_type))
        counter += 1
    #
    # f.write('\n# Dihedral Coeffs\n#\n')
    # dihedral_coeff = {}
    # counter = 1
    # for dihedral_type in unique_dihedral_types:
    #     dihedral_type = ' '.join(str(val) for val in dihedral_type)
    #     dihedral_coeff[dihedral_type] = counter
    #     f.write('# {} {} \n'.format(counter, dihedral_type))
    #     counter += 1
    #
    # f.write('\n# Improper Coeffs\n#\n')
    # improper_coeff = {}
    # counter = 1
    # for improper_type in unique_improper_types:
    #     improper_type = ' '.join(str(val) for val in improper_type)
    #     improper_coeff[improper_type] = counter
    #     f.write('# {} {} \n'.format(counter, improper_type))
    #     counter += 1

    # Masses Section
    f.write("\nMasses\n")
    f.write("# Mass Style: atom_id, mass\n")
    masses = get_masses()
    for i in range(len(unique_atom_types)):
        atom_type = unique_atom_types[i]
        atom_type_for_mass = atom_type[0:2]
        if len(atom_type_for_mass) > 1 and atom_type_for_mass[1] == "_":
            atom_type_for_mass = atom_type_for_mass[0]
        mass = masses[atom_type_for_mass]
        f.write("{} {} #{}\n".format(i + 1, mass, atom_type))

    # Atoms Section
    # N.B. LAMMPS start indexing at 1
    f.write("\nAtoms\n")
    f.write("# Atom Style = Full: atom_id, mol_id, atom_type, charge, x, y, z\n")
    for i in range(len(atoms)):
        atom = atoms[i]
        atom_symbol = ff_atom_types[i]
        f.write(
            "{} {} {} {} {} {} {} #{}\n".format(
                atom.index + 1,
                int(mol_ids[i] + 1),
                pair_coeff[atom_symbol],
                0.0,
                *np.round(atom.position, 6),
                atom_symbol
            )
        )

    # Bonds Section
    f.write("\nBonds\n")
    f.write("# Bond: bond_id, bond_type, atom1_id, atom2_id\n")
    for i in range(len(all_bonds)):
        bond = all_bonds[i]
        bond = [val + 1 for val in bond]
        bond_type = "-".join(str(val) for val in all_bond_types[i])
        f.write(
            "{} {} {} {} #{}\n".format(i + 1, bond_coeff[bond_type], *bond, bond_type)
        )

    # Angles Section
    f.write("\nAngles\n")
    f.write(
        "# Angle: angle_id, atom1_id, atom2_id, atom3_id, angle_type - Order may matter here, check this!!!\n"
    )
    for i in range(len(all_angles)):
        angle = all_angles[i][1]
        angle = [val + 1 for val in angle]
        angle_type = all_angle_types[i]
        f.write(
            "{} {} {} {} {} #{}\n".format(
                i + 1, angle_coeff[angle_type], *angle, angle_type
            )
        )
    #
    # # Dihedrals Section
    # f.write('\nDihedrals\n')
    # f.write('# Dihedrals: dihedral_id, atom1_id, atom2_id, atom3_id, atom4_id, dihedral_type - Order may matter here, check this!!!\n')
    # for i in range(len(all_dihedrals)):
    #     dihedral = all_dihedrals[i]
    #     dihedral = [val+1 for val in dihedral]
    #     dihedral_type = ' '.join(str(val) for val in all_dihedral_types[i])
    #     f.write('{} {} {} {} {} {} #{}\n'.format(i+1, dihedral_coeff[dihedral_type], *dihedral, dihedral_type))
    #
    # # Impropers Section
    # f.write('\nImpropers\n')
    # f.write('# Impropers: improper_id, atom1_id, atom2_id, atom3_id, atom4_id, improper_type  - Order may matter here, check this!!!\n')
    # for i in range(len(all_impropers)):
    #     improper = all_impropers[i][1]
    #     improper = [val+1 for val in improper]
    #     improper_type = ' '.join(str(val) for val in all_improper_types[i])
    #     f.write('{} {} {} {} {} {} #{}\n'.format(i+1, improper_coeff[improper_type], *improper, improper_type))

    f.close()
