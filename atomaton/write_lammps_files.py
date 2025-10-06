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

#  micelle example good.
# --- Header Info
# The private functions simply write long strings that will be written with write lines.
def write_header_comments(header):
    header.append("# COMMENT LINE 1 \n")
    header.append("# COMMENT LINE 2 \n")

    return header


def write_multibody_terms_header(header, 
                                  atoms, atom_types,
                                  bonds, bond_types,
                                  angles, angle_types, 
                                  dihedrals, dihedral_types,
                                  impropers, improper_types):
    header.append('\n')
    header.append("{} atoms\n".format(len(atoms)))
    header.append("{} bonds\n".format(len(bonds)))
    header.append("{} angles\n".format(len(angles)))
    header.append("{} dihedrals\n".format(len(dihedrals)))
    header.append("{} impropers\n".format(len(impropers)))

    unique_atom_types, _ = get_unique_items(atom_types)
    unique_bond_types, _ = get_unique_items(bond_types)
    unique_angle_types, _ = get_unique_items(angle_types)
    unique_dihedral_types, _ = get_unique_items(dihedral_types)
    unique_improper_types, _ = get_unique_items(improper_types)

    header.append('\n')
    header.append("{} atom types\n".format(len(unique_atom_types)))
    header.append("{} bond types\n".format(len(unique_bond_types)))
    header.append("{} angle types\n".format(len(unique_angle_types)))
    header.append("{} dihedral types\n".format(len(unique_dihedral_types)))
    header.append("{} improper types\n".format(len(unique_improper_types)))

    return header


def write_box_info(header, cell_lengths, cell_angles):
    xhi, yhi, zhi, xy, xz, yz = get_lammps_box_parameters(
        cell_lengths, cell_angles, degrees=True
    )
    header.append('\n')
    header.append("{} {} xlo xhi\n".format(0.0, np.round(xhi, 6)))
    header.append("{} {} ylo yhi\n".format(0.0, np.round(yhi, 6)))
    header.append("{} {} zlo zhi\n".format(0.0, np.round(zhi, 6)))
    header.append("{} {} {} xy xz yz\n".format(0.0, *np.round([xy, xz, yz], 6)))

    return header


def write_header(cell_lengths, cell_angles, 
                 atoms, atom_types,
                 bonds, bond_types,
                 angles, angle_types, 
                 dihedrals, dihedral_types,
                 impropers, improper_types):
    header = []
    header = write_header_comments(header)
    header = write_multibody_terms_header(header, 
                                          atoms, atom_types,
                                          bonds, bond_types,
                                          angles, angle_types, 
                                          dihedrals, dihedral_types,
                                          impropers, improper_types)
    header = write_box_info(header, cell_lengths, cell_angles)

    return header


# --- Forcefield Info (Coeffs)
# TODO: Consider adding labelmap support (maps int id to alphanumerics ids)
def write_mass_section(atom_types):
    unique_atom_types, _ = get_unique_items(atom_types)
    mass_section = []
    mass_section.append("\nMasses\n")
    mass_section.append("# Mass Style: atom_id, mass\n")
    masses = get_masses()
    for i, atom_type in enumerate(unique_atom_types):
        # Converts C_sp2 to C for assigning masses
        atom_type_for_mass = atom_type[0:2]
        if len(atom_type_for_mass) > 1 and atom_type_for_mass[1] == "_":
            atom_type_for_mass = atom_type_for_mass[0]
        mass = masses[atom_type_for_mass]
        mass_section.append("{} {} # {}\n".format(i + 1, mass, atom_type))
    
    return mass_section


def write_pair_coeff_section(atom_type_params):
    """Writes lammps data file section for the individual pair parameters for lennard-jones bonding.

    Args:
        atom_type_params (dict[str, list[float...]]): lennard-jones sigma and epsilon for the individual atom type

    Returns:
        list[str]: pair coeff lines
    """
    pair_coeff_section = []
    pair_coeffs = {}

    pair_coeff_section.append("\nPair Coeffs\n \n")
    counter = 1
    for atom_type, params in atom_type_params.items:
        pair_coeffs[atom_type] = counter
        params_str = " ".join(str(val) for val in params)
        pair_coeff_section.append(f"{counter} {params_str} # {atom_type} \n")
        counter += 1
    
    return pair_coeff_section, pair_coeffs


def write_bond_coeff_section(bond_type_params):
    bond_coeff_section = []
    bond_coeffs = {}

    bond_coeff_section.append("\nBond Coeffs\n \n")
    counter = 1
    for bond_type, params in bond_type_params.items:
        bond_coeffs[bond_type] = counter
        params_str = " ".join(str(val) for val in params)
        bond_coeff_section.append(f"{counter} {params_str} # {bond_type} \n")
        counter += 1
    
    return bond_coeff_section, bond_coeffs


def write_angle_coeff_section(angle_type_params):
    angle_coeff_section = []
    angle_coeffs = {}

    angle_coeff_section.append("\nAngle Coeffs\n#\n")
    counter = 1
    for angle_type, params in angle_type_params:
        angle_coeffs[angle_type] = counter
        params_str = " ".join(str(val) for val in params)
        angle_coeff_section.append(f"{counter} {params_str} # {angle_type} \n")
        counter += 1
    
    return angle_coeff_section, angle_coeffs


# def write_dihedral_coeff_section():
#     f.write('\n# Dihedral Coeffs\n#\n')
#     dihedral_coeff = {}
#     counter = 1
#     for dihedral_type in unique_dihedral_types:
#         dihedral_type = ' '.join(str(val) for val in dihedral_type)
#         dihedral_coeff[dihedral_type] = counter
#         f.write('# {} {} \n'.format(counter, dihedral_type))
#         counter += 1


# def write_improper_coeff_section():
#     f.write('\n# Improper Coeffs\n#\n')
#     improper_coeff = {}
#     counter = 1
#     for improper_type in unique_improper_types:
#         improper_type = ' '.join(str(val) for val in improper_type)
#         improper_coeff[improper_type] = counter
#         f.write('# {} {} \n'.format(counter, improper_type))
#         counter += 1



# --- Structure Info
def write_atom_section(atom_indicies, molecule_indicies, symbols, positions, charges, pair_coeffs):
    """Currently only supports atom type, Full.
    """
    atom_section = []
    # N.B. LAMMPS start indexing at 1
    atom_section.append("\nAtoms\n")
    atom_section.append("# Atom Style = Full: atom_id, mol_id, atom_type, charge, x, y, z\n")
    for (atom_id, mol_id, sym, pos, q) in zip(atom_indicies, molecule_indicies, symbols, positions, charges):
        atom_section.append(
            "{} {} {} {} {} {} {} #{}\n".format(
                atom_id + 1,
                mol_id,
                pair_coeffs[sym],
                0.0,
                *np.round(pos, 6),
                sym
            )
        )
    
    return atom_section


def write_bond_section(bond_ids, bond_types, bonds, bond_coeffs):
    bond_section = []
    bond_section.append("\nBonds\n")
    bond_section.append("# Bond: bond_id, bond_type, atom1_id, atom2_id\n")
    for i in range(len(all_bonds)):
        bond = all_bonds[i]
        bond = [val + 1 for val in bond]
        bond_type = "-".join(str(val) for val in all_bond_types[i])
        bond_section.append(
            "{} {} {} {} #{}\n".format(i + 1, bond_coeffs[bond_type], *bond, bond_type)
        )
    
    return bond_section


def write_angle_section(angle_ids, angle_types, angles, angle_coeffs):
    angle_section = []
    angle_section.append("\nAngles\n")
    # atom ids are in connection order; atom2 is center atom.
    angle_section.append(
        "# Angle: angle_id, angle_type, atom1_id, atom2_id, atom3_id\n"
    )
    for i in range(len(all_angles)):
        angle = all_angles[i][1]
        angle = [val + 1 for val in angle]
        angle_type = all_angle_types[i]
        angle_section.append(
            "{} {} {} {} {} #{}\n".format(
                i + 1, angle_coeffs[angle_type], *angle, angle_type
            )
        )
    
    return angle_section


# def write_dihedral_section():
#     # Dihedrals Section
#     f.write('\nDihedrals\n')
#     f.write('# Dihedrals: dihedral_id, atom1_id, atom2_id, atom3_id, atom4_id, dihedral_type - Order may matter here, check this!!!\n')
#     for i in range(len(all_dihedrals)):
#         dihedral = all_dihedrals[i]
#         dihedral = [val+1 for val in dihedral]
#         dihedral_type = ' '.join(str(val) for val in all_dihedral_types[i])
#         f.write('{} {} {} {} {} {} #{}\n'.format(i+1, dihedral_coeff[dihedral_type], *dihedral, dihedral_type))
    

# def write_improper_section():
#     # Impropers Section
#     f.write('\nImpropers\n')
#     f.write('# Impropers: improper_id, atom1_id, atom2_id, atom3_id, atom4_id, improper_type  - Order may matter here, check this!!!\n')
#     for i in range(len(all_impropers)):
#         improper = all_impropers[i][1]
#         improper = [val+1 for val in improper]
#         improper_type = ' '.join(str(val) for val in all_improper_types[i])
#         f.write('{} {} {} {} {} {} #{}\n'.format(i+1, improper_coeff[improper_type], *improper, improper_type))


# --- Complete File
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
    """Generate a LAMMPS data file from various structure information.

    Args:
        filename (str): path, including file name, of the file to generate
        atoms (Atoms): 
        ff_atom_types (list[str, str]): _description_
        atom_type_params (_type_): _description_
        mol_ids (_type_): _description_
        cell_lengths (_type_): _description_
        cell_angles (_type_): _description_
        all_bonds (_type_): _description_
        all_bond_types (_type_): _description_
        bond_type_params (_type_): _description_
        all_angles (_type_): _description_
        all_angle_types (_type_): _description_
        angle_type_params (_type_): _description_
        all_dihedrals (_type_): _description_
        all_dihedral_types (_type_): _description_
        all_impropers (_type_): _description_
        all_improper_types (_type_): _description_
        degrees (bool, optional): _description_. Defaults to True.
    """
    f = open(filename, "w")
    f.close()


# For testing / development. Delete later.
def write_lammps_data_file_alt(filename, cell_lengths, cell_angles):
    f = open(filename, "w")
    header = write_header(cell_lengths, cell_angles)
    f.writelines(header)
    f.close()

