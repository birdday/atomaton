import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import copy
import numpy as np
import glob
from collections import OrderedDict


from lammps.helper import (get_unique_items)


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
    yz = (b*c*np.cos(alpha) - xy*xz)/ly
    lz = np.abs(np.sqrt(c**2-xz**2-yz**2))

    return lx, ly, lz, xy, xz, yz


def write_lammps_input_file(filename):
    with open (filename, 'w') as fdata:

        # Comment Line
        f.write('COMMENT LINE \n')

        for atom in atoms:
            if 'position_fixed':
                fdata.write('fix {} {} setforce 0.0 0.0 0.0\n'.format(id, group_id))


def write_lammps_data_file(filename, atoms, mol_ids, cell_lengths, cell_angles, all_bonds, all_bond_types, all_angles, all_angle_types, all_dihedrals, all_dihedral_types, all_impropers, all_improper_types, degrees=True):
    f = open(filename, 'w')

    xhi, yhi, zhi, xy, xz, yz = get_lammps_box_parameters(cell_lengths, cell_angles, degrees=degrees)

    unique_atom_types, _ = get_unique_items(atoms.get_chemical_symbols())

    unique_bonds, bi = get_unique_items(all_bonds)
    filtered_bond_types = [all_bond_types[i] for i in bi]
    unique_bond_types, _ = get_unique_items(all_bond_types)

    unique_angles, ai = get_unique_items(all_angles)
    filtered_angle_types = [all_angle_types[i] for i in ai]
    unique_angle_types, _ = get_unique_items(all_angle_types)

    unique_dihedrals, di = get_unique_items(all_dihedrals)
    filtered_dihedral_types = [all_dihedral_types[i] for i in di]
    unique_dihedral_types, _ = get_unique_items(all_dihedral_types)

    unique_impropers, ii = get_unique_items(all_impropers)
    filtered_improper_types = [all_improper_types[i] for i in ii]
    unique_improper_types, _ = get_unique_items(all_improper_types)

    # Comment Line
    f.write('#COMMENT LINE \n')

    # Header - Specify num_atoms, atom_types, box_dimensions
    f.write('{} atoms\n'.format(len(atoms)))
    f.write('{} bonds\n'.format(len(all_bonds)))
    f.write('{} angles\n'.format(len(all_angles)))
    f.write('{} dihedrals\n'.format(len(all_dihedrals)))
    f.write('{} impropers\n'.format(len(all_impropers)))

    f.write('{} atom types\n'.format(len(unique_atom_types)))
    f.write('{} bond types\n'.format(len(unique_bond_types)))
    f.write('{} angle types\n'.format(len(unique_angle_types)))
    f.write('{} dihedral types\n'.format(len(unique_dihedral_types)))
    f.write('{} improper types\n'.format(len(unique_improper_types)))

    f.write('{} {} xlo xhi\n'.format(0.0, np.round(xhi,6)))
    f.write('{} {} ylo yhi\n'.format(0.0, np.round(yhi,6)))
    f.write('{} {} zlo zhi\n'.format(0.0, np.round(zhi,6)))
    f.write('{} {} {} xy xz yz\n'.format(0.0, *np.round([xy, xz, yz], 6)))

    # Note the different atom, bond, angle, dihedral, and improper types, along with index
    f.write('\n# Pair Coeffs\n#\n')
    pair_coeff = {}
    counter = 1
    for atom_type in unique_atom_types:
        pair_coeff[atom_type] = counter
        f.write('# {} {} \n'.format(counter, atom_type))
        counter += 1

    f.write('\n# Bond Coeffs\n#\n')
    bond_coeff = {}
    counter = 1
    for bond_type in unique_bond_types:
        bond_type = '-'.join(str(val) for val in bond_type)
        bond_coeff[bond_type] = counter
        f.write('# {} {} \n'.format(counter, bond_type))
        counter += 1

    f.write('\n# Angle Coeffs\n#\n')
    angle_coeff = {}
    counter = 1
    for angle_type in unique_angle_types:
        angle_type = '-'.join(str(val) for val in angle_type)
        angle_coeff[angle_type] = counter
        f.write('# {} {} \n'.format(counter, angle_type))
        counter += 1

    f.write('\n# Dihedral Coeffs\n#\n')
    dihedral_coeff = {}
    counter = 1
    for dihedral_type in unique_dihedral_types:
        dihedral_type = '-'.join(str(val) for val in dihedral_type)
        dihedral_coeff[dihedral_type] = counter
        f.write('# {} {} \n'.format(counter, dihedral_type))
        counter += 1

    f.write('\n# Improper Coeffs\n#\n')
    improper_coeff = {}
    counter = 1
    for improper_type in unique_improper_types:
        improper_type = '-'.join(str(val) for val in improper_type)
        improper_coeff[improper_type] = counter
        f.write('# {} {} \n'.format(counter, improper_type))
        counter += 1

    # Atoms Section
    # N.B. LAMMPS start indexing at 1
    f.write('\nAtoms\n')
    f.write('# Atom Style = Full: atom_id, mol_id, atom_type, charge, x, y, z\n')
    for i in range(len(atoms)):
        atom = atoms[i]
        f.write('{} {} {} {} {} {} {} #{}\n'.format(atom.index+1, int(mol_ids[i]+1), pair_coeff[atom.symbol], 0.0, *np.round(atom.position,6), atom.symbol))

    # Bonds Section
    f.write('\nBonds\n')
    f.write('# Bond: bond_id, atom1_id, atom2_id, bond_type\n')
    for i in range(len(unique_bonds)):
        bond = unique_bonds[i]
        bond_type = '-'.join(str(val) for val in filtered_bond_types[i])
        f.write('{} {} {} {} #{}\n'.format(i+1, *bond, bond_coeff[bond_type], bond_type))

    # Angles Section
    f.write('\nAngles\n')
    f.write('# Angle: angle_id, atom1_id, atom2_id, atom3_id, angle_type - Order may matter here, check this!!!\n')
    for i in range(len(unique_angles)):
        angle = unique_angles[i][1]
        angle_type = '-'.join(str(val) for val in filtered_angle_types[i])
        f.write('{} {} {} {} {} #{}\n'.format(i+1, *angle, angle_coeff[angle_type], angle_type))

    # Dihedrals Section
    f.write('\nDihedrals\n')
    f.write('# Dihedrals: dihedral_id, atom1_id, atom2_id, atom3_id, atom4_id, dihedral_type - Order may matter here, check this!!!\n')
    for i in range(len(unique_dihedrals)):
        dihedral = unique_dihedrals[i]
        dihedral_type = '-'.join(str(val) for val in filtered_dihedral_types[i])
        f.write('{} {} {} {} {} {} #{}\n'.format(i+1, *dihedral, dihedral_coeff[dihedral_type], dihedral_type))

    # Impropers Section
    f.write('\nImpropers\n')
    f.write('# Impropers: improper_id, atom1_id, atom2_id, atom3_id, atom4_id, improper_type  - Order may matter here, check this!!!\n')
    for i in range(len(unique_impropers)):
        improper = unique_impropers[i][1]
        improper_type = '-'.join(str(val) for val in filtered_improper_types[i])
        f.write('{} {} {} {} {} {} #{}\n'.format(i+1, *improper, improper_coeff[improper_type], improper_type))

    f.close()
