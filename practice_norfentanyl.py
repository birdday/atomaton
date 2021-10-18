import ase as ase
from ase import Atom, Atoms, io, spacegroup, build, visualize
import copy
import csv
import numpy as np
import glob
from collections import OrderedDict
import time
import os

from lammps_tools.helper import (
    mod,
    column,
    get_unique_items,
    get_center_of_positions,
    get_center_of_cell,
    convert_to_fractional,
    convert_to_cartesian,
    )
from lammps_tools.build_structure import (
    build_supercell,
    insert_molecule
    )
from lammps_tools.analyze_structure import (
    calculate_distance,
    create_extended_cell,
    create_extended_cell_minimal,
    guess_bonds,
    guess_angles,
    guess_dihedrals_and_impropers,
    get_bonds_on_atom,
    update_bond_or_dihedral_types,
    update_angle_or_improper_types,
    sort_bond_angle_dihedral_type_list,
    sort_improper_type_list,
    get_bond_properties,
    get_angle_properties,
    split_by_property,
    wrap_atoms_outside_cell
    )
from lammps_tools.forcefields.uff.parameterize import (
    assign_forcefield_atom_types,
    search_for_aromatic_carbons,
    search_for_secondary_aromatics,
    get_pair_potentials,
    get_bond_parameters,
    get_angle_parameters
    )
from lammps_tools.write_lammps_files import (
    write_lammps_data_file
    )
from lammps_tools.visualize import (
    view_structure,
    convert_images_to_gif
    )

# ------------------------------------------------------------

mof_files = glob.glob('/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/*.cif')

mof_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/UiO-66_v2_EQeq.cif'
mof_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/UiO-66-OH.cif'
mof_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/UiO-66-OH-2.cif'
mof_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/UiO-66-NH2.cif'
mof_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/UiO-66-NH2-2.cif'

mof_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/UiO-67.cif'
mof_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/UiO-67-OH.cif'
mof_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/UiO-67-OH-2.cif'
mof_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/UiO-67-NH2.cif'
mof_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/mofs/UiO-67-NH2-2.cif'


for j in range(5):
    atom_set = ase.io.read(mof_file)
    mol_ids = [0 for atom in atom_set]
    mof_name = mof_file.split('/')[-1].split('.')[0]

    molecule_file = '/Users/brian_day/Desktop/Norfentyl-Sensing/molecules/norfentanyl.xyz'
    atoms = ase.io.read(molecule_file)

    # UiO-66, -67 Norfentanyl Locations
    molecule_name = molecule_file.split('/')[-1].split('.')[0]
    cell_lengths_and_angles = list(atom_set.cell.cellpar())

    # Add Argons
    # Center Pore
    t_mat = [0.5, 0.5, 0.5]
    atoms_ar = Atoms( [Atom('Ar', [0,0,0])] )
    atoms_ar.translate([t_mat[i]*cell_lengths_and_angles[i] for i in range(3)])
    atoms_ar.translate([0,0,4])
    atom_set += atoms_ar
    mol_ids += [1 for atom in atoms]

    # Top X
    t_mat = [0.5, 0, 1.0]
    atoms_ar = Atoms( [Atom('Ar', [0,0,0])] )
    atoms_ar.translate([t_mat[i]*cell_lengths_and_angles[i] for i in range(3)])
    atoms_ar.translate([0,0,4])
    atom_set += atoms_ar
    mol_ids += [2 for atom in atoms]

    # Top Y
    t_mat = [0, 0.5, 1.0]
    atoms_ar = Atoms( [Atom('Ar', [0,0,0])] )
    atoms_ar.translate([t_mat[i]*cell_lengths_and_angles[i] for i in range(3)])
    atoms_ar.translate([0,0,4])
    atom_set += atoms_ar
    mol_ids += [3 for atom in atoms]

    # Center Pore, Quartered
    t_mat = [0, 0, 0.5]
    atoms_ar = Atoms( [Atom('Ar', [0,0,0])] )
    atoms_ar.translate([t_mat[i]*cell_lengths_and_angles[i] for i in range(3)])
    atoms_ar.translate([0,0,4])
    atom_set += atoms_ar
    mol_ids += [4 for atom in atoms]

    for i in range(j):

        if i == 0:
            # Molecule 1 - Center Pore
            atoms = ase.io.read(molecule_file)
            atoms.translate(-get_center_of_positions(atoms))
            t_mat = [0.5, 0.5, 0.5]
            atoms.translate([t_mat[i]*cell_lengths_and_angles[i] for i in range(3)])
            atom_set += atoms
            mol_ids += [5 for atom in atoms]

        if i == 1:
            # Molecule 2 - Top X
            atoms = ase.io.read(molecule_file)
            atoms.translate(-get_center_of_positions(atoms))
            t_mat = [0.5, 0, 1.0]
            atoms.translate([t_mat[i]*cell_lengths_and_angles[i] for i in range(3)])
            atom_set += atoms
            mol_ids += [6 for atom in atoms]

        if i == 2:
            # Molecule 3 - Top Y
            atoms = ase.io.read(molecule_file)
            atoms.translate(-get_center_of_positions(atoms))
            t_mat = [0, 0.5, 1.0]
            atoms.translate([t_mat[i]*cell_lengths_and_angles[i] for i in range(3)])
            atom_set += atoms
            mol_ids += [7 for atom in atoms]

        if i == 3:
            # Molecule 4 - Center Pore, Quarted
            atoms = ase.io.read(molecule_file)
            atoms.translate(-get_center_of_positions(atoms))
            t_mat = [0, 0, 0.5]
            atoms.translate([t_mat[i]*cell_lengths_and_angles[i] for i in range(3)])
            atom_set += atoms
            mol_ids += [8 for atom in atoms]


    atom_set = wrap_atoms_outside_cell(atom_set, cell_lengths_and_angles[0:3])
    # atom_set = build_supercell(atom_set, [2,2,2], filename=None)

    # view_structure(atom_set, [], [], show_unit_cell=True, filename=None, interactive=True, opacity=1.0)
    ase.io.write('/Users/brian_day/Desktop/'+mof_name+'_'+str(j)+'_norfentanyl.cif', atom_set)

    # Determine Bonds, Angles, Dihedrals, and Impropers
    bond_dict = {'C-C':[0,1.75], 'O-Zr':[0,2.4], 'default':[0, 1.75]}
    print('Calculating Bonds')
    atoms_alt, all_bonds, all_bonds_alt, all_bond_types, all_bonds_across_boundary, extra_atoms_for_plot, extra_bonds_for_plot = guess_bonds(atom_set, mol_ids, cutoff=bond_dict, periodic='xyz')

    # Visulaization Add-Ons
    # all_atoms_for_plot = copy.deepcopy(atom_set)
    # all_atoms_for_plot += extra_atoms_for_plot
    # all_bonds_for_plot = copy.deepcopy(all_bonds)
    # all_bonds_for_plot.extend(extra_bonds_for_plot)
    # view_structure(all_atoms_for_plot, all_bonds_for_plot, all_bonds_across_boundary, show_unit_cell=True, filename=None, interactive=True)

    print('Calculating Angles')
    all_angles, all_angles_alt, all_angle_types = guess_angles(atom_set, all_bonds, all_bonds_alt)
    all_dihedrals, all_dihedrals_alt, all_dihedral_types, all_impropers, all_impropers_alt, all_improper_types = guess_dihedrals_and_impropers(atom_set, all_bonds, all_bonds_alt, all_angles, all_angles_alt, improper_tol=0.1)

    # Calculate Bond Types
    # N.B. This section has extra steps since bonds are used to assign atom types
    print('Calculating Atom Types - Initial')
    all_bonds, bi = get_unique_items(all_bonds)
    bond_count, bonds_present, bonds_with = get_bonds_on_atom(atom_set, all_bonds)         # Extra Step
    ff_atom_types = assign_forcefield_atom_types(atom_set, bonds_with)                     # Extra Step
    print('Calculating Atom Types - Aromatics')
    ff_atom_types = search_for_aromatic_carbons(atom_set, all_dihedrals, ff_atom_types)    # Extra Step
    print('Calculating Atom Types - Secondary Aromatics')
    ff_atom_types = search_for_secondary_aromatics(atom_set, bonds_present, ff_atom_types)
    unique_atom_types, _ = get_unique_items(ff_atom_types)                              # Extra Step
    all_bond_types = update_bond_or_dihedral_types(all_bonds, ff_atom_types)
    all_bond_types = sort_bond_angle_dihedral_type_list(all_bond_types)

    # Calculate Angle Types
    all_angles, ai = get_unique_items(all_angles)
    all_angle_types = update_angle_or_improper_types(all_angles, ff_atom_types)
    all_angle_types = sort_bond_angle_dihedral_type_list(all_angle_types)

    # # Calculate Dihedral Types
    # all_dihedrals, di = get_unique_items(all_dihedrals)
    # all_dihedral_types = update_bond_or_dihedral_types(all_dihedrals, ff_atom_types)
    # all_dihedral_types = sort_bond_angle_dihedral_type_list(all_dihedral_types)
    #
    # # Calculate Improper Types
    # # Update later, currently initial atom types or ordered properly, but ones using uff atom types are not.
    # all_impropers, ii = get_unique_items(all_impropers)
    # all_improper_types = update_angle_or_improper_types(all_impropers, ff_atom_types)
    # all_improper_types = sort_improper_type_list(all_improper_types)

    # Get properties of atoms/bonds/angles/dihedrals/impropers and subdivide if necessarys
    print('Calculating Parameters')
    atom_type_params = get_pair_potentials(ff_atom_types)

    bond_type_indicies, bond_values = get_bond_properties(atoms_alt, all_bonds_alt, all_bond_types)
    bt_final, bp_final, bi_final = split_by_property(bond_type_indicies, bond_values, tol=0.05) # Separated by bond length
    bond_type_params = get_bond_parameters(bt_final, form='harmonic')
    print('Bond Type Params')
    for pset in bond_type_params:
        print('\t', pset+':', bond_type_params[pset])

    angle_type_indicies, angle_values, mag_ij, mag_jk, mag_ik = get_angle_properties(atoms_alt, all_angles_alt, all_angle_types)
    at_final, ap_final, ai_final = split_by_property(angle_type_indicies, angle_values, tol=2.5) # Separated by angle, theta
    angle_type_params = get_angle_parameters(at_final, ap_final, angle_tol=3, degrees=True)

    all_angle_types_final = ['' for angle in all_angles]
    for key in ai_final:
        for val in ai_final[key]:
            all_angle_types_final[val] = key

    print('Angle Type Params')
    for pset in angle_type_params:
        print('\t', pset+':', angle_type_params[pset])

    # Check Bonds
    # print(len(all_bonds), len(all_bond_types), len(unique_bond_types))
    # print(len(all_angles), len(all_angle_types), len(unique_angle_types))
    # print(len(all_dihedrals), len(all_dihedral_types), len(unique_dihedral_types))
    # print(len(all_impropers), len(all_improper_types), len(unique_impropers_types))

    # 3. Write LAMMPS files
    cell_lengths = atom_set.cell.cellpar()[0:3]
    cell_angles =  atom_set.cell.cellpar()[3::]

    print('Writings Lammps File')

    filename = '/Users/brian_day/Desktop/in.'+mof_name+'_'+str(j)+'_norfentanyl'
    write_lammps_data_file(filename, atom_set, ff_atom_types, atom_type_params, np.zeros(len(atom_set)), cell_lengths, cell_angles, all_bonds, all_bond_types, bond_type_params, all_angles, all_angle_types_final, angle_type_params, all_dihedrals, all_dihedral_types, all_impropers, all_improper_types)

    print('DONE!!!')










# ------------------------------------------------------------


atom_types = {'1': 'O', '2': 'N', '3': 'N', '4': 'C', '5': 'C', '6': 'H'}
pos_files = glob.glob('/Users/brian_day/Desktop/dump.lammpstrj*')
image_dir = '/Users/brian_day/Desktop/norf-vis-test/'

i=0
for i in range(0,1):

    file = pos_files[i]
    file_as_text = []

    with open(file, newline='') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
         for row in spamreader:
            file_as_text.extend([row])
    num_atoms = int(file_as_text[3][0])
    num_cycles = int(len(file_as_text)/(num_atoms+9))

    for j in range(num_cycles):

        # Determine the Unit Cell
        bounding_box = file_as_text[j*(num_atoms+9)+4:j*(num_atoms+9)+8]
        xlo, xhi, yz = [float(val) for val in bounding_box[1]]
        ylo, yhi, xz = [float(val) for val in bounding_box[2]]
        zlo, zhi, xy = [float(val) for val in bounding_box[3]]
        origin = [xlo, ylo, zlo]
        a = (xhi-xlo,0,0)
        b = (xy,yhi-ylo,0)
        c = (xz,yz,zhi-zlo)

        # Convert Atoms to ase.Atoms Object
        positions = file_as_text[j*(num_atoms+9)+8:(j+1)*(num_atoms+9)]
        positions_atoms_obj = Atoms([ atom_types[row[2]] for row in positions[1::]], [row[3:6] for row in positions[1::]])
        positions_atoms_obj.translate([-xlo, -ylo, -zlo])
        positions_atoms_obj.set_cell([xhi-xlo, yhi-ylo, zhi-zlo, 90, 90, 90])

        # Set Camera
        # elevation = 90
        # if elevation == 45:
        #     azimuth = 45
        # else:
        #     azimuth = 0
        # camera = {'azimuth': azimuth, 'elevation': elevation, 'distance': 50, 'parallel': True, 'focalpoint': (0.5*(xhi-xlo), 0.5*(yhi-ylo), 0.5*(zhi-zlo))}

        # Create Image
        # filename = '/Users/brian_day/Desktop/norf-min-test/'+str(j)+'.png'
        view_structure(positions_atoms_obj, [], [], show_unit_cell=True,  interactive=True)
