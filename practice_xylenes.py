import ase as ase
from ase import Atom, Atoms, io, spacegroup, build, visualize
import copy
import csv
import numpy as np
import glob
from collections import OrderedDict
import time
import os
import re

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

def file_to_array(file):
    file_as_array = []
    with open (file) as f:
        reader = csv.reader(f)
        for row in reader:
            file_as_array.extend(row)

    return file_as_array


def process_movie(file):
    file_as_array = file_to_array(file)
    box_by_cycle = []
    pos_by_cycle = []
    cycle = []
    for i in range(len(file_as_array)):
        row = file_as_array[i]
        row_split = re.split(' +', row)
        if row_split[0] == 'MODEL':
            if i != 0:
                pos_by_cycle.extend([cycle])
            cycle = []
        elif row_split[0] == 'CRYST1':
            box_by_cycle.extend([[float(val) for val in row_split[1::]]])
        elif row_split[0] == 'ATOM':
            cycle.extend([[int(row_split[1]), row_split[2], *[float(val) for val in row_split[4:7]] ]])
        elif row_split[0] == 'ENDMDL':
            continue
        else:
            print('INVALID ROW!')

    return box_by_cycle, pos_by_cycle

# ------------------------------------------------------------


# mof_files = glob.glob('/Users/brian_day/Desktop/Xylene_Sensing/Input_Files/structures/*.cif')
mof_file = '/Users/brian_day/Desktop/Xylene_Sensing/Input_Files/structures/Cu-BDC.cif'
# mof_file = '/Users/brian_day/Desktop/Xylene_Sensing/Input_Files/structures/UIO-66_v2_EQeq.cif'

movie_files = glob.glob('/Users/brian_day/Desktop/Xylene_Sensing/MOF_Results/Xylene_Movies/Cu-BDC_?-xylene/*/Movies/*/*xylene*.pdb')
# movie_files = glob.glob('/Users/brian_day/Desktop/Xylene_Sensing/MOF_Results/Xylene_Movies/UIO-66_?-xylene/*/Movies/*/*xylene*.pdb')
# movie_file = glob.glob('/Users/brian_day/Desktop/Xylene_Sensing/MOF_Results/Xylene_Movies/UIO-66_o-xylene/1e-1/Movies/*/*xylene*.pdb')[0]


for movie_file in movie_files:
    box_by_cycle, pos_by_cycle = process_movie(movie_file)

    # mof_atoms = ase.io.read(mof_file)
    # mof_atoms = build_supercell(mof_atoms, num_cells=[2,2,3])
    # slice_size = mof_atoms.cell.cellpar()[0:3]
    mof_atoms = ase.io.read(mof_file)
    mof_atoms = build_supercell(mof_atoms, num_cells=[3,3,5])
    # for atom in mof_atoms:
    #     atom.symbol = 'N'

    box = box_by_cycle[-0]
    atoms = pos_by_cycle[-0]
    positions_atoms_obj = Atoms()
    for atom in atoms:
        symbol = atom[1]
        if symbol == 'CH3':
            symbol = 'C'
        pos = atom[2::]
        positions_atoms_obj.extend(Atom(symbol, pos))

    # positions_atoms_obj = Atoms([row[1] for row in atoms], [row[2::] for row in atoms])
    positions_atoms_obj.set_cell([box[0], box[1], box[2], 90, 90, 90])
    positions_atoms_obj += mof_atoms

    # atoms_slice = Atoms()
    # for atom in positions_atoms_obj:
    #     if np.any(atom.position >= slice_size) != True:
    #         atoms_slice.extend(atom)
    # atoms_slice.set_cell([*slice_size, 90, 90, 90])

    # view_structure(positions_atoms_obj, [], [], show_unit_cell=True,  interactive=True, opacity=0.5)
    filename = '_'.join(movie_file.split('/')[-5:-3])
    ase.io.write(f'/Users/brian_day/Desktop/Snapshots/{filename}.pdb', positions_atoms_obj)
