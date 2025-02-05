import ase
import copy
import itertools
import random
import numpy as np

from atomaton.models import Atoms, Crystal, SimulationBox
from atomaton.write_lammps_files import write_lammps_data_file


# ---- Ethanol in Box
ethanol = Atoms.bind_from_file("ethanol.xyz")
ethanol.cell_lengths = np.array([30, 30, 30])
ethanol.cell_angles = np.array([90, 90, 90])
ethanol.center_in_cell()
ethanol.calculate_bonds(cutoffs={"C-C": [0, 2.0], "O-H": [0, 1.0]})
objects = {
        "atom_sf": 0.10,
        "bond_r": 0.05,
        "cell_r": 0.5,
        "bond_color": (0.4, 0.4, 0.4),
        "cell_color": (0.4, 0.4, 0.4),
    }
ethanol.view(show_unit_cell=True, objects=objects)


# --- Generate LAMMPS Data Files
# Note, all lammps input files are simple text files and may have arbitrary extensions
out_file = "ethanol.lammps"

ff_atom_types = ethanol.symbols
write_lammps_data_file(
    out_file, # filename,
    ethanol.symbols, # atoms,
    ethanol.bond_types, # ff_atom_types,
    # atom_type_params,
    # mol_ids,
    # cell_lengths,
    # cell_angles,
    # all_bonds,
    # all_bond_types,
    # bond_type_params,
    # all_angles,
    # all_angle_types,
    # angle_type_params,
    # all_dihedrals,
    # all_dihedral_types,
    # all_impropers,
    # all_improper_types,
    # degrees=True,
):



