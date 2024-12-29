import numpy as np
from atomaton.analyze import guess_bonds
from atomaton.visualize import view_structure
from atomaton.models import Atoms

bond_dict = {'default':[0,2.0], 'C-C':[0, 1.8], 'C-Cl':[0, 1.8], 'C-S':[0, 1.9], 'O-S':[0, 1.9], 'H-H':[0, 0]}
ethane = Atoms.bind_from_file("examples/ethane.xyz")
ethane.cell_lengths = np.array([12, 12, 12])
ethane.calculate_bonds(cutoffs={"default": [0,2], "H-H": [1.0]})
ethane.calculate_angles()
ethane.calculate_dihedrals_and_impropers()
ethane.center_atom_in_cell()
ethane.view_structure()
