import numpy as np
import itertools
from atomaton.models import Atoms, SimulationBox, UnitCell

cell_lengths = np.array([30, 30, 30])
cell_angles = np.array([90, 90, 90])
ppx = 11

xyz = np.array(list(itertools.product(np.linspace(0,1,ppx)[:-1], np.linspace(0,1,ppx)[:-1], np.linspace(0,1,ppx)[:-1])))
xyz = xyz*np.array(cell_lengths)
syms = np.array(['H' for _ in range(len(xyz))])

lattice = Atoms(symbols=syms, positions=xyz)
lattice.cell_lengths = cell_lengths
lattice.cell_angles = cell_angles
lattice.center_atom_in_cell()
lattice_xyz = lattice.positions

simbox = SimulationBox()
simbox.cell_lengths = cell_lengths
simbox.cell_angles = cell_angles
simbox.unit_cell = UnitCell(simbox.cell_lengths, simbox.cell_angles)

h2o = Atoms.bind_from_file("examples/water.xyz")
h2o.cell_lengths = np.array([5,5,5])
h2o.cell_angles = np.array([90,90,90])
h2o.calculate_bonds()

for xyz in lattice_xyz:
    simbox.insert_atoms(h2o, position=xyz)

simbox.view()
