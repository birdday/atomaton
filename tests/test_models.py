import ase
import numpy as np

from atomaton.analyze import guess_bonds
from atomaton.visualize import view_structure
from atomaton.models import Atoms, Crystal


def test_ethane():
    ethane = Atoms.bind_from_file("examples/ethane.xyz")
    ethane.cell_lengths = np.array([20, 20, 20])
    ethane.calculate_bonds(cutoffs={"default": [0,2], "H-H": [1.0]})
    
    assert len(ethane.symbols) == 8
    assert len(ethane.bonds) == 7


def test_irmof():
    irmof = Crystal.bind_from_file("examples/IRMOF-1.cif")
    irmof.calculate_bonds()
    irmof222 = irmof.build_supercell([2,2,2])
    
    assert len(irmof.symbols) == 424
    assert len(irmof222.symbols) == 424*8
