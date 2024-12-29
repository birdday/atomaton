import ase
import numpy as np

from atomaton.analyze import guess_bonds
from atomaton.visualize import view_structure
from atomaton.models import Atoms

# ase has no attribute io if imported after ase locally??
from atomaton.models import Atoms

def test_ethane_atoms_properly_parsed():
    ethane = Atoms.bind_from_file("examples/ethane.xyz")
    ethane.cell_lengths = np.array([20, 20, 20])
    ethane.calculate_bonds(cutoffs={"default": [0,2], "H-H": [1.0]})
    
    assert len(ethane.symbols) == 8
    assert len(ethane.bonds) == 7

    