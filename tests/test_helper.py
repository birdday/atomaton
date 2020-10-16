import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import numpy as np
import pytest
from pytest import approx
from collections import OrderedDict

from lammps_tools.helper import (
    mod,
    get_unique_items,
    get_center_of_positions,
    get_center_of_cell,
    convert_to_fractional,
    convert_to_cartesian,
    )

def test_center_of_cell_calculated_correctly():
    coc = get_center_of_cell([10,10,10], [90,90,90])
    
    assert np.testing.assert_array_equal(coc, [5,5,5]) == None
