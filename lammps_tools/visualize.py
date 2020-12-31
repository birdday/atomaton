import mayavi.mlab as mlab
import numpy as np

from lammps_tools.helper import (
    mod,
    get_unique_items,
    get_center_of_positions,
    get_center_of_cell,
    convert_to_fractional,
    convert_to_cartesian,
    )

plotting_parameters = {
    'H'  : {'cov_radius':3.1 , 'resolution':20, 'color':(1.00000, 1.00000, 1.00000), 'scale_mode':'none'},
    'C'  : {'cov_radius':7.3 , 'resolution':20, 'color':(0.56500, 0.56500, 0.56500), 'scale_mode':'none'},
    'N'  : {'cov_radius':7.1 , 'resolution':20, 'color':(0.18800, 0.31400, 0.97300), 'scale_mode':'none'},
    'O'  : {'cov_radius':6.6 , 'resolution':20, 'color':(1.00000, 0.05100, 0.05100), 'scale_mode':'none'},
    'Ar' : {'cov_radius':10.6, 'resolution':20, 'color':(0.50200, 0.82000, 0.89000), 'scale_mode':'none'},
    'Ni' : {'cov_radius':14.4, 'resolution':20, 'color':(0.31400, 0.81600, 0.31400), 'scale_mode':'none'},
    'Cu' : {'cov_radius':13.2, 'resolution':20, 'color':(0.78400, 0.50200, 0.20000), 'scale_mode':'none'}
    }

def draw_atoms(atom_config_new, cell_lengths, cell_angles, fractional_in=True, sf=0.125):

    symbols = atom_config_new.get_chemical_symbols()
    unique_symbols = {symbol:[] for symbol in set(symbols)}
    for atom in atom_config_new:
        unique_symbols[atom.symbol].extend([atom.index])

    if fractional_in == True:
        atom_positions = convert_to_cartesian(atom_config_new, cell_lengths, cell_angles, degrees=True).get_positions().transpose()
    else:
        atom_positions = atom_config_new.get_positions.transpose()

    for key in unique_symbols:
        values = unique_symbols[key]
        x = atom_positions[0][values]
        y = atom_positions[1][values]
        z = atom_positions[2][values]
        pts = mlab.points3d(x, y, z,
            resolution=plotting_parameters[key]['resolution'],
            scale_factor=plotting_parameters[key]['cov_radius']*sf,
            color=plotting_parameters[key]['color']
            )

def draw_bonds(atom_config_new, cell_lengths, cell_angles, all_bonds, all_bonds_across_boundary, fractional_in=True, degrees=True):
    if fractional_in == True:
        atom_positions = convert_to_cartesian(atom_config_new, cell_lengths, cell_angles, degrees=True).get_positions().transpose()
    else:
        atom_positions = atom_config_new.get_positions().transpose()
    x,y,z = atom_positions
    connections = list(tuple(bond) for bond in all_bonds if bond not in all_bonds_across_boundary)
    all_pts = mlab.points3d(x, y, z, resolution=8, scale_factor=0)
    all_pts.mlab_source.dataset.lines = np.array(connections)
    tube = mlab.pipeline.tube(all_pts, tube_radius=0.15)
    mlab.pipeline.surface(tube, color=(0.4, 0.4, 0.4))

def draw_unit_cell(cell_lengths, cell_angles, degrees=True):

    unit_cell_corners = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]
    a, b, c = cell_lengths
    if degrees == True:
        alpha, beta, gamma = np.deg2rad(cell_angles)
    else:
        alpha, beta, gamma = cell_angles

    omega = a*b*c*(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))**0.5
    frac_to_cart_matrix = [ [a, b*np.cos(gamma), c*np.cos(beta)],
        [0, b*np.sin(gamma), c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)],
        [0, 0, omega/(a*b*np.sin(gamma))] ]

    x, y, z = np.matmul(frac_to_cart_matrix, np.array(unit_cell_corners).transpose())
    corner_points = mlab.points3d(x, y, z, resolution=8, scale_factor=0)
    corner_points.mlab_source.dataset.lines = np.array(((0,1), (0,2), (0,3), (1,4), (2,4), (1,5), (3,5), (2,6), (3,6), (4,7), (5,7), (6,7)))
    corner_tube = mlab.pipeline.tube(corner_points, tube_radius=0.15)
    mlab.pipeline.surface(corner_tube, color=(0.4, 0.4, 0.4))

def view_structure(atoms, bonds, bonds_across_boundary, cell_lengths, cell_angles, fractional_in=True, degrees=True, show_unit_cell=True):

    mlab.figure(1, bgcolor=(1,1,1), size=(350,350))
    mlab.clf()

    draw_atoms(atoms, cell_lengths, cell_angles, fractional_in=True, sf=0.125)
    draw_bonds(atoms, cell_lengths, cell_angles, bonds, bonds_across_boundary, fractional_in=False, degrees=True)

    if show_unit_cell == True:
        draw_unit_cell(cell_lengths, cell_angles, degrees=True)

    mlab.show()
