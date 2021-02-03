import imageio
import mayavi.mlab as mlab
import numpy as np


plotting_parameters = {
    'No' : {'cov_radius':0.0 , 'resolution':20, 'color':(1.00000, 1.00000, 1.00000)},
    'H'  : {'cov_radius':3.1 , 'resolution':20, 'color':(1.00000, 1.00000, 1.00000)},
    'C'  : {'cov_radius':7.3 , 'resolution':20, 'color':(0.56500, 0.56500, 0.56500)},
    'N'  : {'cov_radius':7.1 , 'resolution':20, 'color':(0.18800, 0.31400, 0.97300)},
    'O'  : {'cov_radius':6.6 , 'resolution':20, 'color':(1.00000, 0.05100, 0.05100)},
    'Ar' : {'cov_radius':10.6, 'resolution':20, 'color':(0.50200, 0.82000, 0.89000)},
    'Ni' : {'cov_radius':12.4, 'resolution':20, 'color':(0.31400, 0.81600, 0.31400)},
    'Cu' : {'cov_radius':13.2, 'resolution':20, 'color':(0.78400, 0.50200, 0.20000)},
    'Zn' : {'cov_radius':12.2, 'resolution':20, 'color':(0.49000, 0.50200, 0.69000)},
    'Cs' : {'cov_radius':24.2, 'resolution':20, 'color':(0.78400, 0.50200, 0.20000)}
    }


def draw_atoms(atom_config_new, sf=0.125):

    atom_positions = atom_config_new.get_positions().transpose()

    symbols = atom_config_new.get_chemical_symbols()
    unique_symbols = {symbol:[] for symbol in set(symbols)}
    for atom in atom_config_new:
        unique_symbols[atom.symbol].extend([atom.index])

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


def draw_bonds(atom_config_new, all_bonds, all_bonds_across_boundary, bond_r=0.15, color=(0.4, 0.4, 0.4)):

    atom_positions = atom_config_new.get_positions().transpose()
    x,y,z = atom_positions

    connections = list(tuple(bond) for bond in all_bonds if bond not in all_bonds_across_boundary)
    all_pts = mlab.points3d(x, y, z, resolution=8, scale_factor=0)
    all_pts.mlab_source.dataset.lines = np.array(connections)
    tube = mlab.pipeline.tube(all_pts, tube_radius=bond_r)
    mlab.pipeline.surface(tube, color=color)


def draw_unit_cell(cell_lengths, cell_angles, cell_r=0.15, color=(0.4, 0.4, 0.4)):

    unit_cell_corners = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]

    a, b, c = cell_lengths
    alpha, beta, gamma = np.deg2rad(cell_angles)

    omega = a*b*c*(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))**0.5
    frac_to_cart_matrix = [ [a, b*np.cos(gamma), c*np.cos(beta)],
        [0, b*np.sin(gamma), c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)],
        [0, 0, omega/(a*b*np.sin(gamma))] ]

    x, y, z = np.matmul(frac_to_cart_matrix, np.array(unit_cell_corners).transpose())
    corner_points = mlab.points3d(x, y, z, resolution=8, scale_factor=0)
    corner_points.mlab_source.dataset.lines = np.array(((0,1), (0,2), (0,3), (1,4), (2,4), (1,5), (3,5), (2,6), (3,6), (4,7), (5,7), (6,7)))
    corner_tube = mlab.pipeline.tube(corner_points, tube_radius=cell_r)
    mlab.pipeline.surface(corner_tube, color=color)


def view_structure(atoms, bonds, bonds_across_boundary, show_unit_cell=True, filename=None, interactive=False, figure={}, objects={}, camera={}):

    # Update all default parameter sets
    figure_default = {'bgcolor':(0,0,0), 'size':(1000,1000)}
    figure_default.update(figure)
    figure = figure_default
    bgcolor, size = figure['bgcolor'], figure['size']

    objects_default = {'atom_sf':0.125, 'bond_r':0.15, 'cell_r':0.15, 'bond_color':(0.4, 0.4, 0.4), 'cell_color':(0.4, 0.4, 0.4)}
    objects_default.update(objects)
    objects = objects_default
    atom_sf, bond_r, cell_r, bond_color, cell_color = objects['atom_sf'], objects['bond_r'], objects['cell_r'], objects['bond_color'], objects['cell_color']

    camera_default = {'azimuth': None, 'elevation': None, 'distance': None, 'parallel': False}
    camera_default.update(camera)
    camera = camera_default
    azimuth, elevation, distance, parallel = camera['azimuth'], camera['elevation'], camera['distance'], camera['parallel']

    # Render offscreen for proper sizing
    if interactive == False:
        mlab.options.offscreen = True

    # Create figure
    mlab.figure(1, bgcolor=bgcolor, size=size)
    mlab.clf()
    draw_atoms(atoms, sf=atom_sf)
    draw_bonds(atoms, bonds, bonds_across_boundary, bond_r=bond_r, color=bond_color)
    if show_unit_cell == True:
        a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
        cell_lengths, cell_angles = [a, b, c], [alpha, beta, gamma]
        draw_unit_cell(cell_lengths, cell_angles, cell_r=cell_r, color=cell_color)

    # Set Camera Parameters
    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)
    mlab.gcf().scene.parallel_projection=parallel

    # Save and/or view
    if filename != None:
        mlab.savefig(filename)
        mlab.close()
    if interactive == True:
        mlab.show()


def convert_images_to_gif(filenames, filename_final=None, fps=10):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(filename_final, images, fps=fps)
