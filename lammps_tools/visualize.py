import imageio
import mayavi.mlab as mlab
import numpy as np
import matplotlib.cm


plotting_parameters = {
    'No' : {'cov_radius':0.0 , 'resolution':20, 'color':(1.00000, 1.00000, 1.00000)},
    'H'  : {'cov_radius':3.1 , 'resolution':20, 'color':(1.00000, 1.00000, 1.00000)},
    'Li' : {'cov_radius':12.8, 'resolution':20, 'color':(0.80000, 0.50200, 1.00000)},
    'B'  : {'cov_radius':8.4 , 'resolution':20, 'color':(1.00000, 0.71000, 0.71000)},
    'C'  : {'cov_radius':7.3 , 'resolution':20, 'color':(0.56500, 0.56500, 0.56500)},
    'N'  : {'cov_radius':7.1 , 'resolution':20, 'color':(0.18800, 0.31400, 0.97300)},
    'O'  : {'cov_radius':6.6 , 'resolution':20, 'color':(1.00000, 0.05100, 0.05100)},
    'F'  : {'cov_radius':5.7 , 'resolution':20, 'color':(0.56500, 0.87800, 0.31400)},
    'Na' : {'cov_radius':16.6 , 'resolution':20, 'color':(0.67100, 0.36100, 0.94900)},
    'Mg' : {'cov_radius':14.1, 'resolution':20, 'color':(0.54100, 1.00000, 0.00000)},
    'Al' : {'cov_radius':12.1, 'resolution':20, 'color':(0.74900, 0.65100, 0.65100)},
    'Si' : {'cov_radius':11.1, 'resolution':20, 'color':(0.94100, 0.78400, 0.62700)},
    'P'  : {'cov_radius':10.7, 'resolution':20, 'color':(1.00000, 0.50200, 0.00000)},
    'S'  : {'cov_radius':10.5, 'resolution':20, 'color':(1.00000, 1.00000, 0.18800)},
    'Cl' : {'cov_radius':10.2, 'resolution':20, 'color':(0.12200, 0.94100, 0.12200)},
    'Ar' : {'cov_radius':10.6, 'resolution':20, 'color':(0.50200, 0.82000, 0.89000)},
    'K'  : {'cov_radius':20.3, 'resolution':20, 'color':(0.56100, 0.25100, 0.83100)},
    'Ca' : {'cov_radius':17.6, 'resolution':20, 'color':(0.23900, 1.00000, 0.00000)},
    'V'  : {'cov_radius':15.3, 'resolution':20, 'color':(0.65100, 0.65100, 0.67100)},
    'Fe' : {'cov_radius':14.2, 'resolution':20, 'color':(0.87800, 0.40000, 0.20000)},
    'Co' : {'cov_radius':14.0, 'resolution':20, 'color':(0.94100, 0.56500, 0.62700)},
    'Ni' : {'cov_radius':12.4, 'resolution':20, 'color':(0.31400, 0.81600, 0.31400)},
    'Cu' : {'cov_radius':13.2, 'resolution':20, 'color':(0.78400, 0.50200, 0.20000)},
    'Zn' : {'cov_radius':12.2, 'resolution':20, 'color':(0.49000, 0.50200, 0.69000)},
    'As' : {'cov_radius':11.9, 'resolution':20, 'color':(0.74100, 0.50200, 0.89000)},
    'Br' : {'cov_radius':12.0, 'resolution':20, 'color':(0.65100, 0.16100, 0.16100)},
    'Y'  : {'cov_radius':19.0, 'resolution':20, 'color':(0.58000, 1.00000, 1.00000)},
    'Zr' : {'cov_radius':17.5, 'resolution':20, 'color':(0.58000, 0.87800, 0.87800)},
    'Ag' : {'cov_radius':14.5, 'resolution':20, 'color':(0.75300, 0.75300, 0.75300)},
    'Cd' : {'cov_radius':14.4, 'resolution':20, 'color':(1.00000, 0.85100, 0.56100)},
    'In' : {'cov_radius':14.2, 'resolution':20, 'color':(0.65100, 0.45900, 0.45100)},
    'I'  : {'cov_radius':13.9, 'resolution':20, 'color':(0.58000, 0.00000, 0.58000)},
    'Cs' : {'cov_radius':24.2, 'resolution':20, 'color':(0.78400, 0.50200, 0.20000)},
    'Ba' : {'cov_radius':21.5, 'resolution':20, 'color':(0.00000, 0.78800, 0.00000)},
    'La' : {'cov_radius':20.7, 'resolution':20, 'color':(0.43900, 0.83100, 1.00000)},
    'Ce' : {'cov_radius':20.4, 'resolution':20, 'color':(1.00000, 1.00000, 0.78000)},
    'Nd' : {'cov_radius':20.1, 'resolution':20, 'color':(0.78000, 1.00000, 0.78000)},
    'Eu' : {'cov_radius':19.8, 'resolution':20, 'color':(0.38000, 1.00000, 0.78000)},
    'Tb' : {'cov_radius':19.4, 'resolution':20, 'color':(0.18800, 1.00000, 0.78000)},
    'Dy' : {'cov_radius':19.2, 'resolution':20, 'color':(0.12200, 1.00000, 0.78000)},
    'W'  : {'cov_radius':16.2, 'resolution':20, 'color':(0.12900, 0.58000, 0.83900)}
    }


def draw_atoms(atom_config_new, sf=0.125, opacity=1.0):

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
            color=plotting_parameters[key]['color'],
            opacity=opacity
            )


def draw_bonds(atom_config_new, all_bonds, all_bonds_across_boundary, bond_r=0.15, color=(0.4, 0.4, 0.4), opacity=1.0):

    atom_positions = atom_config_new.get_positions().transpose()
    x,y,z = atom_positions

    connections = list(tuple(bond) for bond in all_bonds if bond not in all_bonds_across_boundary)
    all_pts = mlab.points3d(x, y, z, resolution=8, scale_factor=0)
    all_pts.mlab_source.dataset.lines = np.array(connections)
    tube = mlab.pipeline.tube(all_pts, tube_radius=bond_r)
    mlab.pipeline.surface(tube, color=color, opacity=opacity)


def draw_unit_cell(cell_lengths, cell_angles, cell_r=0.15, color=(0.4, 0.4, 0.4), opacity=1.0):

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
    mlab.pipeline.surface(corner_tube, color=color, opacity=opacity)


def bin_data(x_data, y_data, num_bins=10, val_range=None):
    if val_range == None:
        min_x, max_x = min(x_data), max(x_data)
    else:
        min_x, max_x = val_range
    x_bins = [[] for i in range(num_bins)]
    y_bins = [[] for i in range(num_bins)]
    bin_points = np.linspace(min_x, max_x, num_bins+1)

    for i in range(len(x_data)):
        x, y = x_data[i], y_data[i]
        for j in range(len(bin_points)-1):
            status = 'unbinned'
            if x >= bin_points[j] and x <= bin_points[j+1]:
                x_bins[j].extend([x])
                y_bins[j].extend([y])
                status = 'binned'
                break
        if status == 'unbinned':
            print(x, 'Point not binned!')

    return x_bins, y_bins


def view_structure(atoms, bonds, bonds_across_boundary, show_unit_cell=True, filename=None, interactive=False, figure={}, objects={}, camera={}, bond_energies=None, bond_cmap='Reds', bond_bins=5, e_range=None, opacity=1.0):

    # Update all default parameter sets
    figure_default = {'bgcolor':(0,0,0), 'size':(1000,1000)}
    figure_default.update(figure)
    figure = figure_default
    bgcolor, size = figure['bgcolor'], figure['size']

    objects_default = {'atom_sf':0.125, 'bond_r':0.15, 'cell_r':0.15, 'bond_color':(0.4, 0.4, 0.4), 'cell_color':(0.4, 0.4, 0.4)}
    objects_default.update(objects)
    objects = objects_default
    atom_sf, bond_r, cell_r, bond_color, cell_color = objects['atom_sf'], objects['bond_r'], objects['cell_r'], objects['bond_color'], objects['cell_color']

    camera_default = {'azimuth': None, 'elevation': None, 'distance': None, 'focalpoint': None, 'parallel': False, }
    camera_default.update(camera)
    camera = camera_default
    azimuth, elevation, distance, focalpoint = camera['azimuth'], camera['elevation'], camera['distance'], camera['focalpoint']
    parallel = camera['parallel']

    # Render offscreen for proper sizing
    if interactive == False:
        mlab.options.offscreen = True

    # Create figure
    mlab.figure(1, bgcolor=bgcolor, size=size)
    mlab.clf()
    draw_atoms(atoms, sf=atom_sf, opacity=opacity)
    if bond_energies != None:
        cmap = matplotlib.cm.get_cmap(bond_cmap)
        _, binned_bonds = bin_data(bond_energies, bonds, num_bins=bond_bins, val_range=e_range)
        for i in range(bond_bins):
            bond_color = cmap((i*bond_bins)/bond_bins)[0:3]
            draw_bonds(atoms, binned_bonds[i], bonds_across_boundary, bond_r=bond_r, color=bond_color, opacity=opacity)
    else:
        draw_bonds(atoms, bonds, bonds_across_boundary, bond_r=bond_r, color=bond_color, opacity=opacity)
    if show_unit_cell == True:
        a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
        cell_lengths, cell_angles = [a, b, c], [alpha, beta, gamma]
        draw_unit_cell(cell_lengths, cell_angles, cell_r=cell_r, color=cell_color, opacity=opacity)


    # Set Camera Parameters
    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance, focalpoint=focalpoint)
    mlab.gcf().scene.parallel_projection=parallel

    # Add invisible box to keep field of view a constant size
    if parallel == True and distance != None:
        azm, elev, dist, fp = mlab.view()
        cam, fp = mlab.move()
        xvals = [fp[0]-0.5*distance, fp[0], fp[0]+0.5*distance]
        yvals = [fp[1]-0.5*distance, fp[1], fp[1]+0.5*distance]
        zvals = [fp[2]-0.5*distance, fp[2], fp[2]+0.5*distance]
        xx, yy, zz = np.array([[x, y, z] for x in xvals for y in yvals for z in zvals]).transpose()
        pts = mlab.points3d(xx, yy, zz, resolution=6, scale_factor=0, color=(1.0, 1.0, 1.0))

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
