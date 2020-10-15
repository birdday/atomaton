import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import numpy as np


def mod(a,b):
    remainder = a%b
    return remainder


def get_unique_items(items):
    unique_items = []
    unique_indicies = []

    index = 0
    for item in items:
        if item not in unique_items:
            unique_items.extend([item])
            unique_indicies.extend([index])
        index += 1

    return unique_items, unique_indicies


def get_center_of_positions(ase_atoms, scaled=False):
    num_atoms = len(ase_atoms)
    all_xyz = ase_atoms.get_positions()
    avg_xyz = np.sum(all_xyz, axis=0)/num_atoms
    return avg_xyz


def get_center_of_cell(cell_lengths, cell_angles):
    cell_vectors = list(ase.geometry.Cell.fromcellpar(cell_lengths+cell_angles))
    cell_cop = 0.5*np.sum(cell_vectors, axis=0)

    return cell_cop


def convert_to_fractional(ase_atoms, cell_lengths, cell_angles, degrees=True):
    a, b, c = cell_lengths
    if degrees == True:
        alpha, beta, gamma = np.deg2rad(cell_angles)
    else:
        alpha, beta, gamma = cell_angles

    omega = a*b*c*(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))**0.5
    cart_to_frac_matrix = [ [1/a, -np.cos(gamma)/(a*np.sin(gamma)), b*c*(np.cos(alpha)*np.cos(gamma)-np.cos(beta))/(omega*np.sin(gamma))],
        [0, 1/(b*np.sin(gamma)), a*c*(np.cos(beta)*np.cos(gamma)-np.cos(alpha))/(omega*np.sin(gamma))],
        [0, 0, (a*b*np.sin(gamma))/omega] ]

    all_xyz = ase_atoms.get_positions()
    all_xyx_frac = np.matmul(cart_to_frac_matrix, all_xyz.transpose())
    ase_atoms.set_positions(all_xyx_frac.transpose())

    return ase_atoms


def convert_to_cartesian(ase_atoms, cell_lengths, cell_angles, degrees=True):
    a, b, c = cell_lengths
    if degrees == True:
        alpha, beta, gamma = np.deg2rad(cell_angles)
    else:
        alpha, beta, gamma = cell_angles

    omega = a*b*c*(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))**0.5
    frac_to_cart_matrix = [ [a, b*np.cos(gamma), c*np.cos(beta)],
        [0, b*np.sin(gamma), c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)],
        [0, 0, omega/(a*b*np.sin(gamma))] ]

    all_xyz = ase_atoms.get_positions()
    all_xyx_cart = np.matmul(frac_to_cart_matrix, all_xyz.transpose())
    ase_atoms.set_positions(all_xyx_cart.transpose())

    return ase_atoms
