import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import numpy as np


def mod(a, b):
    remainder = a % b
    return remainder


def column(matrix, i):
    return [row[i] for row in matrix]


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
    avg_xyz = np.sum(all_xyz, axis=0) / num_atoms
    return avg_xyz


def get_center_of_cell(cell_lengths, cell_angles):
    cell_vectors = list(ase.geometry.Cell.fromcellpar(cell_lengths + cell_angles))
    cell_cop = 0.5 * np.sum(cell_vectors, axis=0)

    return cell_cop


def convert_to_fractional(positions, cell_lengths, cell_angles, degrees=True):
    # Load cell lengths and angle
    a, b, c = cell_lengths
    if degrees == True:
        alpha, beta, gamma = np.deg2rad(cell_angles)
    else:
        alpha, beta, gamma = cell_angles

    # Create conversion matrix
    omega = (
        a
        * b
        * c
        * (
            1
            - np.cos(alpha) ** 2
            - np.cos(beta) ** 2
            - np.cos(gamma) ** 2
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        ** 0.5
    )
    cart_to_frac_matrix = [
        [
            1 / a,
            -np.cos(gamma) / (a * np.sin(gamma)),
            b
            * c
            * (np.cos(alpha) * np.cos(gamma) - np.cos(beta))
            / (omega * np.sin(gamma)),
        ],
        [
            0,
            1 / (b * np.sin(gamma)),
            a
            * c
            * (np.cos(beta) * np.cos(gamma) - np.cos(alpha))
            / (omega * np.sin(gamma)),
        ],
        [0, 0, (a * b * np.sin(gamma)) / omega],
    ]

    # Load and change positions
    all_xyz = np.array(positions)
    all_xyx_cart = np.matmul(cart_to_frac_matrix, all_xyz.transpose())
    positions = all_xyx_cart.transpose()

    return ase_atoms


def convert_to_cartesian(positions, cell_lengths, cell_angles, degrees=True):
    # Load cell lengths and angle
    a, b, c = cell_lengths
    if degrees == True:
        alpha, beta, gamma = np.deg2rad(cell_angles)
    else:
        alpha, beta, gamma = cell_angles

    # Create conversion matrix
    omega = (
        a
        * b
        * c
        * (
            1
            - np.cos(alpha) ** 2
            - np.cos(beta) ** 2
            - np.cos(gamma) ** 2
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        ** 0.5
    )
    frac_to_cart_matrix = [
        [a, b * np.cos(gamma), c * np.cos(beta)],
        [
            0,
            b * np.sin(gamma),
            c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
        ],
        [0, 0, omega / (a * b * np.sin(gamma))],
    ]

    # Load and change positions
    all_xyz = np.array(positions)
    all_xyx_cart = np.matmul(frac_to_cart_matrix, all_xyz.transpose())
    positions = all_xyx_cart.transpose()

    return positions


def write_pdb_with_bonds(
    filename, atoms, bonds, spacegroup="P 1", spacegroup_number="1"
):
    f = open(filename, "w")

    f.write("COMPND    UNNAMED\n")
    f.write("AUTHOR    Brian Day - LAMMPS Tools2\n")
    format = "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %s\n"

    f.write(format % (*atoms.get_cell_lengths_and_angles(), spacegroup))

    # Write Atoms
    format = "ATOM  %5d %4s MOL     1    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n"
    for atom in atoms:
        f.write(
            format
            % (
                atom.index + 1,
                atom.symbol.upper(),
                *atom.position,
                1.0,
                0.0,
                atom.symbol,
            )
        )

    # Write Bonds
    format = "CONECT %5d %5d\n"
    for bond in bonds:
        f.write(format % (bond[0] + 1, bond[1] + 1))

    f.write("END")

    f.close()


def atom_in_atoms(atom, atoms):
    for i in range(len(atoms)):
        other_atom = atoms[i]
        if atom.symbol == other_atom.symbol and np.all(
            atom.position == other_atom.position
        ):
            return True, other_atom
        else:
            return False, atom
