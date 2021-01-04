import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import numpy as np

"""
UFF Forcefield as published has some errors which are outlined here: http://towhee.sourceforge.net/forcefields/uff.html.
Text from webpage is also copied to 'uff_corrections.txt' for future reference.
"""

def assign_forcefield_atom_types(atoms, bonds_with):
    uff_symbols = []
    for atom in atoms:
        num_bonds_on_atom = len(bonds_with[str(atom.index)])
        if atom.symbol == 'H':
            if num_bonds_on_atom == 1:
                type='H_'
            if num_bonds_on_atom == 2:
                type='H_b'
            else:
                type='H_'

        if atom.symbol == 'C':
            # Note aromatic carbons are testing for separately and will overwrite anything assigned here.
            if num_bonds_on_atom == 2:
                type='C_1'
            if num_bonds_on_atom == 3:
                type='C_2'
            if num_bonds_on_atom == 4:
                type='C_3'
            else:
                type='C_3'

        if atom.symbol == 'O':
            if num_bonds_on_atom == 1:
                type='O_2'
            if num_bonds_on_atom == 2:
                type='O_3'
            else:
                type='O_3'

        if atom.symbol == 'Ar':
            type = 'Ar4+4'

        if atom.symbol == 'Ni':
            type = 'Ni4+2'

        uff_symbols.extend([type])

    return uff_symbols


def search_for_aromatic_carbons(atoms, all_dihedrals, uff_symbols, ring_tol=0.1):
    # Looks for 6-memebered rings only at the moment.

    # Loop for 6 Membered Carbon Rings
    for i in range(len(all_dihedrals)):
        for j in range(i+1,len(all_dihedrals)):

            # Check end points of the dihedrals
            d1 = all_dihedrals[i]
            d2 = all_dihedrals[j]
            end_points_d1 = [d1[0], d1[-1]]
            end_points_d2 = [d2[0], d2[-1]]
            # Midpoints
            p0, p1, p2, p3 = d1[1], d1[2], d2[1], d2[2]
            # Atom Types
            atom_types = [atoms[d1[0]].symbol, atoms[d1[1]].symbol, atoms[d1[2]].symbol, atoms[d1[3]].symbol,
                atoms[d2[0]].symbol, atoms[d2[1]].symbol, atoms[d2[2]].symbol, atoms[d2[3]].symbol]
            # print(set(atom_types))
            if end_points_d1 == end_points_d2 and len(set((p0, p1, p2, p3))) == 4 and set(atom_types) == {'C'}:
                p0, p1, p2, p3 = atoms[p0].position, atoms[p1].position, atoms[p2].position, atoms[p3].position

                # Calculate distance point between point and plane
                v01 = p0-p1
                v02 = p0-p2
                a,b,c = np.cross(v01, v02)
                d = -1*(a*p0[0] + b*p0[1] + c*p0[2])
                num, den = a*p3[0]+b*p3[1]+c*p3[2]+d, (a**2 + b**2 +c**2)**0.5
                if den != 0:
                    dmin = abs(num/den)
                else:
                    dmin = 0

                if dmin <= ring_tol:
                    for index in d1:
                        uff_symbols[index] = 'C_R'
                    for index in d2:
                        uff_symbols[index] = 'C_R'

    return uff_symbols


# def get_angle_potential(angle_types):
#
#
def load_atom_type_parameters(return_as_dict=True):
    # from lammps_tools.forcefields.uff.parameters.uff import UFF_DATA
    from parameters.uff import UFF_DATA

    if return_as_dict == True:
        header = ['r1', 'theta0', 'x1', 'D1', 'zeta', 'Z1', 'Vi', 'Uj', 'Xi', 'Hard', 'Radius']
        UFF_DATA_as_dict = {}
        for key in UFF_DATA:
            UFF_DATA_as_dict[key] = {}
            for i in range(len(UFF_DATA[key])):
                val = UFF_DATA[key][i]
                key2 = header[i]
                UFF_DATA_as_dict[key][key2] = val

        UFF_DATA = UFF_DATA_as_dict

    return UFF_DATA


def get_pair_potential(atom_types):
    """
    This is simply the non-bonded van Der Waals potential, modelled by Lennard-Jones parameters for the atom types. Typcially, we just specify the I,I pair coefficients and I,J pair coefficients are determined by the mixing rules. I,J pair coefficients can be explicitly listed to overwrite the mixing rules, however.
    A 6-12 Lennard-Jones potential of the following form is used:
        E_vdw = D_ij*(-2*(x_ij/x)**6+(x_ij/x)**12)
    N.B. CHECK HOW THIS IS IMPLEMENTED IN LAMMPS
    """

    atom_type_params = {}
    for atom_type in atom_types:
        lj_sigma = UFF_DATA[atom_type]['x1'] * (2**(-1./6.))
        lj_epsilon = UFF_DATA[atom_type]['D1']
        atom_type_params[atom_type] = [lj_epsilon, lj_sigma]

    return atom_type_params


def guess_bond_orders(bond_types):
    # There are highly sophisticated and complicated schemes for guessing bond order. In theory, bond order could be determined independently of the atom type, but for many cases can be more easily assigned after the fact. At some point consider implementing the more generic, sophisticated bond ordering scheme. This method is 'hacky' at best.
    # This bond order scheme is roughly the same as what is used in Pete Boyd's 'lammps-interface'.
    bond_order_dict = {}
    for bond_type in bond_types:
        key = bond_type[0]+'-'+bond_type[1]
        if len(set(['H_', 'F_', 'Cl', 'Br', 'I_']).intersection(set(bond_type))) != 0:
            bond_order_dict[key] = 1
        elif len(set(['C_3', 'N_3', 'O_3']).intersection(set(bond_type))) != 0:
            bond_order_dict[key] = 1
        elif len(set(bond_type)) == 1 and set(bond_type).issubset(set(['C_2', 'N_2', 'O_2'])):
            bond_order_dict[key] = 2
        elif len(set(bond_type)) == 1 and set(bond_type).issubset(set(['C_R', 'N_R', 'O_R'])):
            bond_order_dict[key] = 1.5
        else:
            print(bond_type, ':Bond order not properly assigned. Using default value of 1.')
            bond_order_dict[key] = 1

    return bond_order_dict


def get_bond_potential(bond_types, form='harmonic'):
    """
    Standard Natural Bond Length, r_ij
      r_ij = r_i + r_j +r_BO - r_EN
        r_i, r_j = atom-type-specific single bond radius
        r_BO = Bond Order Correction - Value of n is non-obvious.
        r_EN = Electronegativity Correction
      N.B. Original paper says '+r_EN'. This is a mistake.
    Force Constant K_ij
    """

    # 1. Calculate r_ij and K_ij
    bond_orders = guess_bond_orders(bond_types)
    bond_type_params = {}
    for bond_type in bond_types:
        atom_i = bond_type[0]
        atom_j = bond_type[1]

        r_i = UFF_DATA[atom_i]['r1']
        r_j = UFF_DATA[atom_j]['r1']
        x_i = UFF_DATA[atom_i]['Xi']
        x_j = UFF_DATA[atom_j]['Xi']
        z_i = UFF_DATA[atom_i]['Z1']
        z_j = UFF_DATA[atom_j]['Z1']

        # 'n' determined by hacky bond order approximation.
        n = bond_orders[atom_i+'-'+atom_j]
        r_BO = -0.1332*(r_i+r_j)*np.log(n)
        r_EN = (r_i*r_j*(x_i**0.5-x_j**0.5)**2) / (x_i*r_i + x_j*r_j)
        r_ij = r_i + r_j +r_BO - r_EN

        g = 332.06
        k_ij = g*z_i*z_j/(r_ij**3)

        key = atom_i+'-'+atom_j
        bond_type_params[key] = [k_ij, r_ij]

        # print(bond_type, k_ij, r_ij)

    # Harmonic Oscillator = f(r_ij, K_ij)
    if form == 'harmonic':
        return bond_type_params

    # Morse Function = f(r_ij, K_ij, D_ij)
    # Requires additional parameter, Bond Dissociation Energy, D_ij.
    elif form == 'morse':
        return bond_type_params
# def get_dihedral_potential(dihedral_types):
#
#
# def get_improper_potential(improper_types):
