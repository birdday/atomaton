import ase as ase
from ase import Atoms, io, spacegroup, build, visualize
import numpy as np


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
            if num_bonds_on_atom == 1:
                type='C_1'
            if num_bonds_on_atom == 2:
                type='C_2'
            if num_bonds_on_atom == 3:
                # Check whether this should be C_3 or C_2
                type='C_3'
            else:
                type='C_3'

        if atom.symbol == 'O':
            if num_bonds_on_atom == 1:
                type='O_1'
            if num_bonds_on_atom == 2:
                type='O_3'
            if num_bonds_on_atom == 3:
                type='O_2'
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
