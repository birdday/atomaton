import ase
import copy
import numpy as np

from atomaton.helper import calculate_distance, column
from atomaton.visualize import view_structure

# General Notes:
# Avoid appending to numpy arrays except at the end of loops, since they weill always allocate
# contiguous memory blocks, which becomes inefficient as blocks become large. Use python lists then 
# append once at the end of looping. 

class Atoms:
    """Collection of Atoms, with methods for calculating bond, angle, dihedral, and improper terms, as well as other
    important simulation parameters.

    Based on (and uses some of) ASE atoms object. Reimplementing is for ease of control, and paedegogical exercise.
    """
    # --- Initialization Methods
    def __init__(self):
        # Atom Info
        self.ase_atoms = ase.Atoms()
        self.indicies = np.array([])
        self.symbols = np.array([])
        self.positions = np.array([])
        self.atomic_numbers = np.array([])
        self.masses = np.array([])
        self.charges = np.array([])
        self.num_atoms = 0

        # Cell Params
        self.cell_lengths = np.array([])
        self.cell_angles = np.array([])
        self.pbc = True

        # Intramolecular Forcefield Terms
        self.forcefield = None
        self.bonds = np.array([])
        self.bond_types = np.array([])
        self.angles = np.array([])
        self.angle_types = np.array([])
        self.dihedrals = np.array([])
        self.dihedral_types = np.array([])
        self.impropers = np.array([])
        self.impropers_types = np.array([])

        # Special Bond Parameters
        self.boundary_bonds = np.array([])

        # Special View Parameters
        self.extra_atom_symbols = np.array([])
        self.extra_atom_positions = np.array([])
        self.extra_bonds = np.array([])

    @classmethod
    def bind_from_ase(cls, ase_atoms):
        atoms = cls()
        atoms.ase_atoms = ase_atoms
        atoms.indicies = np.array([i for i, _ in enumerate(ase_atoms)])
        atoms.symbols = ase_atoms.get_chemical_symbols()
        atoms.atomic_numbers = ase_atoms.get_atomic_numbers()
        atoms.positions = ase_atoms.get_positions()
        atoms.masses = ase_atoms.get_masses()
        atoms.num_atoms = len(ase_atoms)

        cell_lengths_and_angles = ase_atoms.cell.cellpar()
        atoms.cell_lengths = cell_lengths_and_angles[0:3]
        atoms.cell_angles = cell_lengths_and_angles[3::]

        return atoms
    
    @classmethod
    def bind_from_file(cls, file):
        ase_atoms = ase.io.read(file)

        return cls.bind_from_ase(ase_atoms)

    # --- Auxilary Parameter Resolution
    @ staticmethod
    def _resolve_bond_cutoffs_dict(cutoffs):
        """Function which validates the dictionary of bond types.
        If "default" not included, will be added with value of [0, 1.5].

        Args:
            cutoffs (dict): Dictionary of atomType-atomType: [min, max] bond distances in angstrom.

        Raises:
            ValueError: Cutoff must be a single value or an array of two numbers.

        Returns:
            dict: dictionary of bond cutoffs, with single values updated to [0, value] and default added.
        """

        for key, value in cutoffs.items():
            if len(value) == 1:
                cutoffs[key] = [0, value]
            if len(value) > 2:
                raise ValueError("Invalid cutoff!")

        if "default" not in cutoffs.keys():
            cutoffs["default"] = [0, 1.5]

        return cutoffs

    # --- Multibody Terms
    def calculate_bonds(self, cutoffs={"default": [0, 1.5]}):
        """Finds the bonds between atoms, as defined by the cutoffs dict.

        Args:
            cutoffs (dict, optional): Dictionary of bond cutoffs. Keys must match atom types, separated by a -, in alphabetically order. Defaults to {"default": [0, 1.5]}.

        Returns:
            list(list(int, int)): list of bonds, by sorted atom index.
            list(list[str, str)): list of bond types, sorted alphabetically by atom type.
            list[list[int, int]]: list of bonds which cross the cell boundary, by sorted orignal atom index.
            Atoms: list of atoms outside of the cell which are part of bonds crossing the cell.
            list[list[int, int]]: list of bonds which cross the cell boundary, using the extended atoms index.
        """

        # Prepare Atoms Object
        num_atoms = self.num_atoms
        ase_atoms = self.ase_atoms
        ext_atom_symbols, ext_atom_positions, ext_atom_pseudo_indicies = self.create_extended_cell_minimal()

        # Add original atoms to extra atoms info for bond calcs.
        ext_atom_symbols = np.append(self.symbols, ext_atom_symbols)
        ext_atom_positions = np.vstack([self.positions, ext_atom_positions])
        ext_atom_pseudo_indicies = np.append(self.indicies, ext_atom_pseudo_indicies)

        num_ext_atoms = len(ext_atom_symbols)
        cutoff = self._resolve_bond_cutoffs_dict(cutoffs)

        bonds = []
        bond_types = []
        bonds_across_boundary = []

        extra_atom_symbols = []
        extra_atom_positions = []
        extra_bonds = []

        for i in range(num_atoms):
            p1 = ase_atoms[i].position
            type1 = ase_atoms[i].symbol

            for j in range(i + 1, num_ext_atoms):
                p2 = ext_atom_positions[j]
                type2 = ext_atom_symbols[j]

                r = calculate_distance(p1, p2)
                bondtype = "-".join(sorted([type1, type2]))

                if (
                    (
                        bondtype in cutoff.keys()
                        and r >= cutoff[bondtype][0]
                        and r <= cutoff[bondtype][1]
                    )
                    or (
                        bondtype not in cutoff.keys()
                        and r >= cutoff["default"][0]
                        and r <= cutoff["default"][1]
                    )
                ):
                    bond = sorted(set((i, ext_atom_pseudo_indicies[j])))

                    # Check if bond already found (i.e., the second time we encounter a bond crossing the cell boundary)
                    if bond not in bonds:
                        bonds.extend([bond])
                        bond_types.extend([sorted([type1, type2])])

                        # Check if bond crosses boundary and if so, record bond to special category.
                        if j > num_atoms:
                            bonds_across_boundary.extend([bond])

                    # If bond crosses boundary, record extra atom and extra bond.
                    # Note that we do not care if the bond has already been found, since this will be a different instance of that bond with respect to the boundary it crosses.
                    # Also, we must index the extra bond slightly differently since we only record atoms as needed.
                    if j > num_atoms:
                        extra_atom_symbols.extend(type2)
                        extra_atom_positions.append(p2)
                        # We must also subtract 1 since we extend atoms first and python uses 0 indexing. 
                        extra_bonds.extend([[i, num_atoms+len(extra_bonds)-1]])

        self.bonds = np.array(bonds)
        self.bond_types = np.array(bond_types)
        self.boundary_bonds = np.array(bonds_across_boundary)
        self.extra_atoms_symbols = np.array(extra_atom_symbols)
        self.extra_atoms_positions = np.array(extra_atom_positions)
        self.extra_bonds = np.array(extra_bonds)
 
    def calculate_angles(self):
        # TODO: Need to check how angle terms which cross unit cells are defined in lammps and adjust code accordingly.
        # TODO: Include more advanced angle dict like cutoffs.
        num_atoms = self.num_atoms
        atom_symbols = self.symbols
        bonds =  self.bonds

        all_angles = []
        all_angle_types = []

        for i in range(len(bonds)):
            bond1 = set(bonds[i])
            
            for j in range(i + 1, len(bonds)):
                bond2 = set(bonds[j])

                atoms_in_angle = sorted(bond1.union(bond2))
                if len(atoms_in_angle) == 3:
                    # Angle defined in by core atom numbers, used for calcuating dihedrals and impropers and writing lammps files
                    center_atom = sorted(set(bond1).intersection(bond2))
                    end_atoms = sorted(set(atoms_in_angle).difference(center_atom))
                    ordered_atoms_in_angle = copy.deepcopy(end_atoms)
                    ordered_atoms_in_angle.insert(1, *center_atom)
                    ordered_atom_types_in_angle = [
                        atom_symbols[index] for index in end_atoms
                    ]
                    ordered_atom_types_in_angle.insert(
                        1, *[atom_symbols[index] for index in center_atom]
                    )

                    all_angles.extend([[*center_atom, ordered_atoms_in_angle]])
                    all_angle_types.extend([ordered_atom_types_in_angle])

        sorted_indicies = np.argsort(column(all_angles, 0))
        self.angles = np.array([Angle(*all_angles[index]) for index in sorted_indicies])
        self.angle_types = np.array([all_angle_types[index] for index in sorted_indicies])

    def calculate_dihedrals_and_impropers(self, improper_tol=0.1):
        atom_symbols = self.symbols
        atoms_positions = self.positions
        bonds = self.bonds
        center_atoms = [angle.center_atom for angle in self.angles]
        angles = [angle.ordered_atoms for angle in self.angles]

        all_dihedrals = []
        all_dihedral_types = []
        all_impropers = []
        all_improper_types = []

        for i in range(len(angles)):
            center_atom = center_atoms[i]
            angle = set(angles[i])

            for j in range(len(bonds)):
                bond = set(bonds[j])
                atoms_in_group = sorted(angle.union(bond))
                shared_atom = sorted(set(angle).intersection(bond))

                # Dihedral Terms
                if len(atoms_in_group) == 4 and shared_atom != [center_atom]:
                    ordered_atoms = list(
                        set(angle).difference([center_atom]).difference(shared_atom)
                    )
                    ordered_atoms.insert(1, center_atom)
                    ordered_atoms.insert(2, *shared_atom)
                    ordered_atoms.insert(3, *list(set(bond).difference(shared_atom)))

                    if ordered_atoms[0] > ordered_atoms[-1]:
                        ordered_atoms.reverse()
                    all_dihedrals.extend([ordered_atoms])

                    ordered_atom_types = [atom_symbols[index] for index in ordered_atoms]
                    all_dihedral_types.extend([ordered_atom_types])

                # Improper Terms
                if len(atoms_in_group) == 4 and shared_atom == [center_atom]:
                    # Impropers should lie approxiamtely in the same plane
                    ordered_atoms = set(copy.deepcopy(atoms_in_group))
                    ordered_atoms = ordered_atoms.difference([center_atom])
                    ordered_atoms = list(ordered_atoms)

                    # Create two vectors from non-central points
                    # This method likely needs to use alt atom positions...
                    pc = atoms_positions[center_atom]
                    p0 = atoms_positions[ordered_atoms[0]]
                    v01 = atoms_positions[ordered_atoms[1]] - atoms_positions[ordered_atoms[0]]
                    v02 = atoms_positions[ordered_atoms[2]] - atoms_positions[ordered_atoms[0]]
                    a, b, c = np.cross(v01, v02)
                    d = -1 * (a * p0[0] + b * p0[1] + c * p0[2])
                    num, den = (
                        a * pc[0] + b * pc[1] + c * pc[2] + d,
                        (a**2 + b**2 + c**2) ** 0.5,
                    )
                    if den != 0:
                        dmin = abs(num / den)
                    else:
                        dmin = 0

                    if dmin <= improper_tol:
                        all_impropers.extend(
                            [[center_atom, [center_atom, *sorted(ordered_atoms)]]]
                        )
                        ordered_atom_types = sorted(
                            atom_symbols[index] for index in ordered_atoms
                        )
                        ordered_atom_types.insert(0, atom_symbols[center_atom])
                        all_improper_types.extend([ordered_atom_types])

        self.dihedrals = np.array(all_dihedrals)
        self.dihedral_types = np.array(all_dihedral_types)
        self.impropers = np.array(all_impropers)
        self.improper_types = np.array(all_improper_types)

    # --- Other helper functions
    def shift_atoms(self, shift):
        self.positions += shift

    def get_center_of_positions(self):
        return self.positions.mean(axis=0)
    
    def center_atom_in_cell(self):
        unit_cell = UnitCell(self.cell_lengths, self.cell_angles)
        center_of_cell = unit_cell.get_center_of_cell()
        center_of_atoms = self.get_center_of_positions()
        self.shift_atoms(center_of_cell-center_of_atoms)

    def create_extended_cell_minimal(self, max_bond_length=5.0):
        """Creates a minimally extended cell to speed up O(N^2) bond check. This functions is O(N).

        Args:
            max_bond_length (float, optional): Maximum possible bond length, used to determine degree to which cell is extended. Defaults to 5.0.

        Raises:
            TypeError: max_bond_length must be a single value, or dictionary of bond cutoffs.
            ValueError: max_bond_length must be less than half the length of the shortes unit cell dimension.

        Returns:
            ase.Atoms: minimally extended cell
        """

        cell_x, cell_y, cell_z = self.cell_lengths
        extended_atom_symbols = []
        extended_atom_positions = []
        pseudo_indicies = []

        if type(max_bond_length) == dict:
            max_bond_length = max([max(max_bond_length[key]) for key in max_bond_length])
        elif type(max_bond_length) != int and type(max_bond_length) != float:
            raise TypeError("Invalid max_bond_length type.")

        if (
            max_bond_length >= 0.5 * cell_x
            or max_bond_length >= 0.5 * cell_y
            or max_bond_length >= 0.5 * cell_z
        ):
            raise ValueError("max_bond_length greater than half the cell length.")

        for i, symbol, position in zip(self.indicies, self.symbols, self.positions):
            px, py, pz = position
            perx, pery, perz = None, None, None

            # Check X
            if px >= 0 and px <= max_bond_length:
                perx = "+x"
                ext_x = cell_x
                extended_atom_symbols.extend((symbol))
                extended_atom_positions.append([px + ext_x, py, pz])
                pseudo_indicies.extend([i])
            if px >= cell_x - max_bond_length and px <= cell_x:
                perx = "-x"
                ext_x = -cell_x
                extended_atom_symbols.extend((symbol))
                extended_atom_positions.append([px + ext_x, py, pz])
                pseudo_indicies.extend([i])

            # Check Y
            if py >= 0 and py <= max_bond_length:
                pery = "+y"
                ext_y = cell_y
                extended_atom_symbols.extend((symbol))
                extended_atom_positions.append([px, py + ext_y, pz])
                pseudo_indicies.extend([i])
            if py >= cell_y - max_bond_length and py <= cell_y:
                pery = "-y"
                ext_y = -cell_y
                extended_atom_symbols.extend((symbol))
                extended_atom_positions.append([px, py + ext_y, pz])
                pseudo_indicies.extend([i])

            # Check Z
            if pz >= 0 and pz <= max_bond_length:
                perz = "+z"
                ext_z = cell_z
                extended_atom_symbols.extend((symbol))
                extended_atom_positions.append([px, py, pz + ext_z])
                pseudo_indicies.extend([i])
            if pz >= cell_z - max_bond_length and pz <= cell_z:
                perz = "-z"
                ext_z = -cell_z
                extended_atom_symbols.extend((symbol))
                extended_atom_positions.append([px, py, pz + ext_z])
                pseudo_indicies.extend([i])

            # Check XY
            if perx != None and pery != None:
                extended_atom_symbols.extend((symbol))
                extended_atom_positions.append([px + ext_x, py + ext_y, pz])
                pseudo_indicies.extend([i])

            # Check XZ
            if perx != None and perz != None:
                extended_atom_symbols.extend((symbol))
                extended_atom_positions.append([px + ext_x, py, pz + ext_z])
                pseudo_indicies.extend([i])

            # Check YZ
            if pery != None and perz != None:
                extended_atom_symbols.extend((symbol))
                extended_atom_positions.append([px, py + ext_y, pz + ext_z])
                pseudo_indicies.extend([i])

            # Check XYZ
            if perx != None and pery != None and perz != None:
                extended_atom_symbols.extend((symbol))
                extended_atom_positions.append([px + ext_x, py + ext_y, pz + ext_z])
                pseudo_indicies.extend([i])

        extended_atom_symbols = np.array(extended_atom_symbols)
        extended_atom_positions = np.array(extended_atom_positions)
        pseudo_indicies = np.array(pseudo_indicies)

        return extended_atom_symbols, extended_atom_positions, pseudo_indicies

    def view_structure(self, **kwargs):
        view_structure(self, self.bonds, self.boundary_bonds, **kwargs) 


class Crystal(Atoms):
    def __init__(self):
        super().__init__()
    
    def build_supercell(self, num_cells, filename=None):
        crystal = self.ase_atoms
        ao, bo, co = [num_cells[0], 0, 0], [0, num_cells[1], 0], [0, 0, num_cells[2]]
        crystal_new = ase.build.cut(
            crystal, a=ao, b=bo, c=co, origo=(0, 0, 0), tolerance=0.001
        )

        if filename != None:
            ase.io.write(crystal_new)

        return self.bind_from_ase(crystal_new)


class UnitCell:
    def __init__(self, cell_lengths, cell_angles, spacegroup='P1'):
        self.cell_lengths = cell_lengths
        self.cell_angles =  cell_angles
        self.spacegroup = spacegroup
    
    def get_center_of_cell(self):
        a, b, c = self.cell_lengths
        alpha, beta, gamma = np.deg2rad(self.cell_angles)

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

        return np.matmul(frac_to_cart_matrix, np.array([0.5, 0.5, 0.5]))


# --- Intramolecular Forcefield Terms
class Angle:
    def __init__(self, center_atom, ordered_atoms):
        self.center_atom = center_atom
        self.ordered_atoms = ordered_atoms


class Improper:
    def __init__(self, center_atom, ordered_atoms):
        self.center_atom = center_atom
        self.ordered_atoms = ordered_atoms     


class SimulationBox:
    def __init__(self, atoms_objs):
        # Atoms
        self.atoms_objs = atoms_objs

        # Cell Params
        self.cell_lengths = []
        self.cell_angles = []
        self.pbc = True
    
    def _shift_bond_indicies():
        pass
    
    def _insert_atoms():
        pass

    def build_supercell():
        pass
