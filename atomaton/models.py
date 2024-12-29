import ase
import copy
import numpy as np

from atomaton.visualize import view_structure

# General Notes:
# Avoid appending to numpy arrays except at the end of loops, since they weill always allocate
# contiguous memory blocks, which becomes inefficient as blocks become large. Use python lists then 
# append once at the end of looping. 

class Atoms():
    """Collection of Atoms, with methods for calculating bond, angle, dihedral, and improper terms, as well as other
    important simulation parameters.

    Based on (and uses some of) ASE atoms object. Reimplementing is for ease of control, and paedegogical exercise.
    """
    # --- Initialization Methods
    def __init__(self):
        # Atom Info
        self.ase_atoms = ase.Atoms()
        self.symbols = np.array([])
        self.atomic_numbers = np.array([])
        self.indicies = np.array([])
        self.positions = np.array([])
        self.masses = np.array([])
        self.charges = np.array([])
        self.num_atoms = 0

        # Cell Params
        self.cell_lengths = np.array([])
        self.cell_angles = np.array([])
        self.pbc = True

        # Forcefield Terms
        self.forcefield = None
        self.bonds = np.array([])
        self.bond_types = np.array([])
        self.angles = np.array([])
        self.dihedrals = np.array([])
        self.impropers = np.array([])

        # Special Bond Parameters
        self.boundary_bonds = np.array([])

        # Special View Parameters
        self.extra_atom_symbols = np.array([])
        self.extra_atom_positions = np.array([])
        self.extra_bonds = np.array([])

    @classmethod
    def bind_from_file(cls, file):
        ase_atoms = ase.io.read(file)

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
 
    # TODO: Refactor to use no ASE atoms
    def calculate_angles(self):
        atoms = self.atoms
        bonds =  self.bonds
        bonds_alt = self.bond_alt

        # TODO: Need to check how angle terms which cross unit cells are defined in lammps and adjust code accordingly. 
        all_angles = []
        all_angles_alt = []
        all_angle_types = []

        for i in range(len(bonds)):
            bond1 = bonds[i]
            bond1_alt = bonds_alt[i]

            for j in range(i + 1, len(bonds)):
                bond2 = bonds[j]
                bond2_alt = bonds_alt[j]

                atoms_in_angle = sorted(set(bond1 + bond2))
                if len(atoms_in_angle) == 3:
                    # Angle defined in by core atom numbers, used for calcuating dihedrals and impropers and writing lammps files
                    center_atom = sorted(set(bond1).intersection(bond2))
                    end_atoms = sorted(set(atoms_in_angle).difference(center_atom))
                    ordered_atoms_in_angle = copy.deepcopy(end_atoms)
                    ordered_atoms_in_angle.insert(1, *center_atom)
                    ordered_atom_types_in_angle = [
                        atoms[index].symbol for index in end_atoms
                    ]
                    ordered_atom_types_in_angle.insert(
                        1, *[atoms[index].symbol for index in center_atom]
                    )

                    all_angles.extend([[*center_atom, ordered_atoms_in_angle]])
                    all_angle_types.extend([ordered_atom_types_in_angle])

                    # Angle defined by extended atom numbers, used for calculating angle properties
                    # Is this missing a condition for center atom not in either bond??
                    if center_atom[0] in bond1_alt and center_atom[0] in bond2_alt:
                        center_atoms = center_atom
                    elif center_atom[0] in bond1_alt:
                        bond2_center_atom = [i for i in bond2_alt if i >= len(atoms)]
                        center_atoms = center_atom + bond2_center_atom
                    elif center_atom[0] in bond2_alt:
                        bond1_center_atom = [i for i in bond1_alt if i >= len(atoms)]
                        center_atoms = center_atom + bond1_center_atom
                    else:
                        center_atoms = [i for i in bond1_alt if i >= len(atoms)] + [
                            i for i in bond2_alt if i >= len(atoms)
                        ]
                    all_angles_alt.extend([[center_atoms, [bond1_alt, bond2_alt]]])

        sorted_indicies = np.argsort(column(all_angles, 0))
        all_angles_sorted = [all_angles[index] for index in sorted_indicies]
        all_angles_alt_sorted = [all_angles_alt[index] for index in sorted_indicies]
        all_angle_types_sorted = [all_angle_types[index] for index in sorted_indicies]

        return all_angles_sorted, all_angles_alt_sorted, all_angle_types_sorted

    # TODO: Refactor to use no ASE atoms
    def calculate_dihedrals_and_impropers(self, improper_tol=0.1):
        atom_symbols = self.atom_symbols
        atoms_positions = self.atom_positions
        bonds = self.bonds
        bonds_alt = self.bonds_alt
        angles = self.angles
        angles_alt = self.angles_alt

        all_dihedrals = []
        all_dihedrals_alt = []
        all_dihedral_types = []
        all_impropers = []
        all_impropers_alt = []
        all_improper_types = []

        for i in range(len(angles)):
            center_atom = angles[i][0]
            angle = angles[i][1]
            angle_alt = angles_alt[i][1]

            for j in range(len(bonds)):
                bond = bonds[j]
                bond_alt = bonds_alt[j]
                atoms_in_group = sorted(list(set(angle + bond)))
                shared_atom = sorted(set(angle).intersection(bond))

                if len(atoms_in_group) == 4 and shared_atom != [center_atom]:
                    # all_dihedrals.extend([atoms_in_group])

                    ordered_atoms = list(
                        set(angle).difference([center_atom]).difference(shared_atom)
                    )
                    ordered_atoms.insert(1, center_atom)
                    ordered_atoms.insert(2, *shared_atom)
                    ordered_atoms.insert(3, *list(set(bond).difference(shared_atom)))

                    if ordered_atoms[0] > ordered_atoms[-1]:
                        ordered_atoms.reverse()
                    all_dihedrals.extend([ordered_atoms])
                    all_dihedrals_alt.extend([angle_alt + [bond_alt]])

                    ordered_atom_types = [atoms[index].symbol for index in ordered_atoms]
                    all_dihedral_types.extend([ordered_atom_types])

                if len(atoms_in_group) == 4 and shared_atom == [center_atom]:
                    # Impropers should lie approxiamtely in the same plane
                    ordered_atoms = set(copy.deepcopy(atoms_in_group))
                    ordered_atoms = ordered_atoms.difference([center_atom])
                    ordered_atoms = list(ordered_atoms)

                    # Create two vectors from non-central points
                    # This method likely needs to use alt atom positions...
                    pc = atoms[center_atom].position
                    p0 = atoms[ordered_atoms[0]].position
                    v01 = (
                        atoms[ordered_atoms[1]].position - atoms[ordered_atoms[0]].position
                    )
                    v02 = atoms[ordered_atoms[2]].position = atoms[
                        ordered_atoms[0]
                    ].position
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
                            atoms[index].symbol for index in ordered_atoms
                        )
                        ordered_atom_types.insert(0, atoms[center_atom].symbol)
                        all_improper_types.extend([ordered_atom_types])

                        if center_atom in angle_alt:
                            bond_center_atom = [i for i in bond_alt if i >= len(atoms)]
                            all_impropers_alt.extend(
                                [[[center_atom] + bond_center_atom, angle_alt + [bond_alt]]]
                            )
                        elif center_atom in bond_alt:
                            angle_center_atom = [i for i in angle_alt[0] if i >= len(atoms)]
                            all_impropers_alt.extend(
                                [
                                    [
                                        [center_atom] + angle_center_atom,
                                        angle_alt + [bond_alt],
                                    ]
                                ]
                            )

        return (
            all_dihedrals,
            all_dihedrals_alt,
            all_dihedral_types,
            all_impropers,
            all_impropers_alt,
            all_improper_types,
        )

    # --- Other helper functions
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

    # TODO: Refactor to use no ASE atoms
    def view_structure(self, **kwargs):
       # Construct atoms object
       atoms_to_plot = ase.Atoms()
       for symbol, position in zip(self.symbols, self.positions):
           atoms_to_plot += ase.Atom(symbol, position)
       bonds_to_plot = self.bonds

       # Update Unit Cell params
       atoms_to_plot.set_cell(np.array([15, 15, 15, 90, 90, 90]))
       view_structure(atoms_to_plot, bonds_to_plot, np.array([]), **kwargs) 

class SimulationBox:
    def __init__(self):
        # Atoms
        self.framework = None
        self.solvent = []
        self.molecules = []

        # Cell Params
        self.cell_lengths = []
        self.cell_angles = []
        self.pbc = True


def calculate_distance(p1, p2):
    """Function which calculates the distance between two positions in N-dimensional space. 

    Args:
        p1 (list[float...]): position in N-dimensional space
        p2 (list[float...]): position in N-dimensional space

    Returns:
        float: distance between two positions
    """

    dist = np.sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))]) ** 0.5

    return dist