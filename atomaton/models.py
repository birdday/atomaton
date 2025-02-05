import ase
import copy
import numpy as np

from atomaton.helper import calculate_distance, column
from atomaton.visualize import view_structure

# General Notes:
# Avoid appending to numpy arrays except at the end of loops, since they weill always allocate
# contiguous memory blocks, which becomes inefficient as blocks become large. Use python lists then 
# append once at the end of looping. 

# TODO: Convert to Atom/Atoms
# TODO: Rename get methods to avoid confusion with getters/setters?
# TODO: Add function to calculate minimal binding box. Add to calculate bonds function for atoms missing cell data.
# TODO: RESET INDICIES FUNCTION WHICH ACCOUNTS FOR BOND INFO?
class Atoms:
    """Collection of Atoms, with methods for calculating bond, angle, dihedral, and improper terms, as well as other
    important simulation parameters.

    Based on (and uses some of) ASE atoms object. Reimplementing is for ease of control, and paedegogical exercise.
    """
    # --- Magic Methods
    def __len__(self):
        return self.num_atoms

    # --- Initialization Methods
    def __init__(self, symbols=np.array([]), positions=np.array([])):
        assert(len(symbols) == len(positions))

        # Minimall Required / Calculated Atom Info
        # Numpy will use minimal number of bits possible for all arrays (i.e., cannot append 'Ar' to ['C', 'H']).
        # To support 3 character symbols by default, we update the data type of the symbols array here.
        self.num_atoms = len(symbols)
        self.indicies = np.array([i for i in range(self.num_atoms)])
        self.symbols = symbols
        self.symbols = self.symbols.astype("<U3")
        self.positions = positions

        # Atom Info
        self.label = ""
        self.id = 0
        self.ase_atoms = ase.Atoms()
        self.atomic_numbers = np.array([])
        self.masses = np.array([])
        self.charges = np.array([])

        # Cell Params
        self.cell_lengths = np.array([])
        self.cell_angles = np.array([])
        self.unit_cell = None

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

    # TODO: Load Bond info when possible.
    @classmethod
    def bind_from_ase(cls, ase_atoms):
        symbols = np.array(ase_atoms.get_chemical_symbols())
        positions = ase_atoms.get_positions()

        atoms = cls(symbols, positions)
        atoms.ase_atoms = ase_atoms
        atoms.atomic_numbers = ase_atoms.get_atomic_numbers()
        atoms.masses = ase_atoms.get_masses()

        cell_lengths_and_angles = ase_atoms.cell.cellpar()
        atoms.cell_lengths = cell_lengths_and_angles[0:3]
        atoms.cell_angles = cell_lengths_and_angles[3::]
        if (atoms.cell_lengths > 0).all():
            atoms.unit_cell = UnitCell(atoms.cell_lengths, atoms.cell_angles)

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

        # TODO: Enforce data types for all inputs. 
        # Allow for AtomType-Any bonds. Split on "-" and order alphabetically..
        for key, value in cutoffs.items():
            if isinstance(value, int) or isinstance(value, float):
                cutoffs[key] = [0, float(value)]
            elif len(value) == 1:
                cutoffs[key] = [0, float(value[0])]
            if len(value) > 2:
                raise ValueError("Invalid cutoff!")

        if "default" not in cutoffs.keys():
            cutoffs["default"] = [0, 1.5]

        return cutoffs

    # --- Multibody Terms
    # TODO: Add warning if non 0 bond data already exists. Or flag to indicate bonds calculated? but diff cutoff dicts..
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
        symbols, positions = self.symbols, self.positions
        cutoffs = self._resolve_bond_cutoffs_dict(cutoffs)
        max_bond_length = np.array([val for val in cutoffs.values()]).flatten().max()
        ext_atom_symbols, ext_atom_positions, ext_atom_pseudo_indicies = self.create_extended_cell_minimal(
            max_bond_length=max_bond_length
        )
        # Add original atoms to extra atoms info for bond calcs.
        if not ext_atom_symbols.size == 0:
            ext_atom_symbols = np.append(self.symbols, ext_atom_symbols)
            ext_atom_positions = np.vstack([self.positions, ext_atom_positions])
            ext_atom_pseudo_indicies = np.append(self.indicies, ext_atom_pseudo_indicies)
        else:
            ext_atom_symbols, ext_atom_positions, ext_atom_pseudo_indicies =\
                self.symbols, self.positions, self.indicies

        num_ext_atoms = len(ext_atom_symbols)
        cutoff = self._resolve_bond_cutoffs_dict(cutoffs)

        bonds = []
        bond_types = []
        bonds_across_boundary = []

        extra_atom_symbols = []
        extra_atom_positions = []
        extra_bonds = []

        for i in range(num_atoms):
            p1 = positions[i]
            type1 = symbols[i]

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

    # --- Atom Positioning
    def shift_atoms(self, shift):
        self.positions += shift

    def get_center_of_positions(self):
        return self.positions.mean(axis=0)
    
    def center_in_cell(self):
        unit_cell = UnitCell(self.cell_lengths, self.cell_angles)
        center_of_cell = unit_cell.get_center_of_cell()
        center_of_atoms = self.get_center_of_positions()
        self.shift_atoms(center_of_cell-center_of_atoms)

    def center_on_position(self, position):
        center_of_atoms = self.get_center_of_positions()
        self.shift_atoms(position - center_of_atoms)

    # --- Metadata Calculations
    def get_bonds_metadata(self):
        atom_idxs = self.indicies
        symbols = self.symbols
        bonds = self.bonds

        # Initialize Results Dicts
        # Use dict with int keys rather than list since atom nums may be missing, deleted, etc.
        bond_count = {i: 0 for i in atom_idxs}  # Num. of bonds per atom.
        bonds_present = {i: [] for i in atom_idxs}  # List of bonds which include atom.
        bonds_present_idxs = {i: [] for i in atom_idxs} # List of bonds indicies in list.
        bonds_with = {i: [] for i in atom_idxs}  # List of atom types that atom bonds with.

        # Note, bond counts should be double counted, since we are interested in the number of bonds on each atom
        # not the number of total bonds here.
        for bond_idx, bond in enumerate(bonds):
            for i in bond:
                bond_count[i] += 1
                bonds_present[i].extend([bond])
                bonds_present_idxs[i].extend([bond_idx])
                bonds_with[i].extend([symbols[j] for j in bond if j != i])

        return bond_count, bonds_present, bonds_present_idxs, bonds_with

    # ---- Misc. Helper Functions
    def insert_atoms(self, atoms, position=None, new_bonds=None, new_bond_types=None):
        # New bonds use the index of the individual atoms object in order [self, atoms].
        atoms_copy = copy.deepcopy(atoms)
        if position is not None:
            atoms_center = atoms_copy.get_center_of_positions()
            atoms_copy.shift_atoms(position-atoms_center)
        
        # Update all minimal info
        self.symbols = np.concatenate([self.symbols, atoms_copy.symbols])
        self.indicies = np.concatenate([self.indicies, atoms_copy.indicies + self.num_atoms])
        self.positions = np.concatenate([self.positions, atoms_copy.positions])

        # Bonds should always be accompanied by bond types.
        if not (self.bonds.size == 0 or atoms_copy.bonds.size == 0):
            self.bonds = np.concatenate([self.bonds, atoms_copy.bonds + self.num_atoms])
            self.bond_types = np.concatenate([self.bond_types, atoms_copy.bond_types])

        if (new_bonds is not None and new_bond_types is not None):
            if new_bonds.shape[0] != new_bond_types.shape[0]:
                raise ValueError("new_bonds.shape must equal new_bond_types.shape")
            self.bonds = np.concatenate([self.bonds, new_bonds + np.array([0, self.num_atoms])])
            self.bond_types = np.concatenate([self.bond_types, new_bond_types])

        self.num_atoms += atoms_copy.num_atoms
        
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
        elif not (isinstance(max_bond_length, int) or isinstance(max_bond_length, float)):
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

    def calculate_bounding_box(self, padding=0.0, set_cell=False):
        max_xyz = self.positions.max(axis=0) + padding
        if set_cell:
            self.cell_lengths = max_xyz
            self.cell_angles = np.array([90.0, 90.0, 90.0])
            self.unit_cell = UnitCell(self.cell_lengths, self.cell_angles)
            self.center_in_cell()

    def view(self, **kwargs):
        view_structure(self, self.bonds, self.boundary_bonds, **kwargs) 

    def overlap_at_point(self, position, overlap_dist=1.0):
        # Sort positions to increase speed of finding overlap.
        # Sorting breaks this somehow...
        # FIXME: Must also sort the posiition arrays... not just the outer list. Yup! 
        # sorted_positions = np.sort(self.positions)
        for atom_pos in self.positions:
            dist = calculate_distance(atom_pos, position)
            if dist <= overlap_dist:
                return True
        return False

    def delete_atoms(self, atom_idxs, reset_indicies=False):
        # TODO: More elegantly handle other values like masses, charges, etc.
        self.symbols = np.delete(self.symbols, atom_idxs)
        self.positions = np.delete(self.positions, atom_idxs, axis=0)
        self.num_atoms = len(self.symbols)
        if reset_indicies:
            self.indicies = np.array([i for i in range(self.num_atoms)])


class Crystal(Atoms):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # TODO: Type check num cells? Soon we have to do this everywhere... Just document.
    def build_supercell(self, num_cells, filename=None):
        crystal = self.ase_atoms
        ao, bo, co = [num_cells[0], 0, 0], [0, num_cells[1], 0], [0, 0, num_cells[2]]
        crystal_new = ase.build.cut(
            crystal, a=ao, b=bo, c=co, origo=(0, 0, 0), tolerance=0.001
        )

        if filename != None:
            ase.io.write(crystal_new)

        return self.bind_from_ase(crystal_new)


# TODO: Want to use this to draw in visualize, but also want to use visualize in Atoms.
# Could move this class to another file, but feels unnecessary, but avoids circular dependencies...
class UnitCell:
    def __init__(self, cell_lengths, cell_angles, spacegroup='P1'):
        self._validate_cell_lengths_and_angles(cell_lengths, cell_angles)
        self.cell_lengths = cell_lengths
        self.cell_angles =  cell_angles
        self.spacegroup = spacegroup
    
    @staticmethod
    def _validate_cell_lengths_and_angles(cell_lengths, cell_angles):
        assert(len(cell_lengths) == 3)
        assert((0 < cell_lengths).all())
        assert(len(cell_angles) == 3)
        assert((0 < cell_lengths).all())
        assert((180 > cell_lengths).all())

    def _calculate_omega(self):
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

        return omega

    def get_cart_to_frac_matrix(self):
        a, b, c = self.cell_lengths
        alpha, beta, gamma = np.deg2rad(self.cell_angles)
        omega = self._calculate_omega()

        cart_to_frac_matrix = [
            [
                1 / a,
                -np.cos(gamma) / (a * np.sin(gamma)),
                b * c * (np.cos(alpha) * np.cos(gamma) - np.cos(beta)) / (omega * np.sin(gamma))
            ],
            [
                0,
                1 / (b * np.sin(gamma)),
                a * c * (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / (omega * np.sin(gamma))
            ],
            [0, 0, (a * b * np.sin(gamma)) / omega]
        ]

        return cart_to_frac_matrix

    def get_frac_to_cart_matrix(self):
        a, b, c = self.cell_lengths
        alpha, beta, gamma = np.deg2rad(self.cell_angles)
        omega = self._calculate_omega()

        frac_to_cart_matrix = [
            [a, b * np.cos(gamma), c * np.cos(beta)],
            [
                0,
                b * np.sin(gamma),
                c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
            ],
            [0, 0, omega / (a * b * np.sin(gamma))],
        ]

        return frac_to_cart_matrix

    def get_center_of_cell(self):
        frac_to_cart_matrix = self.get_frac_to_cart_matrix()

        return np.matmul(frac_to_cart_matrix, np.array([0.5, 0.5, 0.5]))

    def get_corners(self):
        unit_cell_corners = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ])
        frac_to_cart_matrix = self.get_frac_to_cart_matrix()

        return np.matmul(frac_to_cart_matrix, unit_cell_corners)


# --- Intramolecular Forcefield Terms
# TODO: Update calculate bond functions and input functions.
# Make an optional type? Or separate info into Atoms obj.
class Bond:
    def __init__(self, atoms, bond_type, bond_order=None):
        self.atoms = atoms
        self.bond_type = bond_type
        self.bond_order = bond_order


class Angle:
    def __init__(self, center_atom, ordered_atoms):
        self.center_atom = center_atom
        self.ordered_atoms = ordered_atoms


class Improper:
    def __init__(self, center_atom, ordered_atoms):
        self.center_atom = center_atom
        self.ordered_atoms = ordered_atoms     


# TODO: Add method to convert to Atoms Obj (Already written in view)
class SimulationBox:
    def __init__(self):
        # Atoms
        self.atoms_objs = np.array([])

        # Cell Params
        self.unit_cell = None
        self.cell_lengths = np.array([])
        self.cell_angles = np.array([])
    
    @classmethod
    def create_from_atoms(cls, atoms):
        sim_box = SimulationBox()
        sim_box.atoms_objs = np.array([atoms])
        sim_box.cell_lengths = atoms.cell_lengths
        sim_box.cell_angles = atoms.cell_angles
        sim_box.unit_cell = UnitCell(sim_box.cell_lengths, sim_box.cell_angles)

        return sim_box
    
    # TODO: Allow for positions in fractional coords.
    def insert_atoms(self, atoms, position=None):
        atoms_copy = copy.deepcopy(atoms)
        if position is not None:
            atoms_cop = atoms_copy.get_center_of_positions()
            atoms_copy.shift_atoms(position-atoms_cop)
        self.atoms_objs = np.append(self.atoms_objs, atoms_copy)

    def view(self):
        all_symbols = np.concatenate([obj.symbols for obj in self.atoms_objs])
        all_positions = np.vstack([obj.positions for obj in self.atoms_objs])

        # It is faster to append to lists and convert to numpy arrays
        # and cannot concat an emtpy array and array with definite shape.
        all_bonds = []
        all_boundary_bonds = []

        atom_count = 0
        for obj in self.atoms_objs:
            if obj.bonds.size != 0:
                all_bonds.append(obj.bonds + atom_count)
            if obj.boundary_bonds.size != 0:
                all_boundary_bonds.append(obj.boundary_bonds + atom_count)
            atom_count += obj.num_atoms

        # Convert to numpy arrays, even if no bonds present.
        if len(all_bonds) != 0:
            all_bonds = np.concatenate(all_bonds, axis=0)
        else:
            all_bonds = np.array([])

        if len(all_boundary_bonds) != 0:
            all_boundary_bonds = np.concatenate(all_boundary_bonds, axis=0)
        else:
            all_boundary_bonds = np.array([])

        atoms_obj = Atoms(all_symbols, all_positions)
        atoms_obj.bonds = all_bonds
        atoms_obj.boundary_bonds = all_boundary_bonds
        atoms_obj.cell_lengths = self.cell_lengths
        atoms_obj.cell_angles = self.cell_angles
        atoms_obj.view()

    def build_supercell():
        pass

    def overlap_at_point(self, position, overlap_dist=1.0):
        for atoms_obj in self.atoms_objs:
            if atoms_obj.overlap_at_point(position, overlap_dist=overlap_dist):
                return True

        return False
    
