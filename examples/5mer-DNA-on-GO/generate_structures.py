import ase
import copy
import itertools
import random
import numpy as np

from atomaton.models import Atoms, Crystal, SimulationBox


# TODO: Commit various updates.
# Add export to xyz/cif/pdb/mol functions. Use pybabel as much as possible?

# ---- 5mer DNA
# Structure does not contain hydrogens.
# I think DNA moleules from paper are single stranded..
actga = Atoms.bind_from_file("actga-B.pdb")
actga.cell_lengths = np.array([30, 30, 30])
actga.cell_angles = np.array([90, 90, 90])
actga.center_in_cell()
actga.calculate_bonds(cutoffs={"C-C": [0, 2.0], "O-P": [0, 2.5]})
objects = {
        "atom_sf": 0.20,
        "bond_r": 0.05,
        "cell_r": 0.5,
        "bond_color": (0.4, 0.4, 0.4),
        "cell_color": (0.4, 0.4, 0.4),
    }
actga.view(show_unit_cell=False, objects=objects)


# ---- Basic Graphene
# Bug in bond calculation... Or maybe just a visual artifact. 
# Check extra atoms and bonds across boundaries,
# Build supercell should not how to extend bonding...

# Bonding not needed in this case since graphene atom positions will be fixed.
# Orig paper used a cubic cell, which seems odd for graphene, but proabably doable. 
# Figure out how to generate cubic graphene cell?
# Also figure out function for doping graphene. Make general if possible..
graphene = Crystal.bind_from_file("graphene.cif")
a, b, c = graphene.cell_lengths

graphene = graphene.build_supercell([15,15,1])
graphene.cell_lengths[0] = graphene.cell_lengths[0] + 10
graphene.cell_lengths[1] = graphene.cell_lengths[1] + 10
graphene.cell_lengths[2] = 60.0

cx, cy, cz = graphene.unit_cell.get_center_of_cell()
graphene.center_on_position(np.array([cx, cy, 2]))
graphene.calculate_bonds(cutoffs={"default":[0, 1.2], "C-C": [0, 1.5]})

# Remove carbons with only 1 bond. This is inelegant and slow but works.
bond_counts, bonds_present, bonds_present_idxs, bonds_with = graphene.get_bonds_metadata()
atoms_to_del = [key for key, val in bond_counts.items() if val == 1]
graphene.delete_atoms(atoms_to_del, reset_indicies=True)
graphene.calculate_bonds(cutoffs={"default":[0, 1.2], "C-C": [0, 1.5]})
graphene.view()

# ---- Wrinkled Graphene
# For use in sims, ensure periodicity of the wrinkling is compatible with
# actual cell lengths
def wrinkle_fxn(pos):
    return np.array([pos[0], pos[1],
                     pos[2] + 2*np.sin(pos[0]/6) + 2*np.sin(pos[1]/6) ])

wrinkled_graph = copy.deepcopy(graphene)
pos_new = []
for pos in wrinkled_graph.positions:
    pos_new.append(wrinkle_fxn( pos))
wrinkled_graph.positions = np.array(pos_new)
wrinkled_graph.view()


# ---- Graphene Oxide
# dx.doi.org/10.1021/cr300115g | Chem. Rev. 2012, 112, 6027−6053

# Reasonably high doping density, no specifics given
# Contains epoxides, hydroxyls, carboxyls and carbonyls (Hydroxyl + Carboxyl)
# Carboxyl really only binds along edges
# Oxygen doping converts sp2 to sp3 hybridization, no removal of C atoms necessary.

# Paper includes periodic boundary conditions, but the graphene oxide sheet does NOT bond across the boundaries.
# Thus it does include carboxyl groups.

# TODO: Worth studying the impact of specific functional groups?
# Ratio of functionalization controls band gap.
# Graphene vacancies? - Low conc. but present. 
# Could study affect of defects above / below adsrobing surface.

graphene_oxide = copy.deepcopy(graphene)

hydroxyl = Atoms(
    symbols=np.array(["O", "H"]),
    positions=np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.95]
    ])
)
epoxide = Atoms(symbols=np.array(["O"]), positions=np.array([[0.0, 0.0, 0.0]]))
carboxyl = Atoms(
    symbols=np.array(["C", "O", "O", "H"]),
    positions=np.array([
        [0.0, 0.0, 0.0],
        [-0.75, 0.0, 0.75],
        [0.75, 0.0, 0.75],
        [0.75, 0.0, 1.5]
    ])
)

hydroxyl.calculate_bounding_box(padding=5.0, set_cell=True)
hydroxyl.calculate_bonds(cutoffs={"default":[0, 1.2], "C-C": [0, 1.5]})
# hydroxyl.view()

carboxyl.calculate_bounding_box(padding=5.0, set_cell=True)
carboxyl.calculate_bonds(cutoffs={"default":[0, 1.2], "O-O": [0.0, 0.0]})
# carboxyl.view()

epoxide.calculate_bounding_box(padding=5.0, set_cell=True)
epoxide.calculate_bonds(cutoffs={"default":[0, 1.5], "C-C": [0, 1.5]})
# epoxide.view()

# Anything with 1 or 2 atoms is an edge atom, anything with 3 is basal plane.
bond_counts, bonds_present, bonds_present_idxs, bonds_with = graphene_oxide.get_bonds_metadata()
# Visualize to confirm correct edge detection
for key, val in bond_counts.items():
    if val != 3:
        graphene.symbols[key] = 'Ar'
graphene.view()

positions = graphene_oxide.positions
cell_x = graphene_oxide.cell_lengths[0]
doped_carbons = []

new_bonds = []
new_bond_types = []
# bonds_to_remove = []
random.seed(578294)
for i in range(graphene_oxide.num_atoms):
    atom_pos = positions[i]
    rand_num = random.random()

    # Check if undoped half.
    if atom_pos[0] > cell_x/2:
        continue

    # Check if edge atom
    if bond_counts[i] != 3:
        # Carbonyl
        if (rand_num > 0.5 and rand_num < 0.7) and i not in doped_carbons:
            graphene_oxide.insert_atoms(carboxyl, position=atom_pos+np.array([0,0,1.5]),
                                        new_bonds=np.array([[i, 0]]),
                                        new_bond_types=np.array([['C', 'C']])
                                        ) 
            doped_carbons.extend([i])
        else:
            continue

    # Hydroxyl
    if (rand_num > 0.8 and rand_num < 0.9) and i not in doped_carbons:
        graphene_oxide.insert_atoms(hydroxyl, position=atom_pos+np.array([0,0,1.128]),
                                    new_bonds=np.array([[i, 0]]),
                                    new_bond_types=np.array([['C', 'O']]))
        doped_carbons.extend([i])

    # Epoxide
    if (rand_num > 0.9 and rand_num < 1.0) and i not in doped_carbons:
        # Ugly and only allows for epoxide to check 1 other atom.
        neighbor_atom = [j for j in bonds_present[i][0] if j != i][0]
        if neighbor_atom in doped_carbons:
            continue
        neighber_pos = positions[neighbor_atom]
        avg_pos = (atom_pos+neighber_pos) / 2
        graphene_oxide.insert_atoms(epoxide, position=avg_pos+np.array([0,0,0.7]),
                                    new_bonds=np.array([[i, 0], [neighbor_atom, 0]]),
                                    new_bond_types=np.array([['C', 'O'], ['C', 'O']])
                                    ) 
        # bonds_to_remove.append(bonds_present_idxs[i][0])
        doped_carbons.extend([i])

# new_bond_set = np.delete(graphene_oxide.bonds, bonds_to_remove, axis=0)
# graphene_oxide.bonds = new_bond_set
graphene_oxide.view()


# # --- Passivate Graphene Atoms
# # Edge graphene atoms with 2 or less bonds needs hydrogen attatched.
# symbols = graphene_oxide.symbols
# positions = graphene_oxide.positions
# bond_counts, bonds_present, bonds_present_idxs, bonds_with = graphene_oxide.get_bonds_metadata()

# h1 = Atoms(
#     symbols=np.array(["H"]),
#     positions=np.array([
#         [0.0, 0.0, 0.0],
#     ])
# )

# h2 = Atoms(
#     symbols=np.array(["H", "H"]),
#     positions=np.array([
#         [0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0]
#     ])
# )

# for i in range(graphene_oxide.num_atoms):
#     atom_pos = positions[i]
#     symbol = symbols[i]
#     bond_count = bond_counts[i]

#     # Tests to visualize passivated atoms.
#     # if symbol == "C" and bond_count<3:
#     #     graphene_oxide.symbols[i] = "Ar"

#     if symbol == "C" and bond_count==1:
#         graphene_oxide.insert_atoms(h2, position=atom_pos+np.array([0,0,1.5]),
#                         new_bonds=np.array([[i, 0], [i, 1]]),
#                         new_bond_types=np.array([['C', 'H'], ['C', 'H']])
#                         )         
#     if symbol == "C" and bond_count==2:    
#         graphene_oxide.insert_atoms(h1, position=atom_pos+np.array([0,0,1.5]),
#                         new_bonds=np.array([[i, 0]]),
#                         new_bond_types=np.array([['C', 'H']])
#                         )

# # graphene_oxide.view()

# ---- Simulation Box
sim_box = SimulationBox.create_from_atoms(graphene_oxide)
sim_box.insert_atoms(actga, position=sim_box.unit_cell.get_center_of_cell())
sim_box.view()

# ---- Solvent Generation
# TODO: Allow for specific opacities by molecule in Visulaization.
# TODO: Check for overlap between exisitng molecules and solvent molecules as they are inserted.
# Calculate approx num. of water molecules to get density right. Can still relax in NPT before sim.
# CWe can subdivide the grids to speed up the overlap caclulation. 
cell_lengths = sim_box.cell_lengths
cell_angles = sim_box.cell_angles
ppx = 15

xyz_all = np.array(list(itertools.product(np.linspace(0,1,ppx)[:-1], np.linspace(0,1,ppx)[:-1], np.linspace(0,1,ppx)[:-1])))
xyz_all = xyz_all*np.array(cell_lengths)

# This is slow..... and broken... fix it.
# Sorting positions somehow breaks the overlap fxn... figure this out.
valid_xyz_all = []
for xyz in xyz_all:
    if not sim_box.overlap_at_point(xyz, overlap_dist=3.0):
        valid_xyz_all.extend([xyz])
valid_xyz_all = np.array(valid_xyz_all)

h2o = Atoms.bind_from_file("../water.xyz")
h2o.cell_lengths = np.array([5,5,5])
h2o.cell_angles = np.array([90,90,90])
h2o.calculate_bonds()

# ArAtom = Atoms(
#     symbols=np.array(["K"]),
#     positions=np.array([[0.0, 0.0, 0.0]])
# )
# ArAtom.cell_lengths = np.array([5,5,5])
# ArAtom.cell_angles = np.array([90,90,90])

for xyz in valid_xyz_all:
    sim_box.insert_atoms(h2o, position=xyz+np.array([0,0,0]))

# # ---- View
sim_box.view(opacity=0.5)