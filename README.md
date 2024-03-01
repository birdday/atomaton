[![banner](assets/atomaton.png)]()

ATOMOTON

Overview: Alternative to topotools, with more flexibility.

Features:

- Caclulate Bonds, Angles, Diherals, Impropers
- Visualize structures and programatically generate images and gifs
- Build supercells, and insert molecules into crystals
- Generate LAMMPS input files

To Do List:

- Porous Material Characterization (e.g., Pore Size Distribution, Surface Area (Geometric, Connolly), Powder X-Ray Diffraction, Helium Void Fraction, RDF?).
- Refactor and add classes as needed (or repurpose ASE classes if possible)
  - Bond class? Simulation class (molecule ids, system params?), Forcefield class.
- Doc Strings
- Examples
- Travis CI/CD (?)
- Use RDKit to determine bond types, update drawn bonds to match style.
- Charge approximations (ex. eqEQ, import from dft). Need to better handle atom reasssignment if we are tracking more than positions. Good argument for shift to classes (this is why we design early!).
- SMILES/SELFIES support
- Generate computational chmeistry tool kit on top of ase stuff
- Radial Distribution Function Calculator / Partial Structure
- Energy Calculators
- Solvent Filling Algorithm (Water / Generic Solvent Molecule, Mixed Solvents?).

Stuff to Read:

- Determining bond orders from distances non-trival, and solution didn't even exist in ase or RDKit until recently. This is an open source program to do that: https://greglandrum.github.io/rdkit-blog/posts/2022-12-18-introducing-rdDetermineBonds.html. It also demos a drawing function using IPythonConsole.
- MolecularDynamics in ASE: https://wiki.fysik.dtu.dk/ase/ase/md.html
- Harmonics Calculator: https://wiki.fysik.dtu.dk/ase/ase/calculators/harmonic.html (Also demos FF objects.)
- Structure Optimization: https://wiki.fysik.dtu.dk/ase/ase/optimize.html (Does not require FF?)

Exmaple Ideas:
- Basics: Basics of ASE / ASE Essentials (for Atomaton), Generating Simulation Boxes, Implicit vs Explict Solvent Environments, Thermodynamic Ensembles, Intro to Forcefields, Thermostats / Barostats, Initial Trajectories, Identifying Interaction Terms, N-body interactions,
- Heat Transfer (Green-Kubo Function)
- Solvents: Oil / Water Simulations (Oil droplet formation), Molecule solvation (Molecule interaction in solvent), Self-Diffusion Calculator
- Property Calculator: Diffusion Coefficient, 
- Proteins: Importing Large Molecules
- Visualization
- Advanced(?) Concepts: Reactive MD FFs, Polarizable FF MD
- Computational Biology: Molecular Docking, Protein Folding
- Misc.: Partial charge estimators, Generating LAMMPS Input File
