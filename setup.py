#!/usr/bin/env python

from distutils.core import setup
setup(
    name = 'atomaton',
    version = '0.1.0',
    description = 'Tools for building lammps datafiles',
    author = 'Brian A Day',
    author_email = 'brd84@pitt.edu',
    url = 'https://github.com/birdday/atomaton',
    packages = ['atomaton'],
    install_requires=['numpy', 'ase', 'mayavi', 'imageio'],
    tests_require=['pytest']
)
