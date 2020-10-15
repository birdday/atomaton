#!/usr/bin/env python

from distutils.core import setup
setup(
    name = 'lammps_tools',
    version = '0.1.0',
    description = 'Tools for building lammps datafiles',
    author = 'Brian A Day',
    author_email = 'brd84@pitt.edu',
    url = 'https://github.com/birdday/LAMMPS_tools',
    packages = ['lammps_tools'],
    install_requires=['numpy','ase'],
    tests_require=['pytest']
)
