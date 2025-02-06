#!/usr/bin/env python

from distutils.core import setup

setup(
    name="atomaton",
    version="0.1.0",
    description="Tools for building lammps datafiles",
    author="Brian A. Day",
    author_email="22bday@gmail.com",
    url="https://github.com/birdday/atomaton",
    packages=["atomaton"],
    # NOTE: Pinning version of mayavi causes sphinx docs to fail. On ARM Mac, I use mayavi==4.8.1.
    # Newer combinations of packages may work, but require some troubleshooting.
    install_requires=["numpy==1.22.4", "ase", "vtk==9.2.6", "mayavi", "imageio", "PyQt5", "configobj",
                      "matplotlib==3.8.2"],
    tests_require=["pytest"],
)
