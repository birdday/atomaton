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
    # Other versions may work, but graphics have been tempermental for me.
    install_requires=["numpy==1.22.4", "ase", "vtk==9.2.6", "mayavi==4.8.1", "imageio", "PyQt5", "configobj",
                      "matplotlib==3.8.2"],
    tests_require=["pytest"],
)
