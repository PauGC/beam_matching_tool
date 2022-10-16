# Beam Optics and Matching Tool for the FLASHForward facility at FLASH (DESY).

## Overview

GUI-based Python tool designed to evaluate and dynamically interact with the magnetic lattice of the FLASH 
accelerator to match the beam envelope to particular experimental requirements. The code uses the beam-transport and 
matching routines of the Ocelot toolkit, and requires the use of a separate module containing the definition of a 
MagneticLattice class describing the sequence of beamline elements (flash_lattice). Interaction with machine components 
is implemented with the Python bindings using the C++ DOOCS client API (pydoocs). 

![GUI snapshot](gui_snapshot.png?raw=true "Title")

## Usage

An indepth description of the usage can be currently found in: https://confluence.desy.de/display/XTDS/python+optics+GUI

## TODO
- [ ] In target QComboBox items list (match), set the last element of the list to the default
- [ ] When loading a new optics file, choose which magnets to use
- [ ] When pushing the magnet values to the linac, choose which magnets to load (might not be the full list of magnets...
- [ ] Check the swesch file format... something wrong with it!
