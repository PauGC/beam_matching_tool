# Beam Optics and Matching Tool for the FLASHForward facility at FLASH (DESY).

## Overview

GUI-based Python tool designed to evaluate and dynamically interact with the magnetic lattice of the FLASH 
accelerator to match the beam envelope to particular experimental requirements. The code uses the beam-transport and 
matching routines of the Ocelot toolkit. Interaction with machine componenets is performed with the Python client APIs
for DOOCS (pydoocs). 

![GUI snapshot](gui_snapshot.png?raw=true "Title")

## Usage

I'll complete the README file when somebody starts using the scripts... :-)


## TODO
- [ ] In target QComboBox items list (match), set the last element of the list to the default
- [ ] When loading a new optics file, choose which magnets to use
- [ ] When pushing the magnet values to the linac, choose which magnets to load (might not be the full list of magnets...
- [ ] Check the swesch file format... something wrong with it!
