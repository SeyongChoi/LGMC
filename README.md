# LGMC

**Simulator of Lattice Gas Model Monte Carlo (LGMC)** implemented in **Python and Cython**.  
This project provides a fast and extensible Monte Carlo simulation framework for lattice gas systems,  
supporting both canonical and grand canonical ensembles.

## Overview

`LGMC` is designed to explore thermodynamic and kinetic behaviors of lattice gas systems.  
It allows the study of phase transitions, adsorption, diffusion, and cooperative effects using  
Monte Carlo methods based on **Glauber/Kawasaki dynamics** and **Metropolis algorithms**.

The Cython-optimized backend accelerates spin or occupancy updates on large lattice grids,  
making it suitable for physical and chemical simulations on surfaces or porous materials.

-----
**Status:** *In progress* 
- [ ] Implement the Glauber Dynamics module
-----

## Usage
```bash
git clone https://github.com/SeyongChoi/LGMC.git
cd LGMC
pip install -e .

# run the simulation
lgmc --input_file config/test.yaml

