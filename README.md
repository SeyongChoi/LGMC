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
- [ ] Wetting simulation mode
-----

## Usage
1. Installation
```bash
git clone https://github.com/SeyongChoi/LGMC.git
cd LGMC
pip install -e .
```
2. Make input configurations file (.yaml)
```yaml
# Simulation Mode
mode: 'nucleation'      # 'nucleation' or 'wetting [not yet]'

# For system initialization
r: 10                   # size of lattice(length = 2*r + 1), default: 10
conc: 0.1               # concentration of particle in lattice, default: 0.1
sys: 'homo'             # nulceation system, 'homo' or 'hete'

# For dynamics,
dynamics: 'kawasaki'    # the Method of Dynamics, 'kawasaki' or 'glauber [not yet]'
pbc: [True, True, True] # PBC of each axis [x, y, z], default: [True, True, True]
temp: 0.2               # the dimensionless temperature (T/Tc), default: 0.1
eps_NN: 1.0             # the dimensionless epsilon_nn, default: 1.0
eps_s: 0.1              # the dimensionless epsilon_surf, default: 0.0
mu: None                # the coexistence chemcial potential, default: None

# For MC run,
n_steps: 10000          # # of MC steps, default: 1000
n_sample: 100           # # of sampling interval steps, default: 100
verbose: True           # Checking or Not [tqdm], default: False
save_dir: 'mc_out'      # Directory to save lattice [.xyz], default: './mc_out'

# logging
log: 'log.log'          # log file name, default: {mode}_{sys}_R{r}c{conc}t{temp}_{dynamics}.log
```
3. Run the simulation
```bash
# run the simulation
lgmc --input_file configs/input_nucleation.yaml
```

