import os
import time
import yaml
import logging
import argparse

from lgmc.simulator.nucleation import NucleationSimulator

def flexible_path(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(os.getcwd(), path)

def main():
    # ---------------------------------------------------------------------------------------------
    # Argument parsing
    parser = argparse.ArgumentParser(description="LGMC simulation")
    parser.add_argument(
        '--input',
        type=str,
        default='input.yaml',
        help='Path of the YAML configuration file (default: input.yaml)'
    )
    args = parser.parse_args()

    # Absolute path or Relative path, both of them is ok
    configs_path = flexible_path(args.input)

    # ---------------------------------------------------------------------------------------------
    # Allocate the value from yaml to variables
    # Read the configs(.yaml)
    with open(configs_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # Simulation mode,
    mode = config['mode']          if 'mode' in config else 'nucleation'
    # For system,
    r = config['r']                if 'r' in config else 10
    conc = config['conc']          if 'conc' in config else 0.1
    sys = config['sys']            if 'sys' in config else 'homo'
    l = 2 * r + 1
    lattice = f'{l} x {l} x {l} [r: {r}]'
    # For dynamics,
    pbc = config['pbc']            if 'pbc' in config else [True, True, True]
    temp = config['temp']          if 'temp' in config else 0.1
    eps_NN = config['eps_NN']      if 'eps_NN' in config else 1.0
    eps_s = config['eps_s']        if 'eps_s' in config else 0.0
    dynamics = config['dynamics']  if 'dynamics' in config else 'kawasaki'
    mu = config['mu']              if 'mu' in config else None
    # For MC run,
    n_steps = config['n_steps']    if 'n_steps' in config else 1000
    n_sample = config['n_sample']  if 'n_sample' in config else 100
    verbose = config['verbose']    if 'verbose' in config else False
    save_dir = config['save_dir']  if 'save_dir' in config else './mc_out'
    save_dir = flexible_path(save_dir)

    # For logging,
    log_path = config['log']       if 'log' in config else f'{mode}_{sys}_R{r}c{conc}t{temp}_{dynamics}.log'
    log_path = flexible_path(log_path)
    
    # ---------------------------------------------------------------------------------------------
    # Make the Logging
    # 1) Create logger & set the level
    logger = logging.getLogger('LGMC_Simulation')
    logger.setLevel(logging.INFO)

    # 2) Set the printing format
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3) Set the console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 4) Set the file handler
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # ---------------------------------------------------------------------------------------------
    # Check the Input Information
    logger.info('-' * 162)
    logger.info('|{:^160}|'.format('Conditions of LGMC Simulation'))
    logger.info('-' * 162)
    logger.info(f"|{'Mode':^21}| {str(mode):<136} |")
    logger.info(f"|{'Lattice':^21}| {str(lattice):<136} |")
    logger.info(f"|{'Concentration':^21}| {str(conc):<136} |")
    logger.info(f"|{'System':^21}| {str(sys):<136} |")
    logger.info(f"|{'PBC':^21}| {str(pbc):<136} |")
    logger.info(f"|{'Temperature':^21}| {str(temp):<136} |")
    logger.info(f"|{'eps_{NN}':^21}| {str(eps_NN):<136} |")
    logger.info(f"|{'eps_{surf}':^21}| {str(eps_s):<136} |")
    logger.info(f"|{'Dynamics':^21}| {str(dynamics):<136} |")
    logger.info(f"|{'mu':^21}| {str(mu):<136} |")
    logger.info(f"|{'# of MC step':^21}| {str(n_steps):<136} |")
    logger.info(f"|{'# of sampling':^21}| {str(n_sample):<136} |")
    logger.info(f"|{'Save Dir':^21}| {str(save_dir):<136} |")
    logger.info(f"|{'Verbose':^21}| {str(verbose):<136} |")
    logger.info('-' * 162)
    logger.info('\n')
    # ---------------------------------------------------------------------------------------------
    # Check the Input Information
    # ---------------------------------------------------------------------------------------------
    time_s = time.time()
    logger.info('=' * 162)
    logger.info(f"[{mode}] Run the LGMC Simulation")
    if mode == 'nucleation':
        simulator = NucleationSimulator(r=r, conc=conc, pbc=pbc, sys=sys,
                                        temp=temp, eps_NN=eps_NN,
                                        eps_s=eps_s,mu=mu,
                                        mode=dynamics)
    else:
        raise ValueError('Only "nucleation" is possible. Other mode is not implemented yet.')
    
    simulator.run(n_steps=n_steps, verbose=verbose, n_sample=n_sample, save_dir=save_dir)
    time_e = time.time()
    logger.info(f'Finish the LGMC Simulation(took: {time_e - time_s:.2f}s[{(time_e - time_s)/60:.2f}min])')


if __name__=='__main__':
    main()

