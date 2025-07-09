from typing import Optional, Tuple, List
from lgmc.init.prob import Prob
from lgmc.init.lattice import Lattice
from lgmc.dynamics.kawasaki import KawasakiMC


class NucleationSimulator:
    """
    Class of the LGMC simulator for Nucleation

    Attributes:
        r (int): Lattice radius in each dimension.
        conc (float): Particle concentration (0.0 to 1.0).
        pbc (list of boolean): Periodic boundary condition flags for x, y, z.
        sys (str): 'homo' or 'hete'.
        seed (int): Random seed.
        temp (float): Reduced temperature T* = T / T_c.
        eps_nn (float): Nearest neighbor interaction energy (set to 1.0 by default).
        eps_s (Optional[float]): Surface interaction energy (for heterogeneous systems, relative to eps_nn).
        mu (Optional[float]): Chemical potential (required for Glauber dynamics, relative to eps_nn).
        NN (int): Number of nearest neighbors.
        sys (str): System type ('homo' or 'hete').
        mode (str): Dynamics mode ('glauber' or 'kawasaki').
    """
    def __init__(self,
                 r: int, conc: float,
                 pbc: List[bool] = [True, True, True],
                 sys: str = 'homo', temp: float=0.1,
                 eps_NN: float = 1.0,
                 num_NN: int = 6,
                 eps_s: Optional[float] = None,
                 mode: str = 'kawasaki',
                 mu: Optional[float] = None,seed: int = 1234):
        
        self.lattice_obj = Lattice(r=r, conc=conc, pbc=pbc, sys=sys, seed=seed)
        self.prob_obj = Prob(temp=temp, eps_NN=eps_NN, num_NN=num_NN, eps_s=eps_s,
                             sys=sys, mode=mode, mu=mu)
        
        self.mc = KawasakiMC(lattice_obj=self.lattice_obj, prob_obj=self.prob_obj)

    def run(self, n_steps: int = 1,
            verbose: bool = False,
            n_sample: int = 1,
            save_dir: Optional[str] = None) -> None:
        """
        Args:
            n_steps (int): number of MC steps 
            verbose (bool): print progress bar.
            save_dir (Optional[str]): extxyz 저장할 디렉토리. None이면 저장 안함.
            n_sample (int): number of sampling steps 

        Returns:
            None
        """

        self.mc.step(n_steps=n_steps, verbose=verbose, n_sample=n_sample, save_dir=save_dir)


if __name__=='__main__':
    r = 10
    conc = 0.1
    sys = 'hete'
    pbc = [True, True, True] if sys == 'homo' else [True, True, False]
    temp = 0.2
    eps_NN = 1.0
    eps_s = 0.1 * eps_NN
    mode = 'kawasaki'
    
    simulator = NucleationSimulator(r=r, conc=conc, pbc=pbc, sys=sys,
                                    temp=temp, eps_NN=eps_NN, eps_s=eps_s, mode=mode)

    simulator.run(n_steps=1000, verbose=True, n_sample=1, save_dir='mc_output')