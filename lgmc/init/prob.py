import numpy as np
from typing import Optional
from lgmc.utils.constant import Beta_c


class Prob:
    """
    Class to initialize local energies and transition probabilities 
    for lattice gas Monte Carlo simulations in reduced units.

    Attributes:
        temp (float): Reduced temperature T* = T / T_c.
        eps_nn (float): Nearest neighbor interaction energy (set to 1.0 by default).
        eps_s (Optional[float]): Surface interaction energy (for heterogeneous systems, relative to eps_nn).
        mu (Optional[float]): Chemical potential (required for Glauber dynamics, relative to eps_nn).
        NN (int): Number of nearest neighbors.
        sys (str): System type ('homo' or 'hete').
        mode (str): Dynamics mode ('glauber' or 'kawasaki').
        hi (np.ndarray): Local energy table.
        tprob (Optional[np.ndarray]): Transition probability table (for Glauber mode).
    """
    def __init__(
        self,
        temp: float,
        eps_NN: float = 1.0,
        num_NN: int = 6,
        sys: str = 'homo',
        eps_s: Optional[float] = None,
        mode: str = 'kawasaki',
        mu: Optional[float] = None
    ):
        # Number of nearest neighbors
        self.NN = num_NN

        # Simulation temperature
        self.temp = temp                   # T* = T / Tc
        self.beta = Beta_c / self.temp     # beta* = beta_c / T*

        # Interaction energies
        self.eps_nn = eps_NN
        self.eps_s = eps_s
        self.mu = mu

        # System type
        self.sys = sys
        if sys == 'hete' and eps_s is None:
            raise ValueError("Heterogeneous system must define surface interaction energy eps_s.")

        # Dynamics mode
        self.mode = mode
        if mode == 'glauber' and mu is None:
            raise ValueError("Glauber dynamics requires chemical potential (mu).")

        # Initialize local energy table
        if sys == 'homo':
            self.hi = self._init_local_h_homo()
        elif sys == 'hete':
            self.hi = self._init_local_h_hete()
        else:
            raise ValueError("sys must be either 'homo' or 'hete'.")

        # Initialize transition probability table (for Glauber dynamics)
        if mode == 'glauber':
            if self.sys == 'homo':
                self.tprob = self._init_trans_prob_homo()
            elif self.sys == 'hete':
                self.tprob = self._init_trans_prob_hete()
        elif mode == 'kawasaki':
            self.tprob = None
        else:
            raise ValueError("mode must be either 'kawasaki' or 'glauber'.")

    def _init_local_h_homo(self) -> np.ndarray:
        """
        Initialize local energy hi[ci, cj_sum] for homogeneous system.

        lattice gas model:
            ci = 0 → unoccupied
            ci = 1 → occupied

        Returns:
            np.ndarray: shape (2, NN+1)
        """
        hi = np.zeros((2, self.NN + 1), dtype=np.float64)

        for ci in range(2):
            # maximum number of NN for ci = self.NN,
            # BUT we need local h even when ci doesn't have neighbors
            for cj_sum in range(self.NN + 1):
                # multiply 0.5: avoid double counting for pair-wise interaction
                hi[ci, cj_sum] = -0.5 * self.eps_nn * ci * cj_sum

        return hi

    def _init_local_h_hete(self) -> np.ndarray:
        """
        Initialize local energy hi[ci, cs, cj_sum] for heterogeneous system.

        lattice gas model:
            ci = 0 → unoccupied | 1 → occupied
            cs = 0 → not in contact with surface | 1 → contact with surface

        Returns:
            np.ndarray: shape (2, 2, NN+1)
        """
        hi = np.zeros((2, 2, self.NN + 1), dtype=np.float64)

        for ci in range(2):
            for cs in range(2):
                # maximum number of NN for ci = self.NN,
                # BUT we need local h even when ci doesn't have neighbors
                for cj_sum in range(self.NN + 1):
                    # multiply 0.5: avoid double counting for pair-wise interaction
                    hi[ci, cs, cj_sum] = -0.5 * self.eps_nn * ci * cj_sum - self.eps_s * ci * cs

        return hi
                
    def _init_trans_prob_homo(self) -> np.ndarray:
        """
        Initialize Glauber transition probability tprob[ci, cj_sum]
        for homogeneous system.

        Returns:
            np.ndarray: shape (2, NN+1)
        """
        tprob = np.zeros((2, self.NN + 1), dtype=np.float64)

        for ci in range(2):
            for cj_sum in range(self.NN + 1):
                # delH = H_new - H_old
                # ci=0: 0 → 1 transition → del_h = hi(1, cj_sum) - hi(0, cj_sum) - mu
                # ci=1: 1 → 0 transition → del_h = hi(0, cj_sum) - hi(1, cj_sum) + mu
                del_h = self.hi[1 - ci, cj_sum] - self.hi[ci, cj_sum] - (1 - 2 * ci) * self.mu
                tprob[ci, cj_sum] = np.exp(-self.beta * del_h)

        return tprob

    def _init_trans_prob_hete(self) -> np.ndarray:
        """
        Initialize Glauber transition probability tprob[ci, cs, cj_sum]
        for heterogeneous system.

        Returns:
            np.ndarray: shape (2, 2, NN+1)
        """
        tprob = np.zeros((2, 2, self.NN + 1), dtype=np.float64)

        for ci in range(2):
            for cs in range(2):
                for cj_sum in range(self.NN + 1):
                    # delH = H_new - H_old
                    # ci=0: 0 → 1 transition → del_h = hi(1, cs, cj_sum) - hi(0, cs, cj_sum) - mu
                    # ci=1: 1 → 0 transition → del_h = hi(0, cs, cj_sum) - hi(1, cs, cj_sum) + mu
                    del_h = self.hi[1 - ci, cs, cj_sum] - self.hi[ci, cs, cj_sum] - (1 - 2 * ci) * self.mu
                    tprob[ci, cs, cj_sum] = np.exp(-self.beta * del_h)

        return tprob

if __name__=='__main__':
    temp = 0.5 #K
    eps_NN = 1.0 #kJ/mol
    eps_s = 0.1*eps_NN
    sys='hete'
    mode='kawasaki'
    mu = 0.4*eps_NN
    prob = Prob(temp=temp, eps_NN=eps_NN, eps_s=eps_s, sys=sys, mode=mode, mu=mu)

    print(prob.hi)
    print(prob.tprob)
        
