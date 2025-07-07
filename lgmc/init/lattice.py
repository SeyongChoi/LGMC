import numpy as np
from lgmc.utils.to_xyz import to_xyz
from lgmc.utils.pbc import apply_pbc
from typing import Optional, Tuple, List

class Lattice:
    """
    Class to generate a 3D lattice for LGMC simulations.

    Supports:
    - Homogeneous sys: Random particle distribution in 3D.
    - Heterogeneous sys: Fixed surface layer at z = -r with no periodicity in z.

    Attributes:
        r (int): Lattice radius in each dimension.
        conc (float): Particle concentration (0.0 to 1.0).
        pbc (list of boolean): Periodic boundary condition flags for x, y, z.
        sys (str): 'homo' or 'hete'.
        seed (int): Random seed.
        lattice (np.ndarray): 3D array representing the lattice.
        n_particle (int): Total number of particles placed.
    """

    def __init__(self, r: int, conc: float,
                 pbc: List[bool] = [True, True, True],
                 sys: str = 'homo', seed: int = 1234):
        self.r = r
        self.dim = 2 * (self.r + 1) + 1
        self.conc = conc
        self.sys = sys
        self.seed = seed
        self.pbc = np.array(pbc, dtype=bool)

        if self.sys == 'hete' and self.pbc[2]:
            raise ValueError("Heterogeneous lattice must not use periodic boundary in z-direction.")

        if self.sys == 'homo':
            self.lattice, self.n_particle = self._init_lattice_homo()
        elif self.sys == 'hete':
            self.lattice, self.n_particle = self._init_lattice_hete()
        else:
            raise ValueError("Sys must be either 'homo' or 'hete'.")

        # self.lattice = self._apply_pbc()
        

    def _apply_pbc(self) -> np.ndarray:
        """Apply periodic boundary conditions to the lattice."""
        return apply_pbc(self.lattice, self.pbc)
        

    def _init_lattice_homo(self) -> Tuple[np.ndarray, int]:
        """
        Initialize homogeneous lattice with randomly distributed particles.

        Returns:
            Tuple[np.ndarray, int]: The lattice array and number of inserted particles.
        """
        
        rng = np.random.default_rng(self.seed)

        grid = np.indices((self.dim - 2, self.dim - 2, self.dim - 2)).reshape(3, -1).T + 1
        n_insert = int(self.conc * len(grid))
        print(f"[homo] total sites: {len(grid)}, inserted: {n_insert}, actual conc: {n_insert / len(grid):.4f}")

        chosen = grid[rng.choice(len(grid), size=n_insert, replace=False)]

        lattice = np.zeros((self.dim, self.dim, self.dim), dtype=np.float64)
        lattice[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = 1

        return lattice, n_insert

    def _init_lattice_hete(self) -> Tuple[np.ndarray, int]:
        """
        Initialize heterogeneous lattice with surface layer at z = -r.

        Returns:
            Tuple[np.ndarray, int]: The lattice array and number of inserted particles.
        """
        
        rng = np.random.default_rng(self.seed)

        lattice = np.zeros((self.dim, self.dim, self.dim), dtype=np.float64)
        surface_z_idx = 1
        lattice[:, :, surface_z_idx] = 2  # surface layer at z = 1 (== z = -r)

        grid = np.indices((self.dim - 2, self.dim - 2, self.dim - 2)).reshape(3, -1).T + 1
        available_sites = grid[grid[:, 2] != 1]  # exclude surface

        n_insert = int(self.conc * len(available_sites))
        print(f"[hete] total sites: {len(available_sites)}, inserted: {n_insert}, actual conc: {n_insert / len(available_sites):.4f}")

        chosen = available_sites[rng.choice(len(available_sites), size=n_insert, replace=False)]
        lattice[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = 1

        return lattice, n_insert

    def to_xyz(self, filename: Optional[str] = None) -> str:
        """
        Convert lattice to XYZ format string. Optionally write to file.

        Args:
            filename (Optional[str]): Path to save the XYZ file. If None, only return string.

        Returns:
            str: XYZ format string.
        """
        length = self.dim - 2
        comment = f'lattice="{length}  0  0  0  {length}  0  0  0  {length}"  origin="1 1 1" properties=species:S:1:pos:R:3'

        return to_xyz(self.dim, self.lattice, filename, comment)
        


if __name__=='__main__':
    r = 10   # 버퍼까지 치면 20 + 1 + 2 --> 23 // 버퍼 빼면 21
    conc = 0.01
    pbc = [True, True, False]
    sys = 'hete'
    lattice = Lattice(r=r, conc=conc, pbc=pbc, sys=sys)

    print(lattice.n_particle)
    lattice.to_xyz(f'init_{sys}.xyz')

    count = 0
    for i in range(1,22):
        count += 1
    print(count)