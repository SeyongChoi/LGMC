import os
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List
from lgmc.init.prob import Prob
from lgmc.init.lattice import Lattice
from lgmc.utils.to_xyz import to_xyz
from lgmc.dynamics.kawasaki import move_hete, move_homo, seed_c_rand


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
        
        # seed C rand()
        seed_c_rand(seed)
        # initialize lattice
        self.lattice_obj = Lattice(r=r, conc=conc, pbc=pbc, sys=sys, seed=seed)
        self.lattice = np.require(self.lattice_obj.lattice,
                                  dtype=np.int64,
                                  requirements={'C', 'O', 'W'})
        self.pbc = self.lattice_obj.pbc
        self.sys = self.lattice_obj.sys
        self.dim = self.lattice.shape[0]

        self.prob_obj = Prob(temp=temp, eps_NN=eps_NN, num_NN=num_NN, eps_s=eps_s,
                             sys=sys, mode=mode, mu=mu)
        self.beta = self.prob_obj.beta
        if self.sys == 'homo':
            self.hi_homo = self.prob_obj.hi
        elif self.sys == 'hete':
            self.hi_hete = self.prob_obj.hi
        else:
            raise ValueError()

        # 주변 이웃 오프셋: 6-방향 (up, down, left, right, front, back)
        self.neighbor_offsets = np.require(np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]), dtype=np.int64, requirements={'C', 'O', 'W'})
        
    def step(self, n_steps: int = 1,
             verbose: bool = False,
             save_dir: Optional[str] = None,
             n_sample: Optional[int] = 1) -> None:
        """
        Perform Kawasaki MC steps and optionally save extxyz files per step.

        Args:
            n_steps (int): number of MC steps (1 step = n_particle attempts).
            verbose (bool): print progress bar.
            save_dir (Optional[str]): extxyz 저장할 디렉토리. None이면 저장 안함.

        Returns:
            None
        """
        rng = np.random.default_rng()
        it = tqdm(range(n_steps), desc='Kawasaki MC steps') if verbose else range(n_steps)

        for step_idx in it:
            # 1. 버퍼 층을 제외한 모든 입자 좌표 (ci = 1) 수집 후 셔플
            particle_positions = np.argwhere(self.lattice[1:-1, 1:-1, 1:-1] == 1) + 1 
            positions = np.require(particle_positions.astype(np.int64),
                                            dtype=np.int64,
                                            requirements={'C', 'O', 'W'})
            rng.shuffle(positions)

            # print(particle_positions)
            if self.sys == 'homo':
                accepted=move_homo(self.lattice,
                          self.hi_homo,
                          self.neighbor_offsets,
                          self.beta,
                          self.pbc[0], self.pbc[1], self.pbc[2],
                          positions, particle_positions.shape[0])
            elif self.sys == 'hete':
                accepted=move_hete(self.lattice,
                          self.hi_hete,
                          self.neighbor_offsets,
                          self.beta,
                          self.pbc[0], self.pbc[1], self.pbc[2],
                          positions, particle_positions.shape[0])
            else:
                raise ValueError()
            
            if save_dir is not None and (step_idx % n_sample == 0):
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{save_dir}/lattice_step_{step_idx:07d}.extxyz"
                self.to_xyz(step=step_idx, filename=fname)

            if verbose:
                it.set_postfix(accepted=accepted)
    
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

        self.step(n_steps=n_steps, verbose=verbose, n_sample=n_sample, save_dir=save_dir)
    
    def _get_neighbor_sum(self, x: int, y: int, z: int) -> int:
        """
        현재 위치 (x,y,z)의 입자 상태 ci와 이웃 cj들의 합(cj_sum)을 계산.
        Heterogeneous일 때 표면 접촉 여부 cs 고려 가능.

        Returns:
            int: cj_sum (occupied 이웃 개수)
        """
        cj_sum = 0
        lx, ly, lz = self.lattice.shape

        for dx, dy, dz in self.neighbor_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            # Wrap or skip out-of-bound
            if self.pbc[0]:
                nx = (nx - 1) % (lx - 2) + 1
            elif nx < 1 or nx >= lx - 1:
                continue

            if self.pbc[1]:
                ny = (ny - 1) % (ly - 2) + 1
            elif ny < 1 or ny >= ly - 1:
                continue

            if self.pbc[2]:
                nz = (nz - 1) % (lz - 2) + 1
            elif nz < 1 or nz >= lz - 1:
                continue


            if self.lattice[nx, ny, nz] in (0, 1):
                cj_sum += self.lattice[nx, ny, nz]

        return cj_sum

    def _is_surface_contact(self, x: int, y: int, z: int) -> int:
        """
        Heterogeneous 시스템에서 위치가 표면과 접촉하는지 여부 판단.
        surface layer: z == 1 으로 가정 (Lattice 클래스 기준)
        """
        return int(self.sys == 'hete' and self.lattice[x, y, z - 1] == 2)
    
    def _calculate_total_energy(self) -> float:
        """
        현재 lattice 전체 local energy 합 계산.
        각 site ci 상태, 주변 cj_sum, (hetero면 cs) 이용해 hi에서 계산.

        Returns:
            float: 전체 에너지
        """
        total_energy = 0.0
        for x in range(1, self.dim - 1):
            for y in range(1, self.dim - 1):
                for z in range(1, self.dim - 1):
                    ci = int(self.lattice[x, y, z])
                    if ci != 1:
                        continue
                    cs = int(self._is_surface_contact(x, y, z)) if self.sys == 'hete' else 0
                    cj_sum = int(self._get_neighbor_sum(x, y, z))
                    total_energy += int(self.hi_hete[ci, cs, cj_sum]) if self.sys == 'hete' else int(self.hi_homo[ci, cj_sum])

        return total_energy


    def to_xyz(self, step: int, filename: Optional[str] = None) -> str:
        """
        현재 lattice 상태를 extxyz 포맷으로 저장.

        Returns:
            str: extxyz 포맷 문자열
        """
        length = self.dim - 2
        total_energy = self._calculate_total_energy()

        comment = (
            f'lattice="{length}  0  0  0  {length}  0  0  0  {length}" '
            f'origin="1 1 1" properties=species:S:1:pos:R:3  '
            f'energy={total_energy:.6f}  step={step}'
        )

        return to_xyz(self.dim, self.lattice, filename, comment)
    
if __name__=='__main__':
    r = 20
    conc = 0.1
    sys = 'hete'
    pbc = [True, True, True] if sys == 'homo' else [True, True, False]
    temp = 0.2
    eps_NN = 1.0
    eps_s = 1.1 * eps_NN
    mode = 'kawasaki'
    
    simulator = NucleationSimulator(r=r, conc=conc, pbc=pbc, sys=sys,
                                    temp=temp, eps_NN=eps_NN, eps_s=eps_s, mode=mode)

    simulator.run(n_steps=20000, verbose=True, n_sample=100, save_dir='mc_out')