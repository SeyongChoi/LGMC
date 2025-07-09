import os
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple

from lgmc.init.prob import Prob
from lgmc.init.lattice import Lattice
from lgmc.utils.to_xyz import to_xyz


class KawasakiMC_py:
    """
    Kawasaki dynamics Monte Carlo simulation for lattice gas model.

    Args:
        lattice_obj (Lattice): 초기 lattice 객체 (상태 포함).
        prob_obj (Prob): Prob 클래스 객체 (에너지 및 확률 테이블 포함).
        max_attempts_factor (int): 1 step 당 시도 횟수를 n_particle * max_attempts_factor 로 설정.

    Attributes:
        lattice (np.ndarray): 현재 lattice 상태 (3D).
        prob (Prob): 확률 및 에너지 정보.
        r (int): lattice radius.
        pbc (np.ndarray): pbc flags.
        n_particle (int): 현재 입자 개수.
        max_attempts_factor (int): step 당 시도 횟수 배수.
    """

    def __init__(self, lattice_obj: Lattice, prob_obj: Prob):
        self.seed = lattice_obj.seed
        self.lattice_obj = lattice_obj
        self.lattice = lattice_obj.lattice
        self.prob = prob_obj
        # self.r = lattice_obj.r
        self.pbc = lattice_obj.pbc
        # self.n_particle = lattice_obj.n_particle
        self.sys = lattice_obj.sys
        self.mode = prob_obj.mode
        self.hi = prob_obj.hi
        

        self.dim = self.lattice.shape[0]

        # 주변 이웃 오프셋: 6-방향 (up, down, left, right, front, back)
        self.neighbor_offsets = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])

    def step(self, n_steps: int = 1, verbose: bool = False, save_dir: Optional[str] = None, n_sample: Optional[int] = 1) -> None:
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
            accepted = 0
            ntry = 0
            # 1. 버퍼 층을 제외한 모든 입자 좌표 (ci = 1) 수집 후 셔플
            particle_positions = np.argwhere(self.lattice[1:-1, 1:-1, 1:-1] == 1) + 1 
            rng.shuffle(particle_positions)

            for x1, y1, z1 in particle_positions:
                ci = self.lattice[x1, y1, z1]

                if ci != 1:
                    continue

                # check number of neighbor of i-th particle
                ci_neighbors_sum = self._get_neighbor_sum(x1, y1, z1)
                # check whether contact surface or not
                ci_surface = self._is_surface_contact(x1, y1, z1)

                # j-th particle position
                offset = self.neighbor_offsets[rng.integers(0, len(self.neighbor_offsets))]
                x2, y2, z2 = x1 + offset[0], y1 + offset[1], z1 + offset[2]
                x2, y2, z2 = self._wrap_coords(x2, y2, z2)

                cj = self.lattice[x2, y2, z2]

                if cj == 1 or cj == 2:
                    continue

                ntry += 1
                cj_neighbors_sum = self._get_neighbor_sum(x2, y2, z2)
                cj_surface = self._is_surface_contact(x2, y2, z2)

                del_h = self._calc_delta_H(ci, cj, ci_neighbors_sum, cj_neighbors_sum, ci_surface, cj_surface)

                if del_h <= 0 or rng.random() < np.exp(-self.prob.beta * del_h):
                    self.lattice[x1, y1, z1], self.lattice[x2, y2, z2] = cj, ci
                    self._update_pbc_boundary(x1, y1, z1)
                    self._update_pbc_boundary(x2, y2, z2)
                    accepted += 1

            if save_dir is not None and (step_idx % n_sample == 0):
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{save_dir}/lattice_step_{step_idx:07d}.extxyz"
                self.to_xyz(step=step_idx, filename=fname)

            if verbose:
                it.set_postfix(accepted=accepted)

    def _wrap_coords(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        """PBC 적용하여 좌표 wrap 처리"""
        def wrap(val, axis):
            return (val - 1) % (self.dim - 2) + 1 if self.pbc[axis] else val

        return wrap(x, 0), wrap(y, 1), wrap(z, 2)

    def _update_pbc_boundary(self, x: int, y: int, z: int) -> None:
        """PBC 경계면 복사 처리 (입자 이동 후 update)"""
        if self.pbc[0]:
            if x == 1:
                self.lattice[self.dim - 1, y, z] = self.lattice[x, y, z]
            elif x == self.dim - 2:
                self.lattice[0, y, z] = self.lattice[x, y, z]
        if self.pbc[1]:
            if y == 1:
                self.lattice[x, self.dim - 1, z] = self.lattice[x, y, z]
            elif y == self.dim - 2:
                self.lattice[x, 0, z] = self.lattice[x, y, z]
        if self.pbc[2]:
            if z == 1:
                self.lattice[x, y, self.dim - 1] = self.lattice[x, y, z]
            elif z == self.dim - 2:
                self.lattice[x, y, 0] = self.lattice[x, y, z]

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

    def _calc_delta_H(self, ci: int, cj: int, ci_neighbors_sum: int, cj_neighbors_sum: int,
                      ci_surface: int = 0, cj_surface: int = 0) -> float:
        """
        Kawasaki dynamics에서 두 site 상태(ci, cj)를 교환할 때,
        에너지 변화 ΔH 계산.

        Returns:
            float: ΔH
        """
        # numpy scalar → Python int
        ci = int(ci)
        cj = int(cj)
        ci_neighbors_sum = int(ci_neighbors_sum)
        cj_neighbors_sum = int(cj_neighbors_sum)
        ci_surface = int(ci_surface)
        cj_surface = int(cj_surface)

        n_max = self.hi.shape[-1] - 1
        idx1 = np.clip(ci_neighbors_sum + 1, 0, n_max)
        idx2 = np.clip(cj_neighbors_sum - 1, 0, n_max)

        if self.sys == 'homo':
            H_old = self.hi[ci, ci_neighbors_sum] + self.hi[cj, cj_neighbors_sum]

            # After switching, ci: 1-> 0, ci_neighbor + 1 | cj: 0 -> 1, cj_neighbor - 1
            H_new = self.hi[cj, idx1] + self.hi[ci, idx2]
        elif self.sys == 'hete':
            H_old = self.hi[ci, ci_surface, ci_neighbors_sum] + self.hi[cj, cj_surface, cj_neighbors_sum]

            # After switching, ci: 1-> 0, ci_neighbor + 1 | cj: 0 -> 1, cj_neighbor - 1
            H_new = self.hi[cj, ci_surface, idx1] + self.hi[ci, cj_surface, idx2]
        else:
            raise ValueError("sys must be 'homo' or 'hete'")

        return H_new - H_old

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
                    total_energy += int(self.hi[ci, cs, cj_sum]) if self.sys == 'hete' else int(self.hi[ci, cj_sum])

        return total_energy

    def get_lattice(self) -> np.ndarray:
        """현재 lattice 상태 반환"""
        return self.lattice

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

if __name__ == '__main__':
    from lgmc.dynamics.kawasaki import KawasakiMC

    r = 30
    conc = 0.1
    sys = 'hete'
    pbc = [True, True, True] if sys == 'homo' else [True, True, False]
    
    lattice = Lattice(r=r, conc=conc, pbc=pbc, sys=sys)

    temp = 0.2
    eps_NN = 1.0
    eps_s = 0.1 * eps_NN
    mode = 'kawasaki'

    prob = Prob(temp=temp, eps_NN=eps_NN, eps_s=eps_s, sys=sys, mode=mode)
    
    mc = KawasakiMC(lattice_obj=lattice, prob_obj=prob)

    # MC 100 step 수행, verbose 출력, extxyz 저장 디렉토리 지정
    mc.step(n_steps=5000, verbose=True, save_dir='mc_extxyz_output')

