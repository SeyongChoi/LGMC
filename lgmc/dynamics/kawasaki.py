import os
import numpy as np
from tqdm import tqdm
from typing import Optional, List

from lgmc.init.prob import Prob
from lgmc.init.lattice import Lattice

from lgmc.utils.pbc import apply_pbc
from lgmc.utils.to_xyz import to_xyz


class KawasakiMC:
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

    def __init__(self, lattice_obj: Lattice, prob_obj: Prob, max_attempts_factor: int = 5):
        self.seed = lattice_obj.seed
        self.lattice_obj = lattice_obj
        self.lattice = lattice_obj.lattice
        self.prob = prob_obj
        self.r = lattice_obj.r
        self.pbc = lattice_obj.pbc
        self.n_particle = lattice_obj.n_particle
        self.sys = lattice_obj.sys
        self.mode = prob_obj.mode
        self.hi = prob_obj.hi
        self.max_attempts_factor = max_attempts_factor
        
        self.dim = self.lattice.shape[0]

        # 주변 이웃 오프셋: 6-방향 (up, down, left, right, front, back)
        self.neighbor_offsets = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
    def step(self, n_steps: int = 1, verbose: bool = False, save_dir: Optional[str] = None):
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

        it = range(n_steps)
        if verbose:
            it = tqdm(it, desc='Kawasaki MC steps')

        for step_idx in it:
            accepted = 0
            ntry = 0
            # 1. 버퍼 층을 제외한 모든 입자 좌표 (ci = 1) 수집 후 셔플
            particle_positions = np.argwhere(self.lattice[1:-1, 1:-1, 1:-1] == 1) + 1 
            rng.shuffle(particle_positions)

            for pos in particle_positions:
                # i-th particle position
                x1, y1, z1 = pos
                ci = self.lattice[x1, y1, z1]
                
                # check number of neighbor of i-th particle
                ci_neighbors_sum = self._get_neighbor_sum(x1, y1, z1, self.lattice)
                # check whether contact surface or not
                ci_surface = self._is_surface_contact(x1, y1, z1)
                

                # i-th particle==1인경우만
                if ci == 1:
                    # j-th particle position
                    offset = self.neighbor_offsets[rng.integers(0, len(self.neighbor_offsets))]
                    x2, y2, z2 = x1 + offset[0], y1 + offset[1], z1 + offset[2]
                    if self.pbc[0]:
                        x2 = self._wrap(x2)
                    if self.pbc[1]:
                        y2 = self._wrap(y2)
                    if self.pbc[2]:
                        z2 = self._wrap(z2)

                    cj = self.lattice[x2, y2, z2]

                    # j-th particle이 1 또는 2이면 mc move는 해당 particle은 switch move 없음.
                    if cj == 1 or cj == 2:
                        continue
                    
                    # j-th particle이 0이어야만 mc move가 switch moving을 할수 있음.
                    ntry += 1
                    # before switching, number of neighbor of j-th particle
                    cj_neighbors_sum = self._get_neighbor_sum(x2, y2, z2, self.lattice)
                    cj_surface = self._is_surface_contact(x2, y2, z2)
                    
                    # calculate the del H local
                    del_h = self._calc_delta_H(ci, cj, ci_neighbors_sum, cj_neighbors_sum, ci_surface, cj_surface)

                    if del_h <= 0 or rng.random() < np.exp(-self.prob.beta * del_h):
                        self.lattice[x1, y1, z1], self.lattice[x2, y2, z2] = self.lattice[x2, y2, z2], self.lattice[x1, y1, z1]
                        
                        if self.pbc[0]:
                            if x1 == 1:
                                self.lattice[self.dim - 1, y1, z1] = self.lattice[x1, y1, z1]
                            if x1 == self.dim - 2:
                                self.lattice[0, y1, z1] = self.lattice[x1, y1, z1]
                            if x2 == 1:
                                self.lattice[self.dim - 1, y2, z2] = self.lattice[x2, y2, z2]
                            if x2 == self.dim - 2:
                                self.lattice[0, y2, z2] = self.lattice[x2, y2, z2]
                        if self.pbc[1]:
                            if y1 == 1:
                                self.lattice[x1, self.dim - 1, z1] = self.lattice[x1, y1, z1]
                            if y1 == self.dim - 2:
                                self.lattice[x1, 0, z1] = self.lattice[x1, y1, z1]
                            if y2 == 1:
                                self.lattice[x2, self.dim - 1, z2] = self.lattice[x2, y2, z2]
                            if y2 == self.dim - 2:
                                self.lattice[x2, 0, z2] = self.lattice[x2, y2, z2]
                        if self.pbc[2]:
                            if z1 == 1:
                                self.lattice[x1, y1, self.dim - 1] = self.lattice[x1, y1, z1]
                            if z1 == self.dim - 2:
                                self.lattice[x1, y1, 0] = self.lattice[x1, y1, z1]
                            if z2 == 1:
                                self.lattice[x2, y2, self.dim - 1] = self.lattice[x2, y2, z2]
                            if z2 == self.dim - 2:
                                self.lattice[x2, y2, 0] = self.lattice[x2, y2, z2]
                        accepted += 1
                else:
                    continue 
            # 저장
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                fname = f"{save_dir}/lattice_step_{step_idx:05d}.extxyz"
                self.to_xyz(step=step_idx, filename=fname)

            if verbose:
                it.set_postfix(accepted=accepted)
                    

    def _wrap(self, val):
        """
        Apply PBC
        """
        if val == self.dim - 1:
            return 1
        elif val == 0:
            return self.dim - 2
        else:
            return val
        
    def _get_neighbor_sum(self, x, y, z, lattice):
        """
        현재 위치 (x,y,z)의 입자 상태 ci와 이웃 cj들의 합(cj_sum)을 계산.
        Heterogeneous일 때 표면 접촉 여부 cs 고려 가능.

        Returns:
            int: cj_sum (occupied 이웃 개수)
        """
        
        cj_sum = 0
        for offset in self.neighbor_offsets:
            nx, ny, nz = x + offset[0], y + offset[1], z + offset[2]

            if lattice[nx, ny, nz] == 1 or lattice[nx, ny, nz] == 0:
                cj_sum += lattice[nx, ny, nz]

        return cj_sum

    def _is_surface_contact(self, x, y, z):
        """
        Heterogeneous 시스템에서 위치가 표면과 접촉하는지 여부 판단.
        surface layer: z == 1 으로 가정 (Lattice 클래스 기준)
        """
        if self.sys != 'hete':
            return 0
        nz = z - 1
        if self.lattice[x, y, nz] == 2:
            return 1
        return 0

    def _calc_delta_H(self, ci, cj, ci_neighbors_sum, cj_neighbors_sum, ci_surface=0, cj_surface=0):
        """
        Kawasaki dynamics에서 두 site 상태(ci, cj)를 교환할 때,
        에너지 변화 ΔH 계산.

        ci, cj: 0 or 1 (site states)
        ci_neighbors_sum, cj_neighbors_sum: 해당 site 주변 occupied 개수
        ci_surface, cj_surface: 표면 접촉 여부 (0 또는 1)

        Returns:
            float: ΔH
        """
        # 두 site 상태 교환: (ci, cj) -> (cj, ci)
        # 원래 hi는 local energy 테이블: hi[ci, cs, cj_sum]
        # ΔH = H_new - H_old = (hi[cj, cs, ci_neighbors_sum] + hi[ci, cs, cj_neighbors_sum]) - (hi[ci, cs, ci_neighbors_sum] + hi[cj, cs, cj_neighbors_sum])
        # cs는 sys에 따라 1차원 또는 2차원 배열 사용
        n_max = self.hi.shape[1] - 1
        
        ci = int(ci)
        cj = int(cj)
        ci_neighbors_sum = int(ci_neighbors_sum)
        cj_neighbors_sum = int(cj_neighbors_sum)
        idx1 = np.clip(ci_neighbors_sum + 1, 0, n_max)
        idx2 = np.clip(cj_neighbors_sum - 1, 0, n_max)
        if self.sys == 'homo':
            # hi shape: (2, NN+1)
            H_old = self.hi[ci, ci_neighbors_sum] + self.hi[cj, cj_neighbors_sum]
            # After switching, ci: 1-> 0, ci_neighbor + 1 | cj: 0 -> 1, cj_neighbor - 1
            H_new = self.hi[cj, idx1] + self.hi[ci, idx2]
        elif self.sys == 'hete':
            # hi shape: (2, 2, NN+1)
            H_old = self.hi[ci, ci_surface, ci_neighbors_sum] + self.hi[cj, cj_surface, cj_neighbors_sum]
            # After switching, ci: 1-> 0, ci_neighbor + 1 | cj: 0 -> 1, cj_neighbor - 1
            H_new = self.hi[cj, ci_surface, idx1] + self.hi[ci, cj_surface, idx2]
        else:
            raise ValueError("sys must be 'homo' or 'hete'")

        delta_H = H_new - H_old
        return delta_H
    
    def _calculate_total_energy(self) -> float:
        """
        현재 lattice 전체 local energy 합 계산.
        각 site ci 상태, 주변 cj_sum, (hetero면 cs) 이용해 hi에서 계산.

        Returns:
            float: 전체 에너지
        """
        hi = self.prob.hi
        total_energy = 0.0
        dim = self.dim

        for x in range(1, dim - 1):
            for y in range(1, dim - 1):
                for z in range(1, dim - 1):
                    ci = 1 if self.lattice[x, y, z] == 1 else 0
                    if ci == 0:
                        continue  # 비어있는 site는 에너지 기여 없음
                    cs = int(self._is_surface_contact(x, y, z)) if self.sys == 'hete' else 0
                    cj_sum = int(self._get_neighbor_sum(x, y, z, self.lattice))
                    if self.sys == 'homo':
                        total_energy += hi[ci, cj_sum]
                    else:
                        total_energy += hi[ci, cs, cj_sum]

        # hi 배열의 local energy는 이중 카운팅 방지로 0.5 곱해둠 (확인 필요)
        # 여기서는 각 site local energy를 그대로 더한 상태이므로 전체 에너지로 적합
        return total_energy
    

    def get_lattice(self):
        """현재 lattice 상태 반환"""
        return self.lattice
    
    
    def to_xyz(self, step: int, filename: Optional[str] = None) -> str:
        """
        현재 lattice 상태를 extxyz 포맷으로 저장.

        extxyz format 참고:
        - 첫 줄: 원자 개수
        - 두 번째 줄: 주석(comment) - 여기서는 에너지, 크기 등 메타정보 기록
        - 이후 줄: atom_symbol x y z

        Args:
            step (int): MC step 번호 (파일명 등에 활용).
            filename (Optional[str]): 저장할 파일명. None일 경우 문자열만 반환.

        Returns:
            str: extxyz 포맷 문자열
        """
        length = self.dim - 2
        total_energy = self._calculate_total_energy()

        # 주석에 step, 에너지, lattice 크기 기록
        comment = f'lattice="{length}  0  0  0  {length}  0  0  0  {length}"  origin="1 1 1" properties=species:S:1:pos:R:3  energy={total_energy:.6f}  step={step}'

        return to_xyz(self.dim, self.lattice, filename, comment)
    
if __name__ == '__main__':
    r = 10
    conc = 0.05
    sys = 'hete'
    pbc = [True, True, True] if sys == 'homo' else [True, True, False]
    
    lattice = Lattice(r=r, conc=conc, pbc=pbc, sys=sys)

    temp = 0.2
    eps_NN = 1.0
    eps_s = 0.1 * eps_NN
    mode = 'kawasaki'

    prob = Prob(temp=temp, eps_NN=eps_NN, eps_s=eps_s, sys=sys, mode=mode)
    
    mc = KawasakiMC(lattice_obj=lattice, prob_obj=prob, max_attempts_factor=20)

    # MC 100 step 수행, verbose 출력, extxyz 저장 디렉토리 지정
    mc.step(n_steps=1000, verbose=True, save_dir='mc_extxyz_output')


# def step(self, n_steps: int = 1, verbose: bool = False, save_dir: Optional[str] = None):
#     """
#     Perform Kawasaki MC steps and optionally save extxyz files per step.

#     Args:
#         n_steps (int): number of MC steps (1 step = n_particle attempts).
#         verbose (bool): print progress bar.
#         save_dir (Optional[str]): extxyz 저장할 디렉토리. None이면 저장 안함.

#     Returns:
#         None
#     """
#     dim = self.dim
#     rng = np.random.default_rng()

#     it = range(n_steps)
#     if verbose:
#         it = tqdm(it, desc='Kawasaki MC steps')

#     for step_idx in it:
#         accepted = 0

#         # 1. 모든 입자 좌표 (ci = 1) 수집 후 셔플
#         particle_positions = np.argwhere(self.lattice == 1)
#         rng.shuffle(particle_positions)

#         for pos in particle_positions:
#             x1, y1, z1 = pos

#             offset = self.neighbor_offsets[rng.integers(0, len(self.neighbor_offsets))]
#             x2, y2, z2 = x1 + offset[0], y1 + offset[1], z1 + offset[2]

#             # PBC 처리
#             if self.pbc[0]:
#                 x2 = (x2 - 1) % (dim - 2) + 1
#             elif x2 < 1 or x2 > dim - 2:
#                 continue
#             if self.pbc[1]:
#                 y2 = (y2 - 1) % (dim - 2) + 1
#             elif y2 < 1 or y2 > dim - 2:
#                 continue
#             if self.pbc[2]:
#                 z2 = (z2 - 1) % (dim - 2) + 1
#             elif z2 < 1 or z2 > dim - 2:
#                 continue

#             # surface 고정 원자 제외
#             if self.lattice[x1, y1, z1] == 2 or self.lattice[x2, y2, z2] == 2:
#                 continue

#             ci = 1 if self.lattice[x1, y1, z1] == 1 else 0
#             cj = 1 if self.lattice[x2, y2, z2] == 1 else 0

#             if ci == cj:
#                 continue

#             ci_surface = self._is_surface_contact(x1, y1, z1) if self.sys == 'hete' else 0
#             cj_surface = self._is_surface_contact(x2, y2, z2) if self.sys == 'hete' else 0

#             ci_neighbors_sum = self._get_neighbor_sum(x1, y1, z1, self.lattice, ci_surface)
#             cj_neighbors_sum = self._get_neighbor_sum(x2, y2, z2, self.lattice, cj_surface)

#             delta_H = self._calc_delta_H(ci, cj, ci_neighbors_sum, cj_neighbors_sum, ci_surface, cj_surface)

#             if delta_H <= 0 or rng.random() < np.exp(-self.prob.beta * delta_H):
#                 self.lattice[x1, y1, z1], self.lattice[x2, y2, z2] = self.lattice[x2, y2, z2], self.lattice[x1, y1, y1]
#                 accepted += 1

#         # 저장
#         if save_dir is not None:
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             fname = f"{save_dir}/lattice_step_{step_idx:05d}.extxyz"
#             self.to_xyz(step=step_idx, filename=fname)

#         if verbose:
#             it.set_postfix(accepted=accepted)