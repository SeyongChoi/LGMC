import numpy as np
from tqdm import tqdm
from typing import Optional, List

from lgmc.init.prob import Prob
from lgmc.init.lattice import Lattice



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
        self.max_attempts_factor = max_attempts_factor
        
        self.dim = self.lattice.shape[0]

        # 주변 이웃 오프셋: 6-방향 (up, down, left, right, front, back)
        self.neighbor_offsets = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])

    def _get_neighbor_sum(self, x, y, z, lattice, cs=0):
        """
        현재 위치 (x,y,z)의 입자 상태 ci와 이웃 cj들의 합(cj_sum)을 계산.
        Heterogeneous일 때 표면 접촉 여부 cs 고려 가능.

        Returns:
            int: cj_sum (occupied 이웃 개수)
        """
        
        cj_sum = 0
        for offset in self.neighbor_offsets:
            nx, ny, nz = x + offset[0], y + offset[1], z + offset[2]

            # pbc 처리
            if self.pbc[0]:
                nx = (nx - 1) % (self.dim - 2) + 1
            else:
                if nx < 1 or nx > self.dim - 2:
                    continue
            if self.pbc[1]:
                ny = (ny - 1) % (self.dim - 2) + 1
            else:
                if ny < 1 or ny > self.dim - 2:
                    continue
            if self.pbc[2]:
                nz = (nz - 1) % (self.dim - 2) + 1
            else:
                if nz < 1 or nz > self.dim - 2:
                    continue

            cj_sum += 1 if lattice[nx, ny, nz] == 1 else 0

        return cj_sum

    def _is_surface_contact(self, x, y, z):
        """
        Heterogeneous 시스템에서 위치가 표면과 접촉하는지 여부 판단.
        surface layer: z == 1 으로 가정 (Lattice 클래스 기준)
        """
        if self.sys != 'hete':
            return 0
        # x,y,z는 lattice 인덱스
        # 6개의 이웃 중 하나라도 표면 layer(z=1)와 접촉하면 cs=1
        for offset in self.neighbor_offsets:
            nx, ny, nz = x + offset[0], y + offset[1], z + offset[2]
            if nx < 1 or nx > self.dim - 2:
                continue
            if ny < 1 or ny > self.dim - 2:
                continue
            if nz < 1 or nz > self.dim - 2:
                continue
            if self.lattice[nx, ny, nz] == 2:  # surface layer 표시
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
        hi = self.prob.hi

        if self.sys == 'homo':
            # hi shape: (2, NN+1)
            H_old = hi[ci, ci_neighbors_sum] + hi[cj, cj_neighbors_sum]
            H_new = hi[cj, ci_neighbors_sum] + hi[ci, cj_neighbors_sum]
        elif self.sys == 'hete':
            # hi shape: (2, 2, NN+1)
            H_old = hi[ci, ci_surface, ci_neighbors_sum] + hi[cj, cj_surface, cj_neighbors_sum]
            H_new = hi[cj, ci_surface, ci_neighbors_sum] + hi[ci, cj_surface, cj_neighbors_sum]
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
                    cs = self._is_surface_contact(x, y, z) if self.sys == 'hete' else 0
                    cj_sum = self._get_neighbor_sum(x, y, z, self.lattice, cs)
                    if self.sys == 'homo':
                        total_energy += hi[ci, cj_sum]
                    else:
                        total_energy += hi[ci, cs, cj_sum]

        # hi 배열의 local energy는 이중 카운팅 방지로 0.5 곱해둠 (확인 필요)
        # 여기서는 각 site local energy를 그대로 더한 상태이므로 전체 에너지로 적합
        return total_energy
    
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
        import os
        n_attempts_per_step = self.n_particle * self.max_attempts_factor
        dim = self.dim
        rng = np.random.default_rng()

        it = range(n_steps)
        if verbose:
            it = tqdm(it, desc='Kawasaki MC steps')

        for step_idx in it:
            
            attempts = 0
            accepted = 0
            while attempts < n_attempts_per_step:
                x1 = rng.integers(1, dim - 1)
                y1 = rng.integers(1, dim - 1)
                z1 = rng.integers(1, dim - 1)

                if self.lattice[x1, y1, z1] != 1:
                    attempts += 1
                    continue # ci가 입자가 아니면 패스

                offset = self.neighbor_offsets[rng.integers(0, len(self.neighbor_offsets))]
                x2, y2, z2 = x1 + offset[0], y1 + offset[1], z1 + offset[2]

                # PBC 처리
                if self.pbc[0]:
                    x2 = (x2 - 1) % (dim - 2) + 1
                elif x2 < 1 or x2 > dim - 2:
                    attempts += 1
                    continue
                if self.pbc[1]:
                    y2 = (y2 - 1) % (dim - 2) + 1
                elif y2 < 1 or y2 > dim - 2:
                    attempts += 1
                    continue
                if self.pbc[2]:
                    z2 = (z2 - 1) % (dim - 2) + 1
                elif z2 < 1 or z2 > dim - 2:
                    attempts += 1
                    continue

                # surface 고정 원자는 움직이면 안됨
                if self.lattice[x1, y1, z1] == 2 or self.lattice[x2, y2, z2] == 2:
                    attempts += 1
                    continue

                ci = 1 if self.lattice[x1, y1, z1] == 1 else 0
                cj = 1 if self.lattice[x2, y2, z2] == 1 else 0

                if ci == cj:
                    attempts += 1
                    continue

                ci_surface = self._is_surface_contact(x1, y1, z1) if self.sys == 'hete' else 0
                cj_surface = self._is_surface_contact(x2, y2, z2) if self.sys == 'hete' else 0

                ci_neighbors_sum = self._get_neighbor_sum(x1, y1, z1, self.lattice, ci_surface)
                cj_neighbors_sum = self._get_neighbor_sum(x2, y2, z2, self.lattice, cj_surface)

                delta_H = self._calc_delta_H(ci, cj, ci_neighbors_sum, cj_neighbors_sum, ci_surface, cj_surface)

                if delta_H <= 0 or rng.random() < np.exp(-self.prob.beta * delta_H):
                    self.lattice[x1, y1, z1], self.lattice[x2, y2, z2] = self.lattice[x2, y2, z2], self.lattice[x1, y1, z1]
                    accepted += 1

                attempts += 1

            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                fname = f"{save_dir}/lattice_step_{step_idx:05d}.extxyz"
                self.to_xyz(step=step_idx, filename=fname)

            if verbose:
                it.set_postfix(accepted=accepted)

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
        dim = self.dim
        total_energy = self._calculate_total_energy()

        # 주석에 step, 에너지, lattice 크기 기록
        comment = f"step={step} energy={total_energy:.6f} lattice_dim={dim}"

        # 버퍼 영역 제외한 입자 및 기판 좌표만 선택
        coords = np.argwhere(self.lattice > 0)
        filtered_coords = [
            (x, y, z) for (x, y, z) in coords
            if 1 <= x <= dim - 2 and 1 <= y <= dim - 2 and 1 <= z <= dim - 2
        ]

        n_atoms = len(filtered_coords)
        lines = [str(n_atoms), comment]

        for x, y, z in filtered_coords:
            atom = 'X' if self.lattice[x, y, z] == 1 else 'S'
            lines.append(f"{atom} {x} {y} {z}")

        xyz_str = "\n".join(lines)

        if filename:
            with open(filename, 'w') as f:
                f.write(xyz_str)

        return xyz_str
    
if __name__ == '__main__':
    r = 10
    conc = 0.05
    pbc = [True, True, True]
    sys = 'homo'
    lattice = Lattice(r=r, conc=conc, pbc=pbc, sys=sys)

    temp = 300
    eps_NN = 4.8
    eps_s = 0.1 * eps_NN
    eps_unit = 'kJ/mol'
    mode = 'kawasaki'

    prob = Prob(temp=temp, eps_NN=eps_NN, eps_s=eps_s, sys=sys, mode=mode, eps_unit=eps_unit)
    print("ci, cj, ci_neighbors_sum, cj_neighbors_sum = 1, 0, 3, 2")
    print("H_old:", prob.hi[1, 3] + prob.hi[0, 2])
    print("H_new:", prob.hi[0, 3] + prob.hi[1, 2])
    print("delta_H:", (prob.hi[0, 3] + prob.hi[1, 2]) - (prob.hi[1, 3] + prob.hi[0, 2]))

    mc = KawasakiMC(lattice_obj=lattice, prob_obj=prob, max_attempts_factor=20)

    # MC 100 step 수행, verbose 출력, extxyz 저장 디렉토리 지정
    mc.step(n_steps=1000, verbose=True, save_dir='mc_extxyz_output')
