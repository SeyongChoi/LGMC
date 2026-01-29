import os
import numpy as np
from lgmc.utils.to_xyz import to_xyz

cimport numpy as np
from cython.parallel import prange
# C 매크로 직접 정의 (Cython에서는 이 방식만 호환됨)
cdef extern from *:
    """
    #ifdef _WIN32
    #define WINDOWS 1
    #else
    #define WINDOWS 0
    #endif
    """

# 이 부분은 플랫폼과 상관없이 항상 선언
cdef extern from "time.h":
    unsigned int time(unsigned int *)

cdef extern from "stdlib.h":
    void srand(unsigned int seed)
    int rand()

cdef extern from "math.h":
    double exp(double)

cdef class KawasakiMC:
    cdef:
        object lattice_obj
        object prob
        unsigned long seed
        object rng
        object lattice
        object pbc
        object hi
        object neighbor_offsets
        int dim
        double beta
        bytes sys
        bytes mode

    def __init__(self, lattice_obj, prob_obj):
        
        # Pull lattice array (int) and related attributes
        self.lattice_obj = lattice_obj
        self.seed = lattice_obj.seed
        self.lattice = lattice_obj.lattice
        self.pbc = lattice_obj.pbc
        self.sys = lattice_obj.sys.encode('ascii')
        self.dim = self.lattice.shape[0]

        # Probability helper
        self.prob = prob_obj
        self.hi = prob_obj.hi
        self.mode = prob_obj.mode.encode('ascii')
        self.beta = prob_obj.beta

        # 6 nearest neighbor offsets
        self.neighbor_offsets = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=np.intc)

        # Initialize RNG seed
        srand(<unsigned long> time(NULL) ^ self.seed)
        self.rng = np.random.default_rng(self.seed)

    cpdef void step(self,
                    int n_steps=1,
                    bint verbose=False,
                    str save_dir=None,
                    int n_sample=1):

        # decalre the dtype
        cdef int step_idx, accepted, ntry
        cdef list particle_positions
        cdef object it
        cdef int x1, y1, z1, x2, y2, z2, offset_idx
        cdef int ci, cj, ci_neighbor_sum, cj_neighbor_sum, ci_surface, cj_surface
        cdef double del_h

        rng = self.rng
        if verbose:
            from tqdm import tqdm
            it = tqdm(range(n_steps), desc='Kawasaki MC steps')
        else:
            it = range(n_steps)
        
        for step_idx in it:
            ntry = 0
            accepted = 0
            # 1. 버퍼층을 제외한 모든 입자 좌표 (ci=1) 수집후 셔플
            # gather and shuffle positions
            # particle_positions = np.argwhere(self.lattice[1:-1,1:-1,1:-1] == 1) + 1
            # rng.shuffle(particle_positions)
            particle_positions = self._collect_particles()
            self.shuffle_particle(particle_positions)

            # 2. Check and switch or not for all particles
            for pos in particle_positions:
                x1, y1, z1 = pos[0], pos[1], pos[2]
                # random neighbor (nearest neighbor)
                offset_idx = rng.integers(0, 6)
                x2 = x1 + self.neighbor_offsets[offset_idx, 0]
                y2 = y1 + self.neighbor_offsets[offset_idx, 1]
                z2 = z1 + self.neighbor_offsets[offset_idx, 2]
                # apply the PBC(wrapping)
                x2, y2, z2 = self._wrap_coords(x2, y2, z2)

                ci, cj = self.lattice[x1, y1, z1], self.lattice[x2, y2, z2]
                # ci=1 이어야만 switch move try || cj=1 or 2이면 switch move X
                if ci != 1 or cj in (1, 2):
                    continue

                ntry += 1
                # Check the number of neighbors of i(j)-th particle
                ci_neighbor_sum = self._get_neighbor_sum(x1, y1, z1)
                cj_neighbor_sum = self._get_neighbor_sum(x2, y2, z2)

                # Check whether particle contact with surface or not
                ci_surface = self._is_surface_contact(x1, y1, z1)
                cj_surface = self._is_surface_contact(x2, y2, z2)

                # Calculate the local delta H
                del_h = self._calc_delta_H(ci, cj, ci_neighbor_sum, cj_neighbor_sum, ci_surface, cj_surface)

                # Accroding to Metroplis Algorith, swap the particle
                if del_h <= 0 or rng.random() < exp(-self.beta * del_h):
                    self.lattice[x1, y1, z1], self.lattice[x2, y2, z2] = cj, ci
                    self._update_pbc_boundary(x1, y1, z1)
                    self._update_pbc_boundary(x2, y2, z2)
                    accepted += 1
            if save_dir is not None and step_idx % n_sample == 0:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{save_dir}/lattice_step_{step_idx:07d}.extxyz"
                self.to_xyz(step_idx, fname)
            if verbose:
                it.set_postfix(accepted=accepted)

    cdef list _collect_particles(self):
        cdef list particles = []
        cdef int x, y, z

        for x in range(1, self.dim - 1):
            for y in range(1, self.dim - 1):
                for z in range(1, self.dim - 1):
                    if self.lattice[x, y, z] == 1:
                        particles.append((x,y,z))
        
        return particles

    cdef void shuffle_particle(self, list particles):
        cdef int i, j
        cdef int n = len(particles)
        cdef object tmp

        for i in range(n-1, 0, -1):
            j = rand() % (i + 1)
            tmp = particles[i]
            particles[i] = particles[j]
            particles[j] = tmp

    cdef tuple _wrap_coords(self, int x, int y, int z):
        """PBC 적용하여 좌표 wrap 처리"""
        cdef int wrap_x = (x - 1) % (self.dim - 2) + 1 if self.pbc[0] else x
        cdef int wrap_y = (y - 1) % (self.dim - 2) + 1 if self.pbc[1] else y
        cdef int wrap_z = (z - 1) % (self.dim - 2) + 1 if self.pbc[2] else z
        return wrap_x, wrap_y, wrap_z        

    cdef int _get_neighbor_sum(self, int x, int y, int z):
        cdef int cj_sum, val
        cdef int lx, ly, lz
        cdef int nx, ny, nz

        cj_sum = 0
        lx, ly, lz = self.lattice.shape

        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            # Wrap or skip out-of-bound
            if self.pbc[0]: nx = (nx - 1) % (lx - 2) + 1
            elif nx < 1 or nx >= lx - 1: continue

            if self.pbc[1]: ny = (ny - 1) % (ly - 2) + 1
            elif ny < 1 or ny >= ly - 1: continue

            if self.pbc[2]: nz = (nz - 1) % (lz - 2) + 1
            elif nz < 1 or nz >= lz - 1: continue
            
            val = self.lattice[nx, ny, nz]
            if val == 0 or val == 1: cj_sum += val

        return cj_sum
    
    cdef int _is_surface_contact(self, int x, int y, int z):
        """
        Heterogeneous 시스템에서 위치가 표면과 접촉하는지 여부 판단.
        surface layer: z == 1 으로 가정 (Lattice 클래스 기준)
        """
        return 1 if self.sys==b'hete' and self.lattice[x, y, z-1] == 2 else 0

    cdef double _calc_delta_H(self, int ci, int cj, 
                              int ci_neighbor_sum,
                              int cj_neighbor_sum,
                              int ci_surface, int cj_surface):
        """
        Kawasaki dynamics에서 두 site 상태(ci, cj)를 교환할 때,
        에너지 변화 ΔH 계산.

        Returns:
            float: ΔH
        """
        cdef int n_max, idx1, idx2
        cdef double H_old, H_new

        n_max = <int>self.hi.shape[-1] - 1
        idx1 = ci_neighbor_sum + 1 if ci_neighbor_sum <= n_max else n_max
        idx2 = cj_neighbor_sum - 1 if cj_neighbor_sum >= 0 else 0

        if self.sys == b'homo':
            H_old = self.hi[ci, ci_neighbor_sum] + self.hi[cj, cj_neighbor_sum]
            # After switching, ci: 1-> 0, ci_neighbor + 1 | cj: 0 -> 1, cj_neighbor - 1
            H_new = self.hi[cj, idx1] + self.hi[ci, idx2]
        elif self.sys == b'hete':
            H_old = self.hi[ci, ci_surface, ci_neighbor_sum] + self.hi[cj, cj_surface, cj_neighbor_sum]
            # After switching, ci: 1-> 0, ci_neighbor + 1 | cj: 0 -> 1, cj_neighbor - 1
            H_new = self.hi[cj, ci_surface, idx1] + self.hi[ci, cj_surface, idx2]
        else:
            H_old = H_new = 0.0
        
        return H_new - H_old

    cdef void _update_pbc_boundary(self, int x, int y, int z):
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

    cdef double _calculate_total_energy(self):
        """
        현재 lattice 전체 local energy 합 계산.
        각 site ci 상태, 주변 cj_sum, (hetero면 cs) 이용해 hi에서 계산.

        Returns:
            float: 전체 에너지
        """
        cdef double total = 0.0
        cdef int x, y, z, ci, cj_sum, cs

        for x in range(1, self.dim-1):
            for y in range(1, self.dim-1):
                for z in range(1, self.dim-1):
                    ci = self.lattice[x,y,z]
                    if ci != 1: continue
                    cs = self._is_surface_contact(x,y,z) if self.sys==b'hete' else 0
                    cj_sum = self._get_neighbor_sum(x,y,z)
                    if self.sys==b'homo':
                        total += self.hi[ci, cj_sum]
                    else:
                        total += self.hi[ci, cs, cj_sum]
        return total
    
    cpdef np.ndarray[np.int_t, ndim=3] get_lattice(self):
        """현재 lattice 상태 반환"""

        return self.lattice

    cpdef str to_xyz(self, int step, object filename=None):
        cdef int length = self.dim - 2
        cdef double total_energy = self._calculate_total_energy()
        cdef str comment = (
            f'lattice="{length}  0  0  0  {length}  0  0  0  {length}" '
            f'origin="1 1 1" properties=species:S:1:pos:R:3  '
            f'energy={total_energy:.6f}  step={step}'
        )

        return to_xyz(self.dim, self.lattice, filename, comment)

    