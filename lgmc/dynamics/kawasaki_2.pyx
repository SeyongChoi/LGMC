
cimport cython
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
    int RAND_MAX

cdef extern from "math.h":
    double exp(double)

# PBC wrap helper
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int wrap_coord(int val, int dim, bint do_pbc) nogil:
    if do_pbc:
        return (val - 1) % (dim - 2) + 1
    return val

# ΔH 계산: homogeneous
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double calc_delta_H_homo(int ci, int cj, int ci_sum, int cj_sum,
                                     double[:, :] hi) nogil:
    cdef int n_max = hi.shape[1] - 1
    cdef int idx1 = ci_sum + 1 if ci_sum <= n_max else n_max
    cdef int idx2 = cj_sum - 1 if cj_sum >= 0 else 0
    return hi[cj, idx1] + hi[ci, idx2] - (hi[ci, ci_sum] + hi[cj, cj_sum])

# ΔH 계산: heterogeneous
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double calc_delta_H_hete(int ci, int cj,
                                     int ci_sum, int cj_sum,
                                     int ci_surf, int cj_surf,
                                     double[:, :, :] hi) nogil:
    cdef int n_max = hi.shape[2] - 1
    cdef int idx1 = ci_sum + 1 if ci_sum <= n_max else n_max
    cdef int idx2 = cj_sum - 1 if cj_sum >= 0 else 0
    # H_old = hi[ci, ci_surf, ci_sum] + hi[cj, cj_surf, cj_sum]
    # H_new = hi[cj, ci_surf, idx1] + hi[ci, cj_surf, idx2]
    return (hi[cj, ci_surf, idx1] + hi[ci, cj_surf, idx2]
            - (hi[ci, ci_surf, ci_sum] + hi[cj, cj_surf, cj_sum]))

# neighbor sum helper
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int get_neighbor_sum(int[:, :, :] lattice,
                          int x, int y, int z,
                          bint pbc0, bint pbc1, bint pbc2) nogil:
    cdef int dx, dy, dz, nx, ny, nz, val, total = 0
    cdef int lx = lattice.shape[0]
    cdef int ly = lattice.shape[1]
    cdef int lz = lattice.shape[2]
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        nx = wrap_coord(x+dx, lx, pbc0)
        ny = wrap_coord(y+dy, ly, pbc1)
        nz = wrap_coord(z+dz, lz, pbc2)
        val = lattice[nx, ny, nz]
        if val == 0 or val == 1:
            total += val
    return total

# surface contact helper
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int is_surface(int x, int y, int z,
                            int[:, :, :] lattice) nogil:
    # surface layer z==1 (outside boundary)
    return 1 if lattice[x, y, z-1] == 2 else 0

# Move: homogeneous
cpdef int move_homo(int[:, :, :] lattice,
                    double[:, :] hi,
                    int[:, :] neighbor_offsets,
                    double beta,
                    bint pbc0, bint pbc1, bint pbc2,
                    int[:, :] coords,
                    int n) nogil:
    cdef int i, x, y, z, nx, ny, nz, ox, oy, oz
    cdef int ci, cj, ci_sum, cj_sum, offset_idx, accepted = 0
    cdef double prob, dH
    for i in prange(n, nogil=True):
        x = coords[i,0]; y = coords[i,1]; z = coords[i,2]
        offset_idx = rand() % neighbor_offsets.shape[0]
        ox = neighbor_offsets[offset_idx,0]; oy = neighbor_offsets[offset_idx,1]; oz = neighbor_offsets[offset_idx,2]
        nx = wrap_coord(x+ox, lattice.shape[0], pbc0)
        ny = wrap_coord(y+oy, lattice.shape[1], pbc1)
        nz = wrap_coord(z+oz, lattice.shape[2], pbc2)
        ci = lattice[x,y,z]; cj = lattice[nx,ny,nz]
        if ci != 1 or cj == 1: continue
        ci_sum = get_neighbor_sum(lattice, x,y,z, pbc0,pbc1,pbc2)
        cj_sum = get_neighbor_sum(lattice, nx,ny,nz, pbc0,pbc1,pbc2)
        dH = calc_delta_H_homo(ci, cj, ci_sum, cj_sum, hi)
        prob = rand() / <double>RAND_MAX
        if dH <= 0 or prob < exp(-beta * dH):
            lattice[x,y,z] = cj
            lattice[nx,ny,nz] = ci
            accepted += 1
    return accepted

# Move: heterogeneous
cpdef int move_hete(int[:, :, :] lattice,
                    double[:, :, :] hi,
                    int[:, :] neighbor_offsets,
                    double beta,
                    bint pbc0, bint pbc1, bint pbc2,
                    int[:, :] coords,
                    int n) nogil:
    cdef int i, x, y, z, nx, ny, nz, ox, oy, oz
    cdef int ci, cj, ci_sum, cj_sum, ci_s, cj_s, offset_idx, accepted = 0
    cdef double prob, dH
    for i in prange(n, nogil=True):
        x = coords[i,0]; y = coords[i,1]; z = coords[i,2]
        offset_idx = rand() % neighbor_offsets.shape[0]
        ox = neighbor_offsets[offset_idx,0]; oy = neighbor_offsets[offset_idx,1]; oz = neighbor_offsets[offset_idx,2]
        nx = wrap_coord(x+ox, lattice.shape[0], pbc0)
        ny = wrap_coord(y+oy, lattice.shape[1], pbc1)
        nz = wrap_coord(z+oz, lattice.shape[2], pbc2)
        ci = lattice[x,y,z]; cj = lattice[nx,ny,nz]
        if ci != 1 or cj == 1: continue
        ci_sum = get_neighbor_sum(lattice, x,y,z, pbc0,pbc1,pbc2)
        cj_sum = get_neighbor_sum(lattice, nx,ny,nz, pbc0,pbc1,pbc2)
        ci_s = is_surface(x, y, z, lattice)
        cj_s = is_surface(nx, ny, nz, lattice)
        dH = calc_delta_H_hete(ci, cj, ci_sum, cj_sum, ci_s, cj_s, hi)
        prob = rand() / <double>RAND_MAX
        if dH <= 0 or prob < exp(-beta * dH):
            lattice[x,y,z] = cj
            lattice[nx,ny,nz] = ci
            accepted += 1
    return accepted
