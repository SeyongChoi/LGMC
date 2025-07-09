
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int wrap_coord(int val, int dim, bint do_pbc) nogil:
    if do_pbc:
        return (val - 1) % (dim - 2) + 1
    return val

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double calc_delta_H_homo(int ci, int cj, int ci_sum, int cj_sum,
                                     double[:, :] hi) nogil:
    cdef int n_max = hi.shape[1] - 1
    cdef int idx1 = ci_sum + 1 if ci_sum <= n_max else n_max
    cdef int idx2 = cj_sum - 1 if cj_sum >= 0 else 0
    return hi[cj, idx1] + hi[ci, idx2] - (hi[ci, ci_sum] + hi[cj, cj_sum])

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
    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        nx = wrap_coord(nx, lx, pbc0)
        ny = wrap_coord(ny, ly, pbc1)
        nz = wrap_coord(nz, lz, pbc2)
        val = lattice[nx, ny, nz]
        # count only 0 or 1
        if val == 0 or val == 1:
            total += val
    return total

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple move_one_step(int[:, :, :] lattice,
                          double[:, :] hi,
                          int[:, :] neighbor_offsets,
                          double beta,
                          bint pbc0, bint pbc1, bint pbc2) nogil:
    """
    Perform one MC sweep (Kawasaki exchange attempts over all 1-sites).
    Returns a tuple (accepted_swaps, updated_lattice).
    """
    cdef int nx, ny, nz, ox, oy, oz
    cdef int x, y, z, i, j, idx, accepted = 0
    cdef int ci, cj, ci_sum, cj_sum, offset_idx
    cdef int dim = lattice.shape[0]

    # temporary array of coordinates (could be preallocated and passed in)
    coords = <int[:, :] > cython.malloc(sizeof(int) * (dim-2)*(dim-2)*(dim-2) * 3)
    cdef int n = 0

    # collect positions of ci=1
    for x in range(1, dim-1):
        for y in range(1, dim-1):
            for z in range(1, dim-1):
                if lattice[x, y, z] == 1:
                    coords[n, 0] = x
                    coords[n, 1] = y
                    coords[n, 2] = z
                    n += 1

    # shuffle via rand
    for i in range(n-1, 0, -1):
        j = rand() % (i+1)
        for idx in range(3): coords[i, idx], coords[j, idx] = coords[j, idx], coords[i, idx]

    # attempt swaps in parallel
    for i in prange(n, nogil=True):
        x = coords[i, 0]; y = coords[i, 1]; z = coords[i, 2]
        # pick random neighbor direction
        offset_idx = rand() % neighbor_offsets.shape[0]
        ox = neighbor_offsets[offset_idx, 0]
        oy = neighbor_offsets[offset_idx, 1]
        oz = neighbor_offsets[offset_idx, 2]
        nx = wrap_coord(x + ox, dim, pbc0)
        ny = wrap_coord(y + oy, dim, pbc1)
        nz = wrap_coord(z + oz, dim, pbc2)

        ci = lattice[x, y, z]
        cj = lattice[nx, ny, nz]
        if ci != 1 or cj == 1:
            continue

        ci_sum = get_neighbor_sum(lattice, x, y, z, pbc0, pbc1, pbc2)
        cj_sum = get_neighbor_sum(lattice, nx, ny, nz, pbc0, pbc1, pbc2)

        # Metropolis criterion
        if calc_delta_H_homo(ci, cj, ci_sum, cj_sum, hi) <= 0 or \
           cython.cast(double, rand())/RAND_MAX < exp(-beta * calc_delta_H_homo(ci, cj, ci_sum, cj_sum, hi)):
            # swap
            lattice[x, y, z] = cj
            lattice[nx, ny, nz] = ci
            accepted += 1
    cython.free(coords)
    # Return accepted count and updated lattice
    return accepted, lattice
