cimport cython
from cython.parallel import prange

# 플랫폼 구분을 위한 C 매크로 정의 (Windows인지 여부 판단)
cdef extern from *:
    """
    #ifdef _WIN32
    #define WINDOWS 1
    #else
    #define WINDOWS 0
    #endif
    """
cdef extern from "stdint.h":
    ctypedef long int64_t

# 시간 함수 선언 (seed 설정 등에 사용)
cdef extern from "time.h":
    unsigned int time(unsigned int *)

# 난수 생성 관련 함수들 선언
cdef extern from "stdlib.h":
    void srand(unsigned int seed)
    int rand()
    int RAND_MAX

# 수학 함수 exp 선언 (nogil 환경에서 사용 가능)
cdef extern from "math.h" nogil:
    double exp(double)

# provide a Python-callable seeding function
cpdef void seed_c_rand(unsigned int seed):
    srand(seed)

#----------------------------------------
# PBC (Periodic Boundary Condition) 좌표 래핑 함수
# val: 현재 좌표, dim: 격자 크기, do_pbc: PBC 적용 여부
# PBC가 적용되면 경계 넘어가면 반대편으로 감 (wrap)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int wrap_coord(int val, int dim, bint do_pbc) nogil:
    cdef int core_dim = dim - 2
    if do_pbc:
        return ((val - 1 + core_dim) % core_dim) + 1
        # return (val - 1) % (dim - 2) + 1
    return val


#----------------------------------------
# ΔH 계산: 동질계 (homogeneous system)
# ci, cj: 현재 위치와 이동 대상 위치의 상태값 (예: 0,1,2)
# ci_sum, cj_sum: 각각의 위치 주변 이웃의 합
# hi: 에너지 관련 2D 배열
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double calc_delta_H_homo(int ci, int cj, int ci_sum, int cj_sum,
                                     double[:, :] hi) nogil:
    cdef int n_max = hi.shape[1] - 1
    cdef int idx1 = ci_sum + 1 if ci_sum <= n_max else n_max
    cdef int idx2 = cj_sum - 1 if cj_sum >= 0 else 0
    # 이동에 따른 에너지 변화량 계산
    return hi[cj, idx1] + hi[ci, idx2] - (hi[ci, ci_sum] + hi[cj, cj_sum])


#----------------------------------------
# ΔH 계산: 이질계 (heterogeneous system, 표면 구분 포함)
# ci_s, cj_s: 현재와 이동 대상 위치의 표면 여부 (0 or 1)
# hi: 3D 배열로 표면 정보 포함 에너지 데이터
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
    # 이질계의 경우 표면 여부에 따른 에너지 변화량 계산
    return (hi[cj, ci_surf, idx1] + hi[ci, cj_surf, idx2]
            - (hi[ci, ci_surf, ci_sum] + hi[cj, cj_surf, cj_sum]))


#----------------------------------------
# 이웃 합계 계산 함수
# 특정 격자 좌표 주변 6방향 이웃 상태값을 더함 (val이 0 또는 1일 때만 합산)
# pbc0, pbc1, pbc2: 각 축의 PBC 적용 여부
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int get_neighbor_sum(int64_t[:, :, :] lattice,
                          int x, int y, int z,
                          bint pbc0, bint pbc1, bint pbc2) nogil:
    cdef int dx, dy, dz, nx, ny, nz, val, total = 0
    cdef int lx = lattice.shape[0]
    cdef int ly = lattice.shape[1]
    cdef int lz = lattice.shape[2]

    # 6방향 이웃 벡터
    cdef int dir[6][3]
    dir[0][0] =  1; dir[0][1] =  0; dir[0][2] =  0
    dir[1][0] = -1; dir[1][1] =  0; dir[1][2] =  0
    dir[2][0] =  0; dir[2][1] =  1; dir[2][2] =  0
    dir[3][0] =  0; dir[3][1] = -1; dir[3][2] =  0
    dir[4][0] =  0; dir[4][1] =  0; dir[4][2] =  1
    dir[5][0] =  0; dir[5][1] =  0; dir[5][2] = -1

    # 각 이웃 위치를 PBC에 따라 보정 후 상태값 합산
    for i in range(6):
        dx = dir[i][0]
        dy = dir[i][1]
        dz = dir[i][2]
        # Wrap or skip out-of-bound
        nx = wrap_coord(x + dx, lx, pbc0)
        ny = wrap_coord(y + dy, ly, pbc1)
        nz = wrap_coord(z + dz, lz, pbc2)

        # wrap_coord가 PBC 미적용 시 그대로 val 리턴하므로 경계 체크 추가 필요
        # if not pbc0 and (nx < 1 or nx >= lx - 1): continue
        # if not pbc1 and (ny < 1 or ny >= ly - 1): continue
        # if not pbc2 and (nz < 1 or nz >= lz - 1): continue

        val = lattice[nx, ny, nz]
        if val == 0 or val == 1:
            total += val
    return total


#----------------------------------------
# 표면 접촉 판정 함수
# z 방향으로 한 칸 아래가 2라면 현재 위치는 표면과 접촉한 것으로 간주
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int is_contact_surface(int x, int y, int z,
                            int64_t[:, :, :] lattice) nogil:
    # 표면과 접촉 판정 (z==1 위치 바로 아래가 2인 경우)
    if z - 1 < 0:
        return 0
    return 1 if lattice[x, y, z-1] == 2 else 0


#----------------------------------------
# PBC 경계면 복사 처리 (입자 이동후 update)
# 각 axis의 pbc에 따라 경계면 복사 처리
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void update_pbc_boundary(int64_t[:, :, :] lattice,
                                     int x, int y, int z,
                                     bint pbc0, bint pbc1, bint pbc2) nogil:
    cdef int dimx = lattice.shape[0]
    cdef int dimy = lattice.shape[1]
    cdef int dimz = lattice.shape[2]

    if pbc0:
        if x == 1:
            lattice[dimx - 1, y, z] = lattice[x, y, z]
        elif x == dimx - 2:
            lattice[0, y, z] = lattice[x, y, z]

    if pbc1:
        if y == 1:
            lattice[x, dimy - 1, z] = lattice[x, y, z]
        elif y == dimy - 2:
            lattice[x, 0, z] = lattice[x, y, z]

    if pbc2:
        if z == 1:
            lattice[x, y, dimz - 1] = lattice[x, y, z]
        elif z == dimz - 2:
            lattice[x, y, 0] = lattice[x, y, z]




#----------------------------------------
# 동질계 이동 함수
# lattice: 상태 배열
# hi: 에너지 데이터 (2D)
# neighbor_offsets: 가능한 이동 벡터들
# beta: 역온도 (1/kT)
# pbc0, pbc1, pbc2: 각 축 PBC 적용 여부
# coords: 이동 후보 좌표 리스트
# n: 후보 개수
# 반환값: 수락된 이동 횟수
cpdef int move_homo(int64_t[:, :, :] lattice,
                    double[:, :] hi,
                    int64_t[:, :] neighbor_offsets,
                    double beta,
                    bint pbc0, bint pbc1, bint pbc2,
                    int64_t[:, :] coords,
                    int n) nogil:
    cdef int i, x, y, z, nx, ny, nz, ox, oy, oz
    cdef int ci, cj, ci_sum, cj_sum, offset_idx, accepted = 0
    cdef double prob, dH
    
    for i in prange(n, nogil=False):
        x = coords[i,0]; y = coords[i,1]; z = coords[i,2]

        # 이동 방향 난수 선택은 GIL 필요 (rand 함수 호출 때문)
        with gil:
            offset_idx = rand() % neighbor_offsets.shape[0]
            # print(i)

        ox = neighbor_offsets[offset_idx,0]; oy = neighbor_offsets[offset_idx,1]; oz = neighbor_offsets[offset_idx,2]

        # 이동 후 좌표 PBC 보정
        nx = wrap_coord(x+ox, lattice.shape[0], pbc0)
        ny = wrap_coord(y+oy, lattice.shape[1], pbc1)
        nz = wrap_coord(z+oz, lattice.shape[2], pbc2)

        if not pbc0 and (nx < 1 or nx >= lattice.shape[0] - 1): continue
        if not pbc1 and (ny < 1 or ny >= lattice.shape[1] - 1): continue
        if not pbc2 and (nz < 1 or nz >= lattice.shape[2] - 1): continue

        ci = lattice[x,y,z]
        cj = lattice[nx,ny,nz]

        # 조건: 현재 위치는 1, 이동 대상 위치는 0 이어야 교환 가능
        if ci != 1 or cj != 0: continue

        # 이웃 합 계산
        ci_sum = get_neighbor_sum(lattice, x,y,z, pbc0,pbc1,pbc2)
        cj_sum = get_neighbor_sum(lattice, nx,ny,nz, pbc0,pbc1,pbc2)

        # 에너지 변화량 계산
        dH = calc_delta_H_homo(ci, cj, ci_sum, cj_sum, hi)

        # 이동 확률 결정용 난수 생성
        with gil:
            prob = rand() / <double>RAND_MAX

        # 이동 허용 조건: 에너지 감소하거나 확률 조건 만족 시
        if dH <= 0 or prob < exp(-beta * dH):
            lattice[x,y,z] = cj
            lattice[nx,ny,nz] = ci
            update_pbc_boundary(lattice, x, y, z, pbc0, pbc1, pbc2)
            update_pbc_boundary(lattice, nx, ny, nz, pbc0, pbc1, pbc2)
            accepted += 1

    return accepted


#----------------------------------------
# 이질계 이동 함수 (표면 구분 포함)
# 파라미터는 move_homo와 유사하나,
# hi는 3D 배열, 표면 여부 확인, 조건이 좀 더 까다로움
cpdef int move_hete(int64_t[:, :, :] lattice,
                    double[:, :, :] hi,
                    int64_t[:, :] neighbor_offsets,
                    double beta,
                    bint pbc0, bint pbc1, bint pbc2,
                    int64_t[:, :] coords,
                    int n) nogil:
    cdef int i, x, y, z, nx, ny, nz, ox, oy, oz
    cdef int ci, cj, ci_sum, cj_sum, ci_s, cj_s, offset_idx, accepted = 0
    cdef double prob, dH

    for i in prange(n, nogil=False):
        x = coords[i,0]; y = coords[i,1]; z = coords[i,2]

        with gil:
            offset_idx = rand() % neighbor_offsets.shape[0]
            # print(i)

        ox = neighbor_offsets[offset_idx,0]; oy = neighbor_offsets[offset_idx,1]; oz = neighbor_offsets[offset_idx,2]

        nx = wrap_coord(x+ox, lattice.shape[0], pbc0)
        ny = wrap_coord(y+oy, lattice.shape[1], pbc1)
        nz = wrap_coord(z+oz, lattice.shape[2], pbc2)

        if not pbc0 and (nx < 1 or nx >= lattice.shape[0] - 1): continue
        if not pbc1 and (ny < 1 or ny >= lattice.shape[1] - 1): continue
        if not pbc2 and (nz < 1 or nz >= lattice.shape[2] - 1): continue

        ci = lattice[x,y,z]
        cj = lattice[nx,ny,nz]

        # 조건: 현재 위치는 1, 이동 대상 위치는 0이면서 2(표면 상태)가 아니어야 함
        # if ci != 1 or cj == 1 or cj == 2: continue
        if ci != 1 or cj in (1, 2): continue

        ci_sum = get_neighbor_sum(lattice, x,y,z, pbc0,pbc1,pbc2)
        cj_sum = get_neighbor_sum(lattice, nx,ny,nz, pbc0,pbc1,pbc2)

        # 표면 접촉 여부 판정
        ci_s = is_contact_surface(x, y, z, lattice)
        cj_s = is_contact_surface(nx, ny, nz, lattice)

        # 에너지 변화량 계산 (표면 포함)
        dH = calc_delta_H_hete(ci, cj, ci_sum, cj_sum, ci_s, cj_s, hi)

        with gil:
            prob = rand() / <double>RAND_MAX

        if dH <= 0 or prob < exp(-beta * dH):
            lattice[x,y,z] = cj
            lattice[nx,ny,nz] = ci
            update_pbc_boundary(lattice, x, y, z, pbc0, pbc1, pbc2)
            update_pbc_boundary(lattice, nx, ny, nz, pbc0, pbc1, pbc2)
            accepted += 1

    return accepted
