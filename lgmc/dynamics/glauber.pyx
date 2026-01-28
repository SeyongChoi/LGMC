cimport cython
from cython.parallel import prange
# from libc.stdlib cimport rand, RAND_MAX


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
cdef extern from "stdlib.h" nogil:
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
# 동질계 flip 함수
# lattice: 상태 배열
# tprob: 전이 확률 테이블(2D)
# pbc0, pbc1, pbc2: 각 축 PBC 적용 여부
# n_attempts: 후보 개수
# 반환값: 수락된 이동 횟수
@cython.cdivision(True)
cpdef int flip_homo(int64_t[:, :, :] lattice,
                    double[:, :] tprob,
                    bint pbc0, bint pbc1, bint pbc2,
                    int n_attempts) nogil:
    cdef int i, x, y, z, ci, cj, cj_sum
    cdef int accepted = 0
    cdef int lx = lattice.shape[0]
    cdef int ly = lattice.shape[1]
    cdef int lz = lattice.shape[2]
    cdef double p, rnd_val
    
    # n_attempts 만큼 루피 (보통 lattice volume 만큼 수행하여 1MCS 정의) 
    for i in range(n_attempts):
        # 1. 임의의 사이트 선택 (Buffer 제외: 1 ~ dim-2)
        x = (rand() % (lx - 2)) + 1
        y = (rand() % (ly - 2)) + 1
        z = (rand() % (lz - 2)) + 1

        ci = lattice[x, y, z]
        
        # 0(빈 공간) 또는 1(입자)인 경우만 처리 (2는 기판 등 고정)
        if ci != 0 and ci != 1:
            continue
        
        # 2. 이웃 합 계산
        cj_sum = get_neighbor_sum(lattice, x,y, z, pbc0,pbc1,pbc2)
        
        # 3. 전이 확률 조회 (Prob 클래스에서 미리 계산된 값)
        # tprob[ci, cj_sum] = exp(-beta * dH)
        p = tprob[ci, cj_sum]

        # 4. Metropolis 판정
        # p >= 1.0 (dH <= 0) 이면 무조건 수락
        # p < 1.0 이면 확률적 수락
        if p >= 1.0:
            lattice[x, y, z] = 1 - ci # Flip state
            update_pbc_boundary(lattice, x, y, z, pbc0, pbc1, pbc2)
            accepted += 1
        else:
            rnd_val = rand() / <double>RAND_MAX
            if rnd_val < p:
                lattice[x, y, z] = 1 - ci # Flip state
                update_pbc_boundary(lattice, x, y, z, pbc0, pbc1, pbc2)
                accepted += 1
                
    return accepted


#----------------------------------------
# 이질계 flip 함수 (표면 구분 포함)
# 파라미터는 flip_homo와 유사하나,
# tprob는 3D 배열, 표면 여부 확인, 조건이 좀 더 까다로움
@cython.cdivision(True)
cpdef int flip_hete(int64_t[:, :, :] lattice,
                    double[:, :, :] tprob,
                    bint pbc0, bint pbc1, bint pbc2,
                    int n_attempts) nogil:
    cdef int i, x, y, z, ci, cj_sum, cs
    cdef int accepted = 0
    cdef int lx = lattice.shape[0]
    cdef int ly = lattice.shape[1]
    cdef int lz = lattice.shape[2]
    cdef double p, rnd_val

    for i in range(n_attempts):
        x = (rand() % (lx - 2)) + 1
        y = (rand() % (ly - 2)) + 1
        z = (rand() % (lz - 2)) + 1

        ci = lattice[x, y, z]
        if ci != 0 and ci != 1:
            continue

        cj_sum = get_neighbor_sum(lattice, x, y, z, pbc0, pbc1, pbc2)
        cs = is_contact_surface(x, y, z, lattice) # 표면 접촉 여부

        # tprob[ci, cs, cj_sum]
        p = tprob[ci, cs, cj_sum]

        if p >= 1.0:
            lattice[x, y, z] = 1 - ci
            update_pbc_boundary(lattice, x, y, z, pbc0, pbc1, pbc2)
            accepted += 1
        else:
            
            rnd_val = rand() / <double>RAND_MAX
            if rnd_val < p:
                lattice[x, y, z] = 1 - ci
                update_pbc_boundary(lattice, x, y, z, pbc0, pbc1, pbc2)
                accepted += 1

    return accepted
