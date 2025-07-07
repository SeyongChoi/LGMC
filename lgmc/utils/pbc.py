def apply_pbc(lattice, pbc):
    if pbc[0]:
        # -x-1 == x
        lattice[0, 1:-1, 1:-1] = lattice[-2, 1:-1, 1:-1]
        # +x+1 == -x
        lattice[-1, 1:-1, 1:-1] = lattice[1, 1:-1, 1:-1]
    if pbc[1]:
        # -y-1 == y
        lattice[1:-1, 0, 1:-1] = lattice[1:-1, -2, 1:-1]
        # +y+1 == -y
        lattice[1:-1, -1, 1:-1] = lattice[1:-1, 1, 1:-1]
    if pbc[2]:
        # -z-1 == z
        lattice[1:-1, 1:-1, 0] = lattice[1:-1, 1:-1, -2]
        # +z+1 == -z
        lattice[1:-1, 1:-1, -1] = lattice[1:-1, 1:-1, 1]

    return lattice
