import numpy as np

def to_xyz(dim, lattice, filename, comment=None):
    
    atom_map = {1:'O', 2:'C'}

    # 버퍼 영역 제외한 입자 및 기판 좌표만 선택
    coords = np.argwhere(lattice > 0)
    filtered_coords = [
        (x, y, z) for (x, y, z) in coords
        if 1<= x <= dim - 2 and 1<= y <= dim - 2 and 1<= z <= dim - 2
    ]

    n_atoms = len(filtered_coords)
    lines = [str(n_atoms), comment]

    for x, y, z in filtered_coords:
        atom = atom_map.get(lattice[x, y, z], '?')
        lines.append(f"{atom} {x} {y} {z}")

    xyz_string = "\n".join(lines)

    if filename:
        with open(filename, 'w') as f:
            f.write(xyz_string)

    return xyz_string

