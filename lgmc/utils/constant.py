# Avogadro's number
NA = 6.02214076e23 # [mol⁻¹]
# Boltzman Factor
K_b = 1.380649e-23# [J/K]

def J_to_eV(value, inverse=False):
    factor = 6.241509074e18  # 1 J = 6.241509074e18 eV
    return value / factor if inverse else value * factor

def J_to_kcal_mol(value, inverse=False):
    J_per_kcal = 0.000239006
    factor = NA * J_per_kcal  # 1 J = (NA / 4184) kcal/mol
    return value / factor if inverse else value * factor
    

def J_to_kJ_mol(value, inverse=False):
    factor = 1e-3 * NA  # 1 J = 1e-3 * NA kJ/mol
    return value / factor if inverse else value * factor

def J_to_Hartree(value, inverse=False):
    factor = 2.293712278e17  # 1 J = 2.293712278e17 Hartree
    return value / factor if inverse else value * factor


if __name__=='__main__':
    print("in eV/K:       ", J_to_eV(K_b))
    print("in kcal/mol/K: ", J_to_kcal_mol(K_b))  
    print("in kJ/mol/K:   ", J_to_kJ_mol(K_b))
    print("in Hartree/K:  ", J_to_Hartree(K_b))