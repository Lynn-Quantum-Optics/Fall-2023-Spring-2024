11/23/23:
- added su_parametrize.py and bell.py. su_parametrize expresses U as tensor product of SU(2) and SU(3) and then minimizes the number of overlapping signatures (using code from bell.py).
- problem: this parametrization won't account for all SU(6) matrices. result is that min value for num_overlap was when params = [all 0], so acting with identity matrix. will implement general SU(n) construction.

11/6/23:
- find_basis.py currently finds the local change of basis from standard to hyperentangled basis -> converts to tensor product of single particle basis
    - currently only works for d = 4 to d = 2, but will be generalized to any prime factorization
    - studying if we can recover d = 4 distinguishability by converting the beam splitter matrix to hyperentangled basis
- distinguish.py
    - determines detection signatures for groups of bell states given some SU(2d) matrix
    - working with find_basis.py results to determine generalizability of hyperentangled basis