LO: 11/6/23

Files I'm using as of LO:
- find_basis.py currently finds the local change of basis from standard to hyperentangled basis -> converts to tensor product of single particle basis
    - currently only works for d = 4 to d = 2, but will be generalized to any prime factorization
    - studying if we can recover d = 4 distinguishability by converting the beam splitter matrix to hyperentangled basis
- distinguish.py
    - determines detection signatures for groups of bell states given some SU(2d) matrix
    - working with find_basis.py results to determine generalizability of hyperentangled basis