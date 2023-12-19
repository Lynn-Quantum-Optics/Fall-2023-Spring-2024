# file to use sympy to determine symmetric bell basis

import numpy as np
import sympy as sp
from sympy.physics.quantum import TensorProduct

def sym_condition(d):
    '''Returns the condition for general symmetric state'''
    # make an empty d^2 x 1 vector
    vec = sp.zeros(d**2, 1)
    # fill in the vector
    j_sym_ls = []
    for j in range(d**2):
        L = j // d
        R = j % d

        index1 = d*L + R
        index2 = d*R + L

        if vec[index1] == 0 and vec[index2] == 0:
            j_sym = sp.Symbol(f'a_{L}{R}', complex = True)
            vec[index1] = j_sym
            vec[index2] = j_sym
            j_sym_ls.append(j_sym)

    print('Num free params:', len(j_sym_ls))
    
    return vec, j_sym_ls

def fully_entangled(vec, j_sym_ls):
    '''Computes the trace of the RDM^2 symbolically.'''
    # first get dimension
    d = int(np.sqrt(len(vec)))
    vec = sp.Matrix(vec)
    vec_dagger = vec.H
    
    # Compute the norm (sqrt of the dot product of v and its conjugate transpose)
    norm_v = sp.sqrt(vec_dagger.dot(vec))

    # Normalize the vector
    normalized_vec = vec / norm_v

    # get the DM
    rho = normalized_vec @ normalized_vec.H
    # get RDM
    rho_reduced = sp.Matrix.zeros(d, d)

    Id = sp.eye(d)

    # partial trace over the second subsystem
    for i in range(d):
        # projector for each basis state of the second subsystem
        basis_state = sp.zeros(d,1)
        basis_state[i] = 1
        projector = TensorProduct(basis_state, Id)
        print(projector.shape)

        # add to reduced density matrix
        rho_reduced += projector.T @ rho @ projector

    # normalize!
    rho_reduced = sp.trace(rho_reduced) * rho_reduced
    
    # expect 1/d for fully entangled system
    tr_sq =  sp.trace(rho_reduced @ rho_reduced)
    print('tr_sq', tr_sq)
    # solve for the parameters such that tr_sq = 1/d
    # separate into real and imaginary parts
    # define the equation
    equation = sp.Eq(tr_sq - 1/d, 0)

    # solve the equation
    sol = sp.solve(equation, j_sym_ls)
    print('sol', sol)
    return sol
    


if __name__ == '__main__':
    d = 6
    vec, j_sym_ls = sym_condition(d)
    sp.pprint(vec.T)
    print(fully_entangled(vec, j_sym_ls))
