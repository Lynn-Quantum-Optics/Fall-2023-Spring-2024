# file to create the bell states, in single and joint particle basis

import numpy as np
from math import comb

## define bell states for given d in joint-particle basis ##
def make_bell(d, c, p, b=True, options = 'none', joint=True):
    '''Makes single bell state of correlation class c and phase class c.
    Parameters:
    d (int): dimension of system
    c (int): correlation class
    p (int): phase class
    b (bool): whether to assume bosonic or fermionic statistics
    options (str): whether to include symmetric/antisymmetric part term or both
    single (bool): whether to calculate in joint basis or joint
    '''

    if joint:
        result = np.zeros((d**2, 1), dtype=complex)
        for j in range(d):
            # determine index of j in joint-particle basis
            j_vec = np.zeros((d**2, 1), dtype=complex)
            gamma = (j+c)%d
            first_index = d*j + gamma
            second_index = d*gamma + j

            # basis is in form |j, gamma>, where j is fixed when gamma is varied
            if options=='none':
                # proceed normally
                j_vec[first_index] = np.exp(2*np.pi*1j*j*p/d)

                # add symmetric/antisymmetric part: swap the L and R particles, so now gamma is fixed and j is varied
                if b: 
                    j_vec[second_index] = np.exp(2*np.pi*1j*j*p/d)
                else:
                    j_vec[second_index] = -np.exp(2*np.pi*1j*j*p/d)

            elif options == 'only_first': # only the first term
                j_vec[first_index] = np.exp(2*np.pi*1j*j*p/d)

            elif options == 'only_second': # only the second term
                if b:
                    j_vec[second_index] = np.exp(2*np.pi*1j*j*p/d)
                else:
                    j_vec[second_index] = -np.exp(2*np.pi*1j*j*p/d)
            else:
                raise ValueError(f'Invalid option flag: {options}. Valid options are: none, only_first, only_second')

            # add to result
            result += j_vec
    else:
        result = np.zeros((4*d**2, 1), dtype=complex)
        for j in range(d):
            j_vec = np.zeros((2*d, 1), dtype=complex)
            j_vec[j] = 1
            gamma_vec = np.zeros((2*d, 1), dtype=complex)
            gamma_vec[d+(j+c)%d] = 1

            if options == 'none':
                result += np.exp(2*np.pi*1j*j*p/d) * np.kron(j_vec, gamma_vec)
                # add symmetric/antisymmetric part
                if b:
                    result += np.exp(2*np.pi*1j*j*p/d)* np.kron(gamma_vec, j_vec)
                else:
                    result -= np.exp(2*np.pi*1j*j*p/d) * np.kron(gamma_vec, j_vec)

            elif options == 'only_first': # only the first erm
                result += np.exp(2*np.pi*1j*j*p/d) * np.kron(j_vec, gamma_vec)

            elif options == 'only_second': # only the second term
                # add symmetric/antisymmetric part
                if b:
                    result += np.exp(2*np.pi*1j*j*p/d)* np.kron(gamma_vec, j_vec)
                else:
                    result -= np.exp(2*np.pi*1j*j*p/d) * np.kron(gamma_vec, j_vec)

    # normalize
    result /= np.linalg.norm(result)
    return result

def num_overlapping(vectors):
    '''
    Finds the number of overlapping detection signatures in a set of vectors by converting to binary and dot producting each vector with all other vectors
    '''

    # convert to binary
    vectors = np.where(np.isclose(vectors, 0, rtol=1e-10), 0, 1)
    # print(vectors)
    # go through each vector and dot product with all other vectors
    num_overlapping = 0
    for i in range(vectors.shape[1]):
        for j in range(i+1, vectors.shape[1]):
            num_overlapping += np.dot(vectors[:, i], vectors[:, j])
    return num_overlapping

def CNS(n, k, x):
    '''Combinatorial Number System to convert index i to a k-tuple

    Params:
        n (int): number of elements to choose from
        k (int): size of tuple
        x (int): index to convert

    Returns:
        tuple: tuple representation of i
    '''
    
    result = []
    while k > 0:
        # find largest value v such that comb(v, k) <= x
        v = n
        while comb(v, k) > x:
            v -= 1
        result.append(v)
        x -= comb(v, k)
        k -= 1
        n = v
    return result