# determines detection signature for a given d given an LELM matrix U

import numpy as np
import sympy
from find_basis import * # for hyperentangled basis conversion

def get_signature(d, U, c, p):
    '''Returns signature for a given d, U, c, p'''
    # make bell state
    bell = make_bell_single(d, c, p)

    # get signature
    if U.shape[0] == 2*d:
        U_t = np.kron(np.linalg.inv(U), np.linalg.inv(U))
    else: # if U is 4d^2 x 4d^2
        U_t = np.linalg.inv(U)
    
    meas =  U_t@bell

    # round to 0 if very small
    meas = np.round(meas, 10)
    meas = meas.reshape(4*d**2)
    return meas

def compile_sig(d, U):
    '''Compiles all signatures for a given d and checks if they are linearly independent'''
    sigs = np.zeros((4*d**2, d**2), dtype=complex)
    j = 0
    for c in range(d):
        for p in range(d):
            sig_cp = get_signature(d, U, c, p)
            # append as column vector
            sigs[:, j] = sig_cp
            j += 1
    
    # get rank of matrix
    # rank = np.linalg.matrix_rank(sigs)
    # print(rank)

    # get columns that have no overlapping entries. start with the first and loop through the rest
    sigs_unique = [sigs[:, 0]]
    for j in range(1, sigs.shape[1]):
        sig_j = sigs[:, j]
        sig_j = sig_j.reshape(4*d**2, 1)
        is_unique = True
        for sig_u in sigs_unique:
            sig_u = sig_u.reshape(4*d**2, 1)
            for k in range(sig_j.shape[0]):
                if sig_j[k] == sig_u[k] and sig_j[k] != 0:
                    is_unique = False
                    break
        if is_unique: # must compare against all unique sigs
            sigs_unique.append(sig_j)

    return len(sigs_unique)
       
## define unitaries ##
def get_BS(d):
    '''Beamsplitter matrix for 2d x 2d single particle ket system'''
    BS = np.zeros((2*d, 2*d), dtype=complex) # must be a column vector
    for j in range(2*d):
        if j < d:
            BS[j, j] = 1/np.sqrt(2)
            BS[j+d, j] = 1/np.sqrt(2)
        else:
            BS[j, j] = -1/np.sqrt(2)
            BS[j-d, j] = 1/np.sqrt(2)

    return BS

if __name__ == '__main__':
    # testing converting BS to hyperentangled basis
    np.set_printoptions(threshold=np.inf)

    d= 4
    BS = get_BS(d)
    U = np.kron(np.linalg.inv(BS), np.linalg.inv(BS))
    print('before:')
    print(U)
    print('---------------')
    cob = np.load('local_transform/transformation_matrix_single_4.npy')
    print('cob:')
    print(cob)
    U_trans = cob.conj().T@U@cob
    U_trans = np.round(U_trans, 10)
    print(U_trans)
    print(compile_sig(d, U))


