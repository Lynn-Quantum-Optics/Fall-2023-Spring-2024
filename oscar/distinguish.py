# determines detection signature for a given d given an LELM matrix U

import numpy as np
from find_basis import * # for hyperentangled basis conversion

def get_signature(d, U, c, p):
    '''Returns signature for a given d, U, c, p'''
    # make bell state
    bell = make_bell(d, c, p)

    # convert to single particle basis
    single_particle_bell = get_single_particle(bell, d)

    # get signature
    if U.shape[0] == 2*d:
        U_t = np.kron(np.linalg.inv(U), np.linalg.inv(U))
    else:
        U_t = U
    
    meas =  U_t@single_particle_bell

    # round to 0 if very small
    meas[np.abs(meas) < 1e-10] = 0
    meas = meas.reshape(4*d**2,1)
    return meas

def compile_sig(d, U):
    '''Compiles all signatures for a given d and checks if they are linearly independent'''
    sigs = []
    for c in range(d):
        for p in range(d):
            sigs.append(get_signature(d, U, c, p))

    sig_mat = np.array(sigs)
    # print(sig_mat)

    def rows_to_eliminate(matrix):
        # Get indices of non-zero values for each column
        non_zero_indices = [np.nonzero(matrix[:, col])[0] for col in range(matrix.shape[1])]
        
        # Determine which rows to eliminate
        rows_to_remove = []
        for indices in non_zero_indices:
            # If there's more than one non-zero value in the column
            if len(indices) > 1:
                # check to see if we haven't already removed the row that it disagrees with
                rows_to_remove.extend(indices[1:])
        
        return sorted(set(rows_to_remove))

    # Find rows to eliminate for the given matrix
    rows_to_remove = rows_to_eliminate(sig_mat)
    return d**2 - len(rows_to_remove)

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


