# file to generate random SU(2d) matrices to see if we beat a certain number of distinguishable

import numpy as np
import matplotlib.pyplot as plt

# define bell states for given d
def make_bell(d, c, p, b=True):
    '''Makes single bell state of correlation class c and phase class c.
    Parameters:
    d (int): dimension of system
    c (int): correlation class
    p (int): phase class
    b (bool): whether to assume bosonic or fermionic statistics

    '''
    result = np.zeros((4*d**2, 1), dtype=complex)
    for j in range(d):
        j_vec = np.zeros((2*d, 1), dtype=complex)
        j_vec[j] = 1
        gamma_vec = np.zeros((2*d, 1), dtype=complex)
        gamma_vec[d+(j+c)%d] = 1

        result += np.exp(2*np.pi*1j*j*p/d) * np.kron(j_vec, gamma_vec)

        # add symmetric/antisymmetric part
        if b:
            result += np.exp(2*np.pi*1j*j*p/d)* np.kron(gamma_vec, j_vec)
        else:
            result -= np.exp(2*np.pi*1j*j*p/d) * np.kron(gamma_vec, j_vec)

    result /= np.sqrt(2*d)
    return result

def get_signature(d, U, c, p, b=True):
    '''Returns the signature in detector mode basis for a given bell state.
    Parameters:
    d (int): dimension of system
    c (int): correlation class
    p (int): phase class
    b (bool): whether to assume bosonic or fermionic statistics
    bs (bool): whether to include beamsplitter in U
    do_QFT (bool): whether to include QFT in U
    '''
    bell = make_bell(d, c, p, b)
    U_t = np.kron(np.linalg.inv(U), np.linalg.inv(U))

    meas =  U_t@bell
    # round to 0 if very small
    meas[np.abs(meas) < 1e-10] = 0
    meas = meas.reshape(4*d**2,1)
    return meas

def compile_sig(d, b=True):
    '''Compiles all signatures for a given d and checks if they are linearly independent'''
    sigs = []
    U = gen_SU(2*d)
    # print(U)
    for c in range(d):
        for p in range(d):
            sigs.append(get_signature(d, U, c, p, b=b))

    sig_mat = np.array(sigs)
    print(sig_mat)

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
def givens_rotation(theta, phi):
    """
    Generate a 2x2 Givens rotation matrix with parameters theta and phi.
    """
    return np.array([[np.cos(theta) * np.exp(1j * phi / 2), -np.sin(theta) * np.exp(-1j * phi / 2)],
                     [np.sin(theta) * np.exp(1j * phi / 2), np.cos(theta) * np.exp(-1j * phi / 2)]])

def apply_givens_to_matrix(matrix, givens, i, j):
    """
    Apply a Givens rotation to rows i and j of a matrix.
    """
    n = matrix.shape[0]
    transformed_matrix = matrix.copy()
    for k in range(n):
        transformed_matrix[i, k], transformed_matrix[j, k] = np.dot(givens, [matrix[i, k], matrix[j, k]])
    return transformed_matrix

def gen_SU(n):
    # Ensure n is even for the intended structure
    if n % 2 != 0:
        raise ValueError("n should be even for this function to work as intended.")
    
    # Start with the identity matrix
    U = np.eye(n, dtype=complex)
    
    # For each column, apply a Givens rotation to two randomly chosen rows
    all_indices = np.arange(n)
    for col in range(n):
        i, j = np.random.choice(all_indices, size=2, replace=False)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        givens = givens_rotation(theta, phi)
        U = apply_givens_to_matrix(U, givens, i, j)
    
    # Adjust the phase of U to ensure it's in SU(n)
    U *= np.power(np.linalg.det(U), -1/n)
    
    return U


if __name__ == '__main__':
    # use multiprocessing to generate a bunch of random matrices and check if they are distinguishable for a given d
    import multiprocessing as mp

    d = 4
    max_iter = 1000

    # # use asynchronus pool to run in parallel
    # with mp.Pool(mp.cpu_count()) as pool:
    #     results = [pool.apply_async(compile_sig, args=(d,)) for _ in range(max_iter)]
    #     results = [p.get() for p in results]
    
    # # find the max number of distinguishable states
    # max_dist = max(results)
    # print(f'For d = {d}, max number of distinguishable states is {max_dist} out of {max_iter} random matrices')

    for i in range(max_iter):
        print(compile_sig(d))


    

    
  