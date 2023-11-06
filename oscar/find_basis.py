# file to find local transformation from d = 4 bell states to hyperentangled basis

import numpy as np
import pandas as pd

## define bell states for given d in joint-particle basis ##
def make_bell(d, c, p, b=True):
    '''Makes single bell state of correlation class c and phase class c.
    Parameters:
    d (int): dimension of system
    c (int): correlation class
    p (int): phase class
    b (bool): whether to assume bosonic or fermionic statistics

    '''
    result = np.zeros((d**2, 1), dtype=complex)
    for j in range(d):
        # determine index of j in joint-particle basis
        j_vec = np.zeros((d**2, 1), dtype=complex)
        gamma = (j+c)%d
        index = d*j + gamma
        j_vec[index] = 1

        # # add symmetric/antisymmetric part
        # if b:
        #     sym_index = d*gamma + j
        #     j_vec[sym_index] = 1
        # else:
        #     asym_index = d*gamma + j
        #     j_vec[asym_index] = -1

        result += np.exp(2*np.pi*1j*j*p/d) * j_vec

    result /= np.sqrt(d)
    return result

## for loss function to minimize ##
def reconstruct_bell(coeffs, hyper_basis):
    '''Reconstructs a (big) d bell state from coefficients in hyperentangled basis'''
    bell = hyper_basis @ coeffs
    bell = np.round(bell, 10)
    return bell

# for finding local transformation for particular c, p in d = 4 ##
def make_hyperentangled_basis(d=2):
    '''Makes hyperentangled basis as tensor product of two d bell states'''
    hyper_basis = np.zeros((d**4, d**4))
    j = 0
    for c1 in range(d):
        for c2 in range(d):
            for p1 in range(d):
                for p2 in range(d):
                    state = np.kron(make_bell(d, c1, p1), make_bell(d, c2, p2))
                    # assign to column of hyper_basis
                    hyper_basis[:, j] = state.reshape((d**4))
                    j+=1

    return hyper_basis

def find_trans_one(c, p, hyper_basis, d=4, alpha=0.0001):
    '''Finds local transformation for a d = 4 bell state given c, p with regularization'''
    # make bell state
    bell = make_bell(d, c, p)
    coeffs = np.linalg.lstsq(hyper_basis, bell, rcond=None)[0]
    resid = np.linalg.norm(bell - hyper_basis @ coeffs)
    return coeffs, resid

def find_trans_all(d=4):
    '''Returns matrix for expression for all d = 4 bell states in terms of hyperentangled basis'''
    hyper_basis = make_hyperentangled_basis(int(np.sqrt(d)))
    results = np.zeros((d**2, d**2), dtype=complex)
    resid_ls = []
    j = 0
    for c in range(d):
        for p in range(d):
            cp_trans, resid = find_trans_one(c, p, hyper_basis)
            # append as column to result
            cp_trans = cp_trans.reshape((d**2))
            results[:, j] = cp_trans
            resid_ls.append(resid)
            j+=1
    
    # round resulting matrix
    results = np.round(results, 10)
    results_pd = pd.DataFrame(results)
    results_pd.to_csv(f'local_transform/transformation_matrix_joint_{d}.csv')
    with open(f'local_transform/transformation_matrix_joint_{d}_latex.txt', 'w') as f:
        f.write(np_to_latex(results))
    return results, resid_ls

# function to convert to single particle basis #
# single particle basis is { |0, L>,  |1, L>,  |d-1, L>, |0, R>, |1, R>, |d-1, R> }
def convert_to_single_particle(results, d=4, in_type='coeffs'):
    '''Converts coefficient matrix from hyperentangled basis to single particle basis in the big d system.
    Params:
    results (np.array): matrix of in standard basis
    d (int): dimension of system
    in_type (str): whether we're dealing with special case of coefficient matrix or just some arbitrary bell state
    '''
    results_shape = results.shape
    # make single particle basis
    single_particle_results = np.zeros((4*d**2, results_shape[1]), dtype=complex)
    
    if results_shape[1] > 1:
        for j in range(d**2):
            # convert column to single particle basis
            col = results[:, j]
            col_single = np.zeros((4*d**2, 1), dtype=complex)
            for i in range(d**2):
                if col[i] != 0:
                    left = i % d  # location within joint particle basis
                    right = d+i//d

                    # get corresponding vector in single particle basis
                    left_vec = np.zeros((2*d, 1), dtype=complex)
                    left_vec[left] = 1
                    right_vec = np.zeros((2*d, 1), dtype=complex)
                    right_vec[right] = 1
                    # take tensor product, scale by coefficient

                    col_single += col[i]*np.kron(left_vec, right_vec)
            # append as column to single_particle_results
            single_particle_results[:, j] = col_single.reshape((4*d**2))
        if in_type == 'coeffs':
            single_particle_results = np.round(single_particle_results, 10)
            single_particle_results_pd = pd.DataFrame(single_particle_results)
            single_particle_results_pd.to_csv(f'local_transform/transformation_matrix_single_{d}.csv')

            with open(f'local_transform/transformation_matrix_single_{d}_latex.txt', 'w') as f:
                f.write(np_to_latex(single_particle_results))
    else:
        col = results
        col_single = np.zeros((4*d**2, 1), dtype=complex)
        for i in range(d**2):
            if col[i] != 0:
                left = i // d
                right = d+i%d

                # get corresponding vector in single particle basis
                left_vec = np.zeros((2*d, 1), dtype=complex)
                left_vec[left] = 1
                right_vec = np.zeros((2*d, 1), dtype=complex)
                right_vec[right] = 1

                # take tensor product, scale by coefficient
                col_single += col[i]*np.kron(left_vec, right_vec)
        single_particle_results = col_single.reshape((4*d**2))
        single_particle_results = np.round(single_particle_results, 10)

    return single_particle_results

# ---- helper function to  convert output to bmatrix in latex ---- #
def np_to_latex(array, precision=2):
    '''Converts a numpy array to a LaTeX bmatrix environment'''
    def format_complex(c, precision):
        '''Formats a complex number as a string'''
        if not(np.isclose(c.real, 0)) and not(np.isclose(c.imag, 0)):
            return format(c.real, f".{precision}f") + " + " + format(c.imag, f".{precision}f") + "i"
        elif np.isclose(c.imag, 0) and not(np.isclose(c.real, 0)):
            return format(c.real, f".{precision}f")
        elif np.isclose(c.real, 0) and not(np.isclose(c.imag, 0)):
            return format(c.imag, f".{precision}f") + "i"
        else:
            return "0"
    
    latex_str = "\\begin{bmatrix}\n"
    for row in array:
        row_str = " & ".join(format_complex(x, precision) for x in row)
        latex_str += row_str + " \\\\\n"
    latex_str += "\\end{bmatrix}"
    return latex_str

if __name__ == '__main__':
    d = 4
    results, resid_ls = find_trans_all(4)
    print(results)
    print(resid_ls)

    # convert to single particle basis
    single_particle_results = convert_to_single_particle(results, d)

    # factor into tensor product of U_L and U_R
    A, B = factor_single_particle(single_particle_results, d)
    print(A)
    print(B)
    


    

    ## ----------- testing ----------- ##
    # print(convert_to_single_particle(make_bell(d, 1, 2), d))

    # hyper_basis = make_hyperentangled_basis(2)
    # print(make_bell(4, 1, 0))
    # print(reconstruct_bell(find_trans_one(1, 0, hyper_basis)[0], hyper_basis))
    # print('here')
    # print('------')
    # print(make_bell(4, 0, 1))
    # print(reconstruct_bell(find_trans_one(0, 1, hyper_basis)[0], hyper_basis))

    # print(make_bell(4, 0, 0))
    # print(make_bell(4, 0, 1))
    # print(make_bell(2, 1, 1).shape)
    # print(np.kron(make_bell(2, 0, 0), make_bell(2, 1, 1)).shape)