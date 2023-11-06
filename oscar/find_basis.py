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


        # add symmetric/antisymmetric part
        if b:
            sym_index = d*gamma + j
            j_vec[sym_index] = 1
        else:
            asym_index = d*gamma + j
            j_vec[asym_index] = -1

        result += np.exp(2*np.pi*1j*j*p/d) * j_vec

    result /= np.sqrt(2*d)
    return result

## for loss function to minimize ##
def fidelity(vec1, vec2):
    '''Returns the fidelity of two bell state vectors in single particle basis'''
    state1 = vec1 @ vec1.conj().T
    state2 = vec2 @ vec2.conj().T
    return (np.trace(np.sqrt(np.sqrt(state1)@state2 @ np.sqrt(state1))))**2

def loss(vec1, vec2):
    '''Returns the loss of two bell state vectors in single particle basis'''
    return 1 - np.sqrt(fidelity(vec1, vec2))

# for finding local transformation for particular c, p in d = 4 ##
def make_hyperentangled_basis(d=2):
    '''Makes hyperentangled basis as tensor product of two d bell states'''
    hyper_basis = np.zeros((d**4, d**4))
    print(hyper_basis.shape)
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

def find_trans_one(c, p, hyper_basis, d=4, lambda_reg=1e-3):
    '''Finds local transformation for a d = 4 bell state given c, p with regularization'''
    # make bell state
    bell = make_bell(d, c, p)
    # add regularization term to hyper_basis
    hyper_basis_reg = hyper_basis.T @ hyper_basis + lambda_reg * np.eye(hyper_basis.shape[1])
    # compute the regularized pseudo-inverse of hyper_basis
    hyper_basis_pinv_reg = np.linalg.inv(hyper_basis_reg) @ hyper_basis.T
    # express bell as linear combination of hyperentangled basis; find coefficients
    coeffs = hyper_basis_pinv_reg @ bell
    # confirm solution is correct within a tolerance
    bell_recon = hyper_basis @ coeffs
    if not np.allclose(bell, bell_recon, atol=1e-10):
        print("Warning: the bell state reconstruction is not within the desired tolerance.")
    # compute the residual
    resid = np.linalg.norm(bell - bell_recon)**2
    return coeffs, resid

def find_trans_all(d=4):
    '''Returns matrix for expression for all d = 4 bell states in terms of hyperentangled basis'''
    hyper_basis = make_hyperentangled_basis(d)
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
            if j ==1:
                print(results)
    results_pd = pd.DataFrame(results)
    results_pd.to_csv(f'transformation_matrix_joint_{d}.csv')
    return results, resid_ls

def reconstruct_bell(coeffs, hyper_basis):
    '''Reconstructs a d = 4 bell state from coefficients in hyperentangled basis'''
    bell = hyper_basis @ coeffs
    bell = np.round(bell, 10)
    return bell


if __name__ == '__main__':
    hyper_basis = make_hyperentangled_basis(2)

    print(make_bell(4, 0, 0))
    print(reconstruct_bell(find_trans_one(0, 0, hyper_basis)[0], hyper_basis))
    print('------')
    print(make_bell(4, 0, 1))
    print(reconstruct_bell(find_trans_one(0, 1, hyper_basis)[0], hyper_basis))

    # print(make_bell(4, 0, 0))
    # print(make_bell(4, 0, 1))
    # print(make_bell(2, 1, 1).shape)
    # print(np.kron(make_bell(2, 0, 0), make_bell(2, 1, 1)).shape)