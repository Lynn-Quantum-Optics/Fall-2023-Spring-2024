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
def reconstruct_bell(coeffs, hyper_basis, d):
    '''Reconstructs a d = 4 bell state from coefficients in hyperentangled basis'''
    coeffs_tot = coeffs[:d**2] + 1j*coeffs[d**2:]
    coeffs_tot = coeffs_tot.reshape((d**2,1))
    bell = hyper_basis @ coeffs_tot
    bell = np.round(bell, 10)
    return bell


# def get_fidelity(vec1, vec2):
#     '''Returns the fidelity of two bell state vectors in single particle basis'''
#     # divide each of the vectors by their norm
#     vec1 /= np.linalg.norm(vec1)
#     vec2 /= np.linalg.norm(vec2)
#     state1 = vec1 @ vec1.conj().T
#     state2 = vec2 @ vec2.conj().T
#     # print(state1)
#     # print(state2)
#     return (np.trace(np.sqrt(np.sqrt(state1)@state2 @ np.sqrt(state1))))**2

# def get_loss(coeffs, bell, hyper_basis, d):
#     '''Returns the loss of two bell state vectors in single particle basis'''
#     # convert coeffs to complex
#     coeffs = coeffs.reshape((2*d**2,1))
#     bell = bell.reshape((d**2,1))
#     bell_recon = reconstruct_bell(coeffs, hyper_basis, d)
    
#     # find the sum of the squared differences
#     loss = np.sum(np.abs(bell - bell_recon)**2)
#     return loss

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

# def find_trans_one(c, p, hyper_basis, d=4, max_iter=1000, l_threshold = 1e-5, frac = 0.01, zeta = 0.01, verbose=False):
#     '''Finds local transformation for a d = 4 bell state given c, p with regularization.
#     Params:
#         c (int): correlation class
#         p (int): phase class
#         hyper_basis (np.array): hyperentangled basis
#         d (int): dimension of system
#         max_iter (int): maximum number of iterations for gradient descent
#         f_threshold (float): threshold for fidelity of local transformation
#         frac (float): fraction of max_iter to introduce new random guess if old one hasn't produced new max_fidelity
#         zeta (float): learning rate
#         verbose (bool): whether to print out progress
#     '''
#     # make bell state
#     bell = make_bell(d, c, p)
    
#     # find coefficients in hyperentangled basis using gradient descent
#     def guess():
#         '''Random simplex in d**2 dimensional complex space'''
#         real_part = np.random.rand(d**2)
#         imag_part = np.random.rand(d**2)
#         # Normalize the vector as all real concatenated with all imaginary
#         return np.concatenate((real_part, imag_part)) 

#     def minimize_fid(coeffs):
#         '''Minimizes loss function for given coefficients'''
#         result = minimize(get_loss, coeffs, args=(bell, hyper_basis, d), method='Nelder-Mead', tol=1e-10)

#         return result.x, 1 - result.fun

#     coeffs = guess()
#     coeffs, loss = minimize_fid(coeffs)

#     max_coeffs = coeffs
#     min_loss = loss

#     get_loss_args = partial(get_loss, bell=bell, hyper_basis=hyper_basis, d=d)

#     n = 0
#     index_since_improvement =0
#     while min_loss > l_threshold and n < max_iter:
#         if verbose:
#             print(n, min_loss, coeffs)
        
#         gradient = approx_fprime(coeffs, get_loss_args, epsilon=1e-8) # epsilon is step size in finite difference

#         # update coeff based on gradient
#         coeffs = [max_coeffs[i] - zeta*gradient[i] for i in range(len(max_coeffs))]

#         coeffs, loss = minimize_fid(coeffs)

#         if loss < min_loss:
#             min_loss = loss
#             max_coeffs = coeffs
#             index_since_improvement = 0
#             if loss < l_threshold:
#                 break
#         else:
#             index_since_improvement += 1
#             if index_since_improvement > frac * max_iter:
#                 coeffs = guess()
#                 coeffs, fidelity = minimize_fid(coeffs)
#                 index_since_improvement = 0

#         n += 1

#     return max_coeffs, min_loss


def find_trans_one(c, p, hyper_basis, d=4, lambda_reg=1e-6):
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


if __name__ == '__main__':
    hyper_basis = make_hyperentangled_basis(2)

    print(make_bell(4, 0, 0))
    print(reconstruct_bell(find_trans_one(0, 0, hyper_basis)[0], hyper_basis, 4))
    # print('------')
    # print(make_bell(4, 0, 1))
    # print(reconstruct_bell(find_trans_one(0, 1, hyper_basis)[0], hyper_basis))

    # print(make_bell(4, 0, 0))
    # print(make_bell(4, 0, 1))
    # print(make_bell(2, 1, 1).shape)
    # print(np.kron(make_bell(2, 0, 0), make_bell(2, 1, 1)).shape)