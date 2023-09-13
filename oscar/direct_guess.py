# file based on solve_prep but instead of actually doing the minimization, just the direct solutions based on the intuition.
from scipy.optimize import minimize
from solve_prep import * # import HyperBell class to handle system generation

# def solve_ksys(m, hb):
#     '''Method to direct solve the systems of equations for the mth k-system.'''

#     hb.set_m(m) # set mth k-system
#     d = hb.get_d() # get dimension of system

#     # do guess of solution
#     # need random simplex for q, pick random r

def get_sep(bp=0, d=2):
    '''Get separable state in fock basis.
    Params:
        bp (int): basis position of separable state
        d (int): dimension of system
    '''
    # return np array; 2-hot encoding at index bp and d+bp.
    sep = np.zeros(2*d)
    sep[bp] = 1
    sep[d+bp] = 1
    return sep

def sep_in_ent(d, k=3, sep=None):
    '''Function to express separable state in entangled basis.
    Params:
        d (int): dimension of system
        k (int): number of distinguishable states (not used)
        sep (np array): separable state in fock basis: [0_L, 1_L, ..., d_L, 0_R, 1_R, ..., d_R]
    Returns:
        ent (np array): separable state in entangled basis.    
    '''
    # initialize HyperBell object
    hb = HyperBell()
    hb.init(d,k)
    # get sum of all bell states
    all_bell = hb.get_all_bell()
    # bell_sum = np.sum(all_bell, axis=0) / np.sqrt(d)
    # set up matrix equation; first get bell matrix which is the bell_sum but appended for each column 2d times
    # bell_matrix = np.tile(bell_sum, 2*d)
    bell_matrix = np.array(all_bell)
    bell_matrix = np.tile(bell_matrix, 2)
    # reshape
    bell_matrix = bell_matrix.reshape((2*d, 2*d**2))    
    print('bell matrix')
    print(bell_matrix)
    print('bell matrix shape')
    print(bell_matrix.shape)

    # # compute the pseudo inverse of bell_matrix [computationally intensive for large problems]
    # bm_inv = np.linalg.pinv(bell_matrix)
    # print('bm_inv')
    # print(bm_inv)
    # print('bm_inv shape')
    # print(bm_inv.shape)
    # # apply the pseudo-inverse to the target vector to obtain the `solution'
    # ent = bm_inv.dot(sep)
    # print('ent')
    # print(ent)
    # print('ent shape')
    # print(ent.shape)
    # # check that the solution is correct by multiplying bell_matrix by the solution
    # print('----')
    # print(sep)
    # print('----')
    # print(bell_matrix.dot(ent))

    # get separable state in entangled basis by minimizing the norm of the difference between the separable state and the bell matrix times the entangled state
    # initialize loss function
    def loss(ent):
        '''Loss function to minimize.'''
        # combine real and imaginary parts
        ent_total = np.concatenate([ent[:d], 1j*ent[d:]])
        return np.linalg.norm(bell_matrix.dot(ent_total) - sep)
    # initialize initial guess
    ent0 = np.random.rand(2*d**2)
    # minimize loss function
    ent = minimize(loss, ent0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    ent=ent.x
    print('ent')
    print(ent)
    # apply to bell matrix to get separable state in entangled basis
    ent_total = np.concatenate([ent[:d], 1j*ent[d:]])
    print('evaluate ent')
    print(bell_matrix.dot(ent_total))
    return ent



if __name__ == '__main__':
    # playing around with some stuff
    d=3
    sep = get_sep(bp=1,d=d)
    sep_in_ent(d, sep=sep)