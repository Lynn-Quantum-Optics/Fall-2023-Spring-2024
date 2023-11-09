# determines detection signature for a given d given an LELM matrix U

import numpy as np
from find_basis import * # for hyperentangled basis conversion

def get_signature(d, U, c, p, b=True):
    '''Returns signature for a given d, U, c, p'''
    # make bell state
    bell = make_bell_single(d, c, p, b)

    # get signature
    assert U.shape[0] == 2*d, f'U must be 2d x 2d. Your shape is {U.shape}'
    U_t = np.kron(np.linalg.inv(U), np.linalg.inv(U))
    
    meas =  U_t@bell

    # round to 0 if very small
    meas = np.round(meas, 10)
    meas = meas.reshape(4*d**2)
    return meas

def get_signature_hyper(d, U, cob, j, hyperbasis):
    '''Returns signature for a given d, U, j using hyperentangled basis.
    
    Parameters:
    d (int): dimension of system (sqrt of big d)
    U (np.array): LELM
    cob (np.array): transform from non hyperentangled basis to hyperentangled basis
    j (int): index of signature to get
    '''

    # make bell state
    bell = hyperbasis[:, j]
    bell = bell.reshape(d**2, 1)

    U_t = np.kron(np.linalg.inv(U), np.linalg.inv(U))
    U_t = cob.conj().T @ U_t @ cob

    meas =  U_t@bell

    # round to 0 if very small
    meas = np.round(meas, 10)
    meas = meas.reshape(d**2)
    return meas

def compile_sig(d, U, cob = None, hyperentangled=False):
    '''Compiles all signatures for a given d and checks if they are linearly independent.
    
    Parameters:
    d (int): dimension of system
    U (np.array): LELM matrix
    cob (np.array): change of basis matrix from non hyperentangled basis to hyperentangled basis
    hyperentangled (bool): whether to use hyperentangled basis or not
    
    '''
    
    if not(hyperentangled):
        sigs = np.zeros((4*d**2, d**2), dtype=complex)
        j = 0
        for c in range(d):
            for p in range(d):
                sig_cp = get_signature(d, U, c, p)
            
                # append as column vector
                sigs[:, j] = sig_cp
                j += 1

        # get columns that have no overlapping entries. start with the first and loop through the rest
        sigs_unique = [sigs[:, 0]]
        for j in range(1, sigs.shape[1]):
            sig_j = sigs[:, j]
            sig_j = sig_j.reshape(4*d**2)
            is_unique = True
            for sig_u in sigs_unique:
                sig_u = sig_u.reshape(4*d**2, 1)
                for k in range(sig_j.shape[0]):
                    if sig_j[k] == sig_u[k] and sig_j[k] != 0:
                        is_unique = False
                        break
            if is_unique: # must compare against all unique sigs
                sigs_unique.append(sig_j)
    else:
        assert cob is not None, 'Must provide cob if not using hyperentangled basis'

        sigs = np.zeros((d**2, d**2), dtype=complex)
        for j in range(d**2):
            hyperbasis = make_hyperentangled_basis(int(np.sqrt(d)))
            sig_j = get_signature_hyper(d, U, cob, j, hyperbasis)
            sigs[:, j] = sig_j

        # get columns that have no overlapping entries. start with the first and loop through the rest
        sigs_unique = [sigs[:, 0]]
        
        for j in range(1, sigs.shape[1]):
            sig_j = sigs[:, j]
            sig_j = sig_j.reshape(d**2)
            is_unique = True
            for sig_u in sigs_unique:
                sig_u = sig_u.reshape(d**2, 1)
                for k in range(sig_j.shape[0]):
                    if sig_j[k] == sig_u[k] and sig_j[k] != 0:
                        is_unique = False
                        break
            if is_unique:
                sigs_unique.append(sig_j)

    # convert the elements of sigs which are vectors to the columns of a matrix
    print(sigs_unique)
    sigs_unique = np.array(sigs_unique).T
    
    return sigs_unique.shape[1], sigs_unique
       
## define unitaries ##
def get_BS(d):
    '''Beamsplitter matrix for 2d x 2d single particle ket system'''
    BS = np.zeros((2*d, 2*d), dtype=complex) # 2d x 2d Unitary matrix
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
    with open('local_transform/BS_4.txt', 'w') as f:
        f.write('--------*******-------\n')
        f.write('before:\n')
        f.write('BS\n')
        f.write(np_to_latex(BS))
        f.write('---------------\n')
        f.write('U:\n')
        f.write(np_to_latex(U))
        f.write('---------------\n')
        num, sig = compile_sig(d, BS)
        f.write('num sigs: ' + str(num) + '\n')
        f.write('sigs:\n')
        f.write(np_to_latex(sig))
        f.write('--------*******-------\n')
        cob = np.load('local_transform/transformation_matrix_single_4.npy')
        print('cob:\n')
        print(cob)
        print(cob.shape)
        U_trans = cob.conj().T@U@cob
        f.write('after:\n')
        f.write('U trans:\n')
        f.write(np_to_latex(U_trans))
        f.write('---------------\n')
        num, sig = compile_sig(d, BS, cob, hyperentangled=True)
        f.write('num sigs: ' + str(num) + '\n')
        f.write('sigs:\n')
        f.write(np_to_latex(sig))
        f.write('--------*******-------\n')
       




