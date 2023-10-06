# file to explore possible U matrices to express detector modes as superpositions of bell states tensor producted with U and itself

import sympy as sp
import numpy as np
import itertools

# define bell states for given d
def make_bell(d, c, p, b=True):
    '''Makes single bell state of correlation class c and phase class c.
    Parameters:
    d (int): dimension of system
    c (int): correlation class
    p (int): phase class
    b (bool): whether to assume bosonic or fermionic statistics
    
    '''
    result = sp.Matrix(np.zeros((4*d**2, 1), dtype=complex))
    for j in range(d):
        j_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
        j_vec[j] = 1
        gamma_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
        gamma_vec[d+(j+c)%d] = 1
        # print(j, np.kron(j_vec, gamma_vec))
        result += sp.exp(2*sp.pi*1j*j*p/d) * np.kron(j_vec, gamma_vec)

        # # symmetric case
        # gamma_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
        # gamma_vec[(j+c)%d] = 1

        # j_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
        # j_vec[d+j] = 1

        if b:
            result += sp.exp(2*sp.pi*1j*j*p/d)* np.kron(gamma_vec, j_vec)
        else:
            result -= sp.exp(2*sp.pi*1j*j*p/d) * np.kron(gamma_vec, j_vec)
        # sp.pprint(result.T)

    # convert to Sympy
    result = sp.Matrix(result)
    result /= sp.sqrt(2*d)
    result = sp.simplify(result)
    # sp.pprint(result.T)
    return result

def bell_states(d):
    '''Uses single particle basis'''
    result_ls = []
    for c in range(d):
        for p in range(d):
            result_ls.append(make_bell(d, c, p))
    return result_ls

def QFT(d):
    '''Construct quantum fourier transform matrix for d-dimensional system in single particle basis. NOTE: THIS FORM MATCHES PISENTI'S ET AL MATRIX FOR D = 2 BUT WITH THE SECOND AND THIRD COLUMN SWAPPED FROM WHAT WE THINK THEY SHOULD BE BASED ON OUR DEFINITION OF BASIS. 

    Parameters:
    d (int): dimension of system
    '''
    # define U
    U = sp.Matrix()
    for j in range(d):
        col = sp.Matrix(np.zeros((d, 1)))
        for l in range(d):
            col[l] = sp.exp(2*sp.pi*1j*j*l/d)
        U = U.col_insert(j, col)
    U/= sp.sqrt(d)
    U = sp.simplify(U)

    # now add off diagonal blocks
    U_t = sp.Matrix(np.block([[U, U], [U, -U]]))
    return U_t

# def get_U(d):
#     '''Takes the QFT matrix and converts it to single particle basis'''
#     U = QFT(d)
#     sp.pprint(U)
#     U_t = sp.Matrix(np.kron(U, U))
#     # swap adjacent columns
#     for j in range(1, 2*d, 2):
#         if j < 2*d-1:
#             # sp.pprint(U_t.col(j).T)
#             U_t.col_swap(j, j+1)
#     # sp.pprint(U_t)
#     return U_t

def get_signature(d, c, p, b=True):
    '''Returns the signature in detector mode basis for a given bell state.
    Parameters:
    d (int): dimension of system
    c (int): correlation class
    p (int): phase class
    b (bool): whether to assume bosonic or fermionic statistics
    
    '''
    bell = make_bell(d, c, p, b)
    # print(bell.shape)
    # sp.pprint(bell.T)
    U = QFT(d)
    # print('U shape', U.shape)
    # U_t = np.kron(U, U)
    U_t = sp.Matrix(np.kron(U, U))
    # print('Ut shape', U_t.shape)
    # print('bell shape', bell.shape)
    return sp.simplify(U_t*bell)

def get_all_signatures(d, display=False):
    '''Calls get_signature for all bell states'''
    signatures = []
    for c in range(d):
        for p in range(d):
            sig = sp.simplify(get_signature(d, c, p))
            if display:
                print('----------------')
                print('c = ', c)
                print('p = ', p)
                sp.pprint(sig.T)
            signatures.append(sig)
    return signatures

def distinguish(d,k):
    '''Picks out k states from the d^2 bell states' detectio signatures and finds if they are orthogonal.
    IDEAL: LOOK AT ALL POSSIBLE STATES AND TRY TO FIND AN ORTHOGONAL SET OF k STATES AS OPPOSED TO JUST TRYING IERATIVELY
    
    '''
    signatures = get_all_signatures(d)



    # # find all combinations of k states -- RATHER THAN LOADING ALL STATES, USE CNS
    # combos = list(itertools.combinations(signatures, k))
    # # compare all detection signatures in combo; need to find if they are orthogonal
    # print('total', len(combos[0]))
    # for j, combos_l in enumerate(combos): # go through each combos elements
    #     for i in range(len(combos_l)-1):
    #         if sp.simplify(combos_l[i].T*combos_l[i+1])[0] != 0:
    #             print('not orthogonal', (combos_l[i].T*combos_l[i+1])[0])
    #             if j == len(combos)-1:
    #                 print('last combo, not orthogonal')
    #                 return False
    # print('orthogonal!')
    # return True





if __name__=='__main__':
    d = 3
    k = 3
    get_all_signatures(d, display=True)
    # distinguish(d, k)





    # bells = bell_states(d)
    # for b in bells:
    #     print('----------------')
    #     sp.pprint(b.T)
    # bell = make_bell(d, 0, 1)
    # sp.pprint(bell.T)
    # print(len(bell))

    # T = tsingle_to_joint(d)
    # sp.pprint(T)
    # sp.pprint(np.kron(get_U(d), get_U(d)))
    # sp.pprint(get_all_signatures(d))

    # testing----------------------------------
    # U = 0.5*sp.Matrix(np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]))
    # U = np.linalg.inv(U)
    # UtU = np.kron(U, U)
    # print('sum of col 2', sum(UtU[:, 2]))
    # print('sum of col 3', sum(UtU[:, 3]))
    # print('sum of col 6', sum(UtU[:, 6]))
    # print('sum of col 7', sum(UtU[:, 7]))
    # for c in range(d):
    #     for p in range(d):
    #         bell = make_bell(d, c, p)
           
    #         print('----------------')
    #         print('c = ', c)
    #         print('p = ', p)
    #         # sp.pprint(bell.T)
    #         sp.pprint((UtU*bell).T)



    # print('!!!------------!!!')

    # sp.pprint(U)
    # apply U to bell state, take tensor product with each component
    # bell = make_bell(2, 0, 0)
    # def measure_bell(U, d, c, p):
    #     '''Makes single bell state of correlation class c and phase class c'''
    #     result = sp.Matrix(np.zeros((4*d**2, 1), dtype=complex))
    #     for j in range(d):
    #         j_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
    #         j_vec[j] = 1
    #         gamma_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
    #         gamma_vec[d+(j+c)%d] = 1
    #         result += sp.Matrix(sp.exp(2*sp.pi*1j*j*p/d)*np.kron(U*j_vec, U*gamma_vec))

    #     # convert to Sympy
    #     result = sp.Matrix(result)
    #     result /= sp.sqrt(d)
    #     result = sp.simplify(result)
    #     sp.pprint(result.T)
    #     return result
    # for c in range(d):
    #     for p in range(d):
    #         print('----------------')
    #         print('c = ', c)
    #         print('p = ', p)
    #         measure_bell(U, d, c, p)

    # U = sp.Matrix(0.5*np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]))

    # # b00 = 0.5*np.array([0, 0, 1, 0, 0,0,0, 1, 1, 0,0,0,0,1,0,0])
    # print((sp.Matrix(np.kron(U, U))*make_bell(2, 0, 0, True)).T)
    # print((sp.Matrix(np.kron(U, U))*make_bell(2, 0, 1, True)).T)
    # print((sp.Matrix(np.kron(U, U))*make_bell(2, 1, 0, True)).T)
    # print((sp.Matrix(np.kron(U, U))*make_bell(2, 1, 1, True)).T)
    # make_bell(d, 0, 1, True)