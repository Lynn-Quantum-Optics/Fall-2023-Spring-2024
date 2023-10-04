# file to explore possible U matrices to express detector modes as superpositions of bell states tensor producted with U and itself

import sympy as sp
import numpy as np

# define bell states for given d
def make_bell(d, c, p):
    '''Makes single bell state of correlation class c and phase class c'''
    result = sp.Matrix(np.zeros((4*d**2, 1), dtype=complex))
    for j in range(d):
        j_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
        j_vec[j] = 1
        gamma_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
        gamma_vec[d+(j+c)%d] = 1
        result += sp.Matrix(sp.exp(2*sp.pi*1j*j*p/d)*np.kron(j_vec, gamma_vec))
    # convert to Sympy
    result = sp.Matrix(result)
    result /= sp.sqrt(d)
    result = sp.simplify(result)
    return result

def bell_states(d):
    '''Uses single particle basis'''
    result_ls = []
    for c in range(d):
        for p in range(d):
            result_ls.append(make_bell(d, c, p))
    return result_ls

def QFT(d):
    '''Construct quantum fourier transform matrix for d-dimensional system in joint particle basis'''
    # define U
    U = sp.Matrix()
    for j in range(d):
        col = sp.Matrix(np.zeros((d, 1)))
        for l in range(d):
            col[l] = sp.exp(2*sp.pi*1j*j*l/d)
        U = U.col_insert(j, col)
    
    U/= sp.sqrt(d)
    U = sp.simplify(U)
    return U

def get_U(d):
    '''Takes the QFT matrix and converts it to single particle basis'''
    U = QFT(2*d)
    sp.pprint(U)
    U_t = sp.Matrix(np.kron(U, U))
    # swap adjacent columns
    for j in range(1, 2*d, 2):
        if j < 2*d-1:
            # sp.pprint(U_t.col(j).T)
            U_t.col_swap(j, j+1)
    # sp.pprint(U_t)
    return U_t

def get_signature(d, c, p):
    '''Returns the signature in detector mode basis for a given bell state'''
    bell = make_bell(d, c, p)
    # print(bell.shape)
    # sp.pprint(bell.T)
    U = get_U(d)
    print('U shape', U.shape)
    # U_t = np.kron(U, U)
    U_t = U
    print('Ut shape', U_t.shape)
    print('bell shape', bell.shape)
    return sp.simplify(U_t*bell)

def get_all_signatures(d):
    '''Calls get_signature for all bell states'''
    for c in range(d):
        for p in range(d):
            print('----------------')
            print('c = ', c)
            print('p = ', p)
            sp.pprint(get_signature(d, c, p).T)

if __name__=='__main__':
    d = 2
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
    U = 0.5*sp.Matrix(np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]))
    UtU = np.kron(U, U)
    print('sum of col 2', sum(UtU[:, 2]))
    print('sum of col 3', sum(UtU[:, 3]))
    print('sum of col 6', sum(UtU[:, 6]))
    print('sum of col 7', sum(UtU[:, 7]))
    for c in range(d):
        for p in range(d):
            bell = make_bell(d, c, p)
           
            print('----------------')
            print('c = ', c)
            print('p = ', p)
            # sp.pprint(bell.T)
            sp.pprint((UtU*bell).T)



    print('!!!------------!!!')

    # sp.pprint(U)
    # apply U to bell state, take tensor product with each component
    bell = make_bell(2, 0, 0)
    def measure_bell(U, d, c, p):
        '''Makes single bell state of correlation class c and phase class c'''
        result = sp.Matrix(np.zeros((4*d**2, 1), dtype=complex))
        for j in range(d):
            j_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
            j_vec[j] = 1
            gamma_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
            gamma_vec[d+(j+c)%d] = 1
            result += sp.Matrix(sp.exp(2*sp.pi*1j*j*p/d)*np.kron(U*j_vec, U*gamma_vec))

        # convert to Sympy
        result = sp.Matrix(result)
        result /= sp.sqrt(d)
        result = sp.simplify(result)
        sp.pprint(result.T)
        return result
    for c in range(d):
        for p in range(d):
            print('----------------')
            print('c = ', c)
            print('p = ', p)
            measure_bell(U, d, c, p)