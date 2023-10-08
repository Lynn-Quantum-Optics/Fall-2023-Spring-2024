# file to explore possible U matrices to express detector modes as superpositions of bell states tensor producted with U and itself

import sympy as sp
import numpy as np
import itertools
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
    result = sp.Matrix(np.zeros((4*d**2, 1), dtype=complex))
    for j in range(d):
        j_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
        j_vec[j] = 1
        gamma_vec = sp.Matrix(np.zeros((2*d, 1), dtype=complex))
        gamma_vec[d+(j+c)%d] = 1

        result += sp.exp(2*sp.pi*1j*j*p/d) * np.kron(j_vec, gamma_vec)

        if b:
            result += sp.exp(2*sp.pi*1j*j*p/d)* np.kron(gamma_vec, j_vec)
        else:
            result -= sp.exp(2*sp.pi*1j*j*p/d) * np.kron(gamma_vec, j_vec)

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
    U/= sp.sqrt(2*d)
    U = sp.simplify(U)

    # now add off diagonal blocks
    U_t = sp.Matrix(np.block([[U, U], [U, -U]]))
    return U_t

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
    meas =  sp.simplify(U_t*bell)
    meas = meas.reshape(4*d**2,1)
    return meas

def convert_kets(sig, d):
    '''Converts the signature to the ket representation'''
    ket = ''
    for i in range(len(sig)):
        if sig[i] != 0:
            left = i//(2*d)
            right= i % (2*d)
            if len(ket) != 0:
                ket += f' + {np.round(sig[i],3)}|{left}>|{right}>'
            else:
                ket += f'{np.round(sig[i],3)}|{left}>|{right}>'

    return ket


def get_all_signatures(d, b = True, display=False):
    '''Calls get_signature for all bell states'''
    signatures = []
    cp= []
    for c in range(d):
        for p in range(d):
            sig = sp.Matrix(sp.simplify(get_signature(d, c, p, b=b)))

            # apply b/f statistics---------
            # find if indices like |i>|j> -> |j>|i> are present
            for n in range(len(sig)):
                for q in range(n, len(sig)):
                    if n // (2*d) == q % (2*d) and n % (2*d) == q // (2*d) and n != q and sig[n] != 0 and sig[q] != 0:
                        if b:
                            if n < q:
                                sig[n] += sig[q]
                                sig[q] = 0
                            else:
                                sig[q] += sig[n]
                                sig[n] = 0
                        else:
                            if n < q:
                                sig[n] -= sig[q]
                                sig[q] = 0
                            else:
                                sig[q] -= sig[n]
                                sig[n] = 0
            # -----------------------------
            sig_mag = []
            for i in range(4*d**2):
                val = float((sp.Abs(sig[i]).expand(complex=True))**2)
                if val < 10**(-10): # remove small values
                    val = 0.0
                sig_mag.append(val)

            cp.append((c,p))
            signatures.append(sig_mag)
            if display:
                print('----------------')
                print('c = ', c)
                print('p = ', p)
                print(sig_mag)
    signatures = np.array(signatures, dtype=np.float64)
    if display:
        # remove redundant states
        print(sig)
        signatures_old = signatures
        signatures = np.unique(signatures, axis=0)
        # get the corresponding cp values
        cp = [cp[i] for i in range(len(signatures_old)) if np.isin(signatures_old[i], signatures).all()]
        print('----------------')
        for i in range(len(signatures)):
            print('c = ', cp[i][0])
            print('p = ', cp[i][1])
            print(convert_kets(signatures[i], d))
        
        sig_mat = sp.Matrix(signatures).reshape(4*d**2, len(signatures))
        sp.pprint(sig_mat)
    return signatures

def distinguish(d,k):
    '''Picks out k states from the d^2 bell states' detectio signatures and finds if they are orthogonal.
    IDEAL: LOOK AT ALL POSSIBLE STATES AND TRY TO FIND AN ORTHOGONAL SET OF k STATES AS OPPOSED TO JUST TRYING IERATIVELY:
    - start with the first one, see if can find k orthogonal states
    
    '''
    signatures = get_all_signatures(d)
    # normalize all signatures
    for i in range(len(signatures)):
        signatures[i] /= signatures[i].norm()

    # def is_orthogonal(dot_product):
    #     return np.isclose(dot_product, 0)

    # def find_largest_orthogonal_group(vectors, precomputed_dots, current_group=None, index=0, memo=None):
    #     if current_group is None:
    #         current_group = []
        
    #     if memo is None:
    #         memo = {}

    #     if index == len(vectors):
    #         return current_group

    #     # Early stopping
    #     if len(current_group) + (len(vectors) - index) <= len(memo.get(tuple(sorted(current_group)), [])):
    #         return []

    #     state = tuple(sorted(current_group + [index]))
    #     if state in memo:
    #         return memo[state]

    #     next_vector = vectors[index]

    #     # If the next_vector is orthogonal to all vectors in current_group, 
    #     # then explore the possibility of adding it to the group
    #     if all(is_orthogonal(precomputed_dots[i][index]) for i in current_group):
    #         with_vector = find_largest_orthogonal_group(vectors, precomputed_dots, current_group + [index], index + 1, memo)
    #         without_vector = find_largest_orthogonal_group(vectors, precomputed_dots, current_group, index + 1, memo)
    #         largest = with_vector if len(with_vector) > len(without_vector) else without_vector
    #     else:
    #         largest = find_largest_orthogonal_group(vectors, precomputed_dots, current_group, index + 1, memo)

    #     memo[state] = largest
    #     return largest

    # Precompute dot products
    precomputed_dots = []
    for i in range(len(signatures)):
        i_th_dots = []
        for j in range(len(signatures)):
            if i != j:
                i_th_dots.append(sp.simplify(signatures[i].T*signatures[j])[0])
            else:
                i_th_dots.append(0)
        precomputed_dots.append(i_th_dots)
        # count number of non-zero elements
        i_non_zero= []
        for elem in i_th_dots:
            if elem != 0:
                i_non_zero.append(1)
        
        # print(i_th_dots, sum(i_non_zero))

    

    # visualize as matrix
    dots = sp.Matrix(precomputed_dots).reshape(len(signatures), len(signatures))
    sp.pprint(dots)
    # plt.imshow(dots)
  

    # largest_group_indices = find_largest_orthogonal_group(signatures, precomputed_dots)
    # largest_group = [signatures[i] for i in largest_group_indices]
    # for vec in largest_group:
    #     print(vec)




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
    d = 5
    k = 3
    get_all_signatures(d, display=True, b=False)
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