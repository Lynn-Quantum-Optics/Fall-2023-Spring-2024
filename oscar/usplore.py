# file to explore possible U matrices to express detector modes as superpositions of bell states tensor producted with U and itself

import sympy as sp
import numpy as np
import itertools, sys
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
                ket += f', |{left}>|{right}>: {np.round(sig[i],3)}'
            else:
                ket += f'|{left}>|{right}>: {np.round(sig[i],3)}'

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
                expand = sp.Abs(sig[i].expand(complex=True)).simplify()
                val =(expand**2).simplify()
                val_f = float(val.evalf())
                if val_f < 10**(-10): # remove small values
                    val_f = 0.0
                sig_mag.append(val_f)

            cp.append((c,p))
            signatures.append(sig_mag)
            if display:
                print('----------------')
                print('c = ', c)
                print('p = ', p)
                print(sig_mag)
    signatures = np.array(signatures, dtype=np.float64)

    # normalize signatures by sum since the vector stores probability
    for i in range(len(signatures)):
        signatures[i] /= np.sum(signatures[i])

    # remove redundant states
    signatures_old = signatures
    unique_signatures = []

    for row in signatures_old:
        is_duplicate = False
        for existing_row in unique_signatures:
            if np.allclose(row, existing_row, atol=10**(-10)):
                is_duplicate = True
                print('----')
                print('signatures unique', unique_signatures)
                print('redundant state found', row)
                print('----')
                break
            else:
                print('should be unique', row)
                print('existing row', existing_row)
        
        if not is_duplicate:
            unique_signatures.append(row)

    signatures = unique_signatures

    # get the corresponding cp values
    cp = [cp[i] for i in range(len(signatures_old)) if np.isin(signatures_old[i], signatures).all()]
    if display:
        print('----------------')
        
    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(f'output/{d}_{b}.txt', 'w') as f:
        sys.stdout = f  # Redirect stdout to the file
        if b:
            print('**bosonic**')
        else:
            print('**fermionic**')
        print(f'd = {d}\n')
        for i in range(len(signatures)):
            print('c = ', cp[i][0])
            print('p = ', cp[i][1])
            print(f'{convert_kets(signatures[i], d)}\n')

        print(f'number of unique states = {len(signatures)}. Matrix as detector mode per row, bell state per column\n')
        sig_mat = sp.Matrix(signatures).reshape(4*d**2, len(signatures))
        sp.pprint(sig_mat)
        print('\n')
        print('----------------')
        print('generating QFT matrix: \n')
        sp.pprint(QFT(d))
        print('\n')
        print('----------------')
        print('bell states: \n')
        for c in range(d):
            for p in range(d):
                print(f'c = {c}, p = {p}')
                sp.pprint(make_bell(d, c, p, b).T)
                print('\n')

    sys.stdout = original_stdout  # Reset stdout back to the original
            
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
        
    # visualize as matrix
    dots = sp.Matrix(precomputed_dots).reshape(len(signatures), len(signatures))
    sp.pprint(dots)

if __name__=='__main__':
    for d in [6]:
        for stat in [True, False]:
            get_all_signatures(d, display=True, b=stat)
    # sig = get_signature(6, 1, 1, b=True)
    # for i in range(4*6**2):
    #     # try:
    #     expand = sp.Abs(sig[i].expand(complex=True)).simplify()
    #     val =(expand**2).simplify()
    #     val_f = float(val.evalf())

    #         # val = float((sp.Abs(sig[i]).expand(complex=True).evalf())**2)
    #     # except:
    #     #     print('val', sig[i])
    #     #     print('val abs', sp.Abs(sig[i]).expand(complex=True))
    #     if val_f < 10**(-10): # remove small values
    #         val_f = 0.0
        


    ## manual testing----------
    # U = sp.Matrix(0.5*np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]))

    # print((sp.Matrix(np.kron(U, U))*make_bell(2, 0, 0, True)).T)
    # print((sp.Matrix(np.kron(U, U))*make_bell(2, 0, 1, True)).T)
    # print((sp.Matrix(np.kron(U, U))*make_bell(2, 1, 0, True)).T)
    # print((sp.Matrix(np.kron(U, U))*make_bell(2, 1, 1, True)).T)
    # make_bell(d, 0, 1, True)