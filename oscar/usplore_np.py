# file to explore possible U matrices to express detector modes as superpositions of bell states tensor producted with U and itself
# adapting usplore.py to use numpy instead of sympy

import numpy as np
import sys

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

def QFT(d):
    '''Construct quantum fourier transform matrix for d-dimensional system in single particle basis. NOTE: THIS FORM MATCHES PISENTI'S ET AL MATRIX FOR D = 2 BUT WITH THE SECOND AND THIRD COLUMN SWAPPED FROM WHAT WE THINK THEY SHOULD BE BASED ON OUR DEFINITION OF BASIS. 

    Parameters:
    d (int): dimension of system
    '''
    # define U
    U = np.zeros((d, d), dtype=complex)
    for j in range(d):
        col = np.zeros((d, 1), dtype=complex)
        for l in range(d):
            col[l] = np.exp(2*np.pi*1j*j*l/d)
        U[:, j] = col[:, 0]
    U/= np.sqrt(2*d)

    # now add off diagonal blocks
    U_t = np.block([[U, U], [U, -U]])
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
    U = QFT(d)
    U_t = np.kron(U, U)

    meas =  U_t@bell
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
            sig = get_signature(d, c, p, b=b)

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
                expand = np.abs(sig[i])
                val =expand**2
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
        sig_mat = np.array(signatures).reshape(4*d**2, len(signatures))
        print(sig_mat)
        print('\n')
        print('----------------')
        print('generating QFT matrix: \n')
        print(QFT(d))
        print('\n')
        print('----------------')
        print('bell states: \n')
        for c in range(d):
            for p in range(d):
                print(f'c = {c}, p = {p}')
                print(make_bell(d, c, p, b).T)
                print('\n')

    sys.stdout = original_stdout  # Reset stdout back to the original
            
    return signatures

if __name__=='__main__':
    import matplotlib.pyplot as plt
    num_sig_ls = []
    min_d = 2
    max_d = 20
    for d in range(min_d, max_d+1):
        for stat in [True]:
            sig = len(get_all_signatures(d, display=True, b=stat))
            print(f'd = {d}, b = {stat}, sig = {sig}')
            num_sig_ls.append(sig)

    num_sig_ls = np.array(num_sig_ls)
    print(num_sig_ls)
    np.save(num_sig_ls, f'output/num_sig_{min_d}_{max_d}.npy')

    plt.figure(figsize=(10,10))
    plt.plot(range(min_d, max_d+1), num_sig_ls)
    plt.xlabel('Dimension of Entanglement')
    plt.ylabel('Number of Unique Signatures')
    plt.title('$LELM Disnguishability with \mathcal\{U\}_\\text{QFT}$')
    plt.savefig(f'output/num_sig_{min_d}_{max_d}.pdf')


    


        
