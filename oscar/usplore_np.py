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
    U_t = np.kron(np.linalg.inv(U), np.linalg.inv(U))

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


def get_all_signatures(d, b = True, display=False, ret_len=True):
    '''Calls get_signature for all bell states'''
    signatures = []
    cp= []
    for c in range(d):
        for p in range(d):
            if display:
                print(f'(c,p) = ({c},{p})')
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
                if display:
                    print('----')
                    print('signatures unique', unique_signatures)
                    print('redundant state found', row)
                    print('----')
                break
            else:
                if display:
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
            
    if ret_len:
        return len(signatures)
    else:
        return signatures

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from multiprocessing import Pool, cpu_count
    from functools import partial
    import os

    def compute_num_sig(min_d, max_d, b=True, parallel=True, display=False):
        '''Compute the number of signatures for all d in range(min_d, max_d+1) and save to file.

        Parameters:
        min_d (int): minimum d value
        max_d (int): maximum d value
        b (bool): bosonic or fermionic statistics
        parallel (bool): use multiprocessing or not
        
        '''
        num_sig_ls = []
  
        # do multiprocessing
        if parallel:
            func = partial(get_all_signatures, b=b, display=display, ret_len=True)
            with Pool(cpu_count()) as po:
                num_sig_ls = po.map_async(func, range(min_d, max_d+1)).get()

        # do serial
        else:
            for d in range(min_d, max_d+1):
                for stat in [True]:
                    sig = get_all_signatures(d, display=display, b=stat, ret_len=True)
                    print(f'd = {d}, b = {stat}, sig = {sig}')
                    num_sig_ls.append(sig)
                    np.save(f'output/num_sig_{min_d}_{max_d}.npy', num_sig_ls)

        num_sig_ls = np.array(num_sig_ls)
        print(num_sig_ls)
        np.save(f'output/num_sig_{min_d}_{max_d}.npy', num_sig_ls, allow_pickle=True)

    def plot(min_d_max_d_ls):
        '''Plot results of get_all_signatures given list of tuples (min_d, max_d)'''

        min_min_d = min([min_d for min_d, max_d in min_d_max_d_ls])
        max_max_d = max([max_d for min_d, max_d in min_d_max_d_ls])

        fig, ax = plt.subplots(2, 1, figsize=(10,10))

        for i, (min_d, max_d) in enumerate(min_d_max_d_ls):
            num_sig_ls = np.load(f'output/num_sig_{min_d}_{max_d}.npy')
            # determine the max power of 2 to use in pisenti comparison
            min_2 = int(np.emath.logn(2, min_d))
            max_2 = int(np.emath.logn(2, max_d))
            max_2+=1


            if i ==0:
                # plot number distinguished
                ax[0].scatter(np.arange(min_d, max_d+1, 1), num_sig_ls, label='This work', color='blue')
                # plot even answer: for d = 2^n, 2^{n+1}-1 unique states
                ax[0].scatter(2**(np.arange(min_2, max_2,1)), 2**(np.arange(min_2, max_2,1)+1)-1, label='Pisenti et al. 2011', color='red')
                ax[0].scatter([3], [3], label='Leslie et al. 2019', color='orange')
        
                # plot percent distinguished
                ax[1].scatter(np.arange(min_d, max_d+1, 1), num_sig_ls/(np.arange(min_d, max_d+1, 1)**2), label='This work', color='blue')
                ax[1].scatter(2**(np.arange(min_2, max_2,1)), (2**(np.arange(min_2, max_2,1)+1)-1) / (2**(np.arange(min_2, max_2,1)))**2, label='Pisenti et al. 2011', color='red')
                ax[1].scatter([3], [3/9], label='Leslie et al. 2019', color='orange')
            else:
                # same as above but without labels
                ax[0].scatter(np.arange(min_d, max_d+1, 1), num_sig_ls, color='blue')
                ax[0].scatter(2**(np.arange(min_2, max_2,1)), 2**(np.arange(min_2, max_2,1)+1)-1, color='red')
                ax[0].scatter([3], [3], color='orange')

                ax[1].scatter(np.arange(min_d, max_d+1, 1), num_sig_ls/(np.arange(min_d, max_d+1, 1)**2), color='blue')
                ax[1].scatter(2**(np.arange(min_2, max_2,1)), (2**(np.arange(min_2, max_2,1)+1)-1) / (2**(np.arange(min_2, max_2,1)))**2, color='red')
                ax[1].scatter([3], [3/9], color='orange')
        
        ax[0].set_xlim(min_min_d, max_max_d+1)
        ax[0].set_xlabel('Dimension of Entanglement')
        ax[0].set_ylabel('Number of Unique Signatures')
        ax[1].set_xlim(min_min_d, max_max_d+1)
        ax[1].set_xlabel('Dimension of Entanglement')
        ax[1].set_ylabel('Percent Distinguished')
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='upper right')
        plt.title('LELM Distinguishability with QFT')
        plt.tight_layout()
        plt.savefig(f'output/num_sig_{min_min_d}_{max_max_d}.pdf')

    def combine(subdir):
        '''Extract results from each file in subdir and combine into one file'''
        num_sig_d= []
        min_d = 999
        max_d = 0
        subdir = 'output/' + subdir
        for file in os.listdir(subdir):
            # in the text file, search for line 'unique states = ' and extract the number
            d= int(file.split('_')[0])
            if d < min_d:
                min_d = d
            if d > max_d:
                max_d = d
            with open(f'{subdir}/{file}', 'r') as f:
                for line in f:
                    if 'unique states = ' in line:
                        num = int(line.split('unique states = ')[1].split('.')[0])
                        print(line)
                        print(num)
                        print(file)
                        num_sig_d.append([d, num])
                        break
        # sort based on d
        num_sig_d = sorted(num_sig_d, key=lambda x: x[0])
        # get array of only the number of unique states
        num_sig_ls = np.array([num for d, num in num_sig_d])
        # print(num_sig_ls)
        
        np.save(f'output/num_sig_{min_d}_{max_d}.npy', num_sig_ls)

    # compute_num_sig(38, 60, parallel=True, display=False)
    # combine('new')
    plot([(2, 20), (21, 37)])


    


        
