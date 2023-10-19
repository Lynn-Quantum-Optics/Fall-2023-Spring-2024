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
    U/= np.sqrt(d)

    # now add off diagonal blocks
    U_t = np.block([[U, U], [U, -U]])
    return U_t

def get_bs(d):
    '''Beamsplitter matrix for 2d x 2d single particle ket system'''
    BS = np.zeros((2*d, 2*d), dtype=complex) # must be a column vector
    for j in range(2*d):
        if j < d:
            BS[j, j] = 1/np.sqrt(2)
            BS[j+d, j] = 1/np.sqrt(2)
        else:
            BS[j, j] = -1/np.sqrt(2)
            BS[j-d, j] = 1/np.sqrt(2)

    return BS

def get_U(d, bs=True, do_QFT=True):
    '''Create U from composing beamsplitter with QFT'''
    if do_QFT:
        QFT_m = QFT(d)
    if bs:
        BS = get_bs(d)

    if do_QFT and bs:
        U = QFT_m@BS
    elif do_QFT:
        U = QFT_m
    elif bs:
        U = BS

    return U


def get_signature(d, c, p, b=True, bs = True, do_QFT=True):
    '''Returns the signature in detector mode basis for a given bell state.
    Parameters:
    d (int): dimension of system
    c (int): correlation class
    p (int): phase class
    b (bool): whether to assume bosonic or fermionic statistics
    bs (bool): whether to include beamsplitter in U
    do_QFT (bool): whether to include QFT in U
    '''
    bell = make_bell(d, c, p, b)
    if do_QFT:
        U = get_U(d, bs=bs, do_QFT=do_QFT)
    else:
        U = get_bs(d)
    U_t = np.kron(np.linalg.inv(U), np.linalg.inv(U))

    meas =  U_t@bell
    # round to 0 if very small
    meas[np.abs(meas) < 1e-10] = 0
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

def compile_sig(d, b=True, bs=True, do_QFT=False):
    '''Compiles all signatures for a given d and checks if they are linearly independent'''
    sigs = []
    for c in range(d):
        for p in range(d):
            sigs.append(get_signature(d, c, p, b=b, bs=bs, do_QFT=do_QFT))

    sig_mat = np.array(sigs)

    print(sig_mat)

    def rows_to_eliminate(matrix):
        # Get indices of non-zero values for each column
        non_zero_indices = [np.nonzero(matrix[:, col])[0] for col in range(matrix.shape[1])]
        print(non_zero_indices)
        
        # Determine which rows to eliminate
        rows_to_remove = []
        for indices in non_zero_indices:
            # If there's more than one non-zero value in the column
            if len(indices) > 1:
                # check to see if we haven't already removed the row that it disagrees with
                rows_to_remove.extend(indices[1:])
        
        return sorted(set(rows_to_remove))

    # Find rows to eliminate for the given matrix
    rows_to_remove = rows_to_eliminate(sig_mat)
    print(rows_to_remove)
    return d**2 - len(rows_to_remove)




def get_all_signatures(d, b = True, bs = True, do_QFT=True, display=False, ret_len=True):
    '''Calls get_signature for all bell states'''
    signatures = []
    cp= []
    for c in range(d):
        for p in range(d):
            sig = get_signature(d, c, p, b=b, bs=bs, do_QFT=do_QFT)
            # if display:
            #     print('----------------')
            #     print(f'(c,p) = ({c},{p})')
            #     print('raw signature', convert_kets(sig, d))
            #     print(sig.T)
            #     print('----------------')

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
                print(convert_kets(sig_mag, d))
    signatures = np.array(signatures, dtype=np.float64)

    # normalize signatures by sum since the vector stores probability
    for i in range(len(signatures)):
        signatures[i] /= np.sum(signatures[i])

    # remove redundant states
    signatures_old = signatures
    unique_signatures = []

    # need to impose orthogonality condition
    for row in signatures_old:
        is_duplicate = False
        for existing_row in unique_signatures:
            print('dot product', np.dot(row, existing_row))
            if not(np.allclose(np.dot(row, existing_row), 0, atol=10**(-10))):
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
    from scipy.optimize import curve_fit

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

    def plot_sep(min_d_max_d_ls):
        '''Plot results of get_all_signatures given list of tuples (min_d, max_d)'''

        min_min_d = min([min_d for min_d, max_d in min_d_max_d_ls])
        max_max_d = max([max_d for min_d, max_d in min_d_max_d_ls])

        fig, ax = plt.subplots(2, 1, figsize=(10,10), sharex=True)

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
        ax[0].set_ylabel('Number of Unique Signatures')
        ax[1].set_xlim(min_min_d, max_max_d+1)
        ax[1].set_xlabel('Dimension of Entanglement')
        ax[1].set_ylabel('Percent Distinguished')
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='upper right')
        plt.suptitle('LELM Distinguishability with QFT')
        plt.tight_layout()
        plt.savefig(f'output/num_sig_{min_min_d}_{max_max_d}.pdf')

    def plot_one(min_d, max_d, fit_start=-1):
        '''Plot results of get_all_signatures given list of tuples (min_d, max_d). Fit curves to even/odd entries separately. Assume fit_start is even.'''

        fig, ax = plt.subplots(2, 1, figsize=(10,10), sharex=True)

        num_sig_ls = np.load(f'output/num_sig_{min_d}_{max_d}.npy')
        # determine the max power of 2 to use in pisenti comparison
        min_2 = int(np.emath.logn(2, min_d))
        max_2 = int(np.emath.logn(2, max_d))
        max_2+=1

        if fit_start >= 0:

            # fit curves to even/odd--------------------
            def func(x, a, b, c):
                return a * np.exp(b * x)   + c
            # select only odd entries of num_sig_ls
            num_sig_ls_r = num_sig_ls[fit_start-min_d:]
            num_sig_ls_odd = num_sig_ls_r[1::2]
            # fit curve to odd entries
            # start fitting from start 

            print(np.arange(fit_start+1, max_d+1, 2))
            print(num_sig_ls_odd)

            popt_odd, pcov_odd = curve_fit(func, np.arange(fit_start, max_d, 2), num_sig_ls_odd)
            # fit curve to even entries
            num_sig_ls_even = num_sig_ls_r[::2]
            print(np.arange(fit_start, max_d+1, 2).shape, num_sig_ls_even.shape)
            popt_even, pcov_even = curve_fit(func, np.arange(fit_start, max_d+2, 2), num_sig_ls_even)

            ax[0].plot(np.linspace(fit_start+1, max_d, 1000), func(np.linspace(fit_start, max_d, 1000), *popt_odd), color='gray', linestyle='dashed')

            ax[0].plot(np.linspace(fit_start, max_d+1, 1000), func(np.linspace(fit_start, max_d, 1000), *popt_even), color='gray', linestyle='dotted')

            # ax[0].plot(np.linspace(fit_start+1, max_d, 1000), func(np.linspace(fit_start, max_d, 1000), *popt_odd), color='gray', linestyle='dashed', label='$%.3g e^\{%.3g d + %.3g d^2 \}+%.3g$'%(popt_odd[0], popt_odd[1], popt_odd[2], popt_odd[3]))

            # ax[0].plot(np.linspace(fit_start, max_d+1, 1000), func(np.linspace(fit_start, max_d, 1000), *popt_even), color='gray', label='$%.3g e^\{%.3g d + %.3g d^2\}+%.3g$'%(popt_even[0], popt_even[1], popt_even[2], popt_even[3]), linestyle='dotted')

        # plot number distinguished
        ax[0].scatter(np.arange(min_d, max_d+1, 1), num_sig_ls, label='This work', color='blue')
        # plot stars for prime d
        primes = np.array([3,5,7,11,13,17,19,23,29,31,37,41,43,47])
        primes = primes[(primes >= min_d) & (primes <= max_d)]
        num_sig_ls_primes = num_sig_ls[primes-min_d]
        
        ax[0].scatter(primes, num_sig_ls_primes, marker='*', s=200, color='blue')


        # plot even answer: for d = 2^n, 2^{n+1}-1 unique states
        ax[0].scatter(2**(np.arange(min_2, max_2,1)), 2**(np.arange(min_2, max_2,1)+1)-1, label='Pisenti et al. 2011', color='red')
        ax[0].scatter([3], [3], label='Leslie et al. 2019', color='orange')

        # plot percent distinguished
        ax[1].scatter(np.arange(min_d, max_d+1, 1), num_sig_ls/(np.arange(min_d, max_d+1, 1)**2), label='This work', color='blue')
        # plot stars for prime d
        ax[1].scatter(primes, num_sig_ls_primes/(primes**2), marker='*', s=200, color='blue')
        ax[1].scatter(2**(np.arange(min_2, max_2,1)), (2**(np.arange(min_2, max_2,1)+1)-1) / (2**(np.arange(min_2, max_2,1)))**2, label='Pisenti et al. 2011', color='red')
        ax[1].scatter([3], [3/9], label='Leslie et al. 2019', color='orange')

        print(num_sig_ls/(np.arange(min_d, max_d+1, 1)**2))

        ax[0].set_xlim(min_d, max_d+1)
        ax[0].set_ylabel('Number of Unique Signatures')
        ax[1].set_xlim(min_d, max_d+1)
        ax[1].set_xlabel('Dimension of Entanglement')
        ax[1].set_ylabel('Percent Distinguished')
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='upper right')
 
        # plt.suptitle('LELM Distinguishability with QFT')
        plt.tight_layout()
        plt.savefig(f'output/num_sig_{min_d}_{max_d}.pdf')

    def combine_txt(subdir):
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

    def combine_np(np_files):
        '''Combine all the separate .np files into one array and save'''
        num_sig_com = []
        min_min_d = 999
        max_max_d = 0
        for file in np_files:
            min_d = int(file.split('.')[0].split('_')[2])
            max_d = int(file.split('.')[0].split('_')[3])
            if min_d < min_min_d:
                min_min_d = min_d
            if max_d > max_max_d:
                max_max_d = max_d
            for val in np.load('output/'+file):
                num_sig_com.append(val)

        num_sig_com = np.array(num_sig_com)
        np.save(f'output/num_sig_{min_min_d}_{max_max_d}', num_sig_com)

    # print(get_all_signatures(d=4,display=True))
    # print(get_all_signatures(d=5,display=True))
    print(get_all_signatures(d=4,display=True, bs=True, do_QFT=True))
    # print(QFT(2))

    # compute_num_sig(38, 60, parallel=True, display=False)
    # combine_np(['num_sig_2_20.npy', 'num_sig_21_37.npy', 'num_sig_38_42.npy'])
    # plot_one(2, 42, 2)
    # plot_one(2, 42)





        
