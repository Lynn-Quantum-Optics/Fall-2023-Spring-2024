# file to test new "symmetrized" bell basis
import numpy as np
import itertools
from oscars_toolbox.trabbit import trabbit

## helper functions for generate ideas for vectors ##
def get_comb(total_vec, num_neg):
    ''' generate all combinations of positions for num_neg -1 in a vector of length total_vec'''
    positions = range(total_vec)  # Positions in the vector (0 to 5)
    combinations = list(itertools.combinations(positions, num_neg))

    # generate the vectors based on the combinations
    vectors = []
    for combo in combinations:
        vector = [1]*total_vec  # start with all 1s
        for pos in combo:
            vector[pos] = -1  # set -1s at the specified positions
        vectors.append(vector)
        print(vector)
    return vectors

def structured_gram_schmidt(vectors):
    '''BROKEN'''
    orthogonal_set = []
    for v in vectors:
        for u in orthogonal_set:
            projection = np.dot(u, v) / np.dot(u, u) * u
            # Adjust the projection to maintain the structure
            for i in range(0, len(projection), 2):
                avg = (projection[i] - projection[i + 1]) / 2
                projection[i], projection[i + 1] = avg, -avg
            v -= projection
        orthogonal_set.append(v)
    return orthogonal_set

def get_old_phase(d=3):
    '''Get the phase vectors for d = 3 as in the exisiting definition'''
    return [[np.exp(2 * np.pi * 1j * i * p / d) for i in range(d)] for p in range(d)]

## helper functions to test ideas # 
def check_orthogonal(vec_ls):
    # orthogonal = True
    for i in range(len(vec_ls)):
        for j in range(len(vec_ls)):
            if i != j:
                if not(np.isclose(np.abs(np.array(vec_ls[i]) @ np.array(vec_ls[j]).conj().T), 0, atol=7e-9)):
                    print(f'vector {i} and {j} are not orthogonal and have dot product {np.abs(np.array(vec_ls[i]) @ np.array(vec_ls[j]).conj().T)}')
                    # orthogonal = False
                    return False
    print('all vectors are orthogonal.')
    return True

def check_constraints_and_orthogonality(vectors):
    '''Checks if the basis is eigenstate of particle exchange'''
    # Check orthogonality
    orthogonality = check_orthogonal(vec_ls=vectors)

    # Check the structural constraints: no zero entries and pairing
    no_zero_entries = not np.any(vectors == 0)

    correct_pairing = True
    for v in vectors:
        scale = v[1] / v[0]
        for i in range(2, len(v), 2):
            scale_i = v[i+1] / v[i]
            if not np.isclose(scale, scale_i, atol=1e-33):
                print(f'v: {v}')
                print(f'scale: {scale}')
                print(f'scale_i: {scale_i}')
                correct_pairing = False
                break

    print(f'orthogonality: {orthogonality}')
    print(f'no_zero_entries: {no_zero_entries}')
    print(f'correct_pairing: {correct_pairing}')

    return orthogonality and no_zero_entries and correct_pairing

## learning phase ##
def get_vectors(params):
    '''Convert the learned params into phase vectors'''
    # unpack the params
    real_ls = params[:len(params)//2]
    imag_ls = params[len(params)//2:]

    vectors = [
        np.array([1, 1, 1, 1, 1, 1], dtype=complex),
        np.array([real_ls[1] + 1j*imag_ls[1], (real_ls[0] + 1j*imag_ls[0])*(real_ls[1] + 1j*imag_ls[1]), real_ls[2] + 1j*imag_ls[2], (real_ls[0] + 1j*imag_ls[0])*(real_ls[2] + 1j*imag_ls[2]), real_ls[3]+1j*imag_ls[3], (real_ls[0] + 1j*imag_ls[0])*(real_ls[3] + 1j*imag_ls[3])], dtype=complex),
        np.array([real_ls[5] + 1j*imag_ls[5], (real_ls[4] + 1j*imag_ls[4])*(real_ls[5] + 1j*imag_ls[5]), real_ls[6] + 1j*imag_ls[6], (real_ls[4] + 1j*imag_ls[4])*(real_ls[6] + 1j*imag_ls[6]), real_ls[7]+1j*imag_ls[7], (real_ls[4] + 1j*imag_ls[4])*(real_ls[7] + 1j*imag_ls[7])], dtype=complex),
        np.array([real_ls[9] + 1j*imag_ls[9], (real_ls[8] + 1j*imag_ls[8])*(real_ls[9] + 1j*imag_ls[9]), real_ls[10] + 1j*imag_ls[10], (real_ls[8] + 1j*imag_ls[8])*(real_ls[10] + 1j*imag_ls[10]), real_ls[11]+1j*imag_ls[11], (real_ls[8] + 1j*imag_ls[8])*(real_ls[11] + 1j*imag_ls[11])], dtype=complex),
        np.array([real_ls[13] + 1j*imag_ls[13], (real_ls[12] + 1j*imag_ls[12])*(real_ls[13] + 1j*imag_ls[13]), real_ls[14] + 1j*imag_ls[14], (real_ls[12] + 1j*imag_ls[12])*(real_ls[14] + 1j*imag_ls[14]), real_ls[15]+1j*imag_ls[15], (real_ls[12] + 1j*imag_ls[12])*(real_ls[15] + 1j*imag_ls[15])], dtype=complex),
        np.array([real_ls[17] + 1j*imag_ls[17], (real_ls[16] + 1j*imag_ls[16])*(real_ls[17] + 1j*imag_ls[17]), real_ls[18] + 1j*imag_ls[18], (real_ls[16] + 1j*imag_ls[16])*(real_ls[18] + 1j*imag_ls[18]), real_ls[19]+1j*imag_ls[19], (real_ls[16] + 1j*imag_ls[16])*(real_ls[19] + 1j*imag_ls[19])], dtype=complex),
        ]
    
    # normalize
    for i in range(len(vectors)):
        vectors[i] /= np.linalg.norm(vectors[i])

    # print('comparison')
    # print(vectors[1][0] / vectors[1][1])
    # print(vectors[1][2] / vectors[1][3])
    
    return vectors

def loss_phase(params):
    '''Loss function to find the optimal phases for d = 6'''

    vectors = get_vectors(params)

    # now find inner products
    inner_products = 0
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            inner_products += np.abs(np.dot(vectors[i], vectors[j].conj().T))

    return inner_products

def random_gen(num_params=40):
    '''Generate random parameters'''
    return np.random.uniform(-1, 1, num_params)
    
def optimize_phase(tol=1e-6, x0=None):
    '''Minimize the loss function of sum of abs value of inner products to find the optimal phases'''
    x_best, loss_best = trabbit(loss_func=loss_phase, random_gen=random_gen, alpha=1, tol=tol, temperature=0.01, x0_ls=[x0])
    print(f'best loss: {loss_best}')
    print(f'best params: {list(x_best)}')
    return x_best, loss_best

if __name__ == '__main__':
    # initial_vectors = np.array([
    #     [1, -1, 1, -1, 1, -1],
    #     [1, -1, -1, 1, 1, -1],
    #     [1, -1, -1, 1, -1, 1],
    #     [1, -1, 1, -1, -1, 1],
    #     [1, 1, -1, -1, 1, 1],
    #     [1, 1, 1, 1, -1, -1]
    # ]).astype(float)

    # old_phase = get_old_phase()
    # check_orthogonal(old_phase)
    # print(old_phase)

    # second_vectors = np.array([
    #     [1, 1, 1, 1, 1, 1],
    #     [1, -1, np.exp(2*np.pi*1j/3), -np.exp(2*np.pi*1j/3), np.exp(2*np.pi*1j/3), -np.exp(2*np.pi*1j/3)],
    #     [1, -1, -np.exp(2*np.pi*1j/3), np.exp(2*np.pi*1j/3), np.exp(2*np.pi*1j/3), -np.exp(2*np.pi*1j/3)],
    #     [1, -1, np.exp(2*np.pi*1j/3), -np.exp(2*np.pi*1j/3), -np.exp(2*np.pi*1j/3), np.exp(2*np.pi*1j/3)],
    #     [1, -1, -np.exp(2*np.pi*1j/3), np.exp(2*np.pi*1j/3), -np.exp(2*np.pi*1j/3), np.exp(2*np.pi*1j/3)]
    #     ])

    # check_constraints_and_orthogonality(second_vectors)

    x0 = [-0.26957523753194196, 1.2255067628584213, -1.8140745726231078, 0.5885677693016419, 0.3056393212405134, 1.5853919369836418, -1.7656582706665025, 0.18026631980690383, -1.0000000021188098, 0.9944961719952249, 0.9944961592711796, 0.9944961802796902, -0.7107111105850151, -1.2279809900976428, -0.326568572070583, 1.5545495251219525, 0.5792472360067527, -0.8989998649828175, 0.3166079342532708, 0.5823919132861747, 0.8996296942917451, -1.2898148534195002, 1.005664318472053, 0.28415050815715914, -1.019983235511904, -0.007356466426700959, -0.5464310628467627, 0.5537875102161702, -2.8600226218100746e-09, 1.1214375020607408, 1.121437509414354, 1.1214375329993491, 0.8496152737170141, -0.3668087016935599, -0.8036196417378422, 1.170428324918717, -0.6924576004755217, -1.3212094808403232, -1.0342619236156319, 2.355471380541379]

    best_params, best_loss = optimize_phase(tol=1e-10, x0=x0)

    # best_loss = 9.440345514106494e-07
    # best_params = [-0.1268860616808698, 0.5886238937550582, -0.45847894057688376, -0.1301452181425787, -1.0000003228778174, -1.3451340877544036, -1.345133905794467, -1.3451341168951334, 0.14595266412945068, 0.3838792083640446, -1.0656074601943992, 0.6817281399774296, -0.12688569597412505, -1.9868248626837408, 0.8252053330405361, 1.1616195280756307, 0.14595295594403637, -0.3611445240551306, -0.24961751250677564, 0.6107618844035959, -0.923721498017154, 0.34138732374634495, -1.9095298298805803, 1.5681420397075128, -1.2751183983522213e-08, 0.25461790276576135, 0.25461786277002146, 0.25461785417874, 1.062528779918951, 0.1944171811809395, 1.26447448399162, -1.4588914720163297, -0.9237211925292237, -0.7103536317989507, -0.2379699377925436, 0.9483235116043163, 1.0625288436738816, -2.130928609568798, 1.0939959280252247, 1.036933042184377]

    # best_loss=4.249493263456608e-08
    # best_params= [-0.26957523753194196, 1.2255067628584213, -1.8140745726231078, 0.5885677693016419, 0.3056393212405134, 1.5853919369836418, -1.7656582706665025, 0.18026631980690383, -1.0000000021188098, 0.9944961719952249, 0.9944961592711796, 0.9944961802796902, -0.7107111105850151, -1.2279809900976428, -0.326568572070583, 1.5545495251219525, 0.5792472360067527, -0.8989998649828175, 0.3166079342532708, 0.5823919132861747, 0.8996296942917451, -1.2898148534195002, 1.005664318472053, 0.28415050815715914, -1.019983235511904, -0.007356466426700959, -0.5464310628467627, 0.5537875102161702, -2.8600226218100746e-09, 1.1214375020607408, 1.121437509414354, 1.1214375329993491, 0.8496152737170141, -0.3668087016935599, -0.8036196417378422, 1.170428324918717, -0.6924576004755217, -1.3212094808403232, -1.0342619236156319, 2.355471380541379]

    vectors = get_vectors(best_params)
    print(vectors)
    check_constraints_and_orthogonality(vectors)


