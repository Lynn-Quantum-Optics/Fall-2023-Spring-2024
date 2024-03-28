# exploring if we can permute the old basis definition to the new one
import numpy as np
from new_esym_general import get_correlation_classes
from scipy.linalg import svd

def old_correlation_classes(d, print_result=False):
    '''Prints out the pairs of particles in the old correlation classes'''
    vecs = [[] * d for _ in range(d)]  
    for c in range(d):
        c_vec = []
        for j in range(d):
            c_vec.append((j, (j + c) % d))
        if print_result:
            print(f'c vec {c} = {c_vec}')
        # order the pairs in the c_vec by the first element
        c_vec.sort(key=lambda x: x[0])
        vecs[c] = c_vec
    return vecs

def even_idea(d):
    '''get correlation classes for even d using symmetric framework new idea'''
    vecs = [[] * d for _ in range(d)]  
    for c in range(d):
        c_vec = []
        j = 0
        while j < d:
            c_vec.append((j, (j + c) % d))
            c_vec.append(((j + c) % d, j))
            j+=(j + c) % d+1

        vecs[c] = c_vec
    return vecs
            

## helper function ##
def convert_to_matrix(vec_vec):
    '''Convert a list of vectors of pairs into a matrix of column vectors'''
    # Flatten the vector of vectors into a list of tuples
    flat_list = [str(item[0])+str(item[1]) for sublist in vec_vec for item in sublist]
    # Convert the flat list into a numpy array of objects (tuples in this case)
    matrix_of_tuples = np.array(flat_list, dtype=object).reshape(-1, len(vec_vec))

    # Transpose the matrix to get column vectors
    result_matrix = matrix_of_tuples.T
    return result_matrix

def diff_matrices(m1, m2):
    '''Take the difference between two matrices'''
    # iterate through each element in the matrix and take the difference for L and R particles
    # create new matrix with differences
    result = np.empty(m1.shape, dtype=object)
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            L_diff = int(m1[i,j][0])-  int(m2[i,j][0])
            R_diff = int(m1[i,j][1])-  int(m2[i,j][1])
            print(f'L_diff = {L_diff}, R_diff = {R_diff}')
            result[i,j] = str(L_diff) + str(R_diff)

    return result

def convert_jp(m):
    '''converts a matrix of strings into a matrix of integers'''
    d = m.shape[0]
    result = np.zeros((d**2,d), dtype=int)
    for i in range(d):
        # get this column
        col = m[:,i]
        for j in range(d):
            L = int(col[j][0])
            R = int(col[j][1])

            index = L*d + R
            result[index,i] = 1

    return result

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def get_schmidt_decomp(cvec_mat):
    '''Get the schmidt decomposition of the matrix'''

    d = cvec_mat.shape[0]

    # get the singular value decomposition of the matrix
    for c in range(cvec_mat.shape[1]):
        # create a matrix to store the coefficients
        coeff_matrix = np.zeros((d, d**2))

        for i, state in enumerate(cvec_mat[:,c]):
            # Convert the 'xy' string into a numerical index for the matrix
            index = int(state[0]) * d + int(state[1])
            coeff_matrix[i, index] = 1

        # Perform Singular Value Decomposition (SVD) on the coefficient matrix
        U, s, Vh = svd(coeff_matrix)
        print(f'c = {c}')
        print(f'U = {U}')
        print(f's = {s}')
        print(f'Vh = {Vh}')

if __name__ == '__main__':
    d = 6
    old_cvec = old_correlation_classes(d)
    new_c_vec = get_correlation_classes(d)

    # convert each to a matrix, where each vector is a column but the pairs remain together
    old_cvec_mat = convert_to_matrix(old_cvec)
    new_cvec_mat = convert_to_matrix(new_c_vec)

    # get_schmidt_decomp(old_cvec_mat)
    # print('-------')
    # get_schmidt_decomp(new_cvec_mat)
    

    old_mat = convert_jp(old_cvec_mat)
    new_mat = convert_jp(new_cvec_mat)
    # print(old_mat)
    # print(new_mat)
    # print(old_mat - new_mat)

    # # does there exist linear transformation that takes old_mat to new_mat?
    # # permute the columns of old_mat to match new_mat
    # new_mat = np.random.permutation(new_mat)

  
    alpha = new_mat @ np.linalg.pinv(old_mat)
    # print(alpha.shape)
    print(np.round(alpha, 3))
    # # save as a csv
    # # np.savetxt('alpha.csv', np.round(alpha, 3), delimiter=',')
    # # divide alpha into a 6x6 grid of 6x6 matrices
    alpha_grid = alpha.reshape(d, d, d, d)
    print(alpha_grid)
    print(len(alpha_grid))