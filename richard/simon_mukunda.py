import numpy as np
import math

v1 = [1,0]
v2 = [0,1]
u1 = [0,1/np.sqrt(2) + 1j/np.sqrt(2)]
u2 = [-1/np.sqrt(2) + 1j/np.sqrt(2),0]

def findAngles_gen(v1, v2, u1, u2):
    """
    Accepts an incoming orthogonal basis v1, v2 and returns the values of lambda, epsilon, and phi needed to turn the Simon Mukunda gadget to change the polarization to u1, u2.

    ***Only for demonstration purposes, this is not the function that calculates what the lab will actually use. The correct function for the lab is SMgadget().***
    """
    z1 = v1[0]
    z2 = v2[0]
    zeta1 = u1[0]
    zeta2 = u2[0]

    M = np.array([[z1, z2],[-np.conj(z2),np.conj(z1)]])
    N = np.array([[zeta1,zeta2],[-np.conj(zeta2),np.conj(zeta1)]])

    A = N @ M.conj().T

    a = np.real(A[0,0])
    b = np.imag(A[0,0])
    c = np.real(A[0,1])
    d = np.imag(A[0,1])

    phi = 2*math.acos(math.sqrt(a**2 + c**2))
    l = math.acos(a/math.sqrt(a**2 + c**2))
    epsilon = (math.acos(a/math.sqrt(a**2 + c**2)) + math.acos(-b/math.sqrt(b**2 + d**2)))/2

    Q1 = epsilon + np.pi/4
    Q2 = epsilon + np.pi/4 + phi/2
    H = epsilon + (np.pi + phi)/4 - l/2

    print([Q1,Q2,H])

#findAngles_gen(v1,v2,u1,u2)


def anglestoM(H, Q1, Q2):
    """
    Returns the transformation matrix M given wave plates angles H, Q1, and Q2. Inverse of the SM gadget process.
    """
    
    phi = 2*(Q1 - Q2)
    epsilon = Q2 - np.pi/4
    l = 2*(epsilon + (np.pi + phi)/4 - H)

    a = math.cos(l)*math.cos(phi/2)
    b = -math.cos(2*epsilon - l)*math.sin(phi/2)
    c = -math.sin(l)*math.cos(phi/2)
    d = -math.sin(2*epsilon - l)*math.sin(phi/2)

    zeta1 = a + b*1j
    zeta2 = c + d*1j

    M = [[zeta1, zeta2],[-zeta2.conjugate(),zeta1.conjugate()]]

    return M


#check the original function using specific cases


def HWPmatrix(alpha):

    H = np.array([[np.cos(2*alpha), np.sin(2*alpha)],[np.sin(2*alpha), -np.cos(2*alpha)]])

    return H

def QWPmatrix(alpha):

    Q = np.array([[(1-1j*np.cos(2*alpha))/np.sqrt(2), (-1j*np.sin(2*alpha))/np.sqrt(2)],[(-1j*np.sin(2*alpha))/np.sqrt(2), (1 + 1j*np.cos(2*alpha))/np.sqrt(2)]])

    return Q


def SMgadget(state_basis, measurement_basis):
    """
    Accepts a state and measurement basis and returns the angles of the waveplates needed to produce the result according to the SM algorithm.
    """

    A = measurement_basis.conj().T @ state_basis

    a = np.real(A[0,0])
    b = np.imag(A[0,0])
    c = np.real(A[0,1])
    d = np.imag(A[0,1])

    phi = 2*np.arccos(min(math.sqrt(a**2 + c**2),1))
    re_quadrant = np.arctan2(-c,a)
    im_quadrant = np.arctan2(-b,-d)

    if b**2 + d**2 == 0:
        epsilon = 0
        if re_quadrant >= 0:
            if a >= 0:
                l = np.arccos(min(a,1))
            elif a < 0:
                l = 2*np.pi - np.arccos(a)
        elif re_quadrant < 0:
            if a >= 0:
                l = 2*np.pi - np.arccos(a)
            elif a < 0:
                l = np.arccos(a)
    elif a**2 + c**2 == 0:
        epsilon = 0
        if im_quadrant >= 0:
            if b > 0:
                l = np.arccos(-b)
            elif b < 0:
                l = -np.arccos(-b)
        elif im_quadrant < 0:
            if b > 0:
                l = np.arccos(-b)
            elif b < 0:
                l = -np.arccos(-b)
    else:
        l = np.arccos(a/np.sqrt(a**2 + c**2))
        epsilon = (np.arccos(a/np.sqrt(a**2 + c**2)) + np.arccos(-b/np.sqrt(b**2 + d**2)))/2

    Q1 = epsilon + np.pi/4
    Q2 = epsilon + np.pi/4 + phi/2
    H = epsilon + (np.pi + phi)/4 - l/2

    return [H, Q1, Q2, A]

def basis_generation(min_theta, max_theta, min_phi, max_phi, theta_count, phi_count):

    thetas = np.linspace(min_theta, max_theta, theta_count)
    phis = np.linspace(min_phi, max_phi, phi_count)

    zeta1s = np.kron(np.cos(thetas), np.repeat(1, theta_count))
    zeta2s = np.kron(np.sin(thetas), np.exp(1j*phis))

    bases_set = np.zeros(shape=(theta_count*phi_count,2,2),dtype='complex_')

    for i in range(theta_count*phi_count):
        bases_set[i] = [[zeta1s[i],zeta2s[i]],[-zeta2s[i].conjugate(),zeta1s[i].conjugate()]]

    return bases_set

def stateCreation_check(state_basis, measurement_basis):

    H, Q1, Q2, A = SMgadget(state_basis, measurement_basis)

    HWP = HWPmatrix(H)
    QWP1 = QWPmatrix(Q1)
    QWP2 = QWPmatrix(Q2)

    m1 = HWP
    m2 = QWP1 @ HWP
    m3 = QWP2 @ QWP1 @ HWP

    m1_check = HWP @ HWP.conj().T
    m2_check = QWP1 @ QWP1.conj().T
    m3_check = QWP2 @ QWP2.conj().T


    if (m1 == state_basis).all() or (m2 == state_basis).all() or (m3 == state_basis).all():
        creation_status = True
    else:
        creation_status = False

    return [m1, m2, m3, m1_check, m2_check, m3_check, A, creation_status]

bases_set = basis_generation(0, np.pi/2, 0, 2*np.pi, 10, 10)
state_basis = np.array([[0,-1],[1,0]])
measurement_basis = np.array([[1,0],[0,1]])
# for i in range(len(measurement_basis)):
#     for j in range(len(bases_set)):
#         h, q1, q2, H, Q1, Q2, A, status = stateCreation_check(bases_set[j],measurement_basis[i])
#         #if stateCreation_check(bases_set[j],measurement_basis[i])[-1] == True:
#         print("A:",A)
#         print("Q2:",q2)
#         #print(status)

_,_,_,H, Q1, Q2, A, status = stateCreation_check(state_basis,measurement_basis)
print(state_basis)
print(A)
print(Q2)