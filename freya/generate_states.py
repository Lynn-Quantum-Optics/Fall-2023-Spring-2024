# This is segments of mixedstates.ipynb put in a normal Python file so they can be imported by my other files.
# The wrapper function generate_states lets you pass how many states you want and whether you want (uniform) pure states or
# (almost certainly not uniform) mixed states. make_cs and make_s_mat (change of basis matrix) may be useful as well.
# If you want to use or modify my code, this is the file to import/edit.

import numpy as np
import dataclasses

@dataclasses.dataclass
class Params:
    mu0: float
    mu1: float
    mu2: float
    r: float
    zeta: float
    theta: float
    psi: float
    theta_prime: float
    psi_prime: float
    theta32: float
    psi32: float
    theta21: float
    psi21: float
    theta0: float
    psi0: float

    @property
    def mu3(self):
        return 1 - self.mu0 - self.mu1 - self.mu2

def sample_params(samps):
    unif = np.random.uniform

    # These distributions are verified to work for pure states
    _theta = np.arccos(unif(-1, 1, samps))
    _psi = unif(0, 2 * np.pi, samps)
    _theta_prime = np.arccos(unif(-1, 1, samps))
    _psi_prime = unif(0, 2 * np.pi, samps)
    _r = np.cbrt(unif(0, 1, samps))
    _zeta = unif(0, 2 * np.pi, samps)

    # what follows is purely a guess
    mu = np.random.dirichlet(np.ones(4), samps)  # sample mu's uniformly from the random simplex
    _theta32 = np.arccos(unif(-1, 1, samps))  # sample thetas and psis as for pure states
    _psi32 = unif(0, 2 * np.pi, samps)
    _theta21 = np.arccos(unif(-1, 1, samps))
    _psi21 = unif(0, 2 * np.pi, samps)
    _theta0 = np.arccos(unif(-1, 1, samps))
    _psi0 = unif(0, 2 * np.pi, samps)

    return Params(mu0=mu[:, 0], mu1=mu[:, 1], mu2=mu[:, 2], r=_r, zeta=_zeta, theta=_theta, psi=_psi,
                  theta_prime=_theta_prime, psi_prime=_psi_prime, theta32=_theta32, psi32=_psi32,
                  theta21=_theta21, psi21=_psi21, theta0=_theta0, psi0=_psi0, )

def sample_pure(samps):
    unif = np.random.uniform

    _theta = np.arccos(unif(-1, 1, samps))
    _psi = unif(0, 2 * np.pi, samps)
    _theta_prime = np.arccos(unif(-1, 1, samps))
    _psi_prime = unif(0, 2 * np.pi, samps)
    _r = np.cbrt(unif(0, 1, samps))
    _zeta = unif(0, 2 * np.pi, samps)

    zero = np.zeros(samps)
    _mu0 = np.ones(samps)

    return Params(mu0=_mu0, mu1=zero, mu2=zero, r=_r, zeta=_zeta, theta=_theta, psi=_psi,
                  theta_prime=_theta_prime, psi_prime=_psi_prime, theta32=zero, psi32=zero, theta21=zero,
                  psi21=zero, theta0=zero, psi0=zero, )

def make_cs(theta, psi):
    return np.exp(-1j * psi / 2) * np.cos(theta / 2), np.exp(1j * psi / 2) * np.sin(theta / 2)

def make_s_mat(theta, psi, theta_prime, psi_prime):
    c, s = make_cs(theta, psi)
    c_prime, s_prime = make_cs(theta_prime, psi_prime)

    conj = np.conjugate

    return np.array([
        [c * c_prime, conj(s) * conj(s_prime),
         -c * conj(s_prime), -(conj(s)) * c_prime, ],

        [c * s_prime, -(conj(s)) * conj(c_prime),
         c * conj(c_prime), -(conj(s)) * s_prime, ],

        [s * c_prime, -(conj(c)) * conj(s_prime),
         -s * conj(s_prime), conj(c) * c_prime, ],

        [s * s_prime, conj(c) * conj(c_prime),
         s * conj(c_prime), conj(c) * s_prime, ], ]).transpose((2, 0, 1))
    # Each parameter is a 1D array, so the output is a 3D array consisting of a 2D array (matrix) whose elements are lists.
    # The transpose operation restructures the 3D output array so it is a list of matrices instead

def make_rho_mat(params):
    q_plus = np.sqrt((1 + params.r) / 2)
    q_minus = np.sqrt((1 - params.r) / 2)

    c32, s32 = make_cs(params.theta32, params.psi32)
    c21, s21 = make_cs(params.theta21, params.psi21)
    c0, s0 = make_cs(params.theta0, params.psi0)

    conj = np.conjugate
    abs = np.absolute

    rho00 = ((params.mu0 - params.mu3) * q_plus ** 2 + (params.mu1 - params.mu3) * q_minus ** 2 * abs(c21) ** 2
             + (params.mu2 - params.mu3) * q_minus ** 2 * abs(c0) ** 2 * abs(s21) ** 2 + params.mu3)

    rho01 = ((params.mu0 - params.mu3) * np.exp(-1j * params.zeta) * q_minus * q_plus
             - (params.mu1 - params.mu3) * np.exp(-1j * params.zeta) * q_minus * q_plus * abs(c21) ** 2
             - (params.mu2 - params.mu3) * np.exp(-1j * params.zeta) * q_plus * q_minus * abs(c0) ** 2 * abs(s21) ** 2)

    rho02 = (-(params.mu1 - params.mu3) * np.exp(-1j * params.zeta) * q_minus * c21 * conj(c32) * conj(s21)
             + (params.mu2 - params.mu3) * np.exp(-1j * params.zeta) * q_minus * c0 * conj(s21) * (
                     conj(c0) * c21 * conj(c32) - conj(s0) * s32))

    rho03 = (-(params.mu1 - params.mu3) * np.exp(-1j * params.zeta) * q_minus * c21 * conj(s32) * conj(s21)
             + (params.mu2 - params.mu3) * np.exp(-1j * params.zeta) * q_minus * c0 * conj(s21) * (
                     conj(c0) * c21 * conj(s32) + conj(s0) * c32))

    rho10 = ((params.mu0 - params.mu3) * np.exp(1j * params.zeta) * q_minus * q_plus
             - (params.mu1 - params.mu3) * np.exp(1j * params.zeta) * q_minus * q_plus * abs(c21) ** 2
             - (params.mu2 - params.mu3) * np.exp(1j * params.zeta) * q_minus * q_plus * abs(c0) ** 2 * abs(s21) ** 2)

    rho11 = ((params.mu0 - params.mu3) * q_minus ** 2 + (params.mu1 - params.mu3) * q_plus ** 2 * abs(c21) ** 2
             + (params.mu2 - params.mu3) * q_plus ** 2 * abs(c0) ** 2 * abs(s21) ** 2 + params.mu3)

    rho12 = ((params.mu1 - params.mu3) * q_plus * c21 * conj(c32) * conj(s21)
             - (params.mu2 - params.mu3) * q_plus * c0 * conj(s21) * (
                     conj(c0) * c21 * conj(c32) - conj(s0) * s32))

    rho13 = ((params.mu1 - params.mu3) * q_plus * c21 * conj(s32) * conj(s21)
             - (params.mu2 - params.mu3) * q_plus * c0 * conj(s21) * (
                     conj(c0) * c21 * conj(s32) + conj(s0) * c32))

    rho20 = (-(params.mu1 - params.mu3) * np.exp(1j * params.zeta) * q_minus * conj(c21) * c32 * s21
             + (params.mu2 - params.mu3) * np.exp(1j * params.zeta) * q_minus * conj(c0) * s21 * (
                     c0 * conj(c21) * c32 - s0 * conj(s32)))

    rho21 = ((params.mu1 - params.mu3) * q_plus * conj(c21) * c32 * s21
             - (params.mu2 - params.mu3) * q_plus * conj(c0) * s21 * (c0 * conj(c21) * c32 - s0 * conj(s32)))

    rho22 = ((params.mu1 - params.mu3) * abs(c32) ** 2 * abs(s21) ** 2
             + (params.mu2 - params.mu3) * abs(c0 * conj(c21) * c32 - s0 * conj(s32)) ** 2 + params.mu3)

    rho23 = ((params.mu1 - params.mu3) * abs(s21) ** 2 * c32 * conj(s32) + (params.mu2 - params.mu3) * (
            conj(c0) * c21 * conj(s32) + conj(s0) * c32) * (
                     c0 * conj(c21) * c32 - s0 * conj(s32)))

    rho30 = (-(params.mu1 - params.mu3) * np.exp(1j * params.zeta) * q_minus * conj(c21) * s32 * s21
             + (params.mu2 - params.mu3) * np.exp(1j * params.zeta) * q_minus * conj(c0) * s21 * (
                     c0 * conj(c21) * s32 + s0 * conj(c32)))

    rho31 = ((params.mu1 - params.mu3) * q_plus * conj(c21) * s32 * s21
             - (params.mu2 - params.mu3) * q_plus * conj(c0) * s21 * (c0 * conj(c21) * s32 + s0 * conj(c32)))

    rho32 = ((params.mu1 - params.mu3) * abs(s21) ** 2 * conj(c32) * s32
             + (params.mu2 - params.mu3) * (c0 * conj(c21) * s32 + s0 * conj(c32)) * (
                     conj(c0) * c21 * conj(c32) - conj(s0) * s32))

    rho33 = ((params.mu1 - params.mu3) * abs(s21) ** 2 * abs(s32) ** 2
             + (params.mu2 - params.mu3) * abs(c0 * conj(c21) * s32 + s0 * conj(c32)) ** 2 + params.mu3)

    return np.array([[rho00, rho01, rho02, rho03], [rho10, rho11, rho12, rho13],
                     [rho20, rho21, rho22, rho23], [rho30, rho31, rho32, rho33], ]).transpose((2, 0, 1))
    # Each parameter is a 1D array, so the output is a 3D array consisting of a 2D array (matrix) whose elements are lists.
    # The transpose operation restructures the 3D output array so it is a list of matrices instead

def make_rho_prime_mat(params):
    s_mat = make_s_mat(params.theta, params.psi, params.theta_prime, params.psi_prime)
    rho = make_rho_mat(params)
    s_dagger = np.conjugate(s_mat.transpose((0, 2, 1)))  # This one is just an actual transpose of each matrix in the list
    return np.matmul(s_mat, np.matmul(rho, s_dagger))

def generate_states(samples, pure):
    """ Generates an array of two-qubit density matrices
    :param samples: Integer. How many density matrices to generate
    :param pure: Boolean. If true, generates exclusively pure states """
    if pure:
        rho_prime = make_rho_prime_mat(sample_pure(samples))
    else:
        rho_prime = make_rho_prime_mat(sample_params(samples))
    return rho_prime
