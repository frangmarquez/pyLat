import numpy as np
from numpy.linalg import inv
from scipy.linalg import qr
import random
import sys
sys.path.append('./..')
from basis_reduction import BasisTransformer
from cvp_solver import CVPSolver

class GGH:
    def __init__(self, n, sigma, scale = 10):
        self.scale = scale
        self.n = n
        self.sigma = sigma
        self.U = None
        self.e = None
        self.c = None
        self.v = None
        self.R = self.generate_good_basis()
        self.B = self.generate_bad_basis(self.R)
        

    def generate_good_basis(self):
        # Generate a good basis R with random values
        R_ = np.random.randint(-10, 10, (self.n, self.n)).astype(float)
        while abs(np.linalg.det(R_)) < 0.01:  # Ensure that R is invertible
            R_ = np.random.randint(-10, 10, (self.n, self.n)).astype(float)

        basis_transformer = BasisTransformer()
        R = basis_transformer.reduce_basis(R_, method='LLL')
        return R

    def generate_elementary_unimodular_matrix(self, size):
        # Generate a random elementary unimodular matrix
        U = np.eye(size)
        i, j = random.sample(range(size), 2)
        U[i, j] = random.randint(-10, 10)
        return U

    def generate_random_unimodular_matrix(self, size, m):
        # Start with the identity matrix
        U = np.eye(size)
        for _ in range(m):
            U = np.dot(U, self.generate_elementary_unimodular_matrix(size))

        return U

    def generate_bad_basis(self, R):
        # Generate a unimodular matrix U and compute the bad basis B = R * U
        m = random.randint(5, 10)  # Number of elementary unimodular matrices to multiply
        U = self.generate_random_unimodular_matrix(self.n, m)
        self.U = U
        B = np.dot(R, U)
        return B

    def encrypt(self, m):
        # Encrypt message m
        m = np.array(m)
        e = np.random.randint(-self.sigma, self.sigma + 1, self.n)
        self.e = e
        c = np.dot(m, self.B) + e
        self.c = c
        return c

    def decrypt(self, c):
        # Decrypt ciphertext c
        cvpsolver = CVPSolver(target = c, lattice = self.R)
        v = cvpsolver.solveCVP(method = 'BNP', reduction = False)
        self.v = v
        m_recovered = np.dot(v, inv(self.B)).astype(int)
        return m_recovered

if __name__ == "__main__":

    # Message to encrypt
    m = np.array([32, 0, 13])
    m_recovered = np.array([0, 0, 0]).all()


    print('------------------ \nGGH Example')
    # Example usage
    n = 3  # Dimension of the lattice
    sigma = 10  # Error parameter
    
    ggh = GGH(n, sigma)
    
    # Encryption
    c = ggh.encrypt(m)
    print("Ciphertext:", c)
    
    # Decryption
    m_recovered = ggh.decrypt(c)
    print("Recovered message:", m_recovered)
