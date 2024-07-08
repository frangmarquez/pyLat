import math
import numpy as np
from sympy import Symbol, ZZ, Poly, gcd, invert, GF
from sympy.abc import x

class NTRUEncrypt:
    def __init__(self, N, p, q, df, dg, dm, dr):
        self.N = N
        self.p = p
        self.q = q
        self.df = df
        self.dg = dg
        self.dm = dm
        self.dr = dr
        self.R_poly = Poly(x ** N - 1, x).set_domain(ZZ)
        self.f = self.generate_random_polynomial(N, df, neg_ones_diff=-1)
        self.g = self.generate_random_polynomial(N, dg)
        self.h = None
        self.f_p_inv = None
        self.f_q_inv = None
        self.keygen()

    def generate_random_polynomial(self,length, d, neg_ones_diff=0):
        """Generate a random polynomial with d ones, d minus ones and N-d zeros"""
        polynomial = Poly(np.random.permutation(np.concatenate((np.zeros(length - 2 * d - neg_ones_diff), np.ones(d), -np.ones(d + neg_ones_diff)))),
                          x).set_domain(ZZ)
        return polynomial

    def is_prime(self,n):
        """Checks if a number is prime"""
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True


    def is_2_power(self,n):
        """Checks if a number is a power of 2"""
        return n != 0 and (n & (n - 1) == 0)

    def invert_polinomial(self, f, R_poly, p):
        """Invert a polinomial in Z_p"""
        inv_poly = None
        if self.is_prime(p):
            inv_poly = invert(f, R_poly, domain=GF(p))
        elif self.is_2_power(p):
            inv_poly = invert(f, R_poly, domain=GF(2))
            e = int(math.log(p, 2))
            for i in range(1, e):
                inv_poly = ((2 * inv_poly - f * inv_poly ** 2) % R_poly).trunc(p)
        else:
            raise Exception("Cannot invert polynomial in Z_{}".format(p))
        return inv_poly

    def keygen(self):
        """Generate the public key for the NTRU encryption"""
        while True:
            try:
                self.f_p_inv = self.invert_polinomial(self.f, self.R_poly, self.p)
                self.f_q_inv = self.invert_polinomial(self.f, self.R_poly, self.q)
                break
            except:
                self.f = self.generate_random_polynomial(N, df, neg_ones_diff=-1)

        h_not_reduced = (self.f_q_inv * self.g).trunc(self.q)
        self.h = (h_not_reduced % self.R_poly).trunc(self.q)

    def encrypt(self, m):
        """Encrypt the message"""
        r = self.generate_random_polynomial(N, self.dr)
        e = ((m + self.p * r * self.h) % self.R_poly).trunc(self.q)
        return e

    def decrypt(self, e):
        """Decrypt the message"""
        a = ((self.f * e) % self.R_poly).trunc(self.q)
        m = ((a * self.f_p_inv) % self.R_poly).trunc(self.p)
        return m

if __name__ == '__main__':
    
    # Example usage
    N, p, q = 17, 3, 64
    df, dg, dm, dr = N//3, int(math.sqrt(q)), 113, 5
    ntru = NTRUEncrypt(N, p, q, df, dg, dm, dr)

    # Message polynomial
    #m_coeffs = np.concatenate((np.ones(32), -np.ones(32), np.zeros(N - 64)))
    #np.random.shuffle(m_coeffs)

    m = [1, 0, 1, 0, 0, 1, 1, 1]


    m_coeffs = np.array(m)
    m = Poly(m_coeffs, Symbol('x')).set_modulus(N)

    # Encrypt
    e = ntru.encrypt(m)

    # Decrypt
    decrypted_m = ntru.decrypt(e)

    # Check if the decryption is correct

    print( (decrypted_m - m) == 0 ) 
