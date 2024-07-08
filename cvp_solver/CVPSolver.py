import sys
sys.path.append('.')
from basis_reduction import BasisTransformer
from svp_solver import SVPSolver
import numpy as np

class CVPSolver:

    def __init__(self, lattice, target):
        self.lattice = lattice
        self.target = target

    def BCVSolver(self, lattice, target):

        """Method to solve CVP with BCV"""
        
        real_coefs = np.linalg.solve(lattice, target)
        approx_coefs = np.round(real_coefs)
        solution = np.array(lattice @ approx_coefs, dtype=np.int64)
        return solution


    def BNPSolver(self, lattice, target):

        """Method to solve CVP with BNP"""

        ba = BasisTransformer()

        ortho_basis = ba.orthogonalize_basis(lattice)
        n = ortho_basis.shape[1]
        w = np.zeros(ortho_basis.shape)
        y = np.zeros(ortho_basis.shape, dtype=np.int64)
        
        w[:,n-1] = target

        for i in range(n-1,-1,-1):
            l_i = np.dot(ortho_basis[:,i],w[:,i])/np.dot(ortho_basis[:,i],ortho_basis[:,i])
            y[:,i] = np.round(l_i) * lattice[:,i]
            w[:,i-1] = w[:,i] - (l_i - round(l_i))*ortho_basis[:,i] - y[:,i] 

        solution = np.sum(y, axis=1)
        return solution
    
    def EmbbedingSolver(self, lattice, target, M):

        """Embbeding method to solve SVP"""

        n = lattice.shape[1]
        
        extended_basis = np.zeros((lattice.shape[0]+1, lattice.shape[1]+1))
        extended_basis[:n, :n] = lattice
        extended_basis[:n, -1] = target
        extended_basis[-1, -1] = M

        ba = BasisTransformer()
        extended_basis = ba.lll_reduce(extended_basis)
        
        methodSVP= 'enumeration'
        svp_solver = SVPSolver(extended_basis)
        shortest_vector_in_extended, _, solutions_bag  = svp_solver.solveSVP( method = methodSVP, bag = True )
        
        possible_solutions = []
        for possible_solution_svp in solutions_bag:
            e = possible_solution_svp[:n]
            solution = target - e
            possible_solutions.append(solution)
        
        return possible_solutions

    def enumeration(self,lattice):

        """Enumeration method to solve SVP"""

        target = self.target
        
        ba = BasisTransformer()
        basis = self.lattice
        ortho_basis = ba.orthogonalize_basis(basis)

        ordered_basis = ba.order_columns_by_length(lattice)
        basis = basis[:, ordered_basis]
        ortho_basis = ortho_basis[:, ordered_basis]
        mu = np.zeros(ortho_basis.shape)

        for i in range(ortho_basis.shape[1]):
            for j in range(0,i+1):
                mu[j, i] = np.dot(ortho_basis[:, i], ortho_basis[:, j]) / np.dot(ortho_basis[:, j], ortho_basis[:, j])
                mu[i, j] = mu[j, i]
        
        A = np.dot(ortho_basis[:, 0], ortho_basis[:,0])
        B = np.array([np.dot(ortho_basis[:, i], ortho_basis[:, i]) for i in range(ortho_basis.shape[1])])
        n = basis.shape[1]

        y = np.linalg.solve(ortho_basis, target)
        x = np.zeros(n, dtype=np.int64)
        z = np.zeros(n, dtype=np.float64)

        nearest_vector = None
        nearest_length = float('inf')
        
        def enumerate_vectors(i):
            nonlocal nearest_length,nearest_vector,n
            
            if i == -1:
                v = basis @ x
                print(v)
                e_length = np.linalg.norm(v - target)**2
                if e_length <= nearest_length:
                    nearest_vector = v
                    nearest_length = e_length
                return

            M_i = np.sqrt((A - np.dot((z[i+1:] - y[i+1:])**2, B[i+1:])) / B[i])
            N_i = np.sum([mu[j, i+1] * x[j] for j in range(i+1, n)])
        
            try:

                lower_bound = int(np.ceil(y[i] - M_i - N_i))
                upper_bound = int(np.floor(y[i] + M_i - N_i))
                
                for xi in range(lower_bound, upper_bound + 1):
                    x[i] = xi
                    z[i] = x[i] + np.sum([mu[j, i+1] * x[j] for j in range(i+1, n)])
                    enumerate_vectors(i - 1)
            
            except:
                pass

        enumerate_vectors(n - 1)
        
        return nearest_vector

    def solveCVP(self, method='BNP', reduction = False, M = None):

        """ Method to solve CVP with BNP, BCV, embbeding, or enumeration method. """

        if reduction:
            lattice = BasisTransformer().reduce_basis(self.lattice)
        else:
            lattice = self.lattice

        if method=='BNP':
            solution = self.BNPSolver(lattice, self.target)
        elif method=='BCV':
            solution = self.BCVSolver(lattice, self.target)
        elif method=='embbeding':
            if M is None:
                raise ValueError('M is required for embbeding method.')
            else:
                solution = self.EmbbedingSolver(lattice, self.target, M)
        elif method=='enumeration':
            solution = self.enumeration(lattice)
        else:
            raise ValueError('Unknown method: ' + method)
        return solution
    

if __name__ == '__main__':

    lattice = np.array([[2, 4, 2],
                        [0, 0, 2], 
                        [6, 5, 4]])
    
    target = np.array([35, -13, -82])

    solver = CVPSolver(lattice, target)
    solution1 = solver.solveCVP(method = 'BNP')
    print(f'Comb lin that gives the solution:{np.linalg.solve(lattice, solution1)}')
    print(solution1)
    print('------------------------------')