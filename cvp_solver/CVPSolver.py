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
        real_coefs = np.linalg.solve(lattice, target)
        approx_coefs = np.round(real_coefs)
        solution = np.array(lattice @ approx_coefs, dtype=np.int64)
        return solution


    def BNPSolver(self, lattice, target):
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

    def solveCVP(self, method='BNP', reduction = False, M = None):

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
            pass
        else:
            raise ValueError('Unknown method: ' + method)
        return solution
    

if __name__ == '__main__':



    lattice = np.array([[45, 25, -10],
                        [-77,57,  0], 
                        [-13,13, 27]])
    
    comb = [567, 25, -383]
    target = np.array([23425, -3995, -10015])

    solver = CVPSolver(lattice, target)
    solution = solver.solveCVP(method = 'embbeding', M = 1)[0]
    print(solution)