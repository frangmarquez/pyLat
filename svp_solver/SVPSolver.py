import sys
sys.path.append('.')
from basis_reduction import BasisTransformer
import numpy as np


class SVPSolver:
    def __init__(self,lattice):
        self.lattice = self.toLattice(lattice)

    def order_columns_by_length(self,lat):
        norms = np.linalg.norm(lat, axis=0)
        sorted_indices = np.argsort(norms)
        return sorted_indices

    def toLattice(self, lattice):
        if all(coef.is_integer() for row in lattice for coef in row):
            return lattice.astype(np.int64)
        else:
            raise Exception('Input is not a lattice.')


    def solve(self, method='enumeration'):
        if method == 'enumeration':
            solution = self.enumeration()
            return solution
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def enumeration(self):
        
        ba = BasisTransformer()

        basis = self.lattice
        ortho_basis = ba.orthogonalize_basis(basis)

        ordered_basis = self.order_columns_by_length(basis)
        basis = basis[:, ordered_basis]
        ortho_basis = ortho_basis[:, ordered_basis]
        mu = np.zeros(ortho_basis.shape)

        for i in range(ortho_basis.shape[1]):
            for j in range(0,i+1):
                mu[j, i] = np.dot(ortho_basis[:, i], ortho_basis[:, j]) / np.dot(ortho_basis[:, j], ortho_basis[:, j])
                mu[i, j] = mu[j, i]


        A = np.dot(ortho_basis[:, 0], ortho_basis[:, 0])
        B = np.array([np.dot(ortho_basis[:, i], ortho_basis[:, i]) for i in range(ortho_basis.shape[1])])
        n = basis.shape[1]
        x = np.zeros(n, dtype=int)
        shortest_vector = None
        shortest_length = float('inf')
        tree = list()

        def enumerate_vectors(i,tree_node):
            nonlocal shortest_vector, shortest_length, tree
            
            if i == -1:
                v = basis @ x
                v_length = np.linalg.norm(v)
                if v_length < shortest_length and v_length > 0:
                    shortest_length = v_length
                    shortest_vector = v
                return
            
            M1 = np.sqrt( (A - np.dot( x[i+1:]**2 , B[i+1:] ) ) / B[i])
            M2 = np.sum([mu[j, i+1] * x[j] for j in range(i+1, n)])

            
            lower_bound = int(np.ceil(-M1 - M2))
            upper_bound = int(np.floor(M1 - M2))
                
            for xi in range(lower_bound, upper_bound + 1):
                x[i] = xi
                if i==0:
                    new_tree_node = [xi]
                    enumerate_vectors(i - 1, new_tree_node[0])
                else:
                    new_tree_node = [xi, []]
                    enumerate_vectors(i - 1, new_tree_node[1])
                tree_node.append(new_tree_node)
                
            
        enumerate_vectors(n - 1, tree)

        return shortest_vector, tree
    
    def print_tree(self,tree, level=0):
        for node in tree:
            if len(node)==2:
                value, children = node
                print('  ' * level + str(value))
                self.print_tree(children, level + 1)
            else:
                value = node
                print('  ' * level + str(value))

if __name__ == '__main__':
    lattice = np.array([[3, 4],
                        [1, 0]])
    solver = SVPSolver(lattice)
    solution, tree = solver.solve()
    solver.print_tree(tree)
    

