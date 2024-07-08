import numpy as np

class BasisTransformer():

    def __init__(self):
        pass

    def isIndependent(self, basis):

        """Check if a basis is independent."""
        
        rango = np.linalg.matrix_rank(basis) 
        num_cols = basis.shape[1]
        return rango == num_cols
    
    def isOrthogonal(self, basis):

        """Check if a basis is orthogonal."""
        
        basis = np.array(basis, dtype=float)
        num_vectors = basis.shape[1]
        
        for i in range(num_vectors):
            for j in range(i+1, num_vectors):
                if not np.isclose(np.dot(basis[:, i], basis[:, j]), 0):
                    return False
        return True
    
    def order_columns_by_length(self,lat):

        """Order the columns of a basis matrix by their length."""

        norms = np.linalg.norm(lat, axis=0)
        sorted_indices = np.argsort(norms)
        return sorted_indices

    def areEquivalent(self, basis1, basis2):

        """Check if two bases are equivalent."""

        rango_union = np.linalg.matrix_rank(np.hstack((basis1, basis2)))
        rango_base_1 = np.linalg.matrix_rank(basis1) 
        rango_base_2 = np.linalg.matrix_rank(basis2)
        return rango_union == rango_base_1 == rango_base_2

    def orthogonalize_basis(self, basis, mu_coefs=False):
        
        """Orthogonalize the columns of basis using Gram-Schmidt."""

        basis = np.array(basis, dtype=float)
        n = basis.shape[1]
        ortho_basis = np.zeros_like(basis)
        mu = np.zeros((n, n))
        
        for i in range(n):
            ortho_basis[:, i] = basis[:, i]
            for j in range(i):
                mu[i, j] = np.dot(basis[:, i], ortho_basis[:, j]) / np.dot(ortho_basis[:, j], ortho_basis[:, j])
                ortho_basis[:, i] -= mu[i, j] * ortho_basis[:, j]
        
        if mu_coefs:
            return ortho_basis, mu
        else:
            return ortho_basis
        
    def lll_reduce(self, basis, delta=0.75):

        """Perform LLL reduction on the basis."""
        basis = np.array(basis, dtype=float)
        n = basis.shape[1]

        if not self.isOrthogonal(basis):
            ortho_basis = self.orthogonalize_basis(basis)
        else:
            ortho_basis = basis
        
        k = 1
        while k < n:
            for j in range(k - 1, -1, -1):
                mu_k_j = np.dot(basis[:, k], ortho_basis[:, j]) / np.dot(ortho_basis[:, j], ortho_basis[:, j])
                basis[:, k] = basis[:, k] - round(mu_k_j) * basis[:, j]
                ortho_basis = self.orthogonalize_basis(basis)

            if np.dot(ortho_basis[:, k], ortho_basis[:, k]) >= (delta - (np.dot(ortho_basis[:, k], ortho_basis[:, k-1]) / np.dot(ortho_basis[:, k-1], ortho_basis[:, k-1]))**2) * np.dot(ortho_basis[:, k-1], ortho_basis[:, k-1]):
                k += 1
            else:
                basis[:, [k-1, k]] = basis[:, [k, k-1]]  # Swap columns
                ortho_basis = self.orthogonalize_basis(basis)  # Recompute orthogonal basis
                k = max(k-1, 1)
        
        return basis.astype(np.int64)


    def reduce_basis(self, basis, method='LLL'):
        """Reduce the basis using the specified method (default: LLL)."""
        if method == 'LLL':
            return self.lll_reduce(basis)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

if __name__=='__main__':
    # Create numpy basis with 4 columns of dimension 5 and not orthogonalized
    basis = np.array([[1, 0, 0, 0],
                       [1.6, 0, 0, 0], 
                       [0, 1, 0, 2.2],
                       [0, 7, 1, 0],
                       [0, 0, 4, 1]])


    
    
    ba = BasisTransformer()

    print('Stop')

