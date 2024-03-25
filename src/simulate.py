import numpy as np
import pandas as pd

class Subspace():
    def __init__(self, num_features = None, num_dimensions = None, basis = None):
        if basis is not None:
            self.num_features = basis.shape[1]
            self.ortho_matrix = np.linalg.qr(basis)[0]
        else:
            gaussian_matrix = np.random.normal(0, 1, (num_features, num_dimensions))
            self.ortho_matrix = np.linalg.qr(gaussian_matrix)[0]
            self.num_features = num_features
    
    
    def simulate(self, num_points, var=1):
        random_points = np.random.randn(self.num_features, num_points)
        random_points /= np.linalg.norm(random_points)
        subspace_points = self.ortho_matrix @ self.ortho_matrix.T @ random_points 
        return subspace_points.T 
    

def simulate_data(num_features, num_points, num_subspaces, subspaces_dim):
    subspaces = [Subspace(num_features, subspaces_dim) for i in range(1, num_subspaces+1)]
    data = []
    indices = list(range(num_subspaces))
    labels = np.array([])

    for i in range(num_points):
        label = int(np.random.choice(indices, 1))
        chosen_subspace = subspaces[label]
        labels = np.append(labels, label)
        data.append(np.concatenate([chosen_subspace.simulate(1)]))
    
    data = pd.DataFrame(np.concatenate(data))
    data['labels'] = labels

    return data

        
