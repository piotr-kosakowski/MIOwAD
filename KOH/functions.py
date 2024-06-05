import numpy as np
def e_distance(x1, x2):
    return np.linalg.norm(x1 - x2, 2)

def m_distance(x1, x2):
    return np.linalg.norm(x1 - x2, 1)

def gaussian_neighbourhood_function(x, y, radius, t, dist = 'm'):
    if dist == 'm':
        distance = m_distance(x, y)
    elif dist == 'e':
        distance = e_distance(x, y)
    else:
        raise ValueError('Invalid distance function')
    return np.exp(-(distance * t * radius )**2)

def mexican_hat_neighbourhood_function(x, y, radius, t, dist = 'm'):
    if dist == 'm':
        distance = m_distance(x, y)
    elif dist == 'e':
        distance = e_distance(x, y)
    else:
        raise ValueError('Invalid distance function')
    return (2-4*((distance * t * radius)**2)) * np.exp(-(distance * t * radius)**2)

def exponential_decay(t, n_iterations):
    return np.exp(-t / n_iterations)

def generate_index_matrix(dim, shape = 'rectangle'):
    if shape == 'rectangle':
        return np.indices(dim).transpose(1, 2, 0).astype(float)
    if shape == 'hexagon':
        indices = []
        for i in range(dim[0]):
            for j in range(dim[1]):
                indx = [i, j]
                if i % 2:
                    indx[1] += 0.5
                indx[0] *= np.sqrt(3)/2
                indices.append(indx)
        return np.array(indices).reshape(dim[0], dim[1], len(dim))
    raise ValueError('Invalid shape')

def calculate_distances_to_bmu(index_matrix, bmu, distance_func, *args):
    return np.apply_along_axis(lambda idx: distance_func(idx, bmu, *args), 2, index_matrix)

def calculate_distances_to_bmu2(index_matrix, bmus):
    # index_matrix: (n, m, 2)
    index_matrices = np.broadcast_to(index_matrix, (bmus.shape[0],) + index_matrix.shape) # (k, n, m, 2)
    # bmus: (k, 2)
    bmus = bmus[:, np.newaxis, np.newaxis, :] # (k, 1, 1, 2)
    bmus = np.broadcast_to(bmus, index_matrices.shape) # (k, n, m, 2)
    return np.linalg.norm(index_matrices - bmus, axis=3) # (k, n, m)
    