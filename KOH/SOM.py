import numpy as np
import matplotlib.pyplot as plt

from functions import *
from MinMaxScaler import MinMaxScaler

class SOM():
    def __init__(self, N: int, M: int, dim: int, 
                 learning_rate: float = 0.1, radius: float = 1, shape: str = 'rectangle',
                 metric: str = 'e', neighbor_distance_function: str = 'gaussian', scale_data: bool = True):
        self.N = N
        self.M = M
        self.dim = dim
        self.learning_rate = learning_rate
        self.radius = radius
        self.neurons = np.random.uniform(0,1,(dim, N, M))
        self.scaler = MinMaxScaler()
        self.scale_data = scale_data
        if metric not in ['m', 'e']:
            raise ValueError('Invalid distance function')
        self.metric = metric
        if neighbor_distance_function not in ['gaussian', 'mexican_hat']:
            raise ValueError('Invalid neighbourhood distance function')
        self.neighbor_distance_function = neighbor_distance_function
        self.learning_rate_decay = exponential_decay
        self.bmu = None
        self.index_matrix = generate_index_matrix(self.neurons.shape[1:], shape=shape)
        self.distances = [np.linalg.norm(np.array([i, j]) - self.index_matrix, ord=1 if metric == 'm' else 2, axis=2) for i in range(self.N) for j in range(self.M)]
        self.distances = np.array(self.distances).reshape(self.N, self.M, self.N, self.M)

    def find_bmu(self, x):
        self.bmu = np.array(
            np.unravel_index(
            np.argmin(
                np.linalg.norm(self.neurons.T - x, axis=2).T
            ), (self.N, self.M)
        ) 
        )
    
    def batch_find_bmu(self, X):
        X = self.scaler.fit_transform(X) if self.scale_data else X
        return np.array([
            np.unravel_index(
            np.argmin(
                np.linalg.norm(self.neurons.T - x, axis=2).T
            ), (self.N, self.M)
        ) for x in X
        ])
    
    def scale_neurons(self, X):
            self.scaler.fit(X)
            self.neurons = self.scaler.inverse_transform(self.neurons)   
            return self.neurons
    
    def update_neurons(self, x, t):
        self.find_bmu(x)
        x = np.broadcast_to(x, (self.M, self.N, self.dim)).T
        distances = self.distances[self.bmu[0], self.bmu[1]]
        distances = self.gaussian_neighbourhood_function(distances, t) if self.neighbor_distance_function == 'gaussian' else self.mexican_hat_neighbourhood_function(distances, t)
        distances = np.broadcast_to(distances, (self.dim, self.N, self.M))
        self.neurons += self.learning_rate * distances * (x - self.neurons)
        
    def mexican_hat_neighbourhood_function(self, d, t):
        return (2-4*((d * t * self.radius)**2)) * np.exp(-(d * t * self.radius)**2)
    
    def gaussian_neighbourhood_function(self, d, t):
        return np.exp(-(d * t * self.radius)**2)

    def batch_update_neurons(self, X, t):
        bmus = self.batch_find_bmu(X) # (k, 2)
        distances = self.distances[bmus[:, 0], bmus[:, 1]]
        X = np.broadcast_to(X.T, (self.M, self.N, self.dim, X.shape[0])).T # (k, d, n, m)
        distances = distances[:, np.newaxis, :, :] # (k, 1, n, m)
        distances = self.gaussian_neighbourhood_function(distances, t) if self.neighbor_distance_function == 'gaussian' else self.mexican_hat_neighbourhood_function(distances, t)
        neurons_update = self.learning_rate * distances * (X - self.neurons) # (k, d, n, m)
        neurons_update = np.sum(neurons_update, axis=0) # (d, n, m)
        self.neurons += neurons_update
    
    def train(self, X, epochs=10):
        if self.scale_data:
            X = self.scaler.fit_transform(X)
 
        for t in range(1,1+epochs):
            X = np.random.permutation(X)
            self.learning_rate *= self.learning_rate_decay(t, epochs)
            for x in X:
                self.update_neurons(x, t)
            if t % np.ceil(epochs / 100) == 0:
                print(f'\rIteration {t}/{epochs}', end='')
        self.neurons = self.scaler.inverse_transform(self.neurons.T).T if self.scale_data else self.neurons
        return self.neurons
    
    def batch_train(self, X, epochs=10, batch_size=32):
        if self.scale_data:
            X = self.scaler.fit_transform(X)
        else:
            self.scaler.fit(X)
            self.neurons = self.scaler.inverse_transform(self.neurons.T).T
        for t in range(1,1+epochs):
            X = np.random.permutation(X)
            self.learning_rate *= self.learning_rate_decay(t, epochs)
            for i in range(0, len(X), batch_size):
                self.batch_update_neurons(X[i:i+batch_size], t)
            if t % np.ceil(epochs / 100) == 0:
                print(f'\rIteration {t}/{epochs}', end='')
        self.neurons = self.scaler.inverse_transform(self.neurons.T).T if self.scale_data else self.neurons
        return self.neurons
    
    def predict(self, X):
        X = self.scaler.fit_transform(X) if self.scale_data else X
        predictions = []
        for x in X:
            self.find_bmu(x)
            predictions.append(self.neurons[self.bmu])
        return np.array(predictions)
    
    def predict_labels(self, X):
        X = self.scaler.fit_transform(X) if self.scale_data else X
        return np.array(
            [
                np.argmin(
                    [e_distance(self.neurons.T[j][i], x) for i in range(self.neurons.T.shape[1]) for j in range(self.neurons.T.shape[0])]
                ) 
            for x in X]
        )
    
    def predict_labels_from_y(self, X, y):
        X = self.scaler.fit_transform(X) if self.scale_data else X
        self.assign_clusters_labels(X, y)
        clusters = [self.clusters[i][j] for i in range(self.N) for j in range(self.M)]
        return clusters[np.array(
            [
                np.argmin(
                    [e_distance(self.neurons.T[j][i], x) for i in range(self.neurons.T.shape[1]) for j in range(self.neurons.T.shape[0])]
                ) for x in X]
        )]
    
    def assign_clusters_labels(self, X, y):
        prediction = []
        X = self.scaler.fit_transform(X) if self.scale_data else X
        self.clusters = [[None for _ in range(self.M)] for _ in range(self.N)]
        for i in range(self.neurons.T.shape[1]):
            for j in range(self.neurons.T.shape[0]):
                self.clusters[i][j] = y[np.argmin([e_distance(self.neurons.T[j, i], x) for x in X])]
        for x in X:
            self.find_bmu(x)
            prediction.append(self.clusters[self.bmu[0]][ self.bmu[1]])
        return prediction
    
    def plot_u_matrix(self):
        u_matrix = np.zeros((self.N, self.M))
        for i in range(self.n_neurons):
            for j in range(self.m_neurons):
                if i > 0:
                    u_matrix[i, j] += e_distance(self.neurons[i, j], self.neurons[i-1, j])
                if i < self.N - 1:
                    u_matrix[i, j] += e_distance(self.neurons[i, j], self.neurons[i+1, j])
                if j > 0:
                    u_matrix[i, j] += e_distance(self.neurons[i, j], self.neurons[i, j-1])
                if j < self.M - 1:
                    u_matrix[i, j] += e_distance(self.neurons[i, j], self.neurons[i, j+1])
        plt.imshow(u_matrix, cmap='gray')
        plt.colorbar()