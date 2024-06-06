import numpy as np
from typing import List, Callable
import matplotlib.pyplot as plt

from functions import *


class CustomMLP:
    def __init__(self, layer_sizes: List[int], weights: List[np.array] = None, biases: List[np.array] = None, activation_functions: List[Callable[[np.array], np.array]] = None,
                 activation_functions_derivatives: List[Callable[[np.array], np.array]] = None, loss: Callable[[np.array, np.array], float] = mse,
                 loss_derivative: Callable[[np.array, np.array], np.array] = lambda y_true, y_pred: 2*(y_pred - y_true)/y_true.shape[0]):
        self.layer_sizes = layer_sizes
        self.weights = weights
        self.biases = biases
        self.activation_functions = activation_functions
        self.activation_functions_derivatives = activation_functions_derivatives
        self.layers = len(layer_sizes) - 1    
        self.loss = loss
        self.loss_derivative = loss_derivative    
        if self.activation_functions is None:
            self.activation_functions = [sigmoid] * (len(layer_sizes) - 2) + [linear]
            self.activation_functions_derivatives = [sigmoid_derivative] * (len(layer_sizes) - 2) + [linear_derivative]
        if self.weights is None:
            self.weights = xavier_initialization_uniform(self.layer_sizes)
        if self.biases is None:
            self.biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(len(self.layer_sizes) - 1)]
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]
        self.cache_weights = [np.zeros_like(w) for w in self.weights]
        self.cache_biases = [np.zeros_like(b) for b in self.biases]
        self.function_dict = {'sigmoid': sigmoid, 'linear': linear, 'softmax': softmax, 'reLU': reLU, 'tanh': tanh}
        self.function_dict_derivatives = {'sigmoid': sigmoid_derivative, 'linear': linear_derivative, 'softmax': linear_derivative, 'reLU': reLU_derivative, 'tanh': tanh_derivative}
    
    def __repr__(self):
        return f'CustomMLP(layer_sizes={self.layer_sizes}, activation_functions={[next((key for key, val in self.function_dict.items() if val == value), None) for value in self.activation_functions]})'

    def set_activation_functions(self, functions: List[str]):
        if len(functions) == 1:
            functions = functions * (len(self.layer_sizes) - 2) + ['linear']
        self.activation_functions = [self.function_dict[f] for f in functions]
        self.activation_functions_derivatives = [self.function_dict_derivatives[f] for f in functions]
        if 'reLU' in functions:
            self.weights = he_initialization_uniform(self.layer_sizes)

    def forward_pass(self, a: np.array, backprop: bool = False):
        activations = [a]
        for f, w, b in zip(self.activation_functions, self.weights, self.biases):
            z = np.dot(a, w) + b
            a = f(z)
            activations.append(a)
        if backprop: return activations
        return a
    
    def _backward_pass(self, x: np.array, y: np.array , activations: List[np.array] = None, 
               cost_derivative: Callable[[np.array, np.array], np.array] = lambda y_true, y_pred: 2*(y_pred - y_true)/y_true.shape[0]):
        if activations is None:
            activations= self.forward_pass(x, backprop=True)
        deltas = [None] * self.layers
        deltas[-1] = cost_derivative(y, activations[-1]) * self.activation_functions_derivatives[-1](activations[-1])
        for i in range(self.layers - 2, -1, -1):
            deltas[i] = np.dot(deltas[i+1], self.weights[i+1].T) * self.activation_functions_derivatives[i](activations[i+1])
        return deltas
        
    def _update_params(self, x: np.array, y: np.array, learning_rate: float = 0.01,
                      cost_derivative: Callable[[np.array, np.array], np.array] = lambda y_true, y_pred: 2*(y_pred - y_true)/y_true.shape[0],
                      momentum_rate: float = 0.9, decay_rate: float = 0.9, epsilon: float = 1e-7, alpha: float = 0, penalty: str = 'l2', optimizer: str = 'gd'):
        activations = self.forward_pass(x, backprop=True)
        deltas = self._backward_pass(x, y, activations, cost_derivative)
        for i in range(len(self.weights)):
            s = x.shape[0] if optimizer in ['gd', 'sgd', 'minibatch_gd'] else 1
            grad_weights = np.dot(activations[i].T, deltas[i]) / s 
            grad_biases = np.mean(deltas[i], axis=0, keepdims=1) / s
            grad_weights += alpha * (l2_penalty_grad(self.weights[i]) if penalty == 'l2' else l1_penalty_grad(self.weights[i]))
            if optimizer in ['gd', 'sgd', 'minibatch_gd']:
                self.weights[i] -= learning_rate * grad_weights
                self.biases[i] -= learning_rate * grad_biases
            elif optimizer == 'momentum':
                self.velocity_weights[i] = momentum_rate * self.velocity_weights[i] - learning_rate * grad_weights
                self.velocity_biases[i] = momentum_rate * self.velocity_biases[i] - learning_rate * grad_biases
                self.weights[i] += self.velocity_weights[i]
                self.biases[i] += self.velocity_biases[i]
            elif optimizer == 'rmsprop':
                self.cache_weights[i] = decay_rate * self.cache_weights[i] + (1 - decay_rate) * grad_weights**2
                self.cache_biases[i] = decay_rate * self.cache_biases[i] + (1 - decay_rate) * grad_biases**2
                self.weights[i] -= learning_rate * grad_weights / (np.sqrt(self.cache_weights[i]) + epsilon)
                self.biases[i] -= learning_rate * grad_biases / (np.sqrt(self.cache_biases[i]) + epsilon)
    
    def train(self, x: np.array, y: np.array, learning_rate: float = 0.01, epochs: int = 100, batch_size: int = 32, optimizer: str = 'gd', 
              momentum_rate: float = 0.9, decay_rate: float = 0.9, epsilon: float = 1e-7, alpha: float = 0, penalty: str = 'l2', 
              patience: int = -1, val_x: np.array = None, val_y: np.array = None):
        if optimizer not in ['gd', 'momentum', 'rmsprop', 'sgd', 'minibatch_gd']:
            raise ValueError('Invalid optimizer')
        min_loss = float('inf')
        subtrain_losses = []
        validation_losses = []
        best_biases = None
        best_weights = None
        if patience > 0:
            min_val_loss = float('inf')
            counter = 0
            if val_x is None or val_y is None:
                data = np.hstack((x, y))  
                np.random.shuffle(data)
                x = data[:, :-y.shape[1]]
                y = data[:, -y.shape[1]:]
                subtrain_x = x[:int(0.8 * x.shape[0])]
                subtrain_y = y[:int(0.8 * y.shape[0])]
                val_x = x[int(0.8 * x.shape[0]):]
                val_y = y[int(0.8 * y.shape[0]):]
            else:
                subtrain_x = x
                subtrain_y = y
        else:
            subtrain_x = x
            subtrain_y = y

        batch_size = min(batch_size, x.shape[0])
        batch_size = x.shape[0] if optimizer == 'gd' else batch_size
        batch_size = 1 if optimizer == 'sgd' else batch_size  
        for epoch in range(1, epochs+1):
            mini_batches = create_mini_batches(subtrain_x, subtrain_y, batch_size)
            for mini_batch in mini_batches:
                self._update_params(mini_batch[0], mini_batch[1], learning_rate=learning_rate, cost_derivative=self.loss_derivative, momentum_rate=momentum_rate, 
                                   decay_rate=decay_rate, epsilon=epsilon, alpha=alpha, penalty=penalty, optimizer=optimizer)       
            y_pred = self.forward_pass(subtrain_x)
            loss = self.loss(subtrain_y, y_pred) + alpha * (l1_penalty(weights_to_vector(self.weights)) if penalty == 'l1' else l2_penalty(weights_to_vector(self.weights)))
            min_loss = min(min_loss, loss)
            subtrain_losses.append(loss)
            if patience > 0:
                val_pred = self.forward_pass(val_x)
                val_loss = self.loss(val_y, val_pred) 
                validation_losses.append(val_loss + alpha * (l1_penalty(weights_to_vector(self.weights)) if penalty == 'l1' else l2_penalty(weights_to_vector(self.weights))))
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    counter = 0
                else:
                    counter += 1
                    if counter > patience:
                        print('\nEarly stopping')
                        self.weights = best_weights
                        self.biases = best_biases
                        break
            if epoch % max(1, (epochs // 100)) == 0:
                print(self._training_progress(epoch, loss, min_loss, x, y), end="")
        if best_weights and best_biases:
            self.weights = best_weights
            self.biases = best_biases

        
    def train_early_stopping(self, x: np.array, y: np.array, learning_rate: float = 0.01, epochs: int = 100, batch_size: int = 32, 
              optimizer: str = 'gd', momentum_rate: float = 0.9, decay_rate: float = 0.9, epsilon: float = 1e-7, alpha: float = 0, penalty: str = 'l2', 
              val_x: np.array = None, val_y: np.array = None, patience: int = 10):
        if optimizer not in ['gd', 'momentum', 'rmsprop', 'sgd', 'minibatch_gd']:
            raise ValueError('Invalid optimizer')
        min_loss = float('inf')
        min_val_loss = float('inf')
        counter = 0
        subtrain_losses = []
        validation_losses = []
        if val_x is None or val_y is None:
            data = np.hstack((x, y))  
            np.random.shuffle(data)
            x = data[:, :-y.shape[1]]
            y = data[:, -y.shape[1]:]
            subtrain_x = x[:int(0.8 * x.shape[0])]
            subtrain_y = y[:int(0.8 * y.shape[0])]
            val_x = x[int(0.8 * x.shape[0]):]
            val_y = y[int(0.8 * y.shape[0]):]
        else:
            subtrain_x = x
            subtrain_y = y
        batch_size = min(batch_size, subtrain_x.shape[0])
        batch_size = subtrain_x.shape[0] if optimizer == 'gd' else batch_size
        batch_size = 1 if optimizer == 'sgd' else batch_size  
        for epoch in range(1, epochs+1):   
            mini_batches = create_mini_batches(subtrain_x, subtrain_y, batch_size)
            for mini_batch in mini_batches:
                self._update_params(mini_batch[0], mini_batch[1], learning_rate=learning_rate, cost_derivative=self.loss_derivative, momentum_rate=momentum_rate, 
                                   decay_rate=decay_rate, epsilon=epsilon, alpha=alpha, penalty=penalty, optimizer=optimizer)       
            y_pred = self.forward_pass(subtrain_x)
            loss = self.loss(subtrain_y, y_pred) 
            subtrain_losses.append(loss)
            val_pred = self.forward_pass(val_x)
            val_loss = self.loss(val_y, val_pred) 
            validation_losses.append(val_loss)
            min_loss = min(min_loss, loss)
        
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    print('\nEarly stopping')
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            if epoch % max(1, (epochs // 100)) == 0:
                print(self._training_progress(epoch, loss, min_loss, x, y), end="")
        self.weights = best_weights
        self.biases = best_biases
        validation_losses = []
        plt.plot(subtrain_losses, label='Training Loss')
        if validation_losses: plt.plot(validation_losses, label='Validation Loss')
        plt.legend()
        plt.show()
        print(f'\nFinal Training Loss: {subtrain_losses[-1]}' + (f', Final Validation Loss: {validation_losses[-1]}' if validation_losses else ''))
    
    def _training_progress(self, epoch, loss, min_loss, x=None, y=None):
        return f'\rEpoch: {epoch}, Loss: {loss}, Min_loss: {min_loss}'
                         
class classificationMLP(CustomMLP):
    def __init__(self, layer_sizes: List[int], weights: List[np.array] = None, biases: List[np.array] = None, activation_functions: List[Callable[[np.array], np.array]] = None,
                  activation_functions_derivatives: List[Callable[[np.array], np.array]] = None):
        super().__init__(layer_sizes, weights, biases, activation_functions, activation_functions_derivatives)
        self.activation_functions[-1] = softmax
        self.activation_functions_derivatives[-1] = linear # derivative of softmax is not needed - it is included in the cross-entropy loss
        self.loss = cross_entropy
        self.loss_derivative = cross_entropy_derivative

    def set_loss(self, loss: Callable[[np.array, np.array], float], loss_derivative: Callable[[np.array, np.array], np.array]):
        self.loss = loss
        self.loss_derivative = loss_derivative
    
    def set_activation_functions(self, functions: List[str]):
        if len(functions) == 1:
            functions = functions * (len(self.layer_sizes) - 2) + ['softmax']
        self.activation_functions = [self.function_dict[f] for f in functions]
        self.activation_functions_derivatives = [self.function_dict_derivatives[f] for f in functions]
        if 'reLU' in functions:
            self.weights = he_initialization_uniform(self.layer_sizes)

    def predict(self, x: np.array):
        return np.argmax(self.forward_pass(x), axis=1)
    
    def _training_progress(self, epoch, loss, min_loss, x, y):
        return super()._training_progress(epoch, loss, min_loss) + f', Accuracy: {self.accuracy(x, y)}'

    def accuracy(self, x: np.array, y: np.array):
        return np.mean(self.predict(x) == np.argmax(y, axis=1))
    
    def confusion_matrix(self, x: np.array, y: np.array):
        y_pred = self.predict(x)
        y_true = np.argmax(y, axis=1)
        num_classes = y.shape[1]
        matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                matrix[i, j] = np.sum((y_true == i) & (y_pred == j))
        return matrix

    def _precisions_and_recalls(self, x: np.array, y: np.array):
        num_classes = y.shape[1]
        cm = self.confusion_matrix(x, y)
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        for i in range(num_classes):
            precision[i] = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) != 0 else 0
            recall[i] = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) != 0 else 0
        return precision, recall
    
    def averaged_F1_score(self, x: np.array, y: np.array):
        precisions, recalls = self._precisions_and_recalls(x, y)
        f1_scores = 2 * precisions * recalls / (precisions + recalls)
        f1_scores[np.isnan(f1_scores)] = 0
        return np.mean(f1_scores)
    
    def F1_of_averages(self, x: np.array, y: np.array):
        precisions, recalls = self._precisions_and_recalls(x, y)
        return 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls))
