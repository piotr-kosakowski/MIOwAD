import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    x = np.clip(x, -500, 500)  # Limit x values to avoid overflow
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)


def linear(x):
    return x


def linear_derivative(x):
    return np.ones_like(x)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=1))
    return exps / np.sum(exps, axis=1, keepdims=True)


def softmax_derivative(softmax_output):
    return softmax_output * (1 - softmax_output)


def reLU(x):
    return np.maximum(0, x)


def reLU_derivative(reLu_output):
    return np.sign(reLu_output)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(tanh_output):
    return 1 - tanh_output**2


def mse(y_true: np.array, y_pred: np.array):
    return np.mean((y_true - y_pred)**2)


def plot_matrices_list(matrices):
    fig, ax = plt.subplots(1, len(matrices), figsize=(10, 5))
    global_min = min(matrix.min() for matrix in matrices)
    global_max = max(matrix.max() for matrix in matrices)
    for i in range(len(matrices)):
        ax[i].imshow(matrices[i], cmap='viridis', aspect='auto')
        title = 'Biases'
        if i < len(matrices)/2:
            title = 'Weights'
        ax[i].set_title(f'{title }: {i % (len(matrices)//2) +1}')
        plt.colorbar(ax[i].imshow(matrices[i], cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max), ax=ax[i])


def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))  # Remove the reshape operation for y
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    # i = 0
    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-y.shape[1]]  # Adjust the indexing for X_mini
        Y_mini = mini_batch[:, -y.shape[1]:]  # Adjust the indexing for Y_mini
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-y.shape[1]]  # Adjust the indexing for X_mini
        Y_mini = mini_batch[:, -y.shape[1]:]  # Adjust the indexing for Y_mini
        mini_batches.append((X_mini, Y_mini))
    return mini_batches


def xavier_initialization_uniform(layer_sizes):
    weights = []
    for i in range(len(layer_sizes) - 1):
        limit = np.sqrt(6. / (layer_sizes[i] + layer_sizes[i+1]))  # Calculate the limit for the uniform distribution
        weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1])))
    return weights


def he_initialization_uniform(layer_sizes):
    weights = []
    for i in range(len(layer_sizes) - 1):
        limit = np.sqrt(2. / layer_sizes[i])  # Calculate the limit for the uniform distribution
        weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1])))
    return weights


def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-10))


def cross_entropy_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]


def one_hot_encode(y: np.array):
    encoded = np.zeros((y.size, y.max()+1)) 
    encoded[np.arange(y.size), y] = 1
    return encoded


def l2_penalty(w: np.array):
    return (w ** 2).sum() 


def l1_penalty(w: np.array):
    return np.abs(w).sum()


def l2_penalty_grad(w: np.array):
    return w * 2


def l1_penalty_grad(w: np.array):
    return np.sign(w)


def weights_to_vector(weights):
    return np.concatenate([w.flatten() for w in weights])