import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def one_hot_encode(y, num_classes):
    """Convert labels to one-hot encoding."""
    num_samples = y.shape[0]
    one_hot = np.zeros((num_classes, num_samples))
    one_hot[y, np.arange(num_samples)] = 1
    return one_hot

def preprocess_data(X, y, X_test, y_test):
    """
    Reshape, normalize, and split the dataset.
    (Note: In main.py we do manual splitting to retain the raw labels as well.)
    """
    X = X.reshape(X.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    num_classes = len(np.unique(y_train))
    y_train = one_hot_encode(y_train, num_classes)
    y_val = one_hot_encode(y_val, num_classes)
    y_test = one_hot_encode(y_test, num_classes)
    
    return X_train.T, X_val.T, X_test.T, y_train, y_val, y_test

### Activation Functions ###
class Sigmoid:
    @staticmethod
    def function(x):
        return 1. / (1. + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        fx = Sigmoid.function(x)
        return fx * (1 - fx)

class Relu:
    @staticmethod
    def function(x):
        return np.maximum(0, x)
    
    @staticmethod
    def derivative(x):
        return (x > 0).astype(float)

class Tanh:
    @staticmethod
    def function(x):
        return np.tanh(x)
    
    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x)**2

class Softmax:
    @staticmethod
    def function(x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    @staticmethod
    def derivative(x):
        s = Softmax.function(x)
        return s * (1 - s)

### Optimizer Functions ###
def update_sgd(parameters, gradients, lr=0.01):
    """Standard SGD update."""
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= lr * gradients[f"dW{l}"]
        parameters[f"b{l}"] -= lr * gradients[f"db{l}"]
    return parameters

def update_momentum(parameters, gradients, prev_updates, lr=0.01, beta1=0.9):
    """Momentum-based gradient descent update."""
    L = len(parameters) // 2
    if "v" not in prev_updates:
        prev_updates["v"] = {f"W{l}": np.zeros_like(parameters[f"W{l}"]) for l in range(1, L+1)}
        prev_updates["v"].update({f"b{l}": np.zeros_like(parameters[f"b{l}"]) for l in range(1, L+1)})
    for l in range(1, L + 1):
        prev_updates["v"][f"W{l}"] = beta1 * prev_updates["v"][f"W{l}"] + (1 - beta1) * gradients[f"dW{l}"]
        prev_updates["v"][f"b{l}"] = beta1 * prev_updates["v"][f"b{l}"] + (1 - beta1) * gradients[f"db{l}"]
        parameters[f"W{l}"] -= lr * prev_updates["v"][f"W{l}"]
        parameters[f"b{l}"] -= lr * prev_updates["v"][f"b{l}"]
    return parameters, prev_updates

def update_nesterov(parameters, gradients, prev_updates, lr=0.01, beta1=0.9):
    """Nesterov Accelerated Gradient update."""
    L = len(parameters) // 2
    if "v" not in prev_updates:
        prev_updates["v"] = {f"W{l}": np.zeros_like(parameters[f"W{l}"]) for l in range(1, L+1)}
        prev_updates["v"].update({f"b{l}": np.zeros_like(parameters[f"b{l}"]) for l in range(1, L+1)})
    for l in range(1, L + 1):
        v_prev = prev_updates["v"][f"W{l}"]
        prev_updates["v"][f"W{l}"] = beta1 * v_prev + (1 - beta1) * gradients[f"dW{l}"]
        parameters[f"W{l}"] -= lr * (beta1 * v_prev + (1 - beta1) * gradients[f"dW{l}"])
        prev_updates["v"][f"b{l}"] = beta1 * prev_updates["v"][f"b{l}"] + (1 - beta1) * gradients[f"db{l}"]
        parameters[f"b{l}"] -= lr * prev_updates["v"][f"b{l}"]
    return parameters, prev_updates

def update_rmsprop(parameters, gradients, prev_updates, lr=0.001, beta2=0.999, epsilon=1e-8):
    """RMSprop update."""
    L = len(parameters) // 2
    if "s" not in prev_updates:
        prev_updates["s"] = {f"W{l}": np.zeros_like(parameters[f"W{l}"]) for l in range(1, L+1)}
        prev_updates["s"].update({f"b{l}": np.zeros_like(parameters[f"b{l}"]) for l in range(1, L+1)})
    for l in range(1, L + 1):
        prev_updates["s"][f"W{l}"] = beta2 * prev_updates["s"][f"W{l}"] + (1 - beta2) * (gradients[f"dW{l}"] ** 2)
        prev_updates["s"][f"b{l}"] = beta2 * prev_updates["s"][f"b{l}"] + (1 - beta2) * (gradients[f"db{l}"] ** 2)
        parameters[f"W{l}"] -= lr * gradients[f"dW{l}"] / (np.sqrt(prev_updates["s"][f"W{l}"]) + epsilon)
        parameters[f"b{l}"] -= lr * gradients[f"db{l}"] / (np.sqrt(prev_updates["s"][f"b{l}"]) + epsilon)
    return parameters, prev_updates

def update_adam(parameters, gradients, prev_updates, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Adam optimizer update."""
    L = len(parameters) // 2
    if "v" not in prev_updates:
        prev_updates["v"] = {f"W{l}": np.zeros_like(parameters[f"W{l}"]) for l in range(1, L+1)}
        prev_updates["v"].update({f"b{l}": np.zeros_like(parameters[f"b{l}"]) for l in range(1, L+1)})
    if "s" not in prev_updates:
        prev_updates["s"] = {f"W{l}": np.zeros_like(parameters[f"W{l}"]) for l in range(1, L+1)}
        prev_updates["s"].update({f"b{l}": np.zeros_like(parameters[f"b{l}"]) for l in range(1, L+1)})
    for l in range(1, L + 1):
        prev_updates["v"][f"W{l}"] = beta1 * prev_updates["v"][f"W{l}"] + (1 - beta1) * gradients[f"dW{l}"]
        prev_updates["s"][f"W{l}"] = beta2 * prev_updates["s"][f"W{l}"] + (1 - beta2) * (gradients[f"dW{l}"] ** 2)
        v_corrected = prev_updates["v"][f"W{l}"] / (1 - beta1 ** t)
        s_corrected = prev_updates["s"][f"W{l}"] / (1 - beta2 ** t)
        parameters[f"W{l}"] -= lr * v_corrected / (np.sqrt(s_corrected) + epsilon)
        
        prev_updates["v"][f"b{l}"] = beta1 * prev_updates["v"][f"b{l}"] + (1 - beta1) * gradients[f"db{l}"]
        prev_updates["s"][f"b{l}"] = beta2 * prev_updates["s"][f"b{l}"] + (1 - beta2) * (gradients[f"db{l}"] ** 2)
        v_corrected_b = prev_updates["v"][f"b{l}"] / (1 - beta1 ** t)
        s_corrected_b = prev_updates["s"][f"b{l}"] / (1 - beta2 ** t)
        parameters[f"b{l}"] -= lr * v_corrected_b / (np.sqrt(s_corrected_b) + epsilon)
    return parameters, prev_updates

def update_parameters(optimizer, parameters, gradients, prev_updates, t=1, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Dispatcher function to select the appropriate optimizer update.
    Options: "sgd", "momentum", "nesterov", "rmsprop", "adam", or "nadam" (which uses Adam update with lookahead).
    """
    if optimizer == "sgd":
        return update_sgd(parameters, gradients, lr), prev_updates
    elif optimizer == "momentum":
        return update_momentum(parameters, gradients, prev_updates, lr, beta1)
    elif optimizer == "nesterov":
        return update_nesterov(parameters, gradients, prev_updates, lr, beta1)
    elif optimizer == "rmsprop":
        return update_rmsprop(parameters, gradients, prev_updates, lr, beta2, epsilon)
    elif optimizer == "adam" or optimizer == "nadam":
        return update_adam(parameters, gradients, prev_updates, t, lr, beta1, beta2, epsilon)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
