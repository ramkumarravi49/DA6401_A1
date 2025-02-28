import numpy as np

# Class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# One-hot encoding function
def one_hot_encode(y, num_classes):
    """Convert labels to one-hot encoding."""
    num_samples = y.shape[0]
    one_hot = np.zeros((num_classes, num_samples))
    one_hot[y, np.arange(num_samples)] = 1
    return one_hot

# Data Preprocessing
def preprocess_data(X, y, X_test, y_test):
    """Reshape, normalize, and split the dataset."""
    X = X.reshape(X.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    num_classes = len(np.unique(y_train))
    y_train = one_hot_encode(y_train, num_classes)
    y_val = one_hot_encode(y_val, num_classes)
    y_test = one_hot_encode(y_test, num_classes)

    return X_train.T, X_val.T, X_test.T, y_train, y_val, y_test

# Activation Functions (Moved from NNet.py)
class Sigmoid:
    @staticmethod
    def function(x):
        return 1. / (1. + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        return Sigmoid.function(x) * (1 - Sigmoid.function(x))

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
        return 1 - (np.tanh(x) ** 2)

class Softmax:
    @staticmethod
    def function(x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    @staticmethod
    def derivative(x):
        return Softmax.function(x) * (1 - Softmax.function(x))
