import numpy as np
from helper import Sigmoid, Relu, Tanh, Softmax  # Import activation functions

def Init_Parameters(layer_dims, init_mode="xavier"):
    np.random.seed(42)
    parameters = {}  # Renamed params to parameters
    prev_updates = {}
    for i in range(1, len(layer_dims)):
        if init_mode == 'random_normal':
            parameters[f"W{i}"] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        elif init_mode == 'random_uniform':
            parameters[f"W{i}"] = np.random.rand(layer_dims[i], layer_dims[i-1]) * 0.01
        elif init_mode == 'xavier':
            parameters[f"W{i}"] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(2 / (layer_dims[i] + layer_dims[i-1]))
        parameters[f"b{i}"] = np.zeros((layer_dims[i], 1))
        prev_updates[f"W{i}"] = np.zeros((layer_dims[i], layer_dims[i-1]))
        prev_updates[f"b{i}"] = np.zeros((layer_dims[i], 1))
    return parameters, prev_updates

def Forward_Propogation(X, parameters, activation_f):
    L = len(parameters) // 2 + 1
    layer_op, pre_activation = [None] * L, [None] * L  # Renamed A to layer_op, Z to pre_activation
    layer_op[0] = X
    for l in range(1, L):
        W, b = parameters[f"W{l}"], parameters[f"b{l}"]
        pre_activation[l] = np.matmul(W, layer_op[l-1]) + b
        if l == L-1:
            layer_op[l] = Softmax.function(pre_activation[l])
        else:
            layer_op[l] = Sigmoid.function(pre_activation[l]) if activation_f == 'sigmoid' else Relu.function(pre_activation[l]) if activation_f == 'relu' else Tanh.function(pre_activation[l])
    return layer_op[-1], layer_op, pre_activation

def Back_Propogation(y_hat, y, layer_op, pre_activation, parameters, activation_f, batch_size, loss, lamb):
    L = len(parameters) // 2
    gradients = {}
    
    gradients[f"dZ{L}"] = (layer_op[L] - y) if loss == 'categorical_crossentropy' else (layer_op[L] - y) * Softmax.derivative(pre_activation[L])
    
    for l in range(L, 0, -1):
        gradients[f"dW{l}"] = (np.dot(gradients[f"dZ{l}"], layer_op[l-1].T) + lamb * parameters[f"W{l}"]) / batch_size
        gradients[f"db{l}"] = np.sum(gradients[f"dZ{l}"], axis=1, keepdims=True) / batch_size
        
        if l > 1:
            activation_derivative = Sigmoid.derivative if activation_f == 'sigmoid' else Relu.derivative if activation_f == 'relu' else Tanh.derivative
            gradients[f"dZ{l-1}"] = np.matmul(parameters[f"W{l}"].T, gradients[f"dZ{l}"]) * activation_derivative(pre_activation[l-1])
    
    return gradients


def update_parameters(parameters, gradients, prev_updates, optimizer, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
    updated_parameters = {}
    L = len(parameters) // 2
    
    if optimizer in ["momentum", "nesterov", "adam", "nadam", "rmsprop"]:
        if "v" not in prev_updates:
            prev_updates["v"] = {f"W{l}": np.zeros_like(parameters[f"W{l}"]) for l in range(1, L+1)}
            prev_updates["v"].update({f"b{l}": np.zeros_like(parameters[f"b{l}"]) for l in range(1, L+1)})
    
    if optimizer in ["adam", "nadam", "rmsprop"]:
        if "s" not in prev_updates:
            prev_updates["s"] = {f"W{l}": np.zeros_like(parameters[f"W{l}"]) for l in range(1, L+1)}
            prev_updates["s"].update({f"b{l}": np.zeros_like(parameters[f"b{l}"]) for l in range(1, L+1)})
    
    for l in range(1, L+1):
        W, b = parameters[f"W{l}"], parameters[f"b{l}"]
        dW, db = gradients[f"dW{l}"], gradients[f"db{l}"]
        
        if optimizer == "sgd":
            W -= lr * dW
            b -= lr * db
        elif optimizer == "momentum":
            prev_updates["v"][f"W{l}"] = beta1 * prev_updates["v"][f"W{l}"] + (1 - beta1) * dW
            W -= lr * prev_updates["v"][f"W{l}"]
        elif optimizer == "adam":
            prev_updates["v"][f"W{l}"] = beta1 * prev_updates["v"][f"W{l}"] + (1 - beta1) * dW
            prev_updates["s"][f"W{l}"] = beta2 * prev_updates["s"][f"W{l}"] + (1 - beta2) * (dW ** 2)
            v_corrected = prev_updates["v"][f"W{l}"] / (1 - beta1 ** t)
            s_corrected = prev_updates["s"][f"W{l}"] / (1 - beta2 ** t)
            W -= lr * v_corrected / (np.sqrt(s_corrected) + epsilon)
        
        updated_parameters[f"W{l}"] = W
        updated_parameters[f"b{l}"] = b
    
    return updated_parameters, prev_updates
