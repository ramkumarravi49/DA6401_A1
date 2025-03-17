import numpy as np
from helper import Sigmoid, Relu, Tanh, Softmax
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def Init_Parameters(layer_dims, init_mode="xavier"):
    np.random.seed(42)
    parameters = {}
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
    layer_op = [None] * L
    pre_activation = [None] * L
    layer_op[0] = X
    for l in range(1, L):
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        pre_activation[l] = np.dot(W, layer_op[l-1]) + b
        if l == L - 1:
            layer_op[l] = Softmax.function(pre_activation[l])
        else:
            if activation_f == 'sigmoid':
                layer_op[l] = Sigmoid.function(pre_activation[l])
            elif activation_f == 'relu':
                layer_op[l] = Relu.function(pre_activation[l])
            elif activation_f == 'tanh':
                layer_op[l] = Tanh.function(pre_activation[l])
    return layer_op[-1], layer_op, pre_activation

def Back_Propogation(y_hat, y, layer_op, pre_activation, parameters, activation_f, batch_size, loss, lamb):
    L = len(parameters) // 2
    gradients = {}
    if loss == 'categorical_crossentropy':
        gradients[f"dZ{L}"] = layer_op[L] - y
    else:
        gradients[f"dZ{L}"] = (layer_op[L] - y) * Softmax.derivative(pre_activation[L])
    
    for l in range(L, 0, -1):
        gradients[f"dW{l}"] = (np.dot(gradients[f"dZ{l}"], layer_op[l-1].T) + lamb * parameters[f"W{l}"]) / batch_size
        gradients[f"db{l}"] = np.sum(gradients[f"dZ{l}"], axis=1, keepdims=True) / batch_size
        if l > 1:
            if activation_f == 'sigmoid':
                activation_deriv = Sigmoid.derivative
            elif activation_f == 'relu':
                activation_deriv = Relu.derivative
            elif activation_f == 'tanh':
                activation_deriv = Tanh.derivative
            gradients[f"dZ{l-1}"] = np.dot(parameters[f"W{l}"].T, gradients[f"dZ{l}"]) * activation_deriv(pre_activation[l-1])
    return gradients

def Loss_Fn(Y, Y_hat, batch_size, loss, lamb, parameters):
    if loss == 'categorical_crossentropy':
        cost = -np.sum(Y * np.log(Y_hat + 1e-8)) / batch_size
    elif loss == 'mse':
        cost = np.sum((Y - Y_hat)**2) / (2 * batch_size)
    
    reg_sum = 0
    L = len(parameters) // 2
    for l in range(1, L+1):
        reg_sum += np.sum(np.square(parameters[f"W{l}"]))
    cost += (lamb / (2 * batch_size)) * reg_sum
    return cost



def NN_predict(X, parameters, activation_f):
    output, _, _ = Forward_Propogation(X, parameters, activation_f)
    predictions = np.argmax(output, axis=0)
    return predictions

def NN_evaluate(X_train, y_train, X_test, y_test, parameters, activation_f):
    train_preds = NN_predict(X_train, parameters, activation_f)
    test_preds = NN_predict(X_test, parameters, activation_f)
    print("Training Accuracy: {:.3f}%".format(accuracy_score(y_train, train_preds) * 100))
    print("Test Accuracy: {:.3f}%".format(accuracy_score(y_test, test_preds) * 100))
    print("Classification Report:\n", classification_report(y_test, test_preds))
    return train_preds, test_preds
