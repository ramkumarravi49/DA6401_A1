import numpy as np
import wandb
import argparse
from tensorflow.keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split
from helper import class_names, one_hot_encode, update_parameters, log_epoch_metrics, log_confusion_matrix
from model import (Init_Parameters, Forward_Propogation, Back_Propogation, 
                   NN_predict, Loss_Fn, NN_evaluate)
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

##############################################
# Utility Functions
##############################################

def Train_minibatch(X_batch, Y_batch, parameters, prev_updates, batch_size, 
                      A_function, loss, L2_lamb, optimizer, t, beta, l_r, args):
    """
    Process a single mini-batch: 
    """
    # Use lookahead for Nesterov/Nadam optimizers
    if optimizer in ['nesterov', 'nadam']:
        L_layers = len(parameters) // 2
        look_ahead_param = {}
        # Initialize velocity if not present
        if "v" not in prev_updates:
            prev_updates["v"] = {f"W{l}": np.zeros_like(parameters[f"W{l}"]) for l in range(1, L_layers+1)}
            prev_updates["v"].update({f"b{l}": np.zeros_like(parameters[f"b{l}"]) for l in range(1, L_layers+1)})
        for l in range(1, L_layers+1):
            look_ahead_param[f"W{l}"] = parameters[f"W{l}"] - beta * prev_updates["v"][f"W{l}"]
            look_ahead_param[f"b{l}"] = parameters[f"b{l}"] - beta * prev_updates["v"][f"b{l}"]
        
        output, layer_op, pre_act = Forward_Propogation(X_batch, look_ahead_param, A_function)
        gradients = Back_Propogation(output, Y_batch, layer_op, pre_act,  
                                     look_ahead_param, A_function, batch_size, loss, L2_lamb)
        if optimizer == "nesterov":
            parameters, prev_updates = update_parameters("nesterov", parameters, gradients, prev_updates, lr=l_r, beta1=beta)
        elif optimizer == "nadam":
            parameters, prev_updates = update_parameters("nadam", parameters, gradients, prev_updates, t, 
                                                          lr=l_r, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
            t += 1
    else:
        output, layer_op, pre_act = Forward_Propogation(X_batch, parameters, A_function)
        gradients = Back_Propogation(output, Y_batch, layer_op, pre_act, 
                                     parameters, A_function, batch_size, loss, L2_lamb)
        parameters, prev_updates = update_parameters(optimizer, parameters, gradients, prev_updates, t, 
                                                       lr=l_r, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
        if optimizer == "adam":
            t += 1

    return parameters, prev_updates, t

def Run_epoch(X_train, y_train_one_hot, parameters, prev_updates, batch_size, 
              A_function, loss, L2_lamb, optimizer, t, beta, l_r, args):
    """
    Process an entire epoch by looping through all mini-batches.
    """
    M = X_train.shape[1]
    for i in range(0, M, batch_size):
        current_batch_size = batch_size if i + batch_size <= M else M - i
        X_batch = X_train[:, i:i+current_batch_size]
        Y_batch = y_train_one_hot[:, i:i+current_batch_size]
        parameters, prev_updates, t = Train_minibatch(X_batch, Y_batch, parameters, prev_updates,
                                                      current_batch_size, A_function, loss, L2_lamb,
                                                      optimizer, t, beta, l_r, args)
    return parameters, prev_updates, t

##############################################
# Main Training Function
##############################################

def NN_Train(args):
    """
    # Trains the neural network using hyperparameters provided via args.
    # """
    # # Set hyperparameters from parsed arguments (defaults match hardcoded values)
    # params_config = {
    #     'epochs': args.epochs,                    
    #     'batch_size': args.batch_size,            
    #     'l_r': args.learning_rate,                
    #     'A_function': args.activation,            
    #     'optimizer': args.optimizer,              
    #     'Wt_init': args.weight_init,              
    #     'L2_lamb': args.l2_lamb,                  
    #     'num_neurons': args.hidden_size,          
    #     'num_hidden': args.num_layers,            
    #     'loss_function': args.loss                
    # }
    
    epochs        = args.epochs
    batch_size    = args.batch_size
    l_r           = args.learning_rate
    A_function    = args.activation
    optimizer     = args.optimizer
    Wt_init       = args.weight_init
    L2_lamb       = args.l2_lamb
    num_neurons   = args.hidden_size
    num_hidden    = args.num_layers
    loss          = args.loss
    beta          = args.momentum  # momentum parameter for lookahead optimizers

    run_name = "lr_{}_ac_{}_in_{}_op_{}_bs_{}_L2_{}_ep_{}_nn_{}_nh_{}".format(
                    l_r, A_function, Wt_init, optimizer, batch_size, L2_lamb, epochs, num_neurons, num_hidden)
    print("Run Name:", run_name)
    
    # Access global preprocessed data
    global X_train, X_val, y_train, y_val, y_train_one_hot, y_val_one_hot, num_features, num_classes
    M = X_train.shape[1]
    
    
    layer_dims = [num_features] + [num_neurons] * num_hidden + [num_classes]
    parameters, prev_updates = Init_Parameters(layer_dims, Wt_init)
    
    t = 1  # Timestep for Adam/Nadam
    epoch_costs = []
    
    # Main training loop
    for epoch in range(1, epochs + 1):
        parameters, prev_updates, t = Run_epoch(X_train, y_train_one_hot, parameters, prev_updates,
                                                batch_size, A_function, loss, L2_lamb, optimizer, t, beta, l_r, args)
        
        # Compute full training cost
        train_output, _, _ = Forward_Propogation(X_train, parameters, A_function)
        train_loss = Loss_Fn(y_train_one_hot, train_output, X_train.shape[1], loss, L2_lamb, parameters)
        epoch_costs.append(train_loss)
        
        # Compute validation cost
        val_output, _, _ = Forward_Propogation(X_val, parameters, A_function)
        val_loss = Loss_Fn(y_val_one_hot, val_output, X_val.shape[1], loss, L2_lamb, parameters)
        
        # Compute accuracies
        train_preds = NN_predict(X_train, parameters, A_function)
        val_preds = NN_predict(X_val, parameters, A_function)
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        
        log_epoch_metrics(epoch, train_loss, val_loss, train_acc, val_acc)
        print(f"Epoch {epoch}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, "
              f"Training Acc = {train_acc:.4f}, Validation Acc = {val_acc:.4f}")
    
    return parameters, epoch_costs, A_function

##############################################
# Main Execution with Argument Parsing
##############################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a neural network on MNIST or Fashion MNIST dataset.")
    
    parser.add_argument('-wp', '--wandb_project', type=str, default="DL_A1",
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default="cs24m037-iit-madras",
                        help='Wandb entity used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-d', '--dataset', type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"],
                        help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size used to train neural network')
    parser.add_argument('-l', '--loss', type=str, default="categorical_crossentropy",
                        choices=["categorical_crossentropy", "mse"],
                        help='Loss function to use')
    parser.add_argument('-o', '--optimizer', type=str, default="rmsprop",
                        choices=["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"],
                        help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='Momentum used by momentum and Nesterov optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.999,
                        help='Beta used by RMSprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9,
                        help='Beta1 used by Adam and Nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999,
                        help='Beta2 used by Adam and Nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-8,
                        help='Epsilon used by optimizers')
    # parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
    #                     help='Weight decay used by optimizers')
    parser.add_argument('-w_i', '--weight_init', type=str, default="xavier",
                        choices=["xavier", "random_normal", "random_uniform"],
                        help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                        help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=256,
                        help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', type=str, default="relu",
                        choices=["identity", "sigmoid", "tanh", "relu"],
                        help='Activation function')
    # Additional parameter used in training (L2 regularization lambda)
    parser.add_argument('--l2_lamb', type=float, default=0,
                        help='L2 regularization lambda')
    
    args = parser.parse_args()
    
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    
    # Load dataset based on args.dataset
    if args.dataset == "fashion_mnist":
        (X, y), (X_test, y_test) = fashion_mnist.load_data()
    else:
        (X, y), (X_test, y_test) = mnist.load_data()
    
    # Reshape and normalize data
    X = X.reshape(X.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Dataset statistics
    M, num_features = X_train.shape
    Mval = X_val.shape[0]
    Mtest = X_test.shape[0]
    num_classes = len(np.unique(y_train))
    
    # One-hot encode labels 
    y_train_one_hot = np.zeros((num_classes, M))
    y_train_one_hot[y_train, np.arange(M)] = 1
    y_val_one_hot = np.zeros((num_classes, Mval))
    y_val_one_hot[y_val, np.arange(Mval)] = 1
    y_test_one_hot = np.zeros((num_classes, Mtest))
    y_test_one_hot[y_test, np.arange(Mtest)] = 1
    
    # Transpose data 
    X_train, X_val, X_test = X_train.T, X_val.T, X_test.T

    
    globals().update({
        "X_train": X_train, "X_val": X_val,
        "y_train": y_train, "y_val": y_val,
        "y_train_one_hot": y_train_one_hot, "y_val_one_hot": y_val_one_hot,
        "num_features": num_features, "num_classes": num_classes
    })
    
    
    parameters, epoch_costs, A_function = NN_Train(args)
    
    # Evaluate the model 
    print("\nEvaluating on test data:")
    train_pred, test_pred = NN_evaluate(X_train, y_train, X_test, y_test, parameters, A_function)
    
    # Optionally, log confusion matrices
    # log_confusion_matrix(y_train, train_pred, "Train Confusion Matrix", class_names)
    # log_confusion_matrix(y_test, test_pred, "Test Confusion Matrix", class_names)
