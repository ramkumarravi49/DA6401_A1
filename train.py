import numpy as np
import wandb
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from helper import class_names, one_hot_encode
from model import (Init_Parameters, Forward_Propogation, Back_Propogation, 
                   NN_predict, Loss_Fn, NN_evaluate)
from helper import update_parameters
from helper import log_epoch_metrics
from helper import log_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize wandb for logging
#wandb.init(project="DL_A1", entity="cs24m037-iit-madras", name="A1_Q7")

##############################################
# Utility Functions
##############################################

def Train_minibatch(X_batch, Y_batch, parameters, prev_updates, batch_size, 
                      A_function, loss, L2_lamb, optimizer, t, beta, l_r):
    """
    Process a single mini-batch: forward propagation, compute gradients, and update parameters.
    For optimizers using lookahead (nesterov/nadam), compute lookahead parameters.
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
                                                          lr=l_r, beta1=0.9, beta2=0.999, epsilon=1e-8)
            t += 1
    else:
        output, layer_op, pre_act = Forward_Propogation(X_batch, parameters, A_function)
        gradients = Back_Propogation(output, Y_batch, layer_op, pre_act, 
                                     parameters, A_function, batch_size, loss, L2_lamb)
        parameters, prev_updates = update_parameters(optimizer, parameters, gradients, prev_updates, t, 
                                                       lr=l_r, beta1=0.9, beta2=0.999, epsilon=1e-8)
        if optimizer == "adam":
            t += 1

    return parameters, prev_updates, t

def Run_epoch(X_train, y_train_one_hot, parameters, prev_updates, batch_size, 
                  A_function, loss, L2_lamb, optimizer, t, beta, l_r):
    """
    Process an entire epoch by looping through all mini-batches.
    """
    M = X_train.shape[1]
    for i in range(0, M, batch_size):
        current_batch_size = batch_size if i + batch_size <= M else M - i
        X_batch = X_train[:, i:i+current_batch_size]
        Y_batch = y_train_one_hot[:, i:i+current_batch_size]
        parameters, prev_updates, t = Train_minibatch ( X_batch, Y_batch, parameters, prev_updates, current_batch_size, 
                                                        A_function, loss, L2_lamb, optimizer, t, beta, l_r )
    return parameters, prev_updates, t



##############################################
# Main Training Function
##############################################

def NN_Train():
    """
    Trains the neural network using modularized mini-batch and epoch processing.
    """
    # Define hyperparameters (customize as needed)
    params_config = {
        'epochs': 5,
        'batch_size': 64,
        'l_r': 0.0005,
        'A_function': 'relu',
        'optimizer': 'rmsprop',  # Options: "sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"
        'Wt_init': 'xavier',   # Options: "xavier", "random_normal", "random_uniform"
        'L2_lamb': 0,
        'num_neurons': 256,
        'num_hidden': 3,
        'loss_function': 'categorical_crossentropy'
    }
    
    epochs        = params_config['epochs']
    batch_size    = params_config['batch_size']
    l_r = params_config['l_r']
    A_function  = params_config['A_function']
    optimizer     = params_config['optimizer']
    Wt_init     = params_config['Wt_init']
    L2_lamb       = params_config['L2_lamb']
    num_neurons   = params_config['num_neurons']
    num_hidden    = params_config['num_hidden']
    loss          = params_config['loss_function']
    beta          = 0.9  # Momentum parameter for lookahead optimizers
    
    run_name = "lr_{}_ac_{}_in_{}_op_{}_bs_{}_L2_{}_ep_{}_nn_{}_nh_{}".format(l_r, A_function, Wt_init, optimizer, batch_size, L2_lamb, epochs, num_neurons, num_hidden)
    print(run_name)
    
    # Access global preprocessed data
    global X_train, X_val, y_train, y_val, y_train_one_hot, y_val_one_hot, num_features, num_classes
    M = X_train.shape[1]
    Mval = X_val.shape[1]
    
    # Define network architecture: input layer, hidden layers, output layer
    layer_dims = [num_features] + [num_neurons] * num_hidden + [num_classes]
    parameters, prev_updates = Init_Parameters(layer_dims, Wt_init)
    
    t = 1  # Timestep for Adam/Nadam
    epoch_costs = []
    val_costs   = []
    
    # Main training loop
    for epoch in range(1, epochs + 1):
        parameters, prev_updates, t = Run_epoch( X_train, y_train_one_hot, parameters, prev_updates,
                                                batch_size, A_function, loss, L2_lamb, optimizer, t, beta, l_r)
        
        # Compute full training cost
        train_output, _, _ = Forward_Propogation(X_train, parameters, A_function)
        train_loss = Loss_Fn(y_train_one_hot, train_output, X_train.shape[1], loss, L2_lamb, parameters)
        epoch_costs.append(train_loss)
        
        # Compute validation cost
        val_output, _, _ = Forward_Propogation(X_val, parameters, A_function)
        val_loss = Loss_Fn(y_val_one_hot, val_output, X_val.shape[1], loss, L2_lamb, parameters)
        val_costs.append(val_loss)
        
        # Compute accuracies
        train_preds = NN_predict(X_train, parameters, A_function)
        val_preds = NN_predict(X_val, parameters, A_function)
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        
        #log_epoch_metrics(epoch, train_loss, val_loss, train_acc, val_acc)
        print(f"Epoch {epoch}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, "
              f"Training Acc = {train_acc:.4f}, Validation Acc = {val_acc:.4f}")
    
    return parameters, epoch_costs, A_function

##############################################
# Main Execution
##############################################

if __name__ == '__main__':
    # Load dataset
    (X, y), (X_test, y_test) = fashion_mnist.load_data()
    
    # (Optional) Log raw sample images to wandb
    example_indices = [np.where(y == i)[0][0] for i in range(len(class_names))]
    example_images = [X[idx] for idx in example_indices]
    example_captions = [class_names[y[idx]] for idx in example_indices]
    # wandb.log({"Raw Sample Images": [wandb.Image(img, caption=cap) for img, cap in zip(example_images, example_captions)]})
    
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
    
    # One-hot encode labels for training, validation, and test sets
    y_train_one_hot = np.zeros((num_classes, M))
    y_train_one_hot[y_train, np.arange(M)] = 1
    y_val_one_hot = np.zeros((num_classes, Mval))
    y_val_one_hot[y_val, np.arange(Mval)] = 1
    y_test_one_hot = np.zeros((num_classes, Mtest))
    y_test_one_hot[y_test, np.arange(Mtest)] = 1
    
    # Transpose data so that each column is a sample
    X_train, X_val, X_test = X_train.T, X_val.T, X_test.T

    # Train the model using the modularized training loop
    parameters, epoch_costs, A_function = NN_Train()
    
    # Evaluate the model on test data (NN_evaluate prints accuracy and classification report)
    print("\nEvaluating on test data:")
    train_pred, test_pred = NN_evaluate(X_train, y_train, X_test, y_test, parameters, A_function)
    
    # Log confusion matrices using the custom utility function
    #log_confusion_matrix(y_train, train_pred, "Train Confusion Matrix", class_names)
    #log_confusion_matrix(y_test, test_pred, "Test Confusion Matrix", class_names)
