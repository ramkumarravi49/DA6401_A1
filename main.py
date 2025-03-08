import numpy as np
import wandb
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from helper import class_names, one_hot_encode
from model import Init_Parameters, Forward_Propogation, Back_Propogation, NN_predict, compute_multiclass_loss, NN_evaluate
from helper import update_parameters
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize wandb for logging
wandb.init(project="DL_A1", entity="cs24m037-iit-madras", name="A1_Q7_CE_2")

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

def NN_fit():
    """
    Trains the neural network using hardcoded hyperparameters.
    """
    # # Cross Entropy Config One
    # params_config = {
    #     'epochs': 5,
    #     'batch_size': 32,
    #     'learning_rate': 1e-3,
    #     'activation_f': 'tanh',
    #     'optimizer': 'nadam',  # Options: "sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"
    #     'init_mode': 'xavier',  # Options: "xavier", "random_normal", "random_uniform"
    #     'L2_lamb': 0.0005,
    #     'num_neurons': 64,
    #     'num_hidden': 4,
    #     'loss_function': 'categorical_crossentropy'
    # }

    # Cross Entropy Config two
    params_config = {
        'epochs': 5,
        'batch_size': 64,
        'learning_rate': 0.0005,
        'activation_f': 'relu',
        'optimizer': 'rmsprop',  # Options: "sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"
        'init_mode': 'xavier',  # Options: "xavier", "random_normal", "random_uniform"
        'L2_lamb': 0,
        'num_neurons': 256,
        'num_hidden': 3,
        'loss_function': 'categorical_crossentropy'
    }
    
    epochs        = params_config['epochs']
    batch_size    = params_config['batch_size']
    learning_rate = params_config['learning_rate']
    activation_f  = params_config['activation_f']
    optimizer     = params_config['optimizer']
    init_mode     = params_config['init_mode']
    L2_lamb       = params_config['L2_lamb']
    num_neurons   = params_config['num_neurons']
    num_hidden    = params_config['num_hidden']
    loss          = params_config['loss_function']
    
    run_name = "lr_{}_ac_{}_in_{}_op_{}_bs_{}_L2_{}_ep_{}_nn_{}_nh_{}".format(
        learning_rate, activation_f, init_mode, optimizer, batch_size, L2_lamb, epochs, num_neurons, num_hidden)
    print(run_name)
    
    # Access global preprocessed data
    global X_train, X_val, y_train, y_val, y_train_one_hot, y_val_one_hot, num_features, num_classes
    M = X_train.shape[1]
    Mval = X_val.shape[1]
    
    # Define network architecture: input layer, hidden layers, output layer
    layer_dims = [num_features] + [num_neurons] * num_hidden + [num_classes]
    
    # Initialize parameters and previous updates (for momentum-based optimizers)
    parameters, prev_updates = Init_Parameters(layer_dims, init_mode)
    
    t = 1  # Timestep for Adam/Nadam
    # For lookahead (used in Nesterov and Nadam)
    params_look_ahead = parameters.copy()
    
    epoch_costs = []
    val_costs   = []
    beta = 0.9  # Momentum parameter
    
    for epoch in range(epochs):
        for i in range(0, M, batch_size):
            current_batch_size = batch_size if i + batch_size <= M else M - i
            X_batch = X_train[:, i:i+current_batch_size]
            Y_batch = y_train_one_hot[:, i:i+current_batch_size]
            
            if optimizer == "nesterov" or optimizer == "nadam":
                L_layers = len(parameters) // 2
                for l in range(1, L_layers+1):
                    if "v" not in prev_updates:
                        prev_updates["v"] = {f"W{l}": np.zeros_like(parameters[f"W{l}"]) for l in range(1, L_layers+1)}
                        prev_updates["v"].update({f"b{l}": np.zeros_like(parameters[f"b{l}"]) for l in range(1, L_layers+1)})
                    params_look_ahead[f"W{l}"] = parameters[f"W{l}"] - beta * prev_updates["v"][f"W{l}"]
                    params_look_ahead[f"b{l}"] = parameters[f"b{l}"] - beta * prev_updates["v"][f"b{l}"]
                
                output, layer_op, pre_act = Forward_Propogation(X_batch, params_look_ahead, activation_f)
                gradients = Back_Propogation(output, Y_batch, layer_op, pre_act, params_look_ahead, activation_f, current_batch_size, loss, L2_lamb)
                
                if optimizer == "nesterov":
                    parameters, prev_updates = update_parameters("nesterov", parameters, gradients, prev_updates, lr=learning_rate, beta1=beta)
                elif optimizer == "nadam":
                    parameters, prev_updates = update_parameters("nadam", parameters, gradients, prev_updates, t, lr=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
                    t += 1
            else:
                output, layer_op, pre_act = Forward_Propogation(X_batch, parameters, activation_f)
                gradients = Back_Propogation(output, Y_batch, layer_op, pre_act, parameters, activation_f, current_batch_size, loss, L2_lamb)
                parameters, prev_updates = update_parameters(optimizer, parameters, gradients, prev_updates, t, lr=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
                if optimizer == "adam":
                    t += 1
        
        # Compute full training cost
        train_output, _, _ = Forward_Propogation(X_train, parameters, activation_f)
        epoch_cost = compute_multiclass_loss(y_train_one_hot, train_output, M, loss, L2_lamb, parameters)
        epoch_costs.append(epoch_cost)
        
        # Compute validation cost
        val_output, _, _ = Forward_Propogation(X_val, parameters, activation_f)
        val_cost = compute_multiclass_loss(y_val_one_hot, val_output, Mval, loss, L2_lamb, parameters)
        val_costs.append(val_cost)
        
        # Compute accuracies
        train_preds = NN_predict(X_train, parameters, activation_f)
        val_preds = NN_predict(X_val, parameters, activation_f)
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        
        print(f"Epoch {epoch+1}: Training Loss = {epoch_cost:.4f}, Validation Loss = {val_cost:.4f}, Training Acc = {train_acc:.4f}, Validation Acc = {val_acc:.4f}")
    
    # Optionally, you could plot cost curves here
    # plot_cost_curve(epoch_costs, val_costs)
    
    return parameters, epoch_costs, activation_f

if __name__ == '__main__':
    # Train the model
    parameters, epoch_costs, activation_f = NN_fit()
    
    # Evaluate the model on test data
    print("\nEvaluating on test data:")
    train_pred, test_pred = NN_evaluate(X_train, y_train, X_test, y_test, parameters, activation_f)
    
    # Plot and log confusion matrix for Training Data using seaborn and wandb
    cnf_matrix_train = confusion_matrix(y_train, train_pred, normalize='true')
    fig_train, ax_train = plt.subplots(figsize=(12, 8))
    sns.heatmap(cnf_matrix_train, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax_train)
    ax_train.set_title("Confusion Matrix (Training set)", size=16)
    ax_train.set_xlabel("Predicted Class", size=14)
    ax_train.set_ylabel("True Class", size=14)
    # Log training confusion matrix to wandb
    wandb.log({"Train Confusion Matrix": wandb.Image(fig_train)})
    plt.show()
    
    # Plot and log confusion matrix for Test Data using seaborn and wandb
    cnf_matrix_test = confusion_matrix(y_test, test_pred, normalize='true')
    fig_test, ax_test = plt.subplots(figsize=(12, 8))
    sns.heatmap(cnf_matrix_test, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax_test)
    ax_test.set_title("Confusion Matrix (Test set)", size=16)
    ax_test.set_xlabel("Predicted Class", size=14)
    ax_test.set_ylabel("True Class", size=14)
    # Log test confusion matrix to wandb
    wandb.log({"Test Confusion Matrix": wandb.Image(fig_test)})
    plt.show()
