import numpy as np
import wandb
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from helper import class_names, one_hot_encode, preprocess_data
from NNet import Init_Parameters, Forward_Propogation, Back_Propogation

# Load dataset
(X, y), (X_test, y_test) = fashion_mnist.load_data()

# Initialize wandb for logging
wandb.init(project="dl_assignment1", entity="ee17b154tony", name="assignment_1_log_images")

# Log raw sample images
example_indices = [np.where(y == i)[0][0] for i in range(len(class_names))]
example_images = [X[idx] for idx in example_indices]
example_captions = [class_names[y[idx]] for idx in example_indices]
wandb.log({"Raw Sample Images": [wandb.Image(img, caption=cap) for img, cap in zip(example_images, example_captions)]})

# Preprocess data
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y, X_test, y_test)

# Initialize parameters
layer_dims = [X_train.shape[0], 128, 64, 10]  # Example architecture
parameters, prev_updates = Init_Parameters(layer_dims)

# Training, Optimization, and Testing functions (To be implemented)
# train_model(parameters, ...)
# test_model(parameters, ...)
