# DA6401_A1

## Overview

This repository contains a custom neural network implementation built from scratch using NumPy. It includes all code used for training, evaluating, and experimenting with different optimizers, activation functions, and network architectures. All experiments are tracked using Weights & Biases (wandb).

### Link to Assignmnet Report
 - https://wandb.ai/cs24m037-iit-madras/DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTY4NzU2OQ
 - 
## Project Structure
```bash
DA6401_Assignment1/
├── A1_old/              # code before adding argparse 
├── helper.py            # Activation,Optimizer and helper fn
├── model.py             # Forward Pass , Backward Pass, Loss Fn
├── train.py             # Training script(main entry point)
├── A1_Q1.ipynb          # To run sweeps
├── A1_Q1_MSE.ipynb      # To run sweeps
├── q1.py                # the first file created , just don't wanna delete it ! :)
|── README.md            # Setup and usage instructions
```

## Dataset
The Fashion-MNIST dataset consists of 70,000 grayscale images of 28x28 pixels, divided into 10 classes representing various fashion items (e.g., T-shirts, trousers, etc.). The dataset is split into training and testing set.

## File Summaries

- **helper.py**  
  - Implements utility functions such as one-hot encoding and a collection of activation functions (Sigmoid, ReLU, Tanh, Softmax) with their derivatives.
  - Contains various optimizer implementations (SGD, Momentum, Nesterov, RMSprop, Adam/Nadam) for updating network parameters.
  - Provides logging utilities integrated with Weights & Biases (wandb) for tracking training metrics and generating visualizations.

- **model.py**  
  - Initializes network parameters using different methods (e.g., Xavier initialization).
  - Implements forward propagation, applying activation functions through the network layers, and uses Softmax for the output layer.
  - Handles backward propagation to compute gradients, calculates the loss (categorical crossentropy or MSE with L2 regularization), and provides functions for making predictions and evaluating model performance.

- **train.py**  
  - Loads and preprocesses the dataset (either Fashion MNIST or MNIST), including reshaping, normalizing, and splitting the data into training, validation, and test sets.
  - Contains functions for managing mini-batch training and running the overall training loop over multiple epochs.
  - Parses command-line arguments to customize hyperparameters, trains the model, and finally evaluates it on the test data while logging metrics.




## Requirements

- Python 3.7 or later
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [Weights & Biases (wandb)](https://wandb.ai/)
- [TensorFlow](https://www.tensorflow.org/) (for dataset loading via `tensorflow.keras.datasets`)

```bash
pip install numpy wandb keras tensorflow pandas seaborn
```

## Training the Model

The training script (`train.py`) accepts several command-line arguments as instructed in "** Code Spesifications**" section. The following table lists all supported arguments, their default values (as set in the code) are the best parameters obtained in the sweeps.

### Command-line Arguments

| Argument                  | Default Value              | Description                                                                      |
|---------------------------|----------------------------|----------------------------------------------------------------------------------|
| **-wp, --wandb_project**  | DL_A1                      | Project name used to track experiments in Weights & Biases dashboard             |
| **-we, --wandb_entity**   | cs24m037-iit-madras        | Wandb Entity used to track experiments in the Weights & Biases dashboard         |
| **-d, --dataset**         | fashion_mnist              | Dataset to use. Choices: ["mnist", "fashion_mnist"]                              |
| **-e, --epochs**          | 5                          | Number of training epochs                                                        |
| **-b, --batch_size**      | 32                         | Batch size                                                                       |
| **-l, --loss**            | categorical_crossentropy   | Loss function. Choices: ["categorical_crossentropy", "mse"]                      |
| **-o, --optimizer**       | rmsprop                    | Optimizer. Choices: ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]    |
| **-lr, --learning_rate**  | 0.0005                     | Learning rate                                                                    |
| **-m, --momentum**        | 0.9                        | Momentum used by momentum and Nesterov optimizers                                |
| **-beta, --beta**         | 0.999                      | Beta used by RMSprop optimizer                                                   |
| **-beta1, --beta1**       | 0.9                        | Beta1 used by Adam and Nadam optimizers                                          |
| **-beta2, --beta2**       | 0.999                      | Beta2 used by Adam and Nadam optimizers                                          |
| **-eps, --epsilon**       | 1e-8                       | Epsilon used by optimizers                                                       |
| **-w_d, --weight_decay**  | 0.0                        | Weight decay used by optimizers                                                  |
| **-w_i, --weight_init**   | xavier                     | Weight initialization method. Choices: ["xavier", "random_normal", "random_uniform"]|
| **-nhl, --num_layers**    | 3                          | Number of hidden layers in the feedforward neural network                        |
| **-sz, --hidden_size**    | 256                        | Number of neurons in each hidden layer                                           |
| **-a, --activation**      | relu                       | Activation function. Choices: ["identity", "sigmoid", "tanh", "relu"]            |
| **--l2_lamb**             | 0.0005                     | L2 regularization lambda                                                         |

### Example Training Command

To train the model using the default hyperparameters and log experiments to wandb, run:

```bash
python train.py --wandb_entity cs24m037-iit-madras --wandb_project DL_A1
