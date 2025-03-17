# DA6401_A1

## Project Structure




## Overview

This repository contains a custom neural network implementation built from scratch using NumPy. It includes all code used for training, evaluating, and experimenting with different optimizers, activation functions, and network architectures. All experiments are tracked using Weights & Biases (wandb).

## Requirements

- Python 3.7 or later
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [Weights & Biases (wandb)](https://wandb.ai/)
- [TensorFlow](https://www.tensorflow.org/) (for dataset loading via `tensorflow.keras.datasets`)



## Training the Model

The training script (`train.py`) accepts several command-line arguments to allow flexibility in experiments. The following table lists all supported arguments, their default values, and descriptions.

### Command-line Arguments

| Argument                 | Default Value    | Description                                                           |
|--------------------------|------------------|-----------------------------------------------------------------------|
| **-wp, --wandb_project** | myprojectname    | Project name used to track experiments in Weights & Biases dashboard  |
| **-we, --wandb_entity**  | myname           | Wandb Entity used to track experiments in the Weights & Biases dashboard |
| **-d, --dataset**        | fashion_mnist    | Dataset to use. Choices: ["mnist", "fashion_mnist"]                   |
| **-e, --epochs**         | 1                | Number of training epochs                                             |
| **-b, --batch_size**     | 4                | Batch size                                                            |
| **-l, --loss**           | cross_entropy    | Loss function. Choices: ["mean_squared_error", "cross_entropy"]         |
| **-o, --optimizer**      | sgd              | Optimizer. Choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] |
| **-lr, --learning_rate** | 0.1              | Learning rate                                                         |
| **-m, --momentum**       | 0.5              | Momentum used by momentum and NAG optimizers                          |
| **-beta, --beta**        | 0.5              | Beta used by RMSprop optimizer                                        |
| **-beta1, --beta1**      | 0.5              | Beta1 used by Adam and Nadam optimizers                               |
| **-beta2, --beta2**      | 0.5              | Beta2 used by Adam and Nadam optimizers                               |
| **-eps, --epsilon**      | 0.000001         | Epsilon used by optimizers                                             |
| **-wd, --weight_decay**  | 0.0              | Weight decay used by optimizers                                        |
| **-wi, --weight_init**   | random           | Weight initialization method. Choices: ["random", "Xavier"]            |
| **-nhl, --num_layers**   | 1                | Number of hidden layers in the feedforward neural network              |
| **-sz, --hidden_size**   | 4                | Number of neurons in each hidden layer                                 |
| **-a, --activation**     | sigmoid          | Activation function. Choices: ["identity", "sigmoid", "tanh", "ReLU"]    |

### Example Training Command

To train the model using the default hyperparameters and log experiments to wandb, run:

```bash
python train.py --wandb_entity myname --wandb_project myprojectname
