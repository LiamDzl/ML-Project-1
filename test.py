import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import labels_to_tensor
from neural_network import model, run_mnist_model

# Import Data

training_data = pd.read_csv('archive-2/mnist_train.csv')
training_data = np.array(training_data)
training_data = torch.FloatTensor(training_data)
training_labels = training_data[: , 0] # Save labels
training_data = training_data[: , 1:] # Remove labels from main tensor
training_inputs = training_data / 255 # TENSOR 60000 x 784
training_outputs = labels_to_tensor(training_labels) # 60000 x 10

test_data = pd.read_csv('archive-2/mnist_test.csv')
test_data = np.array(test_data)
test_data = torch.FloatTensor(test_data)
test_labels = test_data[: , 0] # Save labels
test_data = test_data[: , 1:] # Remove labels from main tensor
test_inputs = test_data / 255 
test_inputs.reshape(10000, 784) # TENSOR 10000 x 784
test_outputs = labels_to_tensor(test_labels) # 10000 x 10

nn1 = model(structure = [784, 100, 100, 10], output_function = "softmax")

saved_model = run_mnist_model(
    model=nn1,
    training_inputs=training_inputs,
    test_inputs=test_inputs,
    training_outputs=training_outputs,
    test_outputs=test_outputs,
    test_labels=test_labels,
    epochs=30,
    nabla=2,
    loss="MSE",
    opt="SGD",
    batch_size=100,
    info_at=100,
    plotting="on")
