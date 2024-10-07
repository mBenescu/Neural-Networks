from typing import List
import torch
import torch.nn as nn


def get_mlp(num_input_first_layer: int, num_hidden_layers: int, num_neurons: List[int]) -> torch.nn.Sequential:
    # Check that num_neurons has the correct length
    assert len(num_neurons) == num_hidden_layers + 2, "Length of num_neurons must match the total number of layers"

    model = torch.nn.Sequential()

    model.add_module("linear_0", nn.Linear(num_input_first_layer, num_neurons[0]))
    model.add_module("tanh_0", nn.Tanh())

    for i in range(1, num_hidden_layers):
        model.add_module(f"linear_{i}", nn.Linear(num_neurons[i - 1], num_neurons[i]))
        model.add_module(f"tanh_{i}", nn.Tanh())

    model.add_module("linear_last", nn.Linear(num_neurons[-1], 1))

    return model
