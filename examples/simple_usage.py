from __future__ import annotations

import torch
from activation_tracker.activation import IndexActivationFilter
from activation_tracker.activation import TypeActivationFilter
from activation_tracker.model import ModelWithActivations


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(5, 3)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(3, 1)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return x


if __name__ == '__main__':
    input_ = torch.rand(1, 10)
    model = SimpleModel()
    activation_filters = {
        'only_tanh': [TypeActivationFilter(['Tanh'])],
        'second_linear': [
            TypeActivationFilter(['Linear']),
            IndexActivationFilter([1]),
        ],
        'all_relus': [TypeActivationFilter(['ReLU'])],
    }
    model_with_activations = ModelWithActivations(
        model=model,
        activation_filters=activation_filters,
        example_input=None,
    )
    model(input_)  # we have to call forward pass to register the activations
    print(model_with_activations.activations_values)
