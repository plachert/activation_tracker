# Activation tracker
A simple `torch.nn.Module` that registers all the activations during the forward pass and enables to filter them based on a given strategy.

##  Description
While experimenting with Deep Dream and Neural Style Transfer I thought that it would be nice to have easy access to all the activations of the model and some easy way to filter them based on some strategies e.g. take all outputs of convolutional layers or all ReLUs.
The code is built around three classes:
- `Activation` - just a container for data related to the activation. It stores `layer_type`, `output_shape` and `values`.
- `ActivationFilter` - its subclasses determine strategies of filtering. Currently 3 strategies are implemented:
    - `TypeActivationFilter` - filter by type of the layers, e.g. take all ReLUs
    - `IndexActivationFilter` - simply select the activations by their indices
    - `TargetsActivationFilter` - this one is a little different. It's for classifiers only. It lets you select neurons associated with given classes e.g. the class of a dog.
- `ModelWithActivations` - `torch.nn.Module` that tracks the activations of a given model. It uses the filters to return the activations of interest.

## Getting Started

### Installing
Run the following command in your virtual env.

```shell
(venv) foo@bar:~$ pip install git+https://github.com/plachert/activation_tracker
```

Verify installation:
```shell
(venv) foo@bar:~$ python
>>> import activation_tracker
>>> activation_tracker.__version__
'1.0.0`
>>>
```

### Usage Examples
```python
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
```


## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/plachert/activation_tracker/blob/main/LICENSE)
