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

1. The following example demonstrates how one can combine filters to get the activations of interest. Also, as we can pass more than one combination of filters and group them in a dictionary. I found this very helpful when experimenting with Neural Style Transfer. You can now easily pass a set of activations for `content` and `style`.
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
2. You can also use the tool to easily visualise feature maps.
```python
from activation_tracker.activation import TypeActivationFilter
from activation_tracker.model import ModelWithActivations

model = # e.g. vgg16
input_image = # proper input e.g. (1, 3, 224, 224)
filters = {"feat_maps": TypeActivationFilter(["Conv2d"])}
model_with_activations = ModelWithActivations(
        model=model,
        activation_filters=activation_filters,
        example_input=None,
    )
model(input_image)
feat_maps = model_with_activations.activations_values # list of feature maps for each conv2 layer
# display the maps

```

3. Developing your own filtering strategies, e.g.
```python
from activation_tracker.activation import ActivationFilter


class StrongestActivationFilter(ActivationFilter):
    def __init__(self, percentile: float):
        self.percentile = percentile

    def filter_activations(self, activations):
        """Return the strongest neurons in each layer"""

    @staticmethod
    def list_all_available_parameters(activations):
        """Optional, explained later"""
```

### `list_all_available_parameters`
This method is optional. Its purpose is to return all the values that we can filter in terms of the selected strategy, e.g. calling this method from TypeActivationFilter on some list of activations will return all the types of these activations.
## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/plachert/activation_tracker/blob/main/LICENSE)
