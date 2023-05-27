from __future__ import annotations

from typing import List, Dict

import torch
import torch.nn as nn


class ModelWithActivations(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        activation_filters: Dict[str, List[ActivationFilter]] | None = None,
        example_input: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self._activations = []
        self._register_activation_hook()
        if activation_filters is None:
            activation_filters = {"all": []}
        self.activation_filters = activation_filters
        if example_input is not None:
            self(example_input)  # recon pass

    @property
    def activations(self) -> list[torch.Tensor]:
        """Return activations based on the activation_filters."""
        filtered_activations = {}
        for name, filters in self.activation_filters.items():
            if not filters:
                filtered_activations[name] = self._activations
            activations = self._activations
            for activation_filter in filters:
                activations = activation_filter.filter_activations(activations)
            filtered_activations[name] = activations
        return filtered_activations

    @property
    def activations_values(self):
        """Return values of the filtered activations."""
        activations = self.activations
        activations_values = {}
        for name, activations in activations.items():
            activations_values[name] = [activation.value for activation in activations]
        return activations_values

    def _register_activation_hook(self):
        def activation_hook(module, input_, output):
            layer_type = module.__class__.__name__
            output_shape = output.shape
            value = output
            activation = Activation(layer_type, output_shape, value)
            self._activations.append(activation)
        for layer in flatten_modules(self.model):
            layer.register_forward_hook(activation_hook)

    def forward(self, input_):
        self._activations.clear()
        return self.model.forward(input_)


def flatten_modules(module):
    flattened_modules = []
    for child in module.children():
        if isinstance(child, nn.Sequential):
            flattened_modules.extend(flatten_modules(child))
        else:
            flattened_modules.append(child)
    return flattened_modules