from __future__ import annotations

from dataclasses import dataclass

import torch

SUPPORTED_FILTERS = {}


def register_filter(cls):
    SUPPORTED_FILTERS[cls.__name__] = cls
    return cls


@dataclass(frozen=True)
class Activation:
    layer_type: str
    output_shape: tuple
    value: torch.Tensor


class ActivationFilter:
    """Abstract class for filtering strategies."""

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        raise NotImplementedError

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        """List all available options based on the strategy."""
        raise NotImplementedError


@register_filter
class TypeActivationFilter(ActivationFilter):
    """Filter by type e.g. collect all ReLU activations."""

    def __init__(self, types: list[str]) -> None:
        self.types = types

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        return [activation for activation in activations if activation.layer_type in self.types]

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        return list({activation.layer_type for activation in activations})


@register_filter
class IndexActivationFilter(ActivationFilter):
    """Filter by indices of the activations."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = list(map(int, indices))

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        return [activations[idx] for idx in self.indices]

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        return list(range(len(activations)))


@register_filter
class TargetsActivationFitler(ActivationFilter):
    """Preserve neurons associated with given classes."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = list(map(int, indices))

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        last_activation = activations[-1]  # last layer
        activations = []
        for idx in self.indices:
            # In this case it's just a label of the neuron associated with a given idx
            layer_type = f'target_{idx}'
            value = last_activation.value[:, idx]
            output_shape = value.shape
            activation = Activation(layer_type, output_shape, value)
            activations.append(activation)
        return activations

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        n_classes = activations[-1].output_shape[-1]
        return list(range(n_classes))
