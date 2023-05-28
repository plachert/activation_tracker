"""
This module provides strategies of filtering activations.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

SUPPORTED_FILTERS = {}


def register_filter(cls):
    SUPPORTED_FILTERS[cls.__name__] = cls
    return cls


@dataclass(frozen=True)
class Activation:
    """
    Represents an activation.

    Attributes:
        layer_type (str): The type of layer associated with the activation.
        output_shape (tuple): The shape of the output.
        value (torch.Tensor): The value of the activation.
    """

    layer_type: str
    output_shape: tuple
    value: torch.Tensor


class ActivationFilter:
    """Abstract class for filtering strategies."""

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        """
        Filter the activations.

        Args:
            activations (list[Activation]): List of activations.

        Returns:
            list[Activation]: Filtered list of activations.
        """
        raise NotImplementedError

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        """
        List all available parameters based on the strategy.

        Args:
            activations (list[Activation]): List of activations.

        Returns:
            list: List of available parameters.
        """
        raise NotImplementedError


@register_filter
class TypeActivationFilter(ActivationFilter):
    """Filter by type e.g. collect all ReLU activations."""

    def __init__(self, types: list[str]) -> None:
        """
        Initialize the TypeActivationFilter.

        Args:
            types (list[str]): List of layer types to filter.
        """
        self.types = types

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        """
        Filter the activations by type.

        Args:
            activations (list[Activation]): List of activations.

        Returns:
            list[Activation]: Filtered list of activations.
        """
        return [activation for activation in activations if activation.layer_type in self.types]

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        """
        List all available layer types based on the activations.

        Args:
            activations (list[Activation]): List of activations.

        Returns:
            list: List of available layer types.
        """
        return list({activation.layer_type for activation in activations})


@register_filter
class IndexActivationFilter(ActivationFilter):
    """Filter by indices of the activations."""

    def __init__(self, indices: list[int]) -> None:
        """
        Initialize the IndexActivationFilter.

        Args:
            indices (list[int]): List of indices to filter.
        """
        self.indices = indices

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        """
        Filter the activations by indices.

        Args:
            activations (list[Activation]): List of activations.

        Returns:
            list[Activation]: Filtered list of activations.
        """
        return [activations[idx] for idx in self.indices]

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        """
        List all available indices based on the activations.

        Args:
            activations (list[Activation]): List of activations.

        Returns:
            list: List of available indices.
        """
        return list(range(len(activations)))


@register_filter
class TargetsActivationFilter(ActivationFilter):
    """Preserve neurons associated with given classes."""

    def __init__(self, indices: list[int]) -> None:
        """
        Initialize the TargetsActivationFilter.

        Args:
            indices (list[int]): List of indices to filter.
        """
        self.indices = indices

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        """
        Filter the activations by preserving neurons associated with given classes.

        Args:
            activations (list[Activation]): List of activations.

        Returns:
            list[Activation]: Filtered list of activations.
        """
        last_activation = activations[-1]  # last layer
        result_activations = []
        for idx in self.indices:
            # In this case, it's just a label of the neuron associated with a given idx
            layer_type = f'target_{idx}'
            value = last_activation.value[:, idx:idx+1]
            output_shape = value.shape
            activation = Activation(layer_type, output_shape, value)
            result_activations.append(activation)
        return result_activations

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        """
        List all available indices based on the activations.

        Args:
            activations (list[Activation]): List of activations.

        Returns:
            list: List of available indices.
        """
        n_classes = activations[-1].output_shape[-1]
        return list(range(n_classes))
