from __future__ import annotations

import pytest
import torch
from activation_tracker import activation


@pytest.fixture
def mock_activations():
    mock_types = ['Conv2d', 'ReLU', 'MaxPool2d', 'Conv2d', 'Tanh', 'Linear']
    mock_shapes = [
        (1, 3, 224, 224),
        (1, 3, 224, 224),
        (1, 3, 100, 100),
        (1, 3, 100, 100),
        (1, 3, 100, 100),
        (1, 10),

    ]
    mock_values = [torch.rand(shape) for shape in mock_shapes]
    mock_activation_list = [
        activation.Activation(
            type_, shape_, value_,
        ) for type_, shape_, value_ in zip(mock_types, mock_shapes, mock_values)
    ]
    return mock_activation_list


def test_type_filtering(mock_activations):
    filter_activation = activation.TypeActivationFilter(['Conv2d', 'Tanh'])
    filtered = filter_activation.filter_activations(mock_activations)
    assert len(filtered) == 3
    assert filtered[0] == mock_activations[0]
    assert filtered[1] == mock_activations[3]
    assert filtered[2] == mock_activations[4]


def test_index_filtering(mock_activations):
    filter_activation = activation.IndexActivationFilter([0, 2, 5])
    filtered = filter_activation.filter_activations(mock_activations)
    assert len(filtered) == 3
    assert filtered[0] == mock_activations[0]
    assert filtered[1] == mock_activations[2]
    assert filtered[2] == mock_activations[5]


def test_target_filtering(mock_activations):
    indices = [1, 5, 7, 9]
    filter_activation = activation.TargetsActivationFilter(indices)
    filtered = filter_activation.filter_activations(mock_activations)
    last_activation = mock_activations[-1]
    for i, activation_ in enumerate(filtered):
        type_ = activation_.layer_type
        shape_ = activation_.output_shape
        value_ = activation_.value
        assert type_ == f'target_{indices[i]}'
        assert shape_ == torch.rand(1, 1).shape
        assert value_ == last_activation.value[:, indices[i]:indices[i]+1]


def test_list_type_parameters(mock_activations):
    all_types = set(
        activation.TypeActivationFilter.list_all_available_parameters(
            mock_activations),
    )
    assert all_types == {
        'Conv2d', 'ReLU',
        'MaxPool2d', 'Conv2d', 'Tanh', 'Linear',
    }


def test_list_index_parameters(mock_activations):
    all_indices = activation.IndexActivationFilter.list_all_available_parameters(
        mock_activations,
    )
    assert all_indices == [0, 1, 2, 3, 4, 5]


def test_list_target_parameters(mock_activations):
    all_targets = activation.TargetsActivationFilter.list_all_available_parameters(
        mock_activations,
    )
    assert all_targets == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


if __name__ == '__main__':
    pytest.main()
