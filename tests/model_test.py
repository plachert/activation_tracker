from __future__ import annotations

import pytest
import torch
from activation_tracker import activation
from activation_tracker import model


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(5, 3)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


def test_empty_activations():
    m = model.ModelWithActivations(MockModel())
    with pytest.raises(model.NoActivationsError):
        m.activations


def test_unfiltered():
    m = model.ModelWithActivations(MockModel(), None, torch.rand(1, 10))
    activations = m.activations['all']
    assert len(activations) == 4
    assert activations[0].layer_type == 'Linear'
    assert activations[0].output_shape == torch.Size([1, 5])
    assert activations[1].layer_type == 'ReLU'
    assert activations[1].output_shape == torch.Size([1, 5])
    assert activations[2].layer_type == 'Linear'
    assert activations[2].output_shape == torch.Size([1, 3])
    assert activations[3].layer_type == 'ReLU'
    assert activations[3].output_shape == torch.Size([1, 3])


def test_forward():
    m = model.ModelWithActivations(MockModel(), None, None)
    input_ = torch.rand(1, 10)
    assert torch.equal(m(input_), m.model(input_))


def test_activations():
    activation_filters = {
        'linear': [activation.TypeActivationFilter(['Linear'])],
        'relu': [activation.TypeActivationFilter(['ReLU'])],
    }
    m = model.ModelWithActivations(
        MockModel(), activation_filters, torch.rand(1, 10),
    )
    activations = m.activations
    assert activations['linear'][0].layer_type == 'Linear'
    assert activations['linear'][1].layer_type == 'Linear'
    assert activations['relu'][0].layer_type == 'ReLU'
    assert activations['relu'][1].layer_type == 'ReLU'


def test_combination():
    combined = {
        'first_relu': [
            activation.TypeActivationFilter(['ReLU']),
            activation.IndexActivationFilter([0]),
        ],
        'last_linear': [
            activation.TypeActivationFilter(['Linear']),
            activation.IndexActivationFilter([-1]),
        ],
    }
    m = model.ModelWithActivations(
        MockModel(), combined, torch.rand(1, 10),
    )
    activations = m.activations
    assert len(activations['first_relu']) == 1
    assert len(activations['last_linear']) == 1
    assert activations['first_relu'][0].layer_type == 'ReLU'
    assert activations['last_linear'][0].layer_type == 'Linear'
    assert activations['first_relu'][0].output_shape == torch.Size([1, 5])
    assert activations['last_linear'][0].output_shape == torch.Size([1, 3])


if __name__ == '__main__':
    pytest.main()
