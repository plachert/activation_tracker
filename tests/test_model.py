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

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


def test_empty_activations():
    m = model.ModelWithActivations(MockModel())
    assert m.activations == {'all': []}
    assert m.activations_values == {'all': []}


def test_unfiltered():
    m = model.ModelWithActivations(MockModel(), None, torch.rand(1, 10))
    activations = m.activations['all']
    assert len(activations) == 2
    assert activations[0].layer_type == 'Linear'
    assert activations[0].output_shape == torch.Size([1, 5])
    assert activations[1].layer_type == 'ReLU'
    assert activations[1].output_shape == torch.Size([1, 5])


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
        MockModel(), activation_filters, torch.rand(1, 10))
    activations = m.activations
    assert activations['linear'][0].layer_type == 'Linear'
    assert activations['relu'][0].layer_type == 'ReLU'


if __name__ == '__main__':
    pytest.main()
