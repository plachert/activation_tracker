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

```bash
pip install git+https://github.com/plachert/activation_tracker
```
