"""Scaffold smoke tests: the package and the core stack import on CPU."""

import importlib

import pytest

import btseg

MODULES = [
    "btseg.config",
    "btseg.data.discover",
    "btseg.data.labels",
    "btseg.data.transforms",
    "btseg.data.dataset",
    "btseg.models.factory",
    "btseg.losses",
    "btseg.train",
    "btseg.infer",
    "btseg.postprocess",
    "btseg.metrics",
    "btseg.volumetrics",
    "btseg.evaluate",
    "btseg.viz",
    "btseg.export",
]


def test_version():
    assert btseg.__version__ == "0.1.0"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_core_stack_imports():
    import monai
    import nibabel
    import torch

    assert torch.tensor([1.0, 2.0]).sum().item() == 3.0
    assert monai.__version__
    assert nibabel.__version__
