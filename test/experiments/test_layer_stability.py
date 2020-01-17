import dip.experiments.one.layer_stability as layer_stability
from collections import namedtuple
import pytest


def test_layer_name_from_var():
    FakeVar = namedtuple('FakeVar', 'name')
    dummy_var = FakeVar('inpaint_net/conv1/kernel:0')
    assert 'conv1' == layer_stability.layer_name_from_var(dummy_var)
