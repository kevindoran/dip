import pytest
import tensorflow as tf

import dip.experiments.two.interactive_training as interactive

def test_build_model():
    # Expect no errors.

    x = tf.zeros([1, 16,16, 1])
    out = interactive.build_model(x)
    assert out is not None
    assert out.shape == x.shape
