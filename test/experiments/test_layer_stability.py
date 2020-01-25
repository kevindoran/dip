from collections import namedtuple
import collections
import random

import pytest
import tensorflow as tf

import dip.experiments.one.layer_stability as layer_stability


@pytest.fixture
def tf_graph():
    with tf.Graph().as_default() as g:
        yield g


def test_layer_name_from_var():
    FakeVar = namedtuple('FakeVar', 'name')
    dummy_var = FakeVar('inpaint_net/conv1/kernel:0')
    assert 'conv1' == layer_stability.layer_name_from_var(dummy_var)


def test_TrainingData():
    num_layers = 10
    col_names = [f'l{i}' for i in range(num_layers)]
    fake_data = [random.random() for i in range(num_layers)]
    # Should construct without errors.
    td = layer_stability.TrainingData(
            column_names=col_names,
            steps_per_record=7,
            training_method='sgd',
            learning_rate=0.4)
    
    # Should add the data fine.
    td.add(fake_data)

    # Should fail with incorrect length data. 
    broken_data = fake_data[0:-1]
    with pytest.raises(ValueError):
        assert td.add(broken_data)

    # Test the to_dataframe() method.
    # First add some more data.
    td.add(fake_data)
    td.add(fake_data)
    # Get the dateframe.
    df = td.to_dataframe()
    assert df.shape == (3, num_layers), \
        f"The dataframe shape should be 3 rows and {num_layer} columns."
    assert df.columns.to_list() == col_names
    assert df.at[0,col_names[2]] == fake_data[2], \
        "First row should be the earliest data."



def create_test_model():
    batch_size = 16
    img_shape = (256, 256, 3)
    mask_shape = (1, 256, 256, 1)
    x = tf.ones((batch_size, *img_shape))
    mask = tf.zeros(mask_shape)
    target = tf.random.uniform(img_shape)
    out, loss = layer_stability.create_dip_model(x, mask, target,
                    layer_stability.OutputStage.STAGE_2)
    return out, loss


def test_gradient_by_layer(tf_graph):
    # Setup
    out, loss = create_test_model()
    optimizer, lr = layer_stability.sgd_optimizer()
    grads = optimizer.compute_gradients(loss)

    # Test
    layer_grad_map = layer_stability.gradients_by_layer(grads)
    assert list(layer_grad_map.keys()) == \
        layer_stability.both_stage_layer_names()
    for k, v in layer_grad_map.items():
        assert isinstance(v, collections.Sequence)


def test_summarize_gradients_per_layer(tf_graph):
    # Setup
    out, loss = create_test_model()
    optimizer, lr = layer_stability.sgd_optimizer()
    grads = optimizer.compute_gradients(loss)

    # Test
    layer_grad_map = layer_stability.summarize_gradients_by_layer(grads)
    assert list(layer_grad_map.keys()) == \
        layer_stability.both_stage_layer_names()
    for k, v in layer_grad_map.items():
        assert not isinstance(v, collections.Sequence)


def test_stage_1_grads_only():
    # Setup
    out, loss = create_test_model()
    optimizer, lr = layer_stability.sgd_optimizer()
    grads = optimizer.compute_gradients(loss)
    
    # Test
    grads = layer_stability.stage_1_grads_only(grads)
    layer_names = list(layer_stability.gradients_by_layer(grads).keys())
    assert layer_names == layer_stability.stage1_layer_names()
