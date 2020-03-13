from typing import List
import re
import logging
import collections

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from neuralgym.ops.layers import resize
import xarray as xr

import dip.image_io
import dip.network_vis as net_vis


OUT_DIR = 'out/experiments/2/11'


class MyAdam(tf.train.AdamOptimizer):

    #def __init__(self, *args, **kwargs):
    #    super(MyAdam, self).__init__(*args, **kwargs)

    def get_mul_factor(self, var):
            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")
            beta1_power, beta2_power = self._get_beta_accumulators()
            m_hat = m/(1-beta1_power)
            v_hat = v/(1-beta2_power)
            step = m_hat/(v_hat**0.5 + self._epsilon_t)
            lr = self._lr
            mul_factor = lr*step
            return mul_factor

    def _apply_dense(self, grad, var):
        log_for = {'conv17'}
        logging_enabled = False
        if logging_enabled and layer_name_from_var(var) in log_for:
            # Use a histogram summary to monitor it during training.
            tf.summary.histogram("hist_adam_step", self.get_mul_factor(var))
            tf.summary.histogram("hist_grad", grad)
            tf.summary.histogram("hist_var", var)
        return super(MyAdam,self)._apply_dense(grad, var)


def plot_as_tf_summary(x):
    if len(x.shape) != 3:
        raise ValueError()
    channels = x.shape[2]
    num_cols = 3
    num_rows = 1 + channels // num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols)
            #figsize=(50,50)) #figsize=(10,20)
    axes_f = axes.flatten()
    for i in range(len(axes_f)):
        if i < channels:
            axes_f[i].imshow(x[...,i], cmap='hot', interpolation='nearest')
            axes_f[i].get_yaxis().set_visible(False)
            axes_f[i].get_xaxis().set_visible(False)
        else:
            axes_f[i].set_axis_off()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    img_str = buf.getvalue()
    buf.close()
    plt.close(fig)
    return tf.Summary.Image(encoded_image_string=img_str,
            width=300, height=300)
    #img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    # img = np.expand_dims(img, axis=0)
    #return img


def create_optimizer():
    #lr = 0.005
    lr = 0.002
    optimizer = MyAdam(lr, epsilon=0.01)
    return optimizer, lr


def l2_loss(out_img, target_img):
    diff = out_img - target_img
    loss = tf.reduce_sum(tf.pow(diff, 2))
    return loss


def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='same', activation=tf.nn.elu, training=True):
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    layer = tf.keras.layers.Conv2D(
        cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)
    #if cnum == 3 or activation is None:
        # conv for output
        #return x, layer
    #x, y = tf.split(x, 2, 3)
    #x = activation(x)
    #y = tf.nn.sigmoid(y)
    #x = x * y
    x = layer(x)
    return x, layer


def gen_deconv(x, cnum, ksize, stride, rate, name='upsample', padding='same', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    # Just using these as options to keep the signature the same as gen_conv.
    assert ksize == 3
    assert stride == 1
    assert rate == None
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x, layer = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x, layer

def build_encoder_decoder_model(xin):
    LayerConf = collections.namedtuple('LayerConf', 
            ['func', 'num_filters', 'filter_size', 'stride', 'rate', 'name'])
    filter_factor = 20#48
    layer_conf = [
        #           func   |  num filters    | filter size | stride | rate | name
        LayerConf(gen_conv,   1+filter_factor,       5,         1,       1,    'conv1'),
        #LayerConf(gen_conv,   2*filter_factor,       3,         2,       1,    'conv2_downsample'),
        LayerConf(gen_conv,   2+filter_factor,       3,         1,       1,    'conv2'),           # 32x32
        #LayerConf(gen_conv,   4*filter_factor,       3,         2,       1,    'conv4_downsample'),     
        LayerConf(gen_conv,   4+filter_factor,       3,         1,       1,    'conv3'),           # 16x16
        LayerConf(gen_conv,   4+filter_factor,       3,         2,       1,    'conv4_downsample'),     
        LayerConf(gen_conv,   4+filter_factor,       3,         1,       1,    'conv5'),
        LayerConf(gen_conv,   4+filter_factor,       3,         2,       1,    'conv6_downsample'),     
        LayerConf(gen_conv,   4+filter_factor,       3,         1,       2,    'conv7_atrous'), 
        LayerConf(gen_conv,   4+filter_factor,       3,         1,       4,    'conv8_atrous'),
        LayerConf(gen_conv,   4+filter_factor,       3,         1,       8,    'conv9_atrous'),
        LayerConf(gen_conv,   4+filter_factor,       3,         1,      16,    'conv10_atrous'),
        LayerConf(gen_conv,   4+filter_factor,       3,         1,       1,    'conv11'),
        LayerConf(gen_conv,   4+filter_factor,       3,         1,       1,    'conv12'),
        LayerConf(gen_deconv, 2+filter_factor,       3,         1,       None, 'conv13_upsample'),
        LayerConf(gen_conv,   2+filter_factor,       3,         1,       1,    'conv14'),
        LayerConf(gen_deconv, 1+filter_factor,       3,         1,       None, 'conv15_upsample'),
        LayerConf(gen_conv,     filter_factor,      3,         1,       1,    'conv16')]
        #LayerConf(gen_conv,   1,                     3,         1,       1,    'conv17')]
    keras_layers = []
    layer_inputs = []
    x = xin
    for c in layer_conf:
        layer_inputs.append(x)
        x, l = c.func(x, *list(c)[1:])
        keras_layers.append(l)
    layer_inputs.append(x)
    x, l = gen_conv(x, 1, 3, 1, activation='tanh', name='conv17')
    keras_layers.append(l)

    layer_instrumentation = []
    for i, l in enumerate(keras_layers):
        kernel, bias = l.variables
        if 'kernel' not in kernel.name:
            raise Exception('Unexpected layer variable: {kernel.name}')
        if 'bias' not in bias.name:
            raise Exception('Unexpected layer variable: {bias.name}')
        layer_instrumentation.append(net_vis.InstrumentationData.Layer(
            layer_inputs[i], kernel, bias))

    # Add the last layer (output) instrumentation. 
    layer_instrumentation.append(
            net_vis.InstrumentationData.Layer(x, None, None))
    return x, layer_instrumentation

def build_model(xin):
    """Builds a very simple convolutional neural network.

    The output will have the same dimensions as the input.

    Returns: the output tensor
    """
    x = xin
    padding = 'SAME'
    def relu6(x):
        return tf.keras.activations.relu(x, max_value=6.0)

    # Layer descriptions.
    num_filters = [10, 10, 10, 1]
    filter_size = [3,  3, 3, 1]
    activations = [relu6, relu6, relu6, 'tanh']
    assert len(num_filters) == len(filter_size) == len(activations)

    keras_layers = []
    layer_inputs = []
    for i in range(len(num_filters)):
        name = f'conv{i}'
        layer = tf.keras.layers.Conv2D(
            filters=num_filters[i],
            kernel_size=(filter_size[i], filter_size[i]),
            strides=(1,1),
            # data_format=data_format, Use default, for now.
            # Is it the case that Tensorflow corrects the ordering for GPUs
            # anyway?
            activation=activations[i],
            padding='same',
            use_bias=True,
            name=name)
        layer_inputs.append(x)
        keras_layers.append(layer)
        x = layer(x)
    
    layer_instrumentation = []
    for i, l in enumerate(keras_layers):
        kernel, bias = l.variables
        if 'kernel' not in kernel.name:
            raise Exception('Unexpected layer variable: {kernel.name}')
        if 'bias' not in bias.name:
            raise Exception('Unexpected layer variable: {bias.name}')
        layer_instrumentation.append(net_vis.InstrumentationData.Layer(
            layer_inputs[i], kernel, bias))

    # Add the last layer (output) instrumentation. 
    layer_instrumentation.append(
            net_vis.InstrumentationData.Layer(x, None, None))
    return x, layer_instrumentation


def run(input_img_path, target_img_path):
    input_img = dip.image_io.load_img_greyscale(input_img_path)
    target_img = dip.image_io.load_img_greyscale(target_img_path)
    # Add batch dimension.
    input_img = tf.expand_dims(
            tf.convert_to_tensor(input_img, dtype=tf.float32),
            axis=0)
    target_img = tf.expand_dims(
            tf.convert_to_tensor(target_img, dtype=tf.float32),
            axis=0)
    tf.train.get_or_create_global_step()
    # Build simple model:
    out, layers = build_model(input_img)
    # Build deep_fill model (first half)
    #out, layers = build_encoder_decoder_model(input_img)
    loss = l2_loss(out, target_img)
    loss += 0.001 * tf.add_n([tf.nn.l2_loss(p) for p in tf.trainable_variables()])
    optimizer, learning_rate = create_optimizer()
    grads = optimizer.compute_gradients(loss)
    minimize_op = optimizer.apply_gradients(grads)
    PIXEL_RANGE = 2.0
    psnr = tf.image.psnr(out, target_img, max_val=PIXEL_RANGE)

    steps_per_img_save = 1
    max_step = 1000

    instrumentation = net_vis.InstrumentationData(layers)
    d_instrumentation = instrumentation.as_derivative(loss)
    step_instrumentation = net_vis.InstrumentationData(
            [net_vis.InstrumentationData.Layer(l.layer_in, l.kernel, l.bias) for
                l in layers])
    trainable_var_set = set((v for v in tf.trainable_variables()))
    for i in range(len(step_instrumentation.layers)):
        if step_instrumentation.layers[i].kernel in trainable_var_set:
            step_instrumentation.layers[i].kernel = \
                d_instrumentation.layers[i].kernel * optimizer.get_mul_factor(step_instrumentation.layers[i].kernel)
                  

    try:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            input_img_val, target_img_val = sess.run([input_img[0],
                target_img[0]])
            dip.image_io.print_img(input_img_val, f'{OUT_DIR}/input.png')
            dip.image_io.print_img(target_img_val, f'{OUT_DIR}/target.png')
            while step < max_step:
                _, loss_val, psnr_val= sess.run([minimize_op, loss, psnr])
                print(f'Step: {step}', f'Loss: {loss_val}', f'PSNR: {psnr_val}',
                        sep='\t')
                should_save_img = not step % steps_per_img_save
                if should_save_img:
                    # Collect instrumentation data.
                    all_results = instrumentation.eval_and_record(sess, step)
                    d_instrumentation.eval_and_record(sess, step)
                    step_instrumentation.eval_and_record(sess, step)
                    out_val = all_results[-1]
                    # Remove batch dimension
                    out_val = out_val[0]
                    # Save out image to filesystem.
                    out_path = f'{OUT_DIR}/{step}.png'
                    dip.image_io.print_img(out_val, out_path)
                step += 1
    finally:
        print('Saving instrumentation data...')
        instrumentation.to_xdataset().to_netcdf(f'{OUT_DIR}/instrumentation.nc')
        d_instrumentation.to_xdataset().to_netcdf(f'{OUT_DIR}/d_instrumentation.nc')
        step_instrumentation.to_xdataset().to_netcdf(f'{OUT_DIR}/step_instrumentation.nc')
        print('Done')
    return psnr_val[0]


def main():
    #run(input_img_path='./resources/greyscale_noise_16x16.png',
    #    target_img_path='./resources/greyscale_vertical_bar_16x16.png')
    run(input_img_path='./resources/greyscale_noise_64x64.png',
        target_img_path='./resources/statue1_64_greyscale.png')


if __name__ == '__main__':
    main()


