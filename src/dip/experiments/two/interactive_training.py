from typing import List
import re
import logging

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import xarray as xr

import dip.image_io
import dip.network_vis as net_vis


OUT_DIR = 'out/experiments/2/2'
NET_NAME = 'mini_model'


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
    lr = 0.005
    optimizer = MyAdam(lr, epsilon=0.01)
    return optimizer, lr


def l2_loss(out_img, target_img):
    diff = out_img - target_img
    loss = tf.reduce_sum(tf.pow(diff, 2))
    return loss


def build_model(xin, filters=None):
    """Builds a very simple convolutional neural network.

    The output will have the same dimensions as the input.

    Returns: the output tensor
    """
    x = xin
    net_name = NET_NAME
    padding = 'SAME'
    def relu6(x): 
        return tf.keras.activations.relu(x, max_value=6.0)

    # Layer descriptions.
    num_filters = [10, 10, 10, 1]
    filter_size = [3,   3,  3, 1]
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


def run(input_img_path, target_img_path, filters=None):
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
    out, layers = build_model(input_img, filters)
    loss = l2_loss(out, target_img)
    loss += 0.001 * tf.add_n([tf.nn.l2_loss(p) for p in tf.trainable_variables()])
    optimizer, learning_rate = create_optimizer()
    grads = optimizer.compute_gradients(loss)
    minimize_op = optimizer.apply_gradients(grads)
    PIXEL_RANGE = 2.0
    psnr = tf.image.psnr(out, target_img, max_val=PIXEL_RANGE)

    steps_per_img_save = 1
    max_step = 600

    instrumentation = net_vis.InstrumentationData(layers)
    d_instrumentation = instrumentation.as_derivative(loss)
    step_instrumentation = net_vis.InstrumentationData(layers)
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


def main2():
    res = []
    for i in range(20, 50):
        filters = [i]
        psnr = run(input_img_path='./resources/greyscale_noise_16x16.png',
                    target_img_path='./resources/greyscale_patternA_16x16.png',
                    filters=filters)
        res.append(psnr)
    print(res)


def main():
    #run(input_img_path='./resources/greyscale_noise_16x16.png',
    #    target_img_path='./resources/greyscale_vertical_bar_16x16.png')
    run(input_img_path='./resources/statue1_256_empty_mask.png',
        target_img_path='./resources/statue1_256_noise.jpg')
    #run(target_img_path='./resources/greyscale_noise_16x16.png',
    #    input_img_path='./resources/greyscale_patternA_16x16.png')


if __name__ == '__main__':
    main()


