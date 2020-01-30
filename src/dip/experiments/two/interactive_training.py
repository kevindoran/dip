import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import dip.image_io


OUT_DIR = 'out/experiments/2/1'
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
        if logging_enabled and layer_name_from_var(var) in log_for:                                # Use a histogram summary to monitor it during training.
            tf.summary.histogram("hist_adam_step", self.get_mul_factor(var))   
            tf.summary.histogram("hist_grad", grad)                            
            tf.summary.histogram("hist_var", var)                               
        return super(MyAdam,self)._apply_dense(grad, var) 


def plot_tensor(x):
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
    lr = 0.002
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
    padding = 'SAME'
    net_name = NET_NAME
    inner_filters = [32, 32, 32, 32, 32, 1] if not filters else filters

    for i in range(len(inner_filters)):
        name = f'conv{i}'
        x = tf.keras.layers.Conv2D(
            filters=inner_filters[i],
            kernel_size=(3,3),
            strides=(1,1),
            # data_format=data_format, Use default, for now.
            # Is it the case that Tensorflow corrects the ordering for GPUs
            # anyway?
            activation='relu',
            padding='same',
            use_bias=True,
            name=name)(x)
    last_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3,3),
            strides=(1,1),
            activation='tanh',
            padding='same',
            use_bias=True,
            name='last_layer')
    x = last_layer(x)
    return x, last_layer


def run(input_img_path, target_img_path, filters=None):
    input_img = dip.image_io.load_img_greyscale(input_img_path)
    target_img = dip.image_io.load_img_greyscale(target_img_path)
    #input_img = dip.image_io.load_img(input_img_path)
    #target_img = dip.image_io.load_img(target_img_path)
    input_img = tf.expand_dims(
            tf.convert_to_tensor(input_img, dtype=tf.float32),
            axis=0)
    target_img = tf.expand_dims(
            tf.convert_to_tensor(target_img, dtype=tf.float32),
            axis=0)
    tf.train.get_or_create_global_step()
    out, last_layer = build_model(input_img, filters)
    loss = l2_loss(out, target_img)
    loss += 0.001 * tf.add_n([tf.nn.l2_loss(p) for p in tf.trainable_variables()])
    optimizer, learning_rate = create_optimizer()
    grads = optimizer.compute_gradients(loss)
    minimize_op = optimizer.apply_gradients(grads)
    PIXEL_RANGE = 2.0
    psnr = tf.image.psnr(out, target_img, max_val=PIXEL_RANGE)
    #last_layer_kernel = tf.get_default_graph().get_tensor_by_name(
    #        f'last_layer/kernel:0')
    #with tf.variable_scope('last_layer', reuse=True):
    #    last_layer_kernel = tf.get_variable('kernel')
    last_layer_kernel = tf.trainable_variables()[-2]
    

    steps_per_img_save = 50
    max_step = 50000
    file_writer = tf.summary.FileWriter(f'{OUT_DIR}/last_layer')
    kernel = tf.squeeze(last_layer_kernel, axis=3)
    #img_summary = tf.summary.image('Kernel weights', kernel_img)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        input_img_val, target_img_val = sess.run([input_img[0], target_img[0]])
        dip.image_io.print_img(input_img_val, f'{OUT_DIR}/input.png')
        dip.image_io.print_img(target_img_val, f'{OUT_DIR}/target.png')
        while step < max_step:
            _, loss_val, psnr_val, kernel_val = sess.run([minimize_op, loss,
                psnr, kernel])
            print(f'Step: {step}', f'Loss: {loss_val}', f'PSNR: {psnr_val}',
                    sep='\t')
            #kernel_val = np.squeeze(last_layer.get_weights()[0], axis=3)
            img_summary = tf.Summary(value=[tf.Summary.Value(tag='kernel',
                image=plot_tensor(kernel_val))])
            file_writer.add_summary(img_summary,
                    global_step=step)
            should_save_img = not step % steps_per_img_save 
            if should_save_img:
                out_val = sess.run(out[0])
                out_path = f'{OUT_DIR}/{step}.png'
                dip.image_io.print_img(out_val, out_path)
            step += 1
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
    run(input_img_path='./resources/greyscale_noise_16x16.png',
        target_img_path='./resources/greyscale_patternA_16x16.png')
    #run(target_img_path='./resources/greyscale_noise_16x16.png',
    #    input_img_path='./resources/greyscale_patternA_16x16.png')


if __name__ == '__main__':
    main()


