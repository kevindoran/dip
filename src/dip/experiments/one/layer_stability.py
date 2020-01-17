import dip.image_io
import numpy as np
import tensorflow as tf
import dip.deep_fill.inpaint_model as deep_fill_model
import re
import pprint
from collections import defaultdict

IMG_SHAPE = (256, 256, 3)
BATCH_SIZE = 16
NET_NAME = 'inpaint_net'
DBG_FOLDER_PATH = 'out/experiments/1/dbg'

pretty_printer = pprint.PrettyPrinter()

def stage_1_layer_names():
    names = ['conv1', 'conv2_downsample', 'conv3', 'conv4_downsample',
             'conv5', 'conv6', 'conv7_atrous', 'conv8_atrous', 'conv9_atrous',
             'conv10_atrous', 'conv11', 'conv12', 'conv13_upsample', 'conv14',
             'conv15_upsample', 'conv16', 'conv17']
    return names


def layer_name_from_var(variable):
    name = variable.name
    pattern = r'{}/(\w+)/.*'.format(NET_NAME)
    m = re.match(pattern, name)
    if not m or not m.group(1):
        raise Exception('Error finding layer name from variable: '
                        f'{name}.')
    return m.group(1)


def gradients_by_layer(grad_var_pairs):
    d = defaultdict(list)
    for g,v in grad_var_pairs:
        d[layer_name_from_var(v)].append(g)
    return d

def gradient_by_layer(grad_var_pairs):
    layer_to_grad = {}
    for layer_name, grads in gradients_by_layer(grad_var_pairs).items():
        layer_dist = tf.add_n([tf.nn.l2_loss(v) for v in grads])
        # Normalize by no. of params.
        layer_dist /= len(grads)
        layer_to_grad[layer_name] = layer_dist
    return layer_to_grad
    

def l2_loss(out_img_batch, true_img):
    diff = out_img_batch - true_img
    loss = tf.reduce_sum(tf.pow(diff, 2))
    #loss = tf.reduce_mean(tf.pow(diff/4.0, 2))
    return loss


def weight_loss():
    loss = tf.add_n([0.001 * tf.nn.l2_loss(v) for v in
        tf.trainable_variables()])
    return loss


def sgd_optimizer():
    lr_start = 05.0
    decay_steps = 20
    decay_rate = 0.98
    global_step = tf.train.get_global_step()
    lr = tf.train.exponential_decay(lr_start, global_step, decay_steps, 
            decay_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    return optimizer, lr


def adam_optimizer():
    learning_rate = 0.003
    optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate, epsilon=0.01)
    return optimizer, learning_rate


def create_optimizer():
    #return sgd_optimizer()
    return adam_optimizer()


def run(target_img_path, true_img_path):
    """Run experiment.

    See notebooks/dip.ipynb for details.
    """
    target_img = dip.image_io.load_img(target_img_path)
    true_img = dip.image_io.load_img(true_img_path)
    target_img_batch = tf.expand_dims(
            tf.convert_to_tensor(target_img, dtype=tf.float32),
            axis=0)
    true_img_batch = tf.expand_dims(
            tf.convert_to_tensor(true_img, dtype=tf.float32),
            axis=0)

    # Our global step.
    global_step = tf.train.create_global_step()
    # The above is the same as:
    # global_step = tf.get_variable('global_step', trainable=False,
    # initializer=0)
    model = deep_fill_model.InpaintCAModel()

    noise = tf.random.uniform((1, *IMG_SHAPE), dtype=tf.dtypes.float32)
    input_img = noise # This is normally the incomplete image.
    input_img_batch = tf.tile(input_img, (BATCH_SIZE, 1, 1, 1))
    # If shape of input_img_batch is = (16, 256, 256, 3), then 
    mask_shape = (1, *IMG_SHAPE[:-1], 1)
    mask = tf.ones(mask_shape, dtype=tf.float32)
    stage_1, stage_2, _ = model.build_inpaint_net(x=input_img_batch,
                                                  mask=mask,
                                                  training=True)
    out_img_batch = stage_1
    #loss = l2_loss(out_img_batch, target_img_batch) + weight_loss()
    loss = l2_loss(out_img_batch, target_img_batch) + weight_loss()
    optimizer, learning_rate = create_optimizer()
    grads = optimizer.compute_gradients(loss)
    flattened_grads = tf.concat([tf.reshape(g, [-1]) for g,v in grads],0)
    update_dist = tf.norm(flattened_grads * learning_rate)
    minimize_op = optimizer.minimize(loss)
    # Should we clip or not? If clipping, then we have some limits on our loss,
    # (-1 - 1)^2 No, don't clip the loss, but you can clip the image before
    # saving.
    #tf.clip_by_value(out_img_batch, clip_value_min=-1.0, clip_value_max=1.0)
    img_to_print = out_img_batch[0] 
    dbg_target = true_img_batch[0]
    per_layer_data = gradient_by_layer(grads)

    # Run graph.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 1
        steps_per_eval = 20
        while True:
            _, loss_val, update_dist_val = sess.run([minimize_op, loss,
                update_dist])
            print(f'Step: {step}.\t'
                  f'Loss: {loss_val}.\t'
                  f'Update dist: {update_dist_val}.')
            # Print the output image every so often.
            run_eval = step < 20 or (not step % steps_per_eval)
            if run_eval:
                #import pdb;pdb.set_trace()
                img_to_print_val, dbg_target_val, layer_data_val = \
                    sess.run([img_to_print, dbg_target, per_layer_data])
                pretty_printer.pprint(layer_data_val)
                out_path = f'{DBG_FOLDER_PATH}/{step}.png'
                dip.image_io.print_img((img_to_print_val + 1.0)*127.5, 
                        out_path)
            step += 1


def main():
    run(target_img_path='./resources/statue1_256_noise.jpg',
        true_img_path= './resources/statue1_256.jpg')


if __name__ == '__main__':
    main()
