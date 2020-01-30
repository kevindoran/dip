import re
import enum
from collections import defaultdict

import numpy as np
import tensorflow as tf
import pandas 
from typing import List

import dip.image_io
import dip.deep_fill.inpaint_model as deep_fill_model

IMG_SHAPE = (256, 256, 3)
BATCH_SIZE = 16
NET_NAME = 'inpaint_net'
DBG_FOLDER_PATH = 'out/experiments/1/5/dbg'


def stage1_layer_names():
    names = ['conv1', 'conv2_downsample', 'conv3', 'conv4_downsample',
             'conv5', 'conv6', 'conv7_atrous', 'conv8_atrous', 'conv9_atrous',
             'conv10_atrous', 'conv11', 'conv12', 'conv13_upsample', 'conv14',
             'conv15_upsample', 'conv16', 'conv17']
    return names


def stage2_layer_names():
    names = ['xconv1', 'xconv2_downsample', 'xconv3', 'xconv4_downsample',
            'xconv5', 'xconv6', 'xconv7_atrous', 'xconv8_atrous',
            'xconv9_atrous', 'xconv10_atrous', 'pmconv1', 'pmconv2_downsample',
            'pmconv3', 'pmconv4_downsample', 'pmconv5', 'pmconv6', 'pmconv9',
            'pmconv10', 'allconv11', 'allconv12', 'allconv13_upsample', 
            'allconv14', 'allconv15_upsample', 'allconv16', 'allconv17']
    return names


def both_stage_layer_names():
    return stage1_layer_names() + stage2_layer_names()


class MyDebugAdam(tf.train.AdamOptimizer):

    #def __init__(self, *args, **kwargs):
    #    super(MyDebugAdam, self).__init__(*args, **kwargs)

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
        if layer_name_from_var(var) in log_for:
            # Use a histogram summary to monitor it during training.
            #tf.summary.histogram("hist_adam_step", self.get_mul_factor(var)) 
            #tf.summary.histogram("hist_grad", grad) 
            tf.summary.histogram("hist_var", var) 
        return super(MyDebugAdam,self)._apply_dense(grad, var)    

class TrainingData:

    def __init__(self,
            column_names : List[str],
            steps_per_record : int, 
            training_method : str, 
            learning_rate : str): 
        self.column_names = column_names
        self.steps_per_record = steps_per_record
        self.training_method = training_method
        self.learning_rate = learning_rate
        self._step_data = []

    def add(self, layer_data):
        if len(layer_data) != len(self.column_names):
            raise ValueError(f"Incorrect data length: {len(layer_data)}")
        self._step_data.append(layer_data)

    def to_dataframe(self):
        # Create DataFrame. The rows will be steps; colums will be layers. 
        start = 0
        step = self.steps_per_record
        stop = step * len(self._step_data)
        index = range(start, stop, step)
        df = pandas.DataFrame(self._step_data, index)
        # Use the layer names as column header.
        df.columns = self.column_names
        # Set the index values.
        return df


def layer_name_from_var(variable):
    name = variable.name
    pattern = r'{}/(\w+)/.*'.format(NET_NAME)
    m = re.match(pattern, name)
    if not m or not m.group(1):
        raise Exception('Error finding layer name from variable: '
                        f'{name}.')
    return m.group(1)


def stage_1_grads_only(grads):
    ignore = set(stage2_layer_names())
    filtered_grads = [(g,v) for g,v in grads \
                          if layer_name_from_var(v) not in ignore]
    return filtered_grads


def gradients_by_layer(grad_var_pairs):
    d = defaultdict(list)
    for g,v in grad_var_pairs:
        d[layer_name_from_var(v)].append(g)
    return d


def summarize_gradients_by_layer(grad_var_pairs):
    layer_to_grad = {}
    for layer_name, grads in gradients_by_layer(grad_var_pairs).items():
        layer_flattened = tf.concat([tf.reshape(g, [-1]) for g in grads], 
                axis=0)
        # Normalize by no. of params.
        layer_dist = tf.norm(layer_flattened, ord='euclidean')
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
    global_step = tf.train.get_or_create_global_step()#tf.train.get_global_step()
    lr = tf.train.exponential_decay(lr_start, global_step, decay_steps, 
            decay_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    return optimizer, lr


def learning_rate():
    lr_start = 0.05
    decay_steps = 20
    decay_rate = 0.98
    global_step = tf.train.get_or_create_global_step()#tf.train.get_global_step()
    lr = tf.train.exponential_decay(lr_start, global_step, decay_steps, 
            decay_rate)
    return lr


def adam_optimizer():
    #learning_rate = 0.00375, good but unstable.
    #learning_rate = 0.0005
    lr = learning_rate()
    #optimizer = tf.train.AdamOptimizer(
    #        learning_rate=learning_rate, epsilon=0.01)
    optimizer = MyDebugAdam(
            learning_rate=lr, epsilon=0.01)
    return optimizer, lr


def create_optimizer():
    #return sgd_optimizer()
    return adam_optimizer()


OutputStage = enum.Enum('OutputStage', 'STAGE_1 STAGE_2')
def create_dip_model(x, mask, target, stage):
    model = deep_fill_model.InpaintCAModel()
    stage_1, stage_2, _ = model.build_inpaint_net(x, mask, training=True)
    out = stage_1 if stage == OutputStage.STAGE_1 else stage_2
    loss = l2_loss(out, target) + weight_loss()
    return out, loss


def run(target_img_path, original_img_path):
    """Run experiment.

    See notebooks/dip.ipynb for details.
    """
    target_img = dip.image_io.load_img(target_img_path)
    orig_img = dip.image_io.load_img(original_img_path)
    target_img_batch = tf.expand_dims(
            tf.convert_to_tensor(target_img, dtype=tf.float32),
            axis=0)
    noise = tf.random.uniform((1, *IMG_SHAPE), 
            seed=127,
            dtype=tf.dtypes.float32)
    input_img = noise # Alternatively, an incomplete image for inpainting.
    input_img_batch = tf.tile(input_img, (BATCH_SIZE, 1, 1, 1))
    # If shape of input_img_batch is = (16, 256, 256, 3), then the mask will
    # have shape = (1, 256, 256 1).
    mask_shape = (1, *IMG_SHAPE[:-1], 1)
    # For denoising, we mask nothing (so all 1s in the 'mask'). 
    mask = tf.ones(mask_shape, dtype=tf.float32)
    stage = OutputStage.STAGE_1
    out_img_batch, loss = create_dip_model(input_img_batch, mask,
            target_img_batch, stage)
    optimizer, learning_rate = create_optimizer()
    grads = optimizer.compute_gradients(loss)
    minimize_op = optimizer.apply_gradients(grads)
    if stage == OutputStage.STAGE_1:
        grads = stage_1_grads_only(grads)
    # Replace gradients with step:   
    step_var_pairs = [(optimizer.get_mul_factor(v),v) for g,v in grads]
    #layer_data = list(summarize_gradients_by_layer(grads).values())
    layer_data = list(summarize_gradients_by_layer(step_var_pairs).values())
    # Should we clip or not? If clipping, then we have some limits on our loss,
    # (-1 - 1)^2 No, don't clip the loss, but you can clip the image before
    # saving.
    # tf.clip_by_value(out_img_batch, clip_value_min=-1.0, clip_value_max=1.0)
    img_to_print = out_img_batch[0] 
    PIXEL_RANGE = 2.0
    psnr = tf.image.psnr(img_to_print, orig_img, max_val=PIXEL_RANGE)


    # Run graph.
    try:
        steps_per_record = 1
        steps_per_img_save = steps_per_record * 100
        layer_names = stage1_layer_names() if stage == OutputStage.STAGE_1 \
                            else both_stage_layer_names()
        column_names = ['loss', 'psnr'] + layer_names
        data = TrainingData(column_names=column_names,
                            steps_per_record=steps_per_record,
                            training_method=optimizer.__class__.__name__,
                            learning_rate=learning_rate)
        all_summaries = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(DBG_FOLDER_PATH + '/tf_summary')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            input_img_val, mask_val, target_img_val = sess.run(
                    [input_img_batch[0], mask[0], target_img_batch[0]])
            # Save the input image, the mask and the target image. This is
            # done so that possible bugs with their usage might be spotted.
            dip.image_io.print_img(input_img_val, 
                    f'{DBG_FOLDER_PATH}/dbg_input_noise.png')
            dip.image_io.print_img(mask_val, f'{DBG_FOLDER_PATH}/dbg_mask.png')
            dip.image_io.print_img(target_img_val, 
                    f'{DBG_FOLDER_PATH}/dbg_target_image.png')
            while True:
                _, loss_val, layer_data_val, psnr_val, grads_val, summary_val \
                    = sess.run([minimize_op, loss, layer_data, psnr, grads,
                        all_summaries])
                print(f'Step: {step}.\t'
                      f'Loss: {loss_val}.\t'
                      f'PSNR: {psnr_val}.\t')
                assert np.all(np.array(layer_data_val) >= 0), 'All norm \
                    measures should be zero or more.'
                # Record data.
                should_record = not step % steps_per_record
                if should_record:
                    single_record = [loss_val, psnr_val] + layer_data_val
                    data.add(single_record)
                    file_writer.add_summary(summary_val, step)
                # Print the output image every so often.
                save_img = step < 20 or (not step % steps_per_img_save)
                if save_img:
                    img_to_print_val = sess.run([img_to_print])[0]
                    out_path = f'{DBG_FOLDER_PATH}/{step}.png'
                    dip.image_io.print_img(img_to_print_val, out_path)
                step += 1
    finally:
        print('Saving data...')
        data.to_dataframe().to_pickle(
            f'{DBG_FOLDER_PATH}/data.dataframe_pickle')
        print('Done')


def main():
    run(target_img_path='./resources/statue1_256_noise.jpg',
        original_img_path='./resources/statue1_256.jpg')


if __name__ == '__main__':
    main()
