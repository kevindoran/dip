import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

import dip.deep_fill.inpaint_model as deep_fill_model

#IMG_SHAPE = (416, 680, 3)
IMG_SHAPE = (256, 256, 3)
BATCH_SIZE = 16




def load_mask(path):
    mask = cv2.imread(path)
    assert mask.shape == IMG_SHAPE
    # Only use 1 channel.
    mask = mask[:, :, 0:1]
    # Make mask elements either 0 or 1.
    mask = np.clip(mask, 0, 1)
    # Add batch dimension.
    mask = np.expand_dims(mask, 0)
    # mask = tf.cast(mask > 0, tf.float32)
    return mask


def load_and_mask_image(img_path, mask):
    img = cv2.imread(img_path)
    assert img.shape == IMG_SHAPE
    # Transform to range [-1, 1].
    img = img / 127.5 - 1.
    # Delete masked area.
    img = img * (1.0 - mask[0])
    # Add batch dimension.
    img = np.expand_dims(img, 0)
    return img


def l2_loss(out_img, masked_image, inv_mask):
    # Will this diff broadcast correctly?
    diff = out_img - masked_image
    # Shouldn't this be mask, not inv_mask?
    diff = diff * inv_mask
    return tf.reduce_sum(tf.pow(diff, 2))


def print_img(normalized_img, step):
    #normalized_img = normalized_img.astype(np.uint16)
    #path = f'./dip_debug_out/step_{step}.tiff'
    path = f'./out/step_{step}.png'
    cv2.imwrite(path, normalized_img)
    #img = Image.fromarray(normalized_img[0], 'BGR')
    #img.save(path)
    #img.show()


def dbg_img(img):
    cv2.imwrite('./out/debug_out.png', img)


def fill(mask_path, img_path):
    # Our global step.
    global_step = tf.train.create_global_step()
    # The above is the same as:
    # global_step = tf.get_variable('global_step', trainable=False, initializer=0)
    mask = load_mask(mask_path)
    masked_img = load_and_mask_image(img_path, mask)
    model = deep_fill_model.InpaintCAModel()
    inv_mask = tf.convert_to_tensor(1.0 - mask[0], dtype=tf.float32)
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    masked_img = tf.convert_to_tensor(masked_img, dtype=tf.float32)

    noise = tf.random.uniform((1, *IMG_SHAPE), dtype=tf.dtypes.float32)
    input_img = noise # This is normally the incomplete image.
    input_img_batch = tf.tile(input_img, (BATCH_SIZE, 1, 1, 1))
    stage_1, stage_2, _ = model.build_inpaint_net(x=input_img_batch,
                                                  mask=mask,
                                                  training=True)
    out_img_batch = stage_2
    loss = l2_loss(out_img_batch, masked_img, inv_mask)
    weight_sum = tf.add_n([0.001 * tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    loss += weight_sum
    learning_rate_start = 05.0; decay_steps = 20; decay_rate = 0.98
    learning_rate = tf.train.exponential_decay(learning_rate_start, global_step, decay_steps, decay_rate)
    learning_rate = 3.0
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.01)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads =optimizer.compute_gradients(loss)
    flattened_grads = tf.concat([tf.reshape(g, [-1]) for g,v in grads],0)
    update_dist = tf.norm(flattened_grads * learning_rate)
    #minimize_op = optimizer.minimize(loss)
    minimize_op = optimizer.minimize(loss)
    # Should we clip or not? If clipping, then we have some limits on our loss,
    # (-1 - 1)^2 No, don't clip the loss, but you can clip the image before saving.
    print_img_batch = out_img_batch #tf.clip_by_value(out_img_batch, clip_value_min=-1.0, clip_value_max=1.0)

    # Run graph.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 1
        steps_per_eval = 20
        while True:
            _, loss_val, update_dist_val = sess.run([minimize_op, loss, update_dist])
            print(f'Step: {step}.\t'
                  f'Loss: {loss_val}.\t'
                  f'Update dist: {update_dist_val}.')
            # Print the output image every so often.
            run_eval = step < 20 or (not step % steps_per_eval)
            if run_eval:
                #import pdb;pdb.set_trace()
                print_img_batch_val, masked_img_val, noise_val = sess.run([print_img_batch,
                                                        masked_img, input_img_batch])
                print_img((print_img_batch_val[0] + 1.0)*127.5, step)
            step += 1


def main():
    fill(mask_path='./resources/statue1_256_empty_mask.png',
         img_path= './resources/statue1_256_noise.jpg')


if __name__ == '__main__':
    main()
