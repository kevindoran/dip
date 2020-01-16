import tensorflow as tf
import cv2
import dip
import dip.deep_fill.inpaint_model as deep_fill_model
import neuralgym as ng
import dip.deep_fill.app as deep_fill_app

batch_size = 1
net_name = 'inpaint_net'

def test_build_model():
    with tf.Graph().as_default():
        model = deep_fill_model.InpaintCAModel()
        incomplete_image = tf.zeros([batch_size, 256, 256, 3], tf.float32)
        mask = tf.ones([batch_size, 256, 256, 1], tf.float32)
        stage_1, stage_2, offset_flow = model.build_inpaint_net(
            x=incomplete_image,
            mask=mask,
            training=False,
            name=net_name)


def test_save_and_run_model(tmpdir):
    checkpoint_dir = tmpdir.join('ckpt') # a 'local' object.
    with tf.Graph().as_default():
        # Setup.
        # Create model.
        model = deep_fill_model.InpaintCAModel()
        dummy_img = tf.zeros([batch_size, 256, 256, 3], tf.float32)
        dummy_mask = tf.ones([batch_size, 256, 256, 1], tf.float32)
        stage_1, stage_2, offset_flow = model.build_inpaint_net(
            x=dummy_img,
            mask=dummy_mask,
            training=False,
            name=net_name)
        with tf.Session() as sess:
            # Initialize variables.
            sess.run(tf.global_variables_initializer())

            # Test.
            # 1. Save random model. There should be no errors.
            global_vars = tf.global_variables()
            saver = tf.train.Saver(global_vars)
            saver.save(sess, str(checkpoint_dir))

    # 2. Use random model model to fill.
    #input_image_path = './test/resources/case1_input.png'
    #mask_path = './test/resources/case1_mask.png'
    input_image_path = './waseda_fill/statue1.jpg'
    mask_path = './waseda_fill/statue1_mask2.png'
    out = deep_fill_app.fill(
        image_path=input_image_path,
        mask_path=mask_path,
        checkpoint_dir=str(checkpoint_dir),
        FLAGS=ng.Config('./test/resources/inpaint_test.yml'))
    reference_out = str(tmpdir.join(
        'case1_filled_using_untrained_model.png'))
    cv2.imwrite('waseda_rand.png', out[0][:, :, ::-1])
    (score, diff) = cv2.compare_ssim(out, reference_out, full=True)
    assert score > 0.9
