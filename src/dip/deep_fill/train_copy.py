import os
import glob
import ast

import tensorflow as tf
import neuralgym as ng
import random
from .inpaint_model import InpaintCAModel


def multigpu_graph_def(model, FLAGS, data, gpu_id=0, loss_type='g'):
    with tf.device('/cpu:0'):
        images = data.data_pipeline(FLAGS.batch_size)
    if gpu_id == 0 and loss_type == 'g':
        _, _, losses = model.build_graph_with_losses(
            FLAGS, images, FLAGS, summary=True, reuse=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            FLAGS, images, FLAGS, reuse=True)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')


def train(config_file, max_train_images=None, log_dir=None):
    # training data
    FLAGS = ng.Config(config_file)
    img_shapes = FLAGS.img_shapes
    log_dir = log_dir if log_dir else FLAGS.log_dir
    FLAGS.log_dir = log_dir
    FLAGS.model_restore = log_dir
    fnames = []
    file_list_path = log_dir + '/files_used'
    if os.path.isfile(file_list_path):
        with open(file_list_path) as file_list:
            contents = file_list.readline()
            fnames = ast.literal_eval(contents)
        assert type(fnames) == list
        assert len(fnames)
        assert len(fnames) <= max_train_images
    else:
        assert False
        with open(FLAGS.data_flist[FLAGS.dataset][0]) as f:
            for l in f.read().splitlines():
                fnames.extend(glob.glob(l, recursive=True))
        random.shuffle(fnames)
        if max_train_images:
            fnames = fnames[:min(max_train_images, len(fnames))]
        with open(log_dir + '/files_used', 'w') as file_list_out:
            file_list_out.write(str(fnames))
            assert len(fnames) <= max_train_images
    if FLAGS.guided:
        fnames = [(fname, fname[:-4] + '_edge.jpg') for fname in fnames]
        img_shapes = [img_shapes, img_shapes]
    data = ng.data.DataFromFNames(
        fnames, img_shapes, random_crop=FLAGS.random_crop,
        nthreads=FLAGS.num_cpus_per_job)
    images = data.data_pipeline(FLAGS.batch_size)
    # main model
    model = InpaintCAModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(FLAGS, images)
    # validation images
    val_fnames = []
    if FLAGS.val:
        with open(FLAGS.data_flist[FLAGS.dataset][1]) as f:
            for l in f.read().splitlines():
                val_fnames.extend(glob.glob(l, recursive=True))
        if FLAGS.guided:
            val_fnames = [
                (fname, fname[:-4] + '_edge.jpg') for fname in val_fnames]
        # progress monitor by visualizing static images
        for i in range(FLAGS.static_view_size):
            static_fnames = val_fnames[i:i+1]
            static_images = ng.data.DataFromFNames(
                static_fnames, img_shapes, nthreads=1,
                random_crop=FLAGS.random_crop).data_pipeline(1)
            static_inpainted_images = model.build_static_infer_graph(
                FLAGS, static_images, name='static_view/%d' % i)
    # training settings
    lr = tf.get_variable(
        'lr', shape=[], trainable=False,
        initializer=tf.constant_initializer(1e-4))
    d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
    g_optimizer = d_optimizer
    # train discriminator with secondary trainer, should initialize before
    # primary trainer.
    # discriminator_training_callback = ng.callbacks.SecondaryTrainer(
    discriminator_training_callback = ng.callbacks.SecondaryMultiGPUTrainer(
        num_gpus=FLAGS.num_gpus_per_job,
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=1,
        grads_summary=False,
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'd'},
    )
    # train generator with primary trainer
    # trainer = ng.train.Trainer(
    trainer = ng.train.MultiGPUTrainer(
        num_gpus=FLAGS.num_gpus_per_job,
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=FLAGS.max_iters,
        graph_def=multigpu_graph_def,
        grads_summary=False,
        gradient_processor=None,
        graph_def_kwargs={
            'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'g'},
        spe=FLAGS.train_spe,
        log_dir=log_dir,
    )
    # add all callbacks
    trainer.add_callbacks([
        discriminator_training_callback,
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix=FLAGS.model_restore+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(FLAGS.train_spe, trainer.context['saver'], FLAGS.log_dir+'/snap'),
        ng.callbacks.SummaryWriter((FLAGS.val_psteps//1), trainer.context['summary_writer'], tf.summary.merge_all()),
    ])
    # launch training
    trainer.train()
