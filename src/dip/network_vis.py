import re
from typing import List
import tempfile

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tensorflow as tf
import cv2
import joypy


class InstrumentationData:

    class Layer:
        """Specifies which layer information to record. Also stores this data.
        """
        def __init__(self, layer_in, kernel, bias):
            """
            Args:
                layer_in (tf.Tensor): the input tensor to a network layer.
                kernel (tf.Variable): a convolution kernel variable.
                bias (tf.Variable): a convolution bias variable.
            """
            # Warn that only the first batch element will be used.
            if layer_in is not None:
                if layer_in.shape[0] != 1:
                    logging.warning(f'For the layer ({layer_in.name}), only '
                            'the first entry of the batch will be recorded. '
                            f'The other ({layer_in.shape[0]}) entries will be '
                            'ignored.')
            self.layer_in = layer_in
            self.kernel = kernel
            self.bias = bias
            # The network evaluated values:
            self.layer_in_data = []
            self.kernel_data = []
            self.bias_data = []

        def non_none_vars(self):
            for x in (self.layer_in, self.kernel, self.bias):
                if x is not None:
                    yield x

        def eval_count(self):
            return len(list(self.non_none_vars()))

        def add_values(self, eval_results):
            if len(eval_results) != self.eval_count():
                raise Exception(f'This layer record requires exactly '
                        f'{self.eval_count()} elements, but received '
                        f'{len(eval_results)}.')
            idx = 0
            if self.layer_in is not None:
                # Remove batch dimension.
                # Remove the batch dimension here, rather than when storing
                # the layer_in tensor, as if the latter is done, a new
                # object "Slice" is made, and we don't have access to the
                # original object's gradient information.
                self.layer_in_data.append(eval_results[idx][0])
                idx += 1
            if self.kernel is not None:
                self.kernel_data.append(eval_results[idx])
                idx += 1
            if self.bias is not None:
                self.bias_data.append(eval_results[idx])
                idx += 1

    def as_derivative(self, wrt_tensor):
        derivative_layers = []
        for l in self.layers:
            d_args = map(
                lambda x: None if x is None else tf.gradients(wrt_tensor, x)[0],
                [l.layer_in, l.kernel, l.bias])
            derivative_layers.append(self.Layer(*d_args))
        d_instrumentation = InstrumentationData(derivative_layers)
        return d_instrumentation

    def __init___old(self, model_layer_vars):
        # Remove batch dimension
        layers_sans_batch = [v[0] for v in model_layer_vars]
        self.model_layer_vars = layers_sans_batch
        self.num_inner_layers = len(model_layer_vars) - 1
        self._out_img = []
        self._layer_inputs = [[] for i in range(self.num_inner_layers)]
        self._kernels = [[] for i in range(self.num_inner_layers)]
        self._index = []

    def __init__(self, layers: List[Layer]):
        """Constructs an InstrumentationData object.

        Args:
            layer_specs: describes which layers and what layer data to
                record.
        """
        self.layers = layers

    def eval_list(self):
        res = []
        for l in self.layers:
            res.extend(l.non_none_vars())
        return res

    def eval_and_record(self, sess, step):
        res = sess.run(self.eval_list())
        from_i = 0
        for l in self.layers:
            to_i = from_i + l.eval_count()
            result_slice_for_layer = res[from_i:to_i]
            l.add_values(result_slice_for_layer)
            from_i = to_i
        assert to_i == from_i == len(res), 'There should be no shortage or \
            spare tensors being evaluated.'
        return res

    def _dataset_attrs(self):
        return {'num_layers': len(self.layers)}

    def to_xdataset(self):
        # Todo: store real layer name in a metadata.
        network_data = {}
        for i, l in enumerate(self.layers):
            if l.layer_in_data is not None:
                network_data[layer_in_key(i)] = xr.DataArray(
                        np.asarray(l.layer_in_data),
                        dims=('step', *layer_input_dims(i)))
            if l.kernel is not None:
                kernel_data = np.asarray(l.kernel_data)
                # Kernel data is store with dimension order:
                # (step, kernel, y, x, channel)
                # Move the kernel dimension to be next after step.
                kernel_data = np.transpose(kernel_data, axes=(0, 4, 1, 2, 3))
                network_data[kernel_key(i)] = xr.DataArray(
                        kernel_data,
                        dims=('step', *kernel_dims(i)))
            if l.bias is not None:
                # TODO
                pass
        ds = xr.Dataset(network_data, attrs=self._dataset_attrs())
        return ds

    def eval_and_add(self, sess, step):
        eval_list = []

        # Add inner layers to eval list.
        inner_layers = self.model_layer_vars[:-1]
        out_layer = self.model_layer_vars[-1]
        eval_list.extend(inner_layers)

        # Add kernels to eval list.
        kernel_name_pattern = r'(conv\d+)/kernel:0'
        kernel_vars = []
        for v in tf.trainable_variables():
            m = re.match(kernel_name_pattern, v.name)
            if m:
                # Making the assumption here that the variables are in
                # order of progression from beginning to end of network.
                kernel_vars.append(v)
        if len(kernel_vars) != len(inner_layers):
            raise Exception('Unexpected number of kernels found. Expected '
                    f'{self.num_inner_layers}, but got {len(kernel_vars)}.')
        eval_list.extend(kernel_vars)

        # Add output layer to eval list.
        eval_list.append(out_layer)

        # Evaluate tensors.
        res = sess.run(eval_list)
        layer_input_vals = res[:len(inner_layers)]
        kernel_vals = res[len(inner_layers):-1]
        out_img_val = res[-1]

        # Add data to the storage lists.
        self.add(step, layer_input_vals, kernel_vals, out_img_val)
        return out_img_val

    def add(self, step : int, layer_inputs, kernels, out_img):
        """Add data for a single step."""
        if len(layer_inputs) != len(kernels) != len(self.num_inner_layers):
            raise ValueError('Incorrect data length: '
                             f'({len(layer_inputs)}, {len(kernels)}')
        for i in range(self.num_inner_layers):
            self._kernels[i].append(kernels[i])
            self._layer_inputs[i].append(layer_inputs[i])
        self._index.append(step)
        self._out_img.append(out_img)

    def to_xdataset_old(self):
        layer_data = {}
        for i in range(self.num_inner_layers):
            layer_data[f'l{i}'] = xr.DataArray(
                    np.asarray(self._layer_inputs[i]),
                    dims=('step', f'l{i}_y', f'l{i}_x', f'l{i}_c'))
            # Move the kernel dimension to be next after step.
            kernel_data = np.asarray(self._kernels[i])
            kernel_data = np.transpose(kernel_data, axes=(0, 4, 1, 2, 3))
            layer_data[f'k{i}'] = xr.DataArray(kernel_data,
                    dims=('step', f'k{i}_k', f'k{i}_y', f'k{i}_x', f'l{i}_c'))
        layer_data['out'] = xr.DataArray(self._out_img,
                    dims=('step', 'out_x', 'out_y', 'out_c'))
        ds = xr.Dataset(data_vars=layer_data)
        return ds


def layer_in_key(layer_idx):
    return f'layer_{layer_idx}_input'


def kernel_key(layer_idx):
    return f'layer_{layer_idx}_kernel'


def layer_in_keys(dataset, exclude=None):
    exclude = set(exclude) if exclude else set()
    keys = []
    for l in range(num_layers(dataset)):
        if l not in exclude and layer_in_key(l) in dataset:
            keys.append(layer_in_key(l))
    return keys


def kernel_keys(dataset, exclude=None):
    exclude = set(exclude) if exclude else set()
    keys = []
    for l in range(num_layers(dataset)):
        if l not in exclude and kernel_key(l) in dataset:
            keys.append(kernel_key(l))
    return keys


def kernel_dim(layer_idx):
    return f'layer_{layer_idx}_kernel_d'


def channel_dim(layer_idx):
    return f'l{layer_idx}_c'


def bias_key(layer_idx):
    return f'layer_{layer_idx}_bias'


def layer_input_dims(layer_idx):
    return (f'l{layer_idx}_y', f'l{layer_idx}_x',
            channel_dim(layer_idx))


def kernel_dims(layer_idx):
    return (kernel_dim(layer_idx), f'l{layer_idx}k_y',
            f'l{layer_idx}k_x', channel_dim(layer_idx))


def num_layers(dataset):
    return dataset.attrs['num_layers']


def num_channels(dataset, layer : int):
    """Determines the number of channels in the given layer.

    The channel count for a layeris considered to be number of channels in
    the input tensor and thus the number of depth dimensions of each
    following kernel.
    """
    # TODO: what if the input is not saved?
    ans = dataset[layer_in_key(layer)].sizes[channel_dim(layer)]
    return ans


def num_kernels(dataset, layer : int):
    if kernel_key(layer) in dataset:
        ans = dataset[kernel_key(layer)].sizes[kernel_dim(layer)]
    else:
        ans = 0
    return ans


def min_max_values(dataset, min_quantile=0.05, max_quantile=0.95, 
        single_layer_val=True, single_kernel_val=True):
    layer_min_max = {}
    _num_layers = num_layers(dataset)
    # Input image is assumed to be in range (-1, 1)
    layer_min_max[layer_in_key(0)] = (-1.0, 1.0)
    # The output isn't constrained to (-1, 1), however, using any other
    # limits would result in an output map that might not match the color
    # range of the input image. So, use the same limits as the input.
    layer_min_max[layer_in_key(_num_layers-1)] = (-1.0, 1.0)
    # All other channels are given a shared (min,max)
    ds_min = dataset.quantile(min_quantile)
    ds_max = dataset.quantile(max_quantile)
    other_layers = layer_in_keys(dataset, exclude=[0, _num_layers-1])
    l_min = min([ds_min[l] for l in other_layers])
    l_max = max([ds_max[l] for l in other_layers])
    for l in other_layers:
        if single_layer_val:
            abs_max = max(abs(l_min), abs(l_max))
        else:
            abs_max = max(abs(ds_min[l]), abs(ds_max[l]))
        layer_min_max[l] = (-abs_max, abs_max)
    # Treat kernels like the inner layers: all share common min-max.
    kernel_min_max = {}
    _kernel_keys = kernel_keys(dataset)
    k_min = min([ds_min[k] for k in _kernel_keys])
    k_max = max([ds_max[k] for k in _kernel_keys])
    for k in _kernel_keys:
        if single_kernel_val:
            abs_max = max(abs(k_min), abs(k_max))
        else:
            abs_max = max(abs(ds_min[k]), abs(ds_min[k]))
        kernel_min_max[k] = (-abs_max, abs_max)
    return layer_min_max, kernel_min_max


def density_plot(data_dict, bins=20, overlap=3.0, figsize=(6,6)):
    vmax = max(max(arr) for arr in data_dict.values()) * 1.1
    vmin = min(min(arr) for arr in data_dict.values()) * 1.1
    fig, axes = joypy.joyplot(
	data_dict, 
	figsize=figsize,
	colormap=matplotlib.cm.Pastel1,
	#hist=True,
	kind='normalized_counts',
	bins=bins,
	range_style='own',
	linewidth=1,
	ylim='max', # allow direct comparison. Share same y-limit between plots.
	overlap=overlap,
	grid='both',
	x_range=(vmin, vmax))    
    return fig

def kernel_density_plot(dataset, use_axes=None):
    kernel_data = {l : dataset[l].values.flatten() for l in
            kernel_keys(dataset)}
    return density_plot(kernel_data)


def layer_in_density_plot(dataset, use_axes=None):
    layer_data = {l : dataset[l].values.flatten() for l in
            layer_in_keys(dataset)}
    return density_plot(layer_data)


def fig_to_data(fig):
    canvas = FigureCanvasAgg(fig)
    fig.canvas.draw()
    s, (width, height) = fig.canvas.print_to_buffer()
    #width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    return img


def network_density_plot(dataset, out_path, fig=None):
    if not fig:
        fig = plt.figure(figsize=(6, 6))
    with tempfile.TemporaryDirectory() as temp_dir:
        k_fig = kernel_density_plot(dataset, fig)
        k_path = f'{temp_dir}/k.png'
        l_path = f'{temp_dir}/l.png'
        plt.savefig(k_path)
        plt.close(fig)
        l_fig = layer_in_density_plot(dataset, fig)
        plt.savefig(l_path)
        plt.close(fig)
        k_img = cv2.imread(k_path)
        l_img = cv2.imread(l_path)
        combined = np.concatenate((k_img, l_img), axis=0)
    cv2.imwrite(out_path, combined)

    #k_img = fig_to_data(kernel_density_plot(dataset, fig))
    #l_img = fig_to_data(layer_in_density_plot(dataset, fig))
    #combined_img = np.vstack([k_img, l_img])
    # Might need to reshuffle BGR -> RGB
    #cv2.imwrite(out_path, combined_img)


def print_network_step(dataset, step, network_name=None, fig=None):
    if not network_name:
        network_name = f'Network'
    step_data = dataset.sel(step=step)
    # calculate num_layers. A little hacky...
    _num_layers = num_layers(dataset)
    #layer_min_max, kernel_min_max = min_max_values(dataset)
    layer_min_max, kernel_min_max = min_max_values(dataset, 
            single_layer_val=True, single_kernel_val=True)
            # Have separate color ranges for each kernel and layer?
            #single_layer_val=False, single_kernel_val=False)

    if not fig:
        fig = plt.figure(figsize=(24*4, 12*4))
    fig.suptitle(f'{network_name} (step {step})')
    '''
    +----+----+ ... +
    | l1 | l2 |  ln |
    +----+----+-...-+
    l1 is input image.
    ln is output image.
    '''
    gs = gridspec.GridSpec(nrows=1, ncols=_num_layers, figure=fig)
    for l in range(_num_layers):
        _num_channels = num_channels(dataset, l)
        '''
                         i   k
        +----+         +---+---+
        | l1 |  ->  c1 |   |   |
        +----+         +-------+
                    c2 |   |   |
                       +-------+
                       .   .   .
                       +-------+
                    cn |   |   |
                       +-------+
        i: layer input
        k: kernel
        '''
        gs_channel  = gs[l].subgridspec(nrows=_num_channels, ncols=2)
        '''
                          i       kernels
        +-------+      +----+  |  +----+
        | i | k |  ->  | c1 |  |  | c1 |
        +---+---+      +----+  |  +----+
                       | c2 |  |  | c2 |
                       +----+  |  +----+
                       .    .  |  .    .
        '''
        lmin, lmax = layer_min_max[layer_in_key(l)]
        layer_in_data = step_data[layer_in_key(l)]
        layer_norm_range = matplotlib.colors.Normalize(vmin=lmin, vmax=lmax,
                clip=True)
        _num_kernels = num_kernels(dataset, l)
        if _num_kernels:
            kmin, kmax = kernel_min_max[kernel_key(l)]
            kernel_norm_range = matplotlib.colors.Normalize(vmin=kmin, 
                    vmax=kmax, clip=True)
        for c in range(_num_channels):
            in_ax = fig.add_subplot(gs_channel[c, 0], 
                    xticks=[], yticks=[], xticklabels=[], yticklabels=[],
                    frame_on=False, autoscale_on=False) 
            in_ax.set_axis_off()
            c_slice = dict(((channel_dim(l), c),))
            channel_data = layer_in_data.loc[c_slice]
            in_ax.imshow(channel_data, interpolation='nearest', cmap='gray',
                    norm=layer_norm_range)

            '''
              kernels             k1   k2         kn
              +----+       +----+----+----+ ... +----+
              | c? |       | c? |    |    |     |    |
              +----+  ->   +----+----+----+     +----+
            '''
            if not _num_kernels:
                continue
            gs_kernel = gs_channel[c, 1].subgridspec(nrows=1, ncols=_num_kernels)
            all_kernel_data = step_data[kernel_key(l)]
            for k in range(_num_kernels):
                kernel_ax = fig.add_subplot(gs_kernel[k],
                    xticks=[], yticks=[], xticklabels=[], yticklabels=[],
                    frame_on=False, autoscale_on=False) 
                kernel_ax.set_axis_off()
                k_slice = dict(((kernel_dim(l), k),))
                c_slice = dict(((channel_dim(l), c),))
                kernel_data_by_channel = all_kernel_data.loc[k_slice].loc[c_slice]
                is_last = l == _num_layers - 2
                kernel_ax.imshow(kernel_data_by_channel, 
                        interpolation='nearest', 
                        norm=kernel_norm_range, cmap='PiYG')
    return fig


def save_network_step(dataset, step, out_file_dir, fig=None):
    out_path = f'{out_file_dir}/{step}.png'
    fig = print_network_step(dataset, step, fig=fig)
    plt.savefig(out_path, #bbox_inches='tight', dpi=20, pad_inches=1.0,
            facecolor=(0,0,0))
    plt.close(fig)
    return out_path


def save_network_step_with_density(dataset, step, out_dir,
        density_file_path, fig=None):
    img_path = save_network_step(dataset, step, out_dir, fig)
    network_img  = cv2.imread(img_path)
    density_img = cv2.imread(density_file_path)
    # 3 or 4?
    print(density_img.shape)
    corrected_density_shape = (network_img.shape[0], density_img.shape[1], 3)
    density_img_temp = np.full(corrected_density_shape, 255)
    margin_top = (network_img.shape[0] - density_img.shape[0]) // 2 
    end_img_pos = margin_top + density_img.shape[0]
    density_img_temp[margin_top:end_img_pos,:, :] = density_img
    combined = np.concatenate((network_img, density_img_temp), axis=1)
    cv2.imwrite(img_path, combined)


def print_network(dataset, out_file_dir, max_steps=None):
    max_steps_in = max_steps
    max_steps  = len(dataset['step'])
    if max_steps_in:
        max_steps = min(max_steps, max_steps_in)

    density_path = f'{out_file_dir}/density.png'
    #network_density_plot(dataset, density_path)
    import multiprocessing
    import functools
    #for i in range(max_steps):
        #save_network_step(dataset, i, out_file_dir=out_file_dir)
    with multiprocessing.Pool() as pool:
        pool.map(functools.partial(save_network_step, dataset, 
                                   out_file_dir=out_file_dir),
                 range(0, max_steps))
    plt.close()
