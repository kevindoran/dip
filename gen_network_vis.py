import pathlib
import xarray as xr
import dip.network_vis as net_vis
#ds = xr.Dataset(data_vars=layer_inputs)
import cProfile
import matplotlib.pyplot as plt

#pr = cProfile.Profile()
#pr.enable()
max_step = 100
dataset_outdir_pairs = (
#    ('./out/experiments/2/2/instrumentation.nc', './out/experiments/2/2/network_vis'),
#    ('./out/experiments/2/2/d_instrumentation.nc', './out/experiments/2/2/network_vis_gradient'),
    ('./out/experiments/2/2/step_instrumentation.nc',
        './out/experiments/2/2/network_vis_step'),
)

#ds = xr.open_dataset('./out/experiments/2/2/instrumentation.nc')
#net_vis.save_network_step_with_density(ds, 0, './out/experiments/2/2/network_vis', './out/experiments/2/2/network_vis/network_density.png')


for ds_path, out_dir in dataset_outdir_pairs:
    ds = xr.open_dataset(ds_path)
    net_vis.network_density_plot(ds, f'{out_dir}/network_density.png')
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True) 
    #net_vis.print_network(ds, out_dir, max_step)
#pr.disable()
#pr.print_stats(sort='cumtime')
