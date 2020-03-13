import pathlib
import xarray as xr
import dip.network_vis as net_vis
#ds = xr.Dataset(data_vars=layer_inputs)
import cProfile
import matplotlib.pyplot as plt

#pr = cProfile.Profile()
#pr.enable()
max_step = 1000
DIR = './out/experiments/2/11'
dataset_outdir_pairs = (
    (f'{DIR}/instrumentation.nc', f'{DIR}/network_vis'),
    (f'{DIR}/d_instrumentation.nc', f'{DIR}/network_vis_gradient'),
    (f'{DIR}/step_instrumentation.nc', f'{DIR}/network_vis_step'),
)


for ds_path, out_dir in dataset_outdir_pairs:
    ds = xr.open_dataset(ds_path, cache=False)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True) 
    net_vis.print_network(ds, out_dir, max_step)
#pr.disable()
#pr.print_stats(sort='cumtime')
