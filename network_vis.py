import pathlib

import xarray as xr

import dip.network_vis as net_vis 

ds = xr.open_dataset('./out/experiments/2/2/instrumentation.nc')
out_dir = './out/experiments/2/2/network_vis2'
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True) 
net_vis.print_network(ds, out_dir)
