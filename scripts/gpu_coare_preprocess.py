import xarray as xr
import cupy as cp
import numpy as np
import time

# -----------------------------------
# LOAD DATA
# -----------------------------------

ds = xr.open_dataset(
    "../data/interpolated/coare_ready_dataset.nc"
)

# -----------------------------------
# EXTRACT VARIABLES
# -----------------------------------

wind = ds["wind_speed"].isel(valid_time=0).values
tair = ds["t2m_c"].isel(valid_time=0).values
sst = ds["sst_c"].isel(valid_time=0).values
pressure = ds["sp_hpa"].isel(valid_time=0).values
rh = ds["rh"].isel(valid_time=0).values

ocean_mask = ds["ocean_mask"].isel(valid_time=0).values

# -----------------------------------
# CPU -> GPU
# -----------------------------------

start = time.time()

wind_gpu = cp.asarray(wind, dtype=cp.float32)
tair_gpu = cp.asarray(tair, dtype=cp.float32)
sst_gpu = cp.asarray(sst, dtype=cp.float32)
pressure_gpu = cp.asarray(pressure, dtype=cp.float32)
rh_gpu = cp.asarray(rh, dtype=cp.float32)

mask_gpu = cp.asarray(ocean_mask)

# -----------------------------------
# VALID FILTERING
# -----------------------------------

valid_gpu = (
    cp.isfinite(wind_gpu)
    & cp.isfinite(tair_gpu)
    & cp.isfinite(sst_gpu)
    & cp.isfinite(pressure_gpu)
    & cp.isfinite(rh_gpu)
    & (mask_gpu == 1)
)

# -----------------------------------
# EXTRACT VALID OCEAN POINTS
# -----------------------------------

u_gpu = wind_gpu[valid_gpu]
t_gpu = tair_gpu[valid_gpu]
ts_gpu = sst_gpu[valid_gpu]
p_gpu = pressure_gpu[valid_gpu]
rh_gpu_valid = rh_gpu[valid_gpu]

# -----------------------------------
# SYNCHRONIZE
# -----------------------------------

cp.cuda.Stream.null.synchronize()

end = time.time()

# -----------------------------------
# RESULTS
# -----------------------------------

print("\nGPU preprocessing completed.")

print("\nValid Ocean Points:", len(u_gpu))

print(f"\nExecution Time: {end - start:.4f} seconds")

print("\nSample Wind Values:")
print(cp.asnumpy(u_gpu[:10]))