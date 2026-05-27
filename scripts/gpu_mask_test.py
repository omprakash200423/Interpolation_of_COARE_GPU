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
ocean_mask = ds["ocean_mask"].isel(valid_time=0).values

print("CPU Array Shape:", wind.shape)

# -----------------------------------
# CPU -> GPU TRANSFER
# -----------------------------------

start = time.time()

wind_gpu = cp.asarray(wind, dtype=cp.float32)
mask_gpu = cp.asarray(ocean_mask)

# -----------------------------------
# APPLY MASK ON GPU
# -----------------------------------

# -----------------------------------
# VALIDITY FILTERING
# -----------------------------------

valid_gpu = (
    cp.isfinite(wind_gpu)
    & (mask_gpu == 1)
)

# Apply filtering
filtered_gpu = cp.where(valid_gpu, wind_gpu, cp.nan)

# Synchronize GPU
cp.cuda.Stream.null.synchronize()

end = time.time()

# -----------------------------------
# RESULTS
# -----------------------------------

print("\nGPU masking completed.")
print("GPU Array Shape:", filtered_gpu.shape)
print(f"Execution Time: {end - start:.4f} seconds")

print("\nSample GPU Values:")
valid_values = cp.asnumpy(filtered_gpu)

print("\nNon-NaN Count:", np.count_nonzero(~np.isnan(valid_values)))

print("\nSample Valid Values:")
print(valid_values[300:305, 300:305])