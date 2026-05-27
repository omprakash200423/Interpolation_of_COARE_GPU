import xarray as xr
import numpy as np

# -----------------------------------
# LOAD GPU OUTPUT
# -----------------------------------

ds = xr.open_dataset("../outputs/coare_gpu_output.nc")

print("\nDataset Summary:\n")
print(ds)

# -----------------------------------
# BASIC DIMENSIONS
# -----------------------------------

print("\nDimensions:")
print(ds.dims)

# -----------------------------------
# VARIABLES TO CHECK
# -----------------------------------

variables_to_check = [
    "tau",
    "hsb",
    "hlb",
    "Cd",
    "U10",
    "Evap"
]

# -----------------------------------
# CHECK STATISTICS
# -----------------------------------

print("\nVariable Statistics:\n")

for var in variables_to_check:

    data = ds[var].values

    print(f"{var}")
    print(f"  Shape : {data.shape}")
    print(f"  Min   : {np.nanmin(data):.6f}")
    print(f"  Max   : {np.nanmax(data):.6f}")
    print(f"  Mean  : {np.nanmean(data):.6f}")
    print(f"  NaNs  : {np.isnan(data).sum()}")
    print()