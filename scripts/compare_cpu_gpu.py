import xarray as xr
import numpy as np

# -----------------------------------
# LOAD DATASETS
# -----------------------------------

cpu = xr.open_dataset("../outputs/coare_cpu_output.nc")
gpu = xr.open_dataset("../outputs/coare_gpu_output.nc")

# -----------------------------------
# VARIABLES
# -----------------------------------

variables = list(cpu.data_vars)

print("\nCPU vs GPU COMPARISON\n")

# -----------------------------------
# COMPARE
# -----------------------------------

for var in variables:

    cpu_data = cpu[var].values
    gpu_data = gpu[var].values

    # Valid comparison mask
    valid = np.isfinite(cpu_data) & np.isfinite(gpu_data)

    if np.sum(valid) == 0:
        print(f"{var:10s} : No valid points")
        continue

    diff = cpu_data[valid] - gpu_data[valid]

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_err = np.max(np.abs(diff))

    cpu_nan = np.isnan(cpu_data).sum()
    gpu_nan = np.isnan(gpu_data).sum()

    print(
        f"{var:10s} | "
        f"MAE={mae:.6e} | "
        f"RMSE={rmse:.6e} | "
        f"MAX={max_err:.6e} | "
        f"CPU_NaN={cpu_nan} | "
        f"GPU_NaN={gpu_nan}"
    )
# -----------------------------------
# SW_NET SAMPLE CHECK
# -----------------------------------

cpu_sw = cpu["sw_net"].values
gpu_sw = gpu["sw_net"].values

valid = np.isfinite(cpu_sw) & np.isfinite(gpu_sw)

cpu_vals = cpu_sw[valid]

gpu_vals = gpu_sw[valid]

idx = np.where(np.abs(cpu_vals) > 100)[0][:10]

cpu_vals = cpu_vals[idx]
gpu_vals = gpu_vals[idx]

print("\nSW_NET SAMPLE CHECK\n")

for i in range(len(cpu_vals)):
    print(
        f"{i}: CPU={cpu_vals[i]:.6f} "
        f"GPU={gpu_vals[i]:.6f} "
        f"DIFF={cpu_vals[i]-gpu_vals[i]:.6f}"
    )

print("\nComparison completed.")
