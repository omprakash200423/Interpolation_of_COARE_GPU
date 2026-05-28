import xarray as xr
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import sys
import time
import warnings
sys.path.append("../coare")

# -----------------------------------
# IMPORT COARE
# -----------------------------------

from coare36vn_zrf_era5 import coare36vn_zrf_et

# -----------------------------------
# OPEN DATASET
# -----------------------------------

ds = xr.open_dataset(
    "../../data/interpolated/coare_ready_dataset.nc",
    chunks={
        "valid_time": 1,
        "latitude": 100,
        "longitude": 100
    }
)
ds = ds.isel(valid_time=slice(0,12))
ds = ds.compute()

# -----------------------------------
# CONVERT TO FLOAT32
# -----------------------------------

vars_needed = [
    "wind_speed",
    "t2m_c",
    "sst_c",
    "sp_hpa",
    "rh",
    "ssrd",
    "strd"
]

for var in vars_needed:
    ds[var] = ds[var].astype("float32")

# -----------------------------------
# APPLY OCEAN MASK
# -----------------------------------

ocean_mask = ds["ocean_mask"] == 1

for var in vars_needed:
    ds[var] = ds[var].where(ocean_mask)

# -----------------------------------
# CONSTANTS
# -----------------------------------

ZU = 10.0
ZT = 2.0
ZQ = 2.0

ZI = 600.0
RAIN = 0.0
SS = 35.0

# Radiation input mode:
# - "auto": infer from values (large values treated as accumulated J m^-2)
# - "accumulated_j_per_m2": convert to W m^-2 by dividing by TIME_STEP_SECONDS
# - "w_per_m2": use values as-is
RADIATION_INPUT_MODE = "auto"
TIME_STEP_SECONDS = 3600.0


def sanitize_coare_outputs(result):
    """Set non-physical COARE outputs to NaN (result shape: [N, 50])."""
    if result.size == 0:
        return result

    # 1-based -> 0-based indices
    i_zo = 9
    i_zot = 10
    i_zoq = 11
    i_u_component = [20, 23, 24, 31, 32, 33]  # Urf, UrfN, UN, U10, U10N, etc.
    i_rh = [23, 44]  # RHrf, RH10
    i_q = [22, 26, 38, 42, 43]  # Qrf, QrfN, Qs, Q10, Q10N
    i_p10 = 45

    result[:, i_zo] = np.where(result[:, i_zo] > 0.0, result[:, i_zo], np.nan)
    result[:, i_zot] = np.where(result[:, i_zot] > 0.0, result[:, i_zot], np.nan)
    result[:, i_zoq] = np.where(result[:, i_zoq] > 0.0, result[:, i_zoq], np.nan)

    for idx in i_u_component:
        result[:, idx] = np.where(result[:, idx] >= 0.0, result[:, idx], np.nan)

    for idx in i_rh:
        v = result[:, idx]
        result[:, idx] = np.where((v >= 0.0) & (v <= 100.0), v, np.nan)

    for idx in i_q:
        result[:, idx] = np.where(result[:, idx] >= 0.0, result[:, idx], np.nan)

    p10 = result[:, i_p10]
    result[:, i_p10] = np.where((p10 >= 800.0) & (p10 <= 1100.0), p10, np.nan)

    return result

# -----------------------------------
# PROCESS SINGLE TIME STEP
# -----------------------------------

def process_timestep(t_index):

    print(f"\nProcessing timestep: {t_index}")

    wind = ds["wind_speed"].isel(valid_time=t_index).values
    tair = ds["t2m_c"].isel(valid_time=t_index).values
    sst = ds["sst_c"].isel(valid_time=t_index).values
    pressure = ds["sp_hpa"].isel(valid_time=t_index).values
    rh = ds["rh"].isel(valid_time=t_index).values
    sw = ds["ssrd"].isel(valid_time=t_index).values
    lw = ds["strd"].isel(valid_time=t_index).values

    valid_time_value = ds["valid_time"].values[t_index]

    lat2d, lon2d = np.meshgrid(
        ds["latitude"].values,
        ds["longitude"].values,
        indexing="ij"
    )

    # -----------------------------------
    # VALID OCEAN POINTS
    # -----------------------------------

    valid = (
        np.isfinite(wind)
        & np.isfinite(tair)
        & np.isfinite(sst)
        & np.isfinite(pressure)
        & np.isfinite(rh)
        & np.isfinite(sw)
        & np.isfinite(lw)
    )

    # Flatten valid points
    u = wind[valid]
    u = np.maximum(u, 0.2)
    t = tair[valid]
    ts = sst[valid]
    temp_valid = ts >= -2.0
    u = u[temp_valid]
    t = t[temp_valid]
    ts = ts[temp_valid]
    P = pressure[valid]
    RH = rh[valid]
    SW = sw[valid]
    LW = lw[valid]
    P = P[temp_valid]
    RH = RH[temp_valid]
    SW = SW[temp_valid]
    LW = LW[temp_valid]
    lat_valid = lat2d[valid][temp_valid]
    lon_valid = lon2d[valid][temp_valid]

    # Physical screening on inputs prior to COARE call.
    RH = np.clip(RH, 0.0, 100.0)
    in_phys = (
        (P >= 800.0) & (P <= 1100.0)
        & (t >= -80.0) & (t <= 60.0)
        & (ts >= -2.0) & (ts <= 45.0)
        & np.isfinite(u) & np.isfinite(t) & np.isfinite(ts)
        & np.isfinite(P) & np.isfinite(RH)
        & np.isfinite(SW) & np.isfinite(LW)
    )
    u = u[in_phys]
    t = t[in_phys]
    ts = ts[in_phys]
    P = P[in_phys]
    RH = RH[in_phys]
    SW = SW[in_phys]
    LW = LW[in_phys]
    lat_valid = lat_valid[in_phys]
    lon_valid = lon_valid[in_phys]

    if RADIATION_INPUT_MODE == "accumulated_j_per_m2":
        SW = SW / TIME_STEP_SECONDS
        LW = LW / TIME_STEP_SECONDS
    elif RADIATION_INPUT_MODE == "auto":
        # Typical fluxes are O(10^0-10^3) W m^-2; much larger values are likely accumulated J m^-2.
        if np.nanmedian(SW) > 1500 or np.nanmedian(LW) > 1500:
            SW = SW / TIME_STEP_SECONDS
            LW = LW / TIME_STEP_SECONDS

    SW = np.maximum(SW, 0.0)
    LW = np.maximum(LW, 0.0)

    LAT = lat_valid.astype(np.float32)
    LON = lon_valid.astype(np.float32)

    dt64 = np.datetime64(valid_time_value)
    year_start = dt64.astype("datetime64[Y]")
    day_part = (dt64.astype("datetime64[D]") - year_start).astype("timedelta64[D]").astype(np.int64)
    hour_part = (dt64 - dt64.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.float64) / 86400.0
    jd_scalar = float(day_part + hour_part)
    JD = np.full(u.shape, jd_scalar, dtype=np.float32)

    # -----------------------------------
    # RUN VECTORIZED COARE
    # -----------------------------------

        # -----------------------------------
    # RUN COARE IN BATCHES
    # -----------------------------------

    batch_size = 50000

    results_list = []

    total_points = len(u)
    if total_points == 0:
        return [np.full(wind.shape, np.nan, dtype=np.float32) for _ in range(50)]

    for start in range(0, total_points, batch_size):

        end = start + batch_size

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log")
            result_batch = coare36vn_zrf_et(
                u=u[start:end],
                zu=ZU,
                t=t[start:end],
                zt=ZT,
                rh=RH[start:end],
                zq=ZQ,
                P=P[start:end],
                ts=ts[start:end],
                sw_dn=SW[start:end],
                lw_dn=LW[start:end],
                lat=LAT[start:end],
                lon=LON[start:end],
                jd=JD[start:end],
                zi=ZI,
                rain=RAIN,
                Ss=SS
            )

        results_list.append(result_batch)

    result = np.vstack(results_list)
    result = sanitize_coare_outputs(result)
    
    # -----------------------------------
    # OUTPUT VARIABLES
    # -----------------------------------

        # -----------------------------------
    # STORE ALL 50 VARIABLES
    # -----------------------------------

    output_grids = []
    valid_indices = np.where(valid)
    valid_i = valid_indices[0][temp_valid][in_phys]
    valid_j = valid_indices[1][temp_valid][in_phys]

    for i in range(50):

        grid = np.full(wind.shape, np.nan, dtype=np.float32)
        grid[valid_i, valid_j] = result[:, i]

        output_grids.append(grid)

    return output_grids

# -----------------------------------
# MAIN EXECUTION
# -----------------------------------

if __name__ == "__main__":

    start_time = time.time()

    nt = ds.sizes["valid_time"]

    all_outputs = [[] for _ in range(50)]

    with ProcessPoolExecutor(max_workers=4) as executor:

        results = []

        for result in tqdm(
            executor.map(process_timestep, range(nt)),
            total=nt
        ):
            results.append(result)

    for timestep_output in results:

        for i in range(50):

            all_outputs[i].append(timestep_output[i])

    stacked_outputs = []

    for i in range(50):

        stacked_outputs.append(
            np.stack(all_outputs[i])
        )

    # -----------------------------------
    # SAVE OUTPUT
    # -----------------------------------
    var_names = [
        "usr","tau","hsb","hlb","hbb","hsbb","hlwebb","tsr","qsr",
        "zo","zot","zoq","Cd","Ch","Ce","L","zeta","dT_skinx",
        "dq_skinx","dz_skin","Urf","Trf","Qrf","RHrf","UrfN",
        "TrfN","QrfN","lw_net","sw_net","Le","rhoa","UN","U10",
        "U10N","Cdn_10","Chn_10","Cen_10","hrain","Qs",
        "Evap","T10","T10N","Q10","Q10N","RH10","P10",
        "rhoa10","gust","wc_frac","Edis"
    ]
    data_vars = {}

    for i in range(50):

        data_vars[var_names[i]] = (
            ("valid_time", "latitude", "longitude"),
            stacked_outputs[i]
        )

    out_ds = xr.Dataset(

        data_vars,

        coords={
            "valid_time": ds["valid_time"][:nt],
            "latitude": ds["latitude"],
            "longitude": ds["longitude"]
        }
    )
    out_ds.to_netcdf(
        "../../outputs/coare_cpu_output.nc"
    )
    print("\nSUCCESS: COARE processing completed.")
    end_time = time.time()

    total_time = end_time - start_time

    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    print(f"Total Execution Time: {total_time/60:.2f} minutes")
