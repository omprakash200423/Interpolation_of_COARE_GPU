import xarray as xr
import cupy as cp
import numpy as np
import time

from coare36vn_zrf_gpu import coare36vn_zrf_et

# -----------------------------------
# LOAD DATA
# -----------------------------------

ds = xr.open_dataset(
    "../data/interpolated/coare_ready_dataset.nc"
)
print(dict(ds.sizes))
all_results = []
start = time.time()

# -----------------------------------
# PREALLOCATE CONSTANT GPU ARRAYS
# -----------------------------------

lat2d, lon2d = np.meshgrid(
    ds["latitude"].values,
    ds["longitude"].values,
    indexing="ij"
)

# -----------------------------------
# SELECT ONE TIMESTEP
# -----------------------------------
for t_idx in range(ds.sizes["valid_time"]):
    print(f"\nProcessing timestep {t_idx+1}/12")
    wind = ds["wind_speed"].isel(valid_time=t_idx).values
    tair = ds["t2m_c"].isel(valid_time=t_idx).values
    sst = ds["sst_c"].isel(valid_time=t_idx).values
    pressure = ds["sp_hpa"].isel(valid_time=t_idx).values
    rh = ds["rh"].isel(valid_time=t_idx).values

    sw = ds["ssrd"].isel(valid_time=t_idx).values
    lw = ds["strd"].isel(valid_time=t_idx).values


    ocean_mask = ds["ocean_mask"].isel(valid_time=t_idx).values
    current_time = np.datetime64(
    ds["valid_time"].values[t_idx]
    )

    year_start = current_time.astype("datetime64[Y]")

    jd_value = (
        current_time.astype("datetime64[D]") - year_start
    ).astype(int) + 1

# -----------------------------------
# FILTER VALID OCEAN POINTS
# -----------------------------------

    valid = (
        np.isfinite(wind)
        & np.isfinite(tair)
        & np.isfinite(sst)
        & np.isfinite(pressure)
        & np.isfinite(rh)
        & (ocean_mask == 1)
    )

    u = wind[valid]
    lat_valid = lat2d[valid]
    lon_valid = lon2d[valid]
    t = tair[valid]
    ts = sst[valid]
    P = pressure[valid]
    RH = rh[valid]

    sw_dn = sw[valid]
    lw_dn = lw[valid]

    # -----------------------------------
    # GPU TRANSFER (dynamic variables)
    # -----------------------------------

    u = cp.asarray(u)
    t = cp.asarray(t)
    ts = cp.asarray(ts)
    P = cp.asarray(P)
    RH = cp.asarray(RH)

    sw_dn = cp.asarray(sw_dn)
    lw_dn = cp.asarray(lw_dn)
    jd = cp.full(
        u.shape[0],
        float(jd_value)
    )
    zu = cp.full(u.shape[0], 10.0)
    zt = cp.full(u.shape[0], 2.0)
    zq = cp.full(u.shape[0], 2.0)

    zi = cp.full(u.shape[0], 600.0)

    rain = cp.zeros(u.shape[0])

    Ss = cp.full(u.shape[0], 35.0)
    lat_flat = cp.asarray(lat_valid)
    lon_flat = cp.asarray(lon_valid)

    # -----------------------------------
    # RUN GPU COARE
    # -----------------------------------

    print("\nStarting GPU COARE test...\n")
    
    # u = cp.asnumpy(u)
    # t = cp.asnumpy(t)
    # ts = cp.asnumpy(ts)
    # P = cp.asnumpy(P)
    # RH = cp.asnumpy(RH)

    # sw_dn = cp.asnumpy(sw_dn)
    # lw_dn = cp.asnumpy(lw_dn)

    # zu = cp.asnumpy(zu)
    # zt = cp.asnumpy(zt)
    # zq = cp.asnumpy(zq)

    # zi = cp.asnumpy(zi)

    # rain = cp.asnumpy(rain)
    # Ss = cp.asnumpy(Ss)

    result = coare36vn_zrf_et(
        u, zu,
        t, zt,
        RH, zq,
        P, ts,
        sw_dn, lw_dn,
        lat_flat, lon_flat,
        jd, zi,
        rain, Ss
    )
    all_results.append(cp.asnumpy(result))
output_names = [
    "usr","tau","hsb","hlb","hbb","hsbb","hlwebb",
    "tsr","qsr","zo","zot","zoq","Cd","Ch","Ce",
    "L","zeta","dT_skinx","dq_skinx","dz_skin",
    "Urf","Trf","Qrf","RHrf","UrfN","TrfN","QrfN",
    "lw_net","sw_net","Le","rhoa","UN","U10","U10N",
    "Cdn_10","Chn_10","Cen_10","hrain","Qs","Evap",
    "T10","T10N","Q10","Q10N","RH10","P10",
    "rhoa10","gust","wc_frac","Edis"
]
# -----------------------------------
# STACK RESULTS
# -----------------------------------

all_results = np.stack(all_results)

# Shape:
# (time, ocean_points, 50)

# -----------------------------------
# CREATE OUTPUT DATASET
# -----------------------------------

output_ds = xr.Dataset()

lat_size = ds.sizes["latitude"]
lon_size = ds.sizes["longitude"]
time_size = ds.sizes["valid_time"]

# Rebuild full ERA5 grids
for i, name in enumerate(output_names):

    full_grid = np.full(
        (time_size, lat_size, lon_size),
        np.nan,
        dtype=np.float32
    )

    for t_idx in range(time_size):

        ocean_mask = (
            ds["ocean_mask"]
            .isel(valid_time=t_idx)
            .values == 1
        )

        full_grid[t_idx][ocean_mask] = all_results[t_idx, i, :]

    output_ds[name] = (
        ("valid_time", "latitude", "longitude"),
        full_grid
    )

# -----------------------------------
# ASSIGN COORDINATES
# -----------------------------------

output_ds = output_ds.assign_coords(
    valid_time=ds["valid_time"],
    latitude=ds["latitude"],
    longitude=ds["longitude"]
)

# -----------------------------------
# SAVE NETCDF
# -----------------------------------

encoding = {
    var: {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32"
    }
    for var in output_ds.data_vars
}

output_path = "../outputs/coare_gpu_output.nc"

output_ds.to_netcdf(
    output_path,
    encoding=encoding
)

print(f"\nSaved GPU COARE output to: {output_path}")

cp.cuda.Stream.null.synchronize()
free_mem, total_mem = cp.cuda.runtime.memGetInfo()

print(f"GPU Free Memory : {free_mem / 1024**3:.2f} GB")
print(f"GPU Total Memory: {total_mem / 1024**3:.2f} GB")
end = time.time()

print(f"\nExecution Time: {end - start:.2f} seconds")
print("\nGPU COARE completed.")
print("All Results Shape:", all_results.shape)

print("\nSample Output:")
print(cp.asnumpy(result[:5, :5]))