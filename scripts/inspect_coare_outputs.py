import numpy as np

from coare36vn_zrf_era5 import coare36vn_zrf_et
from coare36vn_zrf_gpu import coare36vn_zrf_gpu

# -----------------------------------
# DUMMY TEST INPUTS
# -----------------------------------

N = 5

u = np.full(N, 5.0)
t = np.full(N, 28.0)
rh = np.full(N, 80.0)
P = np.full(N, 1010.0)
ts = np.full(N, 29.0)

sw_dn = np.full(N, 300.0)
lw_dn = np.full(N, 400.0)

lat = np.full(N, 15.0)
lon = np.full(N, 70.0)

jd = np.full(N, 150.5)

zu = 10.0
zt = 2.0
zq = 2.0

zi = 600.0
rain = 0.0
Ss = 35.0

# -----------------------------------
# RUN CPU VERSION
# -----------------------------------

cpu_result = coare36vn_zrf_et(
    u=u,
    zu=zu,
    t=t,
    zt=zt,
    rh=rh,
    zq=zq,
    P=P,
    ts=ts,
    sw_dn=sw_dn,
    lw_dn=lw_dn,
    lat=lat,
    lon=lon,
    jd=jd,
    zi=zi,
    rain=rain,
    Ss=Ss
)

# -----------------------------------
# RUN GPU VERSION
# -----------------------------------

gpu_result = coare36vn_zrf_gpu(
    u=u,
    zu=zu,
    t=t,
    zt=zt,
    rh=rh,
    zq=zq,
    P=P,
    ts=ts,
    sw_dn=sw_dn,
    lw_dn=lw_dn,
    lat=lat,
    lon=lon,
    jd=jd,
    zi=zi,
    rain=rain,
    Ss=Ss
)

# -----------------------------------
# PRINT SHAPES
# -----------------------------------

print("\nCPU OUTPUT SHAPE:")
print(cpu_result.shape)

print("\nGPU OUTPUT SHAPE:")
print(gpu_result.shape)

# -----------------------------------
# PRINT FIRST ROW
# -----------------------------------

print("\n================ CPU FIRST ROW ================\n")

for i, val in enumerate(cpu_result[0]):
    print(f"{i:02d} : {val}")

print("\n================ GPU FIRST ROW ================\n")

for i, val in enumerate(gpu_result[0]):
    print(f"{i:02d} : {val}")