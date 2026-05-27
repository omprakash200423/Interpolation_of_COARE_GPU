import cupy as cp
import time

# -----------------------------------
# GPU INFO
# -----------------------------------

print("CuPy Version:", cp.__version__)

gpu = cp.cuda.Device(0)
print("GPU Name:", gpu.attributes)

# -----------------------------------
# SIMPLE GPU ARRAY TEST
# -----------------------------------

N = 10_000_000

start = time.time()

x = cp.random.random(N, dtype=cp.float32)
y = cp.random.random(N, dtype=cp.float32)

z = x + y
cp.cuda.Stream.null.synchronize()

end = time.time()

print(f"\nGPU computation completed.")
print(f"Array size: {N}")
print(f"Execution Time: {end - start:.4f} seconds")