import time
import subprocess

start = time.time()

subprocess.run(
    ["python", "../cpu_reference/coare_cpu_optimized.py"],
    check=True
)

end = time.time()

print(f"\nTotal Runtime: {end - start:.2f} seconds")