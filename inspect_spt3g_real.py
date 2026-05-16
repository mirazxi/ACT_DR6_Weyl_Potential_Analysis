import numpy as np
from pathlib import Path
base = Path(".")
print("Current folder:", base.resolve())
# Ignore macOS metadata and notebook checkpoint files
candidates = [
    p for p in base.rglob("bandpowers.dat")
    if "__MACOSX" not in str(p)
    and ".ipynb_checkpoints" not in str(p)
]
print("\nReal bandpowers.dat candidates:")
for p in candidates:
    print(" -", p)
if not candidates:
    raise FileNotFoundError("No real bandpowers.dat found.")
bp = candidates[0]
print("\nUsing:", bp)
print("\nFirst 15 lines:")
with open(bp, "r", encoding="utf-8", errors="replace") as f:
    for i in range(15):
        line = f.readline()
        print(repr(line))
data = np.loadtxt(bp)
print("\nLoaded bandpowers successfully.")
print("Shape:", data.shape)
print("First rows:")
print(data[:10])
