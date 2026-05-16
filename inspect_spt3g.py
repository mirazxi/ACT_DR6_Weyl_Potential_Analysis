import numpy as np
from pathlib import Path
base = Path(".")
print("Current folder:", base.resolve())
print("\nAll files:")
for p in base.rglob("*"):
    if p.is_file():
        print(" -", p)
print("\nSearching for bandpower files...")
candidates = list(base.rglob("*bandpower*"))
if not candidates:
    print("No bandpower-like files found.")
else:
    for c in candidates:
        print("Found:", c)
    bp = candidates[0]
    print("\nTrying to read:", bp)
    try:
        data = np.loadtxt(bp)
        print("Loaded with np.loadtxt")
    except Exception as e:
        print("np.loadtxt failed:", e)
        data = np.genfromtxt(bp, comments="#")
        print("Loaded with np.genfromtxt")
    print("Shape:", data.shape)
    print("First rows:")
    print(data[:10])
