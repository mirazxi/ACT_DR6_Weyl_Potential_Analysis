import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# Find real SPT-3G bandpowers.dat file
base = Path(".")
candidates = [
    p for p in base.rglob("bandpowers.dat")
    if "__MACOSX" not in str(p)
    and ".ipynb_checkpoints" not in str(p)
]
if not candidates:
    raise FileNotFoundError("No real bandpowers.dat found.")
bp = candidates[0]
print("Using:", bp)
data = np.loadtxt(bp)
bin_id = data[:,0]
lmin = data[:,1]
lmax = data[:,2]
L = data[:,3]
PP = data[:,4]
err = data[:,5]
Ahat = data[:,6]
# Save cleaned CSV
np.savetxt(
    "SPT3G_2018_lensing_bandpowers.csv",
    np.column_stack([bin_id, lmin, lmax, L, PP, err, Ahat]),
    delimiter=",",
    header="bin,l_min,l_max,L_av,PP,error,Ahat",
    comments=""
)
# Plot PP bandpowers
plt.figure(figsize=(7,5))
plt.errorbar(L, PP, yerr=err, fmt="o", capsize=3)
plt.xlabel(r"$L$")
plt.ylabel(r"SPT-3G lensing bandpower")
plt.title("SPT-3G 2018 CMB Lensing Bandpowers")
plt.tight_layout()
plt.savefig("SPT3G_2018_lensing_bandpowers.png", dpi=300)
# Plot Ahat values
plt.figure(figsize=(7,5))
plt.axhline(1.0, linestyle="--")
plt.errorbar(L, Ahat, fmt="o", capsize=3)
plt.xlabel(r"$L$")
plt.ylabel(r"$\hat{A}$")
plt.title("SPT-3G 2018 Lensing Bandpower Amplitudes")
plt.tight_layout()
plt.savefig("SPT3G_2018_lensing_Ahat.png", dpi=300)
print("Saved: SPT3G_2018_lensing_bandpowers.csv")
print("Saved: SPT3G_2018_lensing_bandpowers.png")
print("Saved: SPT3G_2018_lensing_Ahat.png")
print("Number of SPT-3G bandpowers:", len(L))
print("Mean Ahat:", np.mean(Ahat))
