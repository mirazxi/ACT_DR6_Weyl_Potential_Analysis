import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt(
    "ACT_DR6_amplitude_fit.csv",
    delimiter=",",
    skiprows=1
)
L = data[:,0]
y = data[:,1]
sigma = data[:,2]
theory = data[:,3]
bestfit = data[:,4]
residual = y - theory
residual_best = y - bestfit
plt.figure(figsize=(7,5))
plt.axhline(0, linestyle="--")
plt.errorbar(L, residual, yerr=sigma, fmt="o", capsize=3, label="Data - fiducial")
plt.xlabel(r"$L$")
plt.ylabel(r"$\Delta C_L^{\kappa\kappa}$")
plt.title("ACT DR6 Lensing Residuals Relative to Fiducial Theory")
plt.legend()
plt.tight_layout()
plt.savefig("ACT_DR6_lensing_residuals.png", dpi=300)
print("Saved: ACT_DR6_lensing_residuals.png")
