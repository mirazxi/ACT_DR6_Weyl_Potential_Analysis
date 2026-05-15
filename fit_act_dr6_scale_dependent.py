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
cov = np.diag(sigma**2)
cinv = np.linalg.inv(cov)
L0 = 300.0
x = np.log(L / L0)
# Model: y = A0 * theory + B * x * theory
# where B = A0 * alpha
X = np.column_stack([theory, x * theory])
# Weighted least squares
F = X.T @ cinv @ X
p = np.linalg.inv(F) @ (X.T @ cinv @ y)
pcov = np.linalg.inv(F)
A0 = p[0]
B = p[1]
sigma_A0 = np.sqrt(pcov[0,0])
sigma_B = np.sqrt(pcov[1,1])
alpha = B / A0
# Error propagation for alpha = B/A0
var_alpha = (
    pcov[1,1] / A0**2
    + (B**2 * pcov[0,0]) / A0**4
    - 2 * B * pcov[0,1] / A0**3
)
sigma_alpha = np.sqrt(var_alpha)
model = X @ p
chi2 = (y - model) @ cinv @ (y - model)
# AIC and BIC
N = len(y)
k = 2
AIC = chi2 + 2*k
BIC = chi2 + k*np.log(N)
with open("ACT_DR6_scale_dependent_fit_results.txt", "w") as f:
    f.write("ACT DR6 scale-dependent lensing modification fit\n")
    f.write("------------------------------------------------\n")
    f.write(f"A0 = {A0:.6f} +/- {sigma_A0:.6f}\n")
    f.write(f"alpha = {alpha:.6f} +/- {sigma_alpha:.6f}\n")
    f.write(f"chi2 = {chi2:.6f}\n")
    f.write(f"AIC = {AIC:.6f}\n")
    f.write(f"BIC = {BIC:.6f}\n")
    f.write(f"N = {N}\n")
    f.write(f"k = {k}\n")
np.savetxt(
    "ACT_DR6_scale_dependent_fit.csv",
    np.column_stack([L, y, sigma, theory, model]),
    delimiter=",",
    header="L_center,ACT_C_L_kappakappa,error,fiducial_theory,scale_dependent_model",
    comments=""
)
plt.figure(figsize=(7,5))
plt.errorbar(L, y, yerr=sigma, fmt="o", capsize=3, label="ACT DR6 data")
plt.plot(L, theory, "--", label="Fiducial theory")
plt.plot(L, model, "-", label="Scale-dependent fit")
plt.xlabel(r"$L$")
plt.ylabel(r"$C_L^{\kappa\kappa}$")
plt.title("ACT DR6 Scale-Dependent Lensing Modification")
plt.legend()
plt.tight_layout()
plt.savefig("ACT_DR6_scale_dependent_fit.png", dpi=300)
plt.figure(figsize=(7,5))
plt.axhline(0, linestyle="--")
plt.errorbar(L, y - model, yerr=sigma, fmt="o", capsize=3)
plt.xlabel(r"$L$")
plt.ylabel(r"Data $-$ model")
plt.title("Residuals for Scale-Dependent Lensing Fit")
plt.tight_layout()
plt.savefig("ACT_DR6_scale_dependent_residuals.png", dpi=300)
print("Saved: ACT_DR6_scale_dependent_fit_results.txt")
print("Saved: ACT_DR6_scale_dependent_fit.csv")
print("Saved: ACT_DR6_scale_dependent_fit.png")
print("Saved: ACT_DR6_scale_dependent_residuals.png")
print(f"A0 = {A0:.6f} +/- {sigma_A0:.6f}")
print(f"alpha = {alpha:.6f} +/- {sigma_alpha:.6f}")
print(f"chi2 = {chi2:.6f}")
print(f"AIC = {AIC:.6f}")
print(f"BIC = {BIC:.6f}")
