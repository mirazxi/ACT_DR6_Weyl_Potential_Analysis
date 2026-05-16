import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
base = Path(".")
# Locate real files, ignoring macOS metadata
bp_files = [
    p for p in base.rglob("bandpowers.dat")
    if "__MACOSX" not in str(p)
    and ".ipynb_checkpoints" not in str(p)
]
cov_files = [
    p for p in base.rglob("cov.dat")
    if "__MACOSX" not in str(p)
    and ".ipynb_checkpoints" not in str(p)
]
if not bp_files:
    raise FileNotFoundError("No real bandpowers.dat found.")
if not cov_files:
    raise FileNotFoundError("No real cov.dat found.")
bp_file = bp_files[0]
cov_file = cov_files[0]
print("Using bandpowers:", bp_file)
print("Using covariance:", cov_file)
# bandpowers.dat columns:
# bin, L_min, L_max, L_av, PP, Error, Ahat
bp = np.loadtxt(bp_file)
cov_pp = np.loadtxt(cov_file)
bin_id = bp[:, 0]
Lmin = bp[:, 1]
Lmax = bp[:, 2]
L = bp[:, 3]
PP = bp[:, 4]
err_pp = bp[:, 5]
Ahat = bp[:, 6]
print("Bandpower shape:", bp.shape)
print("Covariance shape:", cov_pp.shape)
if cov_pp.shape != (len(L), len(L)):
    raise ValueError("Covariance shape does not match number of bandpowers.")
# Check whether cov.dat corresponds to the PP bandpower covariance
diag_err = np.sqrt(np.diag(cov_pp))
rel_diff = np.max(np.abs(diag_err - err_pp) / err_pp)
print("Max relative difference between sqrt(diag(cov)) and Error column:", rel_diff)
# Infer fiducial PP spectrum from Ahat = PP / PP_fid
PP_fid = PP / Ahat
# Transform PP covariance into Ahat covariance:
# Ahat_i = PP_i / PP_fid_i
cov_A = cov_pp / np.outer(PP_fid, PP_fid)
cinv_A = np.linalg.inv(cov_A)
sigma_Ahat = np.sqrt(np.diag(cov_A))
N = len(Ahat)
# -----------------------------
# Model 1: fiducial Ahat = 1
# -----------------------------
m_fid = np.ones_like(Ahat)
res_fid = Ahat - m_fid
chi2_fid = res_fid @ cinv_A @ res_fid
k_fid = 0
AIC_fid = chi2_fid + 2 * k_fid
BIC_fid = chi2_fid + k_fid * np.log(N)
# -----------------------------
# Model 2: amplitude-only Ahat = A
# -----------------------------
one = np.ones_like(Ahat)
den = one @ cinv_A @ one
num = one @ cinv_A @ Ahat
A = num / den
sigma_A = 1.0 / np.sqrt(den)
m_amp = A * one
res_amp = Ahat - m_amp
chi2_amp = res_amp @ cinv_A @ res_amp
k_amp = 1
AIC_amp = chi2_amp + 2 * k_amp
BIC_amp = chi2_amp + k_amp * np.log(N)
# -----------------------------
# Model 3: scale-dependent
# Ahat(L) = A0 [1 + alpha ln(L/L0)]
# Linearized as Ahat = A0 + B ln(L/L0), B=A0 alpha
# -----------------------------
L0 = 300.0
x = np.log(L / L0)
X = np.column_stack([np.ones_like(L), x])
F = X.T @ cinv_A @ X
pcov = np.linalg.inv(F)
p = pcov @ (X.T @ cinv_A @ Ahat)
A0 = p[0]
B = p[1]
alpha = B / A0
sigma_A0 = np.sqrt(pcov[0, 0])
var_alpha = (
    pcov[1, 1] / A0**2
    + (B**2 * pcov[0, 0]) / A0**4
    - 2 * B * pcov[0, 1] / A0**3
)
sigma_alpha = np.sqrt(var_alpha)
m_scale = X @ p
res_scale = Ahat - m_scale
chi2_scale = res_scale @ cinv_A @ res_scale
k_scale = 2
AIC_scale = chi2_scale + 2 * k_scale
BIC_scale = chi2_scale + k_scale * np.log(N)
# Effective Weyl-potential constraints
Sigma_A = np.sqrt(A) - 1.0
sigma_Sigma_A = 0.5 * sigma_A / np.sqrt(A)
Sigma0 = np.sqrt(A0) - 1.0
sigma_Sigma0 = 0.5 * sigma_A0 / np.sqrt(A0)
Sigma1 = 0.5 * alpha * np.sqrt(A0)
sigma_Sigma1 = 0.5 * np.sqrt(A0) * sigma_alpha
# Save result table
with open("SPT3G_2018_model_comparison_fullcov.txt", "w") as f:
    f.write("SPT-3G 2018 CMB lensing model comparison using full covariance\n")
    f.write("===============================================================\n")
    f.write(f"N = {N}\n")
    f.write(f"Pivot L0 = {L0}\n\n")
    f.write("Fiducial model, A=1\n")
    f.write(f"chi2 = {chi2_fid:.6f}\n")
    f.write(f"AIC = {AIC_fid:.6f}\n")
    f.write(f"BIC = {BIC_fid:.6f}\n\n")
    f.write("Amplitude-only model\n")
    f.write(f"A = {A:.6f} +/- {sigma_A:.6f}\n")
    f.write(f"Sigma_A = {Sigma_A:.6f} +/- {sigma_Sigma_A:.6f}\n")
    f.write(f"chi2 = {chi2_amp:.6f}\n")
    f.write(f"AIC = {AIC_amp:.6f}\n")
    f.write(f"BIC = {BIC_amp:.6f}\n\n")
    f.write("Scale-dependent model\n")
    f.write(f"A0 = {A0:.6f} +/- {sigma_A0:.6f}\n")
    f.write(f"alpha = {alpha:.6f} +/- {sigma_alpha:.6f}\n")
    f.write(f"Sigma0 = {Sigma0:.6f} +/- {sigma_Sigma0:.6f}\n")
    f.write(f"Sigma1 = {Sigma1:.6f} +/- {sigma_Sigma1:.6f}\n")
    f.write(f"chi2 = {chi2_scale:.6f}\n")
    f.write(f"AIC = {AIC_scale:.6f}\n")
    f.write(f"BIC = {BIC_scale:.6f}\n")
# Save CSV
np.savetxt(
    "SPT3G_2018_Ahat_fit.csv",
    np.column_stack([L, Ahat, sigma_Ahat, m_fid, m_amp, m_scale]),
    delimiter=",",
    header="L,Ahat,error_Ahat,fiducial,amplitude_fit,scale_dependent_fit",
    comments=""
)
# Plot amplitude-space comparison
plt.figure(figsize=(7, 5))
plt.axhline(1.0, linestyle="--", label="Fiducial A=1")
plt.errorbar(L, Ahat, yerr=sigma_Ahat, fmt="o", capsize=3, label="SPT-3G data")
plt.plot(L, m_amp, "-", label=f"Amplitude fit, A={A:.3f}")
plt.plot(L, m_scale, "-", label="Scale-dependent fit")
plt.xlabel(r"$L$")
plt.ylabel(r"$\hat{A}$")
plt.title("SPT-3G 2018 CMB Lensing Amplitude-Space Fit")
plt.legend()
plt.tight_layout()
plt.savefig("SPT3G_2018_Ahat_fullcov_fit.png", dpi=300)
# Plot residuals
plt.figure(figsize=(7, 5))
plt.axhline(0.0, linestyle="--")
plt.errorbar(L, Ahat - m_scale, yerr=sigma_Ahat, fmt="o", capsize=3)
plt.xlabel(r"$L$")
plt.ylabel(r"$\hat{A} - \hat{A}_{\rm model}$")
plt.title("SPT-3G 2018 Scale-Dependent Fit Residuals")
plt.tight_layout()
plt.savefig("SPT3G_2018_Ahat_fullcov_residuals.png", dpi=300)
print("")
print("Saved: SPT3G_2018_model_comparison_fullcov.txt")
print("Saved: SPT3G_2018_Ahat_fit.csv")
print("Saved: SPT3G_2018_Ahat_fullcov_fit.png")
print("Saved: SPT3G_2018_Ahat_fullcov_residuals.png")
print("")
print("Fiducial A=1:")
print(f"chi2 = {chi2_fid:.6f}, AIC = {AIC_fid:.6f}, BIC = {BIC_fid:.6f}")
print("")
print("Amplitude-only:")
print(f"A = {A:.6f} +/- {sigma_A:.6f}")
print(f"Sigma_A = {Sigma_A:.6f} +/- {sigma_Sigma_A:.6f}")
print(f"chi2 = {chi2_amp:.6f}, AIC = {AIC_amp:.6f}, BIC = {BIC_amp:.6f}")
print("")
print("Scale-dependent:")
print(f"A0 = {A0:.6f} +/- {sigma_A0:.6f}")
print(f"alpha = {alpha:.6f} +/- {sigma_alpha:.6f}")
print(f"Sigma0 = {Sigma0:.6f} +/- {sigma_Sigma0:.6f}")
print(f"Sigma1 = {Sigma1:.6f} +/- {sigma_Sigma1:.6f}")
print(f"chi2 = {chi2_scale:.6f}, AIC = {AIC_scale:.6f}, BIC = {BIC_scale:.6f}")
