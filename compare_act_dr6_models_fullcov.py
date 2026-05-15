import numpy as np
import act_dr6_lenslike as alike
# Load ACT DR6 full likelihood data
d = alike.load_data(
    "act_baseline",
    lens_only=True,
    like_corrections=False
)
# Load previous amplitude-fit CSV
data = np.loadtxt(
    "ACT_DR6_amplitude_fit.csv",
    delimiter=",",
    skiprows=1
)
L = data[:,0]
y = data[:,1]
theory = data[:,3]
cinv = d["cinv"]
N = len(y)
# -----------------------------
# Model 1: Fiducial, A = 1
# -----------------------------
res_fid = y - theory
chi2_fid = res_fid @ cinv @ res_fid
k_fid = 0
AIC_fid = chi2_fid + 2*k_fid
BIC_fid = chi2_fid + k_fid*np.log(N)
# -----------------------------
# Model 2: Amplitude-only
# y = A * theory
# -----------------------------
den = theory @ cinv @ theory
num = theory @ cinv @ y
A = num / den
sigma_A = 1.0 / np.sqrt(den)
model_amp = A * theory
res_amp = y - model_amp
chi2_amp = res_amp @ cinv @ res_amp
k_amp = 1
AIC_amp = chi2_amp + 2*k_amp
BIC_amp = chi2_amp + k_amp*np.log(N)
# -----------------------------
# Model 3: Scale-dependent
# y = A0 * [1 + alpha ln(L/L0)] theory
# equivalently y = A0*theory + B*ln(L/L0)*theory
# -----------------------------
L0 = 300.0
x = np.log(L / L0)
X = np.column_stack([theory, x * theory])
F = X.T @ cinv @ X
pcov = np.linalg.inv(F)
p = pcov @ (X.T @ cinv @ y)
A0 = p[0]
B = p[1]
alpha = B / A0
sigma_A0 = np.sqrt(pcov[0,0])
var_alpha = (
    pcov[1,1] / A0**2
    + (B**2 * pcov[0,0]) / A0**4
    - 2 * B * pcov[0,1] / A0**3
)
sigma_alpha = np.sqrt(var_alpha)
model_scale = X @ p
res_scale = y - model_scale
chi2_scale = res_scale @ cinv @ res_scale
k_scale = 2
AIC_scale = chi2_scale + 2*k_scale
BIC_scale = chi2_scale + k_scale*np.log(N)
# Save table
with open("ACT_DR6_model_comparison_fullcov.txt", "w") as f:
    f.write("ACT DR6 model comparison using full covariance\n")
    f.write("================================================\n")
    f.write(f"N = {N}\n\n")
    f.write("Fiducial model, A = 1\n")
    f.write(f"chi2 = {chi2_fid:.6f}\n")
    f.write(f"AIC = {AIC_fid:.6f}\n")
    f.write(f"BIC = {BIC_fid:.6f}\n\n")
    f.write("Amplitude-only model\n")
    f.write(f"A_lens = {A:.6f} +/- {sigma_A:.6f}\n")
    f.write(f"chi2 = {chi2_amp:.6f}\n")
    f.write(f"AIC = {AIC_amp:.6f}\n")
    f.write(f"BIC = {BIC_amp:.6f}\n\n")
    f.write("Scale-dependent model\n")
    f.write(f"A0 = {A0:.6f} +/- {sigma_A0:.6f}\n")
    f.write(f"alpha = {alpha:.6f} +/- {sigma_alpha:.6f}\n")
    f.write(f"chi2 = {chi2_scale:.6f}\n")
    f.write(f"AIC = {AIC_scale:.6f}\n")
    f.write(f"BIC = {BIC_scale:.6f}\n")
print("Saved: ACT_DR6_model_comparison_fullcov.txt")
print("")
print("Fiducial A=1:")
print(f"chi2 = {chi2_fid:.6f}, AIC = {AIC_fid:.6f}, BIC = {BIC_fid:.6f}")
print("")
print("Amplitude-only:")
print(f"A_lens = {A:.6f} +/- {sigma_A:.6f}")
print(f"chi2 = {chi2_amp:.6f}, AIC = {AIC_amp:.6f}, BIC = {BIC_amp:.6f}")
print("")
print("Scale-dependent:")
print(f"A0 = {A0:.6f} +/- {sigma_A0:.6f}")
print(f"alpha = {alpha:.6f} +/- {sigma_alpha:.6f}")
print(f"chi2 = {chi2_scale:.6f}, AIC = {AIC_scale:.6f}, BIC = {BIC_scale:.6f}")
