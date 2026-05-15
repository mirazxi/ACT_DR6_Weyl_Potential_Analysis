import numpy as np
import matplotlib.pyplot as plt
import act_dr6_lenslike as alike
# Load ACT DR6 lensing-only data
d = alike.load_data(
    "act_baseline",
    lens_only=True,
    like_corrections=False
)
L = d["bcents_act"]
y = d["data_binned_clkk"]
cov = d["cov"]
cinv = d["cinv"]
sigma = np.sqrt(np.diag(cov))
# Load auxiliary Planck 2018 fiducial theory file
ells, clpp = np.loadtxt(
    "cls_plikHM_TTTEEE_lowl_lowE_dr6_accuracy.txt",
    usecols=[0, 5],
    unpack=True
)
# Convert C_L^phiphi to C_L^kappakappa
clkk = 0.25 * clpp * 2.0 * np.pi
clkk[ells < 2] = 0.0
# Bin theory using ACT binning matrix
n = d["binmat_act"].shape[1]
theory = d["binmat_act"] @ clkk[:n]
# Fit amplitude A_lens
den = theory @ cinv @ theory
num = theory @ cinv @ y
A = num / den
sigma_A = 1.0 / np.sqrt(den)
chi2_best = (y - A * theory) @ cinv @ (y - A * theory)
chi2_fid = (y - theory) @ cinv @ (y - theory)
# Save results
np.savetxt(
    "ACT_DR6_amplitude_fit.csv",
    np.column_stack([L, y, sigma, theory, A * theory]),
    delimiter=",",
    header="L_center,ACT_C_L_kappakappa,error,fiducial_theory,bestfit_A_times_theory",
    comments=""
)
with open("ACT_DR6_amplitude_fit_results.txt", "w") as f:
    f.write("ACT DR6 CMB lensing amplitude fit\n")
    f.write("--------------------------------\n")
    f.write(f"A_lens = {A:.6f} +/- {sigma_A:.6f}\n")
    f.write(f"chi2_bestfit = {chi2_best:.6f}\n")
    f.write(f"chi2_fiducial_A_1 = {chi2_fid:.6f}\n")
    f.write(f"number_of_bandpowers = {len(L)}\n")
# Plot
plt.figure(figsize=(7,5))
plt.errorbar(L, y, yerr=sigma, fmt="o", capsize=3, label="ACT DR6 data")
plt.plot(L, theory, "--", label="Fiducial theory, A=1")
plt.plot(L, A * theory, "-", label=f"Best fit, A={A:.3f}")
plt.xlabel(r"$L$")
plt.ylabel(r"$C_L^{\kappa\kappa}$")
plt.title("ACT DR6 CMB Lensing Amplitude Fit")
plt.legend()
plt.tight_layout()
plt.savefig("ACT_DR6_lensing_amplitude_fit.png", dpi=300)
print("Saved: ACT_DR6_amplitude_fit.csv")
print("Saved: ACT_DR6_amplitude_fit_results.txt")
print("Saved: ACT_DR6_lensing_amplitude_fit.png")
print(f"A_lens = {A:.6f} +/- {sigma_A:.6f}")
print(f"chi2 best-fit = {chi2_best:.6f}")
print(f"chi2 fiducial A=1 = {chi2_fid:.6f}")
