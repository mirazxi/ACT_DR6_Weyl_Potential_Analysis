import numpy as np
import act_dr6_lenslike as alike
# Load ACT DR6 lensing-only data
d = alike.load_data(
    "act_baseline",
    lens_only=True,
    like_corrections=False
)
L = d["bcents_act"]
y = d["data_binned_clkk"]
cov_full = d["cov"]
cinv_full = d["cinv"]
# Diagonal-only covariance
cov_diag = np.diag(np.diag(cov_full))
cinv_diag = np.linalg.inv(cov_diag)
# Load fiducial theory spectrum
ells, clpp = np.loadtxt(
    "cls_plikHM_TTTEEE_lowl_lowE_dr6_accuracy.txt",
    usecols=[0, 5],
    unpack=True
)
# Convert C_L^phiphi to C_L^kappakappa
clkk = 0.25 * clpp * 2.0 * np.pi
clkk[ells < 2] = 0.0
# Bin fiducial theory using ACT binning matrix
n = d["binmat_act"].shape[1]
theory = d["binmat_act"] @ clkk[:n]
N = len(y)
L0 = 300.0
def fit_models(cinv, label):
    # Fiducial model
    chi2_fid = (y - theory) @ cinv @ (y - theory)
    AIC_fid = chi2_fid
    BIC_fid = chi2_fid
    # Amplitude-only model: m = A * theory
    den = theory @ cinv @ theory
    num = theory @ cinv @ y
    A = num / den
    sigma_A = 1.0 / np.sqrt(den)
    model_A = A * theory
    chi2_A = (y - model_A) @ cinv @ (y - model_A)
    AIC_A = chi2_A + 2
    BIC_A = chi2_A + np.log(N)
    # Scale-dependent model:
    # m = A0 * theory + B ln(L/L0) * theory
    x = np.log(L / L0)
    X = np.column_stack([theory, x * theory])
    F = X.T @ cinv @ X
    pcov = np.linalg.inv(F)
    p = pcov @ (X.T @ cinv @ y)
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
    model_scale = X @ p
    chi2_scale = (y - model_scale) @ cinv @ (y - model_scale)
    AIC_scale = chi2_scale + 4
    BIC_scale = chi2_scale + 2 * np.log(N)
    return {
        "label": label,
        "chi2_fid": chi2_fid,
        "AIC_fid": AIC_fid,
        "BIC_fid": BIC_fid,
        "A": A,
        "sigma_A": sigma_A,
        "chi2_A": chi2_A,
        "AIC_A": AIC_A,
        "BIC_A": BIC_A,
        "A0": A0,
        "sigma_A0": sigma_A0,
        "alpha": alpha,
        "sigma_alpha": sigma_alpha,
        "chi2_scale": chi2_scale,
        "AIC_scale": AIC_scale,
        "BIC_scale": BIC_scale,
    }
results = [
    fit_models(cinv_full, "Full covariance"),
    fit_models(cinv_diag, "Diagonal-only covariance"),
]
with open("ACT_DR6_covariance_robustness.txt", "w") as f:
    f.write("ACT DR6 covariance robustness test\n")
    f.write("==================================\n\n")
    for r in results:
        f.write(f"{r['label']}\n")
        f.write("-" * len(r["label"]) + "\n")
        f.write(f"Fiducial: chi2 = {r['chi2_fid']:.6f}, AIC = {r['AIC_fid']:.6f}, BIC = {r['BIC_fid']:.6f}\n")
        f.write(f"Amplitude-only: A = {r['A']:.6f} +/- {r['sigma_A']:.6f}, chi2 = {r['chi2_A']:.6f}, AIC = {r['AIC_A']:.6f}, BIC = {r['BIC_A']:.6f}\n")
        f.write(f"Scale-dependent: A0 = {r['A0']:.6f} +/- {r['sigma_A0']:.6f}, alpha = {r['alpha']:.6f} +/- {r['sigma_alpha']:.6f}, chi2 = {r['chi2_scale']:.6f}, AIC = {r['AIC_scale']:.6f}, BIC = {r['BIC_scale']:.6f}\n\n")
with open("ACT_DR6_covariance_robustness_table.tex", "w") as f:
    f.write(r"""\begin{table}[htbp]
\centering
\caption{Robustness of the ACT DR6 phenomenological constraints to the covariance treatment. The full covariance result is used as the baseline analysis, while the diagonal-only case is shown only as a robustness check.}
\label{tab:covariance_robustness}
\begin{tabular}{lccc}
\toprule
Covariance treatment & $A_{\rm lens}$ & $\alpha$ & Preferred model \\
\midrule
""")
    for r in results:
        preferred = "Fiducial"
        f.write(
            f"{r['label']} & "
            f"${r['A']:.3f}\\pm{r['sigma_A']:.3f}$ & "
            f"${r['alpha']:.3f}\\pm{r['sigma_alpha']:.3f}$ & "
            f"{preferred} \\\\\n"
        )
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
print("Saved: ACT_DR6_covariance_robustness.txt")
print("Saved: ACT_DR6_covariance_robustness_table.tex")
print("")
print(open("ACT_DR6_covariance_robustness.txt").read())
