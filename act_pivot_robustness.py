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
cov = d["cov"]
cinv = d["cinv"]
sigma = np.sqrt(np.diag(cov))
# Load fiducial theory spectrum
ells, clpp = np.loadtxt(
    "cls_plikHM_TTTEEE_lowl_lowE_dr6_accuracy.txt",
    usecols=[0, 5],
    unpack=True
)
# Convert C_L^phiphi to C_L^kappakappa
clkk = 0.25 * clpp * 2.0 * np.pi
clkk[ells < 2] = 0.0
# Bin fiducial theory
n = d["binmat_act"].shape[1]
theory = d["binmat_act"] @ clkk[:n]
N = len(y)
# Fiducial chi-square
chi2_fid = (y - theory) @ cinv @ (y - theory)
AIC_fid = chi2_fid
BIC_fid = chi2_fid
pivots = [200.0, 300.0, 500.0]
rows = []
for L0 in pivots:
    x = np.log(L / L0)
    # model = A0 * theory + B * ln(L/L0) * theory
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
    model = X @ p
    chi2 = (y - model) @ cinv @ (y - model)
    AIC = chi2 + 2 * 2
    BIC = chi2 + 2 * np.log(N)
    # Positivity check for best fit and +-2 sigma alpha values
    alpha_values = [alpha, alpha - 2*sigma_alpha, alpha + 2*sigma_alpha]
    min_factors = []
    for a in alpha_values:
        factor = 1.0 + a * np.log(L / L0)
        min_factors.append(np.min(factor))
    min_factor_best = min_factors[0]
    min_factor_2sigma = min(min_factors)
    rows.append([
        L0, A0, sigma_A0, alpha, sigma_alpha,
        chi2, AIC, BIC, min_factor_best, min_factor_2sigma
    ])
# Save text result
with open("ACT_DR6_pivot_robustness.txt", "w") as f:
    f.write("ACT DR6 pivot robustness test\n")
    f.write("============================\n\n")
    f.write(f"Fiducial model: chi2 = {chi2_fid:.6f}, AIC = {AIC_fid:.6f}, BIC = {BIC_fid:.6f}\n\n")
    f.write("Pivot L0 | A0 +/- sigma_A0 | alpha +/- sigma_alpha | chi2 | AIC | BIC | min factor best | min factor 2sigma\n")
    f.write("---------|------------------|-----------------------|------|-----|-----|-----------------|------------------\n")
    for r in rows:
        L0, A0, sA0, alpha, sa, chi2, AIC, BIC, mf_best, mf_2s = r
        f.write(
            f"{L0:.0f} | {A0:.6f} +/- {sA0:.6f} | "
            f"{alpha:.6f} +/- {sa:.6f} | "
            f"{chi2:.6f} | {AIC:.6f} | {BIC:.6f} | "
            f"{mf_best:.6f} | {mf_2s:.6f}\n"
        )
# Save LaTeX table
with open("ACT_DR6_pivot_robustness_table.tex", "w") as f:
    f.write(r"""\begin{table}[htbp]
\centering
\caption{Robustness of the ACT DR6 scale-dependent fit to the choice of pivot multipole. The positivity column gives the minimum value of $1+\alpha\ln(L/L_0)$ over the fitted ACT DR6 bandpower range for the best-fit parameter value.}
\label{tab:act_pivot_robustness}
\begin{tabular}{ccccc}
\toprule
Pivot $L_0$ & $A_0$ & $\alpha$ & $\chi^2_{\rm min}$ & Minimum factor \\
\midrule
""")
    for r in rows:
        L0, A0, sA0, alpha, sa, chi2, AIC, BIC, mf_best, mf_2s = r
        f.write(
            f"{L0:.0f} & "
            f"${A0:.4f}\\pm{sA0:.4f}$ & "
            f"${alpha:.4f}\\pm{sa:.4f}$ & "
            f"{chi2:.4f} & "
            f"{mf_best:.4f} \\\\\n"
        )
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
print("Saved: ACT_DR6_pivot_robustness.txt")
print("Saved: ACT_DR6_pivot_robustness_table.tex")
print("")
print(open("ACT_DR6_pivot_robustness.txt").read())
