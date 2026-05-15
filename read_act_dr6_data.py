import numpy as np
import matplotlib.pyplot as plt
import act_dr6_lenslike as alike
# Load ACT DR6 lensing-only likelihood data
d = alike.load_data(
    "act_baseline",
    lens_only=True,
    like_corrections=False
)
L = d["bcents_act"]
clkk = d["data_binned_clkk"]
cov = d["cov"]
sigma = np.sqrt(np.diag(cov))
# Save bandpowers to CSV
table = np.column_stack([L, clkk, sigma])
np.savetxt(
    "ACT_DR6_lensing_bandpowers.csv",
    table,
    delimiter=",",
    header="L_center,C_L_kappakappa,error",
    comments=""
)
# Plot ACT DR6 lensing bandpowers
plt.figure(figsize=(7,5))
plt.errorbar(L, clkk, yerr=sigma, fmt="o", capsize=3)
plt.xlabel(r"$L$")
plt.ylabel(r"$C_L^{\kappa\kappa}$")
plt.title("ACT DR6 CMB Lensing Bandpowers")
plt.tight_layout()
plt.savefig("ACT_DR6_lensing_bandpowers.png", dpi=300)
print("Saved: ACT_DR6_lensing_bandpowers.csv")
print("Saved: ACT_DR6_lensing_bandpowers.png")
print("Number of bandpowers:", len(L))
print("Covariance shape:", cov.shape)
