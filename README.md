# ACT DR6 Weyl Potential Analysis

Reproducibility code, figures, and numerical outputs for the manuscript:

**Effective Weyl-Potential Constraints from ACT DR6 CMB Lensing: A Full-Covariance Phenomenological Analysis**

## Summary

This repository contains the Python scripts and output files used to reproduce the ACT DR6 CMB lensing phenomenological analysis.

The analysis tests three models:

1. Fiducial lensing spectrum:
   \[
   C_L^{\kappa\kappa}=C_{L,\rm fid}^{\kappa\kappa}
   \]

2. Amplitude-only model:
   \[
   C_L^{\kappa\kappa}=A_{\rm lens}C_{L,\rm fid}^{\kappa\kappa}
   \]

3. Scale-dependent model:
   \[
   C_L^{\kappa\kappa}
   =
   A_0\left[1+\alpha\ln\left(\frac{L}{300}\right)\right]
   C_{L,\rm fid}^{\kappa\kappa}
   \]

The fitted parameters are then translated into effective projected Weyl-potential constraints.

## Main results

The main numerical results are:

- \(A_{\rm lens}=1.012630\pm0.025607\)
- \(A_0=1.016250\pm0.033801\)
- \(\alpha=-0.017553\pm0.035581\)
- \(\Sigma_0=0.0081\pm0.0168\)
- \(\Sigma_1=-0.0088\pm0.0179\)

The fiducial model is preferred by AIC and BIC.

## Data

The ACT DR6 lensing likelihood data are not redistributed in this repository.

To reproduce the analysis, download the ACT DR6 lensing likelihood release v1.2 from NASA LAMBDA:

`ACT_dr6_likelihood_v1.2.tgz`

After extracting the archive, place the extracted `v1.2` folder inside the installed `act_dr6_lenslike/data/` directory.

The analysis uses:

- ACT DR6 lensing likelihood release v1.2
- ACT baseline lensing bandpowers
- `lens_only=True`
- `like_corrections=False`
- Full ACT DR6 covariance matrix

## Installation

Install the required Python packages:

```bash
python -m pip install numpy scipy matplotlib act_dr6_lenslike
