
### CAMERA CALIBRATION EVALUATION
| Parameter | Ground Truth (GT) | Solved Metrics | Absolute Delta | Percentage Error |
| :--- | :---: | :---: | :---: | :---: |
| **Focal Length $f_x$** (px) | 1150.00 | 1152.97 | 2.97 | **0.26%** |
| **Focal Length $f_y$** (px) | 1150.00 | 1152.97 | 2.97 | **0.26%** |
| **Principal Point $c_x$** (px) | 960.00 | 961.23 | 1.23 | **0.06%** |
| **Principal Point $c_y$** (px) | 540.00 | 538.45 | 1.55 | **0.14%** |
| **Distortion Coefficient $\kappa_1$** | -0.2000 | -0.2098 | 0.0098 | **4.88%** |

**Note on Percentage Error Normalization:**To ensure physical relevance and avoid numerical singularities the percentage errors for estimatedparameters are normalized using independent metrics aligned with the sensor's physical domain:
1. **Focal Length ($f_x, f_y$):** Normalized by their respective ground truth values ($|f_{\text{GT}}|$).
2. **Principal Point Offsets ($c_x, c_y$):** Normalized by the image raster width ($W_{\text{img}}$) and height ($H_{\text{img}}$).
3. **Radial Distortion Coefficient ($\kappa_1$):** Normalized by the ground truth value ($|\kappa_{1,\text{GT}}|$); under ideal flat-lens conditions where $\kappa_{1,\text{GT}} \approx 0$, the error is normalized by the maximum search boundary ($\kappa_{1,\text{max}} = 0.5$).
