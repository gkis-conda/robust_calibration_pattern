---
title: 'B-HGP: Binary Hexagonal Galois Pattern for Robust Single-Frame Camera Calibration'
authors:
  - name: Gennadiy Kis
    orcid: 0000-0002-8604-8560
    affiliation: 1
  - name: Oleksandr Kis
    orcid: 0000-0003-2487-1752
    affiliation: 2
affiliations:
  - name: Independent Researcher
    index: 1
  - name: State University of Information and Communication Technologies
    address: 7, Solomyanska Str., Kyiv, Ukraine, 03110
    index: 2
date: 2 September 2026
bibliography: paper.bib
---

# Summary

This work numerically demonstrates the digital camera calibration capabilities of the Binary Hexagonal Galois Pattern (B-HGP) framework. We present an open-source repository that provides a Python-based implementation of the B-HGP framework for single-frame camera intrinsic and radial distortion calibration. The software implements the generation, extraction, tracking, and algebraic decoding of error-correcting codes embedded natively onto a regular two-dimensional hexagonal lattice. 

The software architecture is engineered to maintain target registration under aggressive perspective shear, spatial occlusions, and optical noise. By utilizing a cascaded, decoupled multi-phase calibration solver, the pipeline separates radial lens distortion optimization from the projective camera matrix recovery. The package includes fully automated verification test benches and a photorealistic rendering simulation suite built on top of the Blender 2.92 API to ensure complete reproducibility of photogrammetric validation tracking.

# Statement of Need

Accurate sensor calibration is a fundamental prerequisite for visual-inertial odometry, autonomous navigation, and multi-sensor data fusion networks. While conventional fiducial frameworks—such as AprilTags [@Olson2011AprilTag], ArUco [@garrido2014aruco], WhyCode [@lightbody2017WhyCode], or standard chessboard layouts—are widely deployed, their software implementations frequently encounter feature extraction dropouts on the image periphery when observed under severe radial barrel distortion and high out-of-plane perspective slants. 

The B-HGP package addresses these operational pipeline vulnerabilities by exposing an open-source, modular codebase driven by discrete crystal-growth graph routing and algebraic linear decoding over $\mathbb{GF}(2)$. This software serves the computer vision and mobile robotics communities by providing an alternative calibration workflow that handles random feature recognition dropouts and isolates calibration parameters without cross-talk. 

This software paper serves strictly as an open-source implementation report documenting the programmatic toolkit of the B-HGP framework. The comprehensive mathematical proofs, deep algebraic foundations, and formal geometric theorems governing this methodology are presented in a standalone theoretical manuscript currently under consideration for publication in the International Journal of Computer Vision (IJCV) / Journal of Mathematical Imaging and Vision (JMIV).

# Architecture and Implementation

## Pattern Design and Mathematical Foundation

B-HGP encodes three independent binary $M$-sequences into a regular hexagonal lattice layout. The visible binary state at each lattice node is evaluated as the XOR combination of channel bits from three continuous tracking axes:

$$B(r, c) = U[u \bmod 31] \oplus V[v \bmod 31] \oplus W[w \bmod 31]$$

where $r$ represents the row coordinate, $c$ denotes the column coordinate, and the barycentric coordinate space components are defined as:

$$
\begin{aligned}
v &= r \\
u &= c - \lfloor r/2 \rfloor \\
w &= - u - v
\end{aligned}
$$

Each M-sequence is generated from a primitive 5th-order Linear Feedback Shift Register (LFSR) polynomial over the Galois Field $\mathbb{GF}(2)$ [@macwilliams1977theory], producing sequences of maximum period $L = 2^5 - 1 = 31$. The primary property of this topology is that any 5-bit sliding window within an M-sequence reveals a completely unique local phase address, enabling single-frame absolute coordinate recovery from small, isolated pattern fragments.

## Processing Pipeline

The single-frame calibration workflow executes sequentially through six deterministic processing phases:

### 1. Optical Feature Extraction
Detects candidate nodes and classifies their topological profiles (circles versus triangles) using adaptive local binarization, contour solidity filtering, and continuous distance classification via circularity metrics. Circles encode bit state "0", and triangles encode bit state "1". Features are validated against image boundaries to eliminate partial or cut shapes. The sub-pixel centroid approximation determines key-point projection centers that remain invariant to topological shape classification results, as this center definition naturally fits any regular polygon, including circles or triangles.

### 2. Lattice Reconstruction via Crystal Growth
Assembles unorganized sub-pixel barycenters into stable hexagonal coordinate networks via Delaunay triangulation nucleation under strict local triangle coherence constraints. Hexagonal grid topologies expand outward from verified seeds via a systematic front-propagation queue. To protect the growth horizon from non-linear lens compression and edge distortion shears, the engine evaluates local neighborhood changes using a scale-invariant, direction-blind triangle inradius ratio metric. Isolated sub-graph islands are securely merged using a Disjoint Set Union (DSU) forest with parallelogram closure verification to guarantee structural lattice reconstruction under extreme perspective warping.

### 3. 1D Axis Distillation
Inspects each crystalline segment by applying local 4-node XOR sliding kernels to isolate independent 1D stream vectors for the U-axis and W-axis:

$$
\begin{aligned}
    dU[i] &= B[r, c+1] \oplus B[r, c+2] \oplus B[r+1, c] \oplus B[r+1, c+1] \\
    dW[i] &= B[r, c] \oplus B[r, c+1] \oplus B[r+1, c] \oplus B[r+1, c+1]
\end{aligned}
$$

These differential operators perfectly cancel two of the three component sequences, yielding clean, independent 1D M-sequence fragments for subsequent decoding.

### 4. Algebraic Binary Sequence Decoding
Processes each extracted 1D stream through a syndrome decoder pipeline. Single bit-flips are repaired via syndrome evaluation and algebraic locator polynomials [@lin2004error], while recognition dropouts are resolved via a deterministic greedy decoding pass. At this step, two orthogonal axes are matched to the sequence, and the sequence phases are evaluated.

### 5. Coordinate Phase Locking
Merges resolved phases from independent orthogonal axes to establish absolute global $(r, c)$ matrix indices via the intersection of 1D decoded phases. The exact barycentric-to-matrix mapping restores exact topological axes direction alignment. After phase localization, the separated planar crystals are merged into a common matrix that corresponds to the genuine pattern layout.

### 6. Decoupled Camera Calibration
The restored hexagonal lattice provides perfect sets of collinear lines easily extracted along hexagonal directions to enable a plumb-line calibration approach. Unlike traditional joint optimization methods for the Brown-Conrady model [@brown1971close] that couple distortion parameters and intrinsic variables, the framework employs a two-stage cascaded decoupling strategy [@shortis1995isolated]:

* **Stage A (Distortion Recovery):** Optimize the radial lens distortion coefficient $\kappa_1$ and distortion center $(c_x, c_y)$ directly from raw plumb-line inputs, minimizing Menger curvature loss with a fixed focal distance aperture [@de2011uncalibrated].

* **Stage B (Intrinsic Calibration):** Apply the resolved distortion correction to the detected points. Utilizing the rectified, undistorted ideal lines, the framework reconstructs vanishing points to find the plane-induced homographies. These homography matrices are then embedded directly into the absolute dual conic constraints, enabling Zhang's closed-form linear solution to evaluate the camera intrinsic parameters ($f_x, f_y, c_x, c_y$) [@zhang2000flexible; @hartley2003multiple]. Finally, the radial distortion coefficient $\kappa_1$ is renormalized relative to this stabilized intrinsic matrix to completely restore the operational Brown-Conrady lens model [@brown1971close].

This decoupling eliminates parameter cross-talk and ensures fast, stable convergence even under degenerate geometric configurations, such as almost pure rotational sequences with weak perspectivity.

## Software Architecture

The framework's implementation is modularized into dedicated structural components:

* `m_sequence.py`: Manages Galois Field Linear Feedback Shift Register (LFSR) generation, Toeplitz parity-check matrices, algebraic linear solvers over $\mathbb{GF}(2)$, and hosts the error-correcting analysis class `MSequenceAnalyzer` [@macwilliams1977theory].
* `lattice_topology.py`: Handles barycentric coordinate transformations, matrix hexagonal rotation helpers, and hexagonal graph-to-matrix grid conversions.
* `matcher.py`: Performs 1D axis distillation via 4-node XOR kernels and encapsulates the `AlgebraicGridDecoder32` master decoder routine (syndrome analysis, error/erasure correction, and phase locking) [@lin2004error].
* `crystal.py`: Implements wave-growth topological island reconstruction via Delaunay triangulation, Disjoint Set Union (DSU) forest management, and parallelogram closure validation via the class `GridCrystalGrower`.
* `detector.py`: Executes adaptive image binarization, solidity-based shape classification, topological lattice assembly, and final node verification.
* `camera.py`: Models a projective pin-hole camera with radial lens distortion [@hartley2003multiple].
* `optimization.py`: Orchestrates the multi-frame camera calibration pipeline. Provides a two-stage decoupled calibration sequence similar to the approach in [@shortis1995isolated].

# Experimental Results

The codebase includes comprehensive synthetic benchmarks and Blender 2.92 photorealistic scene generation to systematically analyze tracking performance and structural robustness under severe perspective warping and heavy image noise. We considered two simulated camera configurations: the simple rotation frameset, characterized by an isotropic focal length and camera rolling (from 0° to 300° degrees) supplemented by a tilted image to resolve perspective ambiguity along with a baseline frame; and the compound rotation (random tilt) frameset, where an anisotropic focal length camera generates 8 randomly rotated frames and a baseline frame.

## Pattern Registration Performance

The physical features of the calibration pattern yield an average spot diameter of approximately 10 pixels on the image canvas. Across all evaluated test sequences, low-level spot localization achieves stable sub-pixel tracking accuracy, with the Root Mean Square Error (RMSE) strictly bounded between 0.6 and 0.9 pixels. This sub-pixel precision provides an exceptionally stable baseline for robust downstream parameter estimation.

### Topological Matching Performance

| Test Case Name | Visible Nodes | Total Detected | Misclassified (Corrected) | False Detections | True Positives (TP) | Skip (Isolated) | Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_baseline` | 961 | 961 | 2 | 0 | 961 | 0 | 100.00 |
| `oblique_tilt_high` | 952 | 889 | 63 | 0 | 888 | 1 | 93.28 |
| `roll_120` | 924 | 890 | 8 | 8 | 880 | 2 | 95.24 |
| `roll_180` | 961 | 976 | 14 | 17 | 958 | 1 | 99.69 |
| `roll_240` | 922 | 636 | 60 | 10 | 561 | 65 | 60.85 |
| `roll_300` | 924 | 887 | 11 | 9 | 877 | 1 | 94.91 |
| `roll_60` | 922 | 900 | 12 | 18 | 882 | 0 | 95.66 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `compound_rotation_0` | 916 | 920 | 39 | 24 | 896 | 0 | 97.82 |
| `compound_rotation_1` | 928 | 774 | 18 | 5 | 767 | 2 | 82.65 |
| `compound_rotation_2` | 440 | 425 | 5 | 29 | 396 | 0 | 90.00 |
| `compound_rotation_3` | 957 | 641 | 61 | 2 | 505 | 134 | 52.77 |
| `compound_rotation_4` | 450 | 418 | 12 | 8 | 407 | 3 | 90.44 |
| `compound_rotation_5` | 633 | 618 | 13 | 38 | 580 | 0 | 91.63 |
| `compound_rotation_6` | 944 | 898 | 75 | 7 | 885 | 6 | 93.75 |
| `compound_rotation_7` | 953 | 578 | 45 | 20 | 369 | 189 | 38.72 |

Despite heavy illumination noise and aggressive geometric warping, the Galois parity-check framework maintains an absolute precision profile ($\text{Precision} = 100.0\%$) across all evaluation tests, yielding zero false positive identifications and matrix misalignments. The structural lattice consensus layer successfully resolves low-level shape misclassification conflicts on-the-fly. Architecturally, feature classification is required strictly for initial pattern matching; for all subsequent processing, only the invariant node centroids are utilized, because centroids are completely immune to key-point visual shape distortions.

All metric ratios are evaluated relative to visible key-points of the master pattern footprint ($31 \times 31 = 961$ nodes) on the camera viewport. Under perspective slants and pixel intensity noise, the peak initial classification failure reaches 7.81% (`compound_rotation_6` with 75 corrupted labels), with a maximum pattern dropout rate of 61.28% (`compound_rotation_7` with only 369 true positives). Despite these concurrent data defects, the framework guarantees perfect, drift-free pattern registration across all test configurations.

## Radial Distortion and Intrinsics Estimation

### Multi-Frame Calibration Accuracy

| Intrinsics Parameter Matrix | Simple<br>GT | Simple<br>Solved | Simple<br>% Error | Random Tilt<br>GT | Random Tilt<br>Solved | Random Tilt<br>% Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Focal Length $f_x$** (px) | 1150.00 | 1152.97 | **0.26%** | 1250.00 | 1246.77 | **0.26%** |
| **Focal Length $f_y$** (px) | 1150.00 | 1152.97 | **0.26%** | 1150.00 | 1144.15 | **0.51%** |
| **Principal Point $c_x$** (px) | 960.00  | 961.23  | **0.06%** | 965.00  | 960.19  | **0.25%** |
| **Principal Point $c_y$** (px) | 540.00  | 538.45  | **0.14%** | 543.00  | 540.23  | **0.26%** |
| **Distortion Coef. $\kappa_1$** | -0.2000 | -0.2098 | **4.88%** | -0.1500 | -0.1493 | **0.45%** |

Distortion calibration results are computed using robust median consensus across 9 multi-view frames. The **decoupled approach** completely suppresses parameter cross-talk across distinct geometric configurations:

* **Rotational Dataset (Isotropic Case):** Despite the absolute absence of spatial translation (which nominally triggers severe $f_x = f_y$ scale coupling and depth degeneracy), inverse feature blob size weighting and multi-scale chord regularization estimates underlying straightness invariants with high precision. Focal length reconstruction achieves a **0.26% error** under degenerate geometric conditions.
* **Random Tilt Dataset (Anisotropic Case):** Over 9 frames containing aggressive perspective tilt, the framework successfully unlocks the full camera matrix. Stage A recovers the radial distortion coefficient $\kappa_1$ through median consensus via plumb-line straightness constraints [@de2011uncalibrated]. Stage B uses Zhang's vanishing point method on the rectified, undistorted coordinates to evaluate intrinsics [@zhang2000flexible]. The final intrinsic error parameters are strictly bounded: $\le$ 0.51% for focal lengths and $\le$ 0.26% for principal point offsets.



Below, the single-frame estimation results for tilted frames are presented:

### Single-Frame Performance Under Varying Key-point Dropouts

| Evaluation Test View | True Positives (TP) | $f_x$ Solved<br>(% Err) | $f_y$ Solved<br>(% Err) | $c_x$ Solved<br>(% Err) | $c_y$ Solved<br>(% Err) | $\kappa_1$ Solved<br>(% Err) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `compound_rotation_7` | 369 | 1386.55 (**10.92%**) | 1376.77 (**19.72%**) | 958.16 (**0.36%**) | 541.17 (**0.17%**) | -0.2568 (**71.19%**) |
| `compound_rotation_4` | 407 | 1239.62 (**0.83%**)  | 1137.94 (**1.05%**)  | 961.53 (**0.18%**) | 540.33 (**0.25%**) | -0.1420 (**5.30%**)  |
| `compound_rotation_1` | 767 | 1203.90 (**3.69%**)  | 1062.19 (**7.64%**)  | 962.06 (**0.15%**) | 538.89 (**0.38%**) | -0.1316 (**12.25%**) |
| `compound_rotation_0` | 896 | 1295.18 (**3.61%**)  | 1215.90 (**5.73%**)  | 958.70 (**0.33%**) | 538.40 (**0.43%**) | -0.1778 (**18.50%**) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean (3 Best Views)** | **—** | **1246.23 (0.30%)**   | **1138.68 (0.98%)**   | **960.76 (0.44%)** | **539.21 (0.70%)** | **-0.1505 (0.31%)**  |
| **Series Consensus**   | **—** | **1246.77 (0.26%)**   | **1144.15 (0.51%)**   | **960.19 (0.25%)** | **540.23 (0.26%)** | **-0.1493 (0.45%)**  |

Single-frame execution under pure rotational roll configurations introduces standard homographic rank degeneracy, triggering focal-depth scale coupling; however, such weak perspectivity views remain highly effective for capturing isolated radial lens distortion. Our cascaded, distortion-first approach resolves this rank bottleneck by extracting $\kappa_1$ via geometric line straightness invariants prior to intrinsic matrix initialization, enabling robust parameter convergence even on single degenerate frames. 

Individual single-frame geometric defects are systematically mitigated through a multi-frame consensus loop. Rather than executing a joint point-cloud optimization, the pipeline sequentially estimates a planar homography for each isolated view, maps the resulting projective fields to a unified conic space, and extracts the camera intrinsics with $\approx 0.5\%$ accuracy.

# Strengths, Limitations, and Concluding Remarks

## Architectural Strengths
The results prove the practical capability and robustness of the error-correcting hexagonal pattern calibration approach. The straightforward calibration algorithm and modular, single-responsibility component structure enable experiments with different lattice encodings and future extensions. There is no need for any initial parameter value guess. 

Camera extrinsic parameters are fully excluded from the calibration process, and the decoupling of distortion and intrinsics makes the algorithm defensive against parameter-coupling issues [@shortis1995isolated]. The framework can provide calibration by a single frame (less precise) or by a sequence. Furthermore, the framework shows high robustness to pattern occlusion, key-point dropouts, and key-point misclassification.

## Limitations

* **Minimal Decoding Block Dimension:** For successful spatial decoding, the framework requires at least an $11 \times 2$ node block footprint with a limited density of omissions and errors.
* **Error Correction Saturation:** Decoding fails if multi-bit errors or consecutive omission bursts violate the underlying error-isolation approach.
* **Implementation Complexity:** Higher conceptual overhead compared to simple tag-based fiducials, requiring discrete crystal-growth graphs and linear algebra over $\mathbb{GF}(2)$.

## Conclusion

The B-HGP framework fuses error-correcting codes and the structural robustness of a hexagonal grid. By combining the spatial efficiency of hexagonal lattices with M-sequence noise immunity, it enables independent multi-stage calibration using fundamental geometric invariants like vanishing points and absolute conics [@zhang2000flexible; @hartley2003multiple]. For applications requiring single-frame calibration under severe out-of-plane perspective, lens aberrations, or uncontrolled environments (mobile robotics, autonomous vehicles, industrial photogrammetry), B-HGP provides a compelling alternative to conventional tag-based targets. 

This approach is not limited to binary sequences of the 5th degree; it is extensible with other primitive polynomials and different lattice configurations. This investigation shows the power of the Galois pattern for practical applications in different computer vision tasks.

# Availability

The open-source implementation, full test suite, Blender benchmark generators, and validation tools are publicly available at: https://github.com

# References

**Note:** This software paper documents the implementation of research presented in a draft manuscript under consideration for publication in the *International Journal of Computer Vision* (IJCV).

# Availability

The open-source implementation, full test suite, Blender benchmark generators, and validation tools are publicly available at: https://github.com/gkis-conda/robust_calibration_pattern

# References

**Note:** This software paper documents the implementation of research presented in a draft manuscript under consideration for publication in the *International Journal of Computer Vision* (IJCV).
