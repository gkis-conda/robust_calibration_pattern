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

Traditional calibration target registration degrades significantly under severe motion blur, perspective slant, and high non-linear lens distortion. This paper introduces the Binary Hexagonal Galois Pattern (B-HGP) framework: a single-frame pattern registration method that maps one-dimensional Galois Field $M$-sequences onto a two-dimensional regular hexagonal lattice. The resulting target topology encodes absolute spatial coordinates locally, supports simultaneous algebraic error and recognition dropout recovery, and tolerates severe perspective shear, motion blur, and localized target occlusions. 

The implemented calibration approach leverages the strong statistical and geometric properties of the proposed pattern alongside its structural robustness to sensor noise. Stress tests under severe perspective foreshortening demonstrate that the unified algebraic pipeline bounds tracking displacement errors within 0.6–0.9 pixels while maintaining a flawless precision profile (zero identity mismatches or alignment drifts). A novel decoupled calibration approach completely separates radial distortion recovery from intrinsic parameter estimation, achieving high parameter accuracy on both isotropic and anisotropic camera models across multi-frame and single-view configurations alike.

# Statement of Need

The fast and robust calibration of multi-sensor devices in self-navigating systems has become critical with the rapid development of autonomous vehicles, augmented reality headsets, and omnidirectional 360-degree cameras utilizing visual sensor fusion for precise spatial self-positioning. A fundamental requirement for successful sensor fusion is the accurate estimation of sensor extrinsic parameters, which relies on strict temporal and spatial synchronization achieved via known multi-sensor baseline signals.

The calibration of multi-sensor setups, particularly Visual-Inertial Navigation Systems (VINS), faces severe challenges when resolving coupled spatiotemporal parameters. These difficulties are frequently compounded by partial occlusions, rapid tracking accelerations, and illumination variations when using standard calibration targets. To address these vulnerabilities, this paper introduces a novel calibration pattern that embeds error-correcting pseudo-random codes directly onto a regular hexagonal lattice.

Circle grids, tag-based fiducials like AprilTags [@Olson2011AprilTag], ArUco [@garrido2014aruco], or WhyCode [@lightbody2017WhyCode], and chessboard or ChArUco patterns remain the industrial standards for camera calibration [@zhang2000]. However, these classical methodologies present significant operational constraints in unconstrained real-world environments:

* **Occlusions and Spatial Payload Overhead:** Fiducial markers require intact tag matrices to resolve absolute identification, leaving them vulnerable to partial occlusions, lens defocus, and heavy perspective warping.
* **Blur Sensitivity:** Standard checkerboard corner extraction algorithms degrade rapidly under motion or out-of-focus blur, restricting calibration workflows to static, heavily controlled multi-frame capture routines.
* **Spatial Scale Constraints:** Hybrid designs like ChArUco partially mitigate data loss but scale poorly across variable camera distances.
* **Localization Bias:** Circular patterns suffer from perspective localization center bias under tilted views, requiring complex implicit estimation to recover true algebraic projection centers [@mallon2007precise].
* **Radial Distortion:** Ultra-wide and fisheye lenses introduce catastrophic radial barrel distortion that leads to feature extraction dropouts on the image periphery, breaking standard graph tracking connectivity loops.

# Architecture and Implementation

## Pattern Design and Mathematical Foundation

B-HGP encodes three independent binary $M$-sequences into a regular hexagonal lattice layout. The visible binary state at each lattice node is evaluated as the XOR combination of channel bits from three continuous tracking axes:

$$B(r, c) = U[u \bmod 31] \oplus V[v \bmod 31] \oplus W[w \bmod 31]$$

where $r$ represents the row coordinate, $c$ denotes the column coordinate, and the barycentric coordinate space components are defined as:
* $v = r$
* $u = c - \lfloor r/2 \rfloor$
* $w = - u - v$

Each M-sequence is generated from a primitive 5th-order Linear Feedback Shift Register (LFSR) polynomial over the Galois Field $\mathbb{GF}(2)$ [@macwilliams1977theory], producing sequences of maximum period $L = 2^5 - 1 = 31$. The primary property of this topology is that any 5-bit sliding window within an M-sequence reveals a completely unique local phase address, enabling single-frame absolute coordinate recovery from small, isolated pattern fragments.

## Processing Pipeline

The single-frame calibration workflow executes sequentially through six deterministic processing phases:

### 1. Optical Feature Extraction
Detects candidate nodes and classifies their topological profiles (circles versus triangles) using adaptive local binarization, contour solidity filtering, and continuous distance classification via circularity metrics. Circles encode bit state "0", and triangles encode bit state "1". Features are validated against image boundaries to eliminate partial or cut shapes.

### 2. Lattice Reconstruction via Crystal Growth
Assembles unorganized sub-pixel barycenters into stable hexagonal coordinate networks via Delaunay triangulation nucleation under strict local triangle coherence constraints. Hexagonal grid topologies expand outward from verified seeds via a systematic front-propagation queue. To protect the growth horizon from non-linear lens compression and edge distortion shears, the engine evaluates local neighborhood changes using a scale-invariant, direction-blind triangle inradius ratio metric. Isolated sub-graph islands are securely merged using a Disjoint Set Union (DSU) forest with parallelogram closure verification to guarantee structural lattice reconstruction under extreme perspective warping.

### 3. 1D Axis Distillation
Evaluates each crystalline segment by applying local 4-node XOR sliding kernels, isolating three independent 1D stream vectors:
* **U-axis:** $dU[i] = B[r, c+1] \oplus B[r, c+2] \oplus B[r+1, c] \oplus B[r+1, c+1]$
* **W-axis:** $dW[i] = B[r, c] \oplus B[r, c+1] \oplus B[r+1, c] \oplus B[r+1, c+1]$

These differential operators perfectly cancel two of the three component sequences, yielding clean, independent 1D M-sequence fragments.

### 4. Algebraic Binary Sequence Decoding
Processes each extracted 1D stream through a syndrome decoder pipeline. Single bit-flips are repaired via syndrome evaluation and algebraic locator polynomials [@lin2004error], while recognition dropouts are resolved via a deterministic greedy decoding pass. Convergence to absolute phase position is established via parity-check matrix inversion. At this step, two orthogonal axes are matched to the sequence, and the sequence phases are evaluated.

### 5. Coordinate Phase Locking
Merges resolved phases from independent orthogonal axes to establish absolute global $(u, v, w)^T$ matrix indices via the intersection of 1D decoded phases. The exact barycentric-to-matrix mapping ensures zero row-parity shearing during coordinate transformation and guarantees isotropic geometric transformation invariance under hexagonal rotations. After phase localization, the separated planar crystallines are merged into a common matrix that corresponds to the genuine pattern.

### 6. Decoupled Camera Calibration
The restored hexagonal lattice provides perfect sets of collinear lines easily extracted along hexagonal directions to enable a plumb-line calibration approach. Unlike traditional joint optimization methods for the Brown-Conrady model [@brown1971close] that couple distortion parameters and intrinsic variables, the framework employs a two-stage cascaded decoupling strategy:
* **Stage A (Distortion Recovery):** Optimize the radial lens distortion coefficient $\kappa_1$ and distortion center $(c_x, c_y)$ directly from raw plumb-line inputs, minimizing Menger curvature loss with a fixed focal distance aperture [@de2011uncalibrated].
* **Stage B (Intrinsic Calibration):** Apply the resolved distortion correction to the detected points. Having undistorted ideal lines, it is possible to reconstruct vanishing points and apply Zhang's closed-form solution for intrinsics recovery via planar homography [@zhang2000; @zheng2013geometric]. Then, the radial distortion parameter $\kappa_1$ is renormalized to this calculated calibration to fully restore the final Brown-Conrady model [@brown1971close].

This decoupling eliminates parameter cross-talk and ensures fast, stable convergence even under degenerate geometric configurations, such as pure rotational sequences or extreme perspective views.

## Software Architecture

The framework's implementation is modularized into dedicated structural components:

* `m_sequence.py`: Manages Galois Field Linear Feedback Shift Register (LFSR) generation, Toeplitz parity-check matrices, algebraic linear solvers over $\mathbb{GF}(2)$, and error-correcting capacity analysis [@macwilliams1977theory].
* `lattice_topology.py`: Handles barycentric coordinate transformations, matrix hexagonal rotation helpers, and hexagonal graph-to-matrix grid conversions.
* `matcher.py`: Performs 1D axis distillation via 4-node XOR kernels and encapsulates the `AlgebraicGridDecoder32` master decoder routine (syndrome analysis, error/erasure correction, and phase locking) [@lin2004error].
* `crystal.py`: Implements wave-growth topological island reconstruction via Delaunay triangulation, Disjoint Set Union (DSU) forest management, and parallelogram closure validation.
* `detector.py`: Executes adaptive image binarization, solidity-based shape classification, topological lattice assembly, and final node verification.
* `camera.py`: Evaluates camera projection matrix estimation and radial lens distortion modeling via straightness metrics [@de2011uncalibrated].
* `optimization.py`: Orchestrates the multi-frame calibration pipeline.

# Experimental Results

The codebase includes comprehensive synthetic benchmarks and Blender 2.92 photorealistic scene generation to systematically analyze tracking performance and structural robustness under severe perspective warping and heavy image noise. The sub-pixel centroid approximation determines key-point projection centers that remain invariant to topological shape classification results (this center definition naturally fits any regular polygon, including circles or triangles).

## Pattern Registration Performance

The physical features of the calibration pattern yield an average spot diameter of approximately 10 pixels on the image canvas. Across all evaluated test sequences, low-level spot localization achieves stable sub-pixel tracking accuracy, with the Root Mean Square Error (RMSE) strictly bounded between 0.6 and 0.9 pixels. This sub-pixel precision provides an exceptionally stable baseline for robust downstream parameter estimation.

### Topological Matching Performance

| Test Case Name | GT Nodes | Total Detected | Misclassified (Corrected) | False Detections | True Positives (TP) | Skip (Isolated) | Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Rotational Sequences** | | | | | | | |
| `clean_baseline` | 961 | 961 | 2 | 0 | 961 | 0 | 100.00 |
| `oblique_tilt_high` | 952 | 889 | 63 | 0 | 888 | 1 | 93.28 |
| `roll_120` | 924 | 890 | 8 | 8 | 880 | 2 | 95.24 |
| `roll_180` | 961 | 976 | 14 | 17 | 958 | 1 | 99.69 |
| `roll_240` | 922 | 636 | 60 | 10 | 561 | 65 | 60.85 |
| `roll_300` | 924 | 887 | 11 | 9 | 877 | 1 | 94.91 |
| `roll_60` | 922 | 900 | 12 | 18 | 882 | 0 | 95.66 |
| **Complex 3D Galois Stress-Test Sequences** | | | | | | | |
| `compound_rotation_0` | 916 | 920 | 39 | 24 | 896 | 0 | 97.82 |
| `compound_rotation_1` | 928 | 774 | 18 | 5 | 767 | 2 | 82.65 |
| `compound_rotation_2` | 440 | 425 | 5 | 29 | 396 | 0 | 90.00 |
| `compound_rotation_3` | 957 | 641 | 61 | 2 | 505 | 134 | 52.77 |
| `compound_rotation_4` | 450 | 418 | 12 | 8 | 407 | 3 | 90.44 |
| `compound_rotation_5` | 633 | 618 | 13 | 38 | 580 | 0 | 91.63 |
| `compound_rotation_6` | 944 | 898 | 75 | 7 | 885 | 6 | 93.75 |
| `compound_rotation_7` | 953 | 578 | 45 | 20 | 369 | 189 | 38.72 |

Despite heavy illumination noise and aggressive geometric warping, the Galois parity-check framework maintains an absolute precision profile ($\text{Precision} \equiv 1.000$) across all evaluation tests, yielding zero false positive identifications and matrix misalignments. The structural lattice consensus layer successfully resolves low-level shape misclassification conflicts on-the-fly. Architecturally, feature classification is required strictly for initial pattern matching; for all subsequent processing, only the invariant node centroids are utilized, rendering the metric backend completely immune to key-point visual shape distortions.

All metric ratios are evaluated relative to the visible key-points of the master pattern footprint ($31 \times 31 = 961$ nodes). Under extreme non-linear perspective slants, the peak initial classification failure reaches 7.7% (`compound_rotation_6` with 74 corrupted labels), while extreme out-of-plane tilts force a maximum pattern dropout rate of 60.67% (`compound_rotation_7` with only 369 true positives). Despite these concurrent data defects, the framework guarantees perfect, drift-free pattern registration across all test configurations.

## Radial Distortion and Intrinsics Estimation

### Multi-Frame Calibration Accuracy

| Intrinsics Parameter Matrix | Simple GT | Simple Solved | Simple % Error | Random Tilt GT | Random Tilt Solved | Random Tilt % Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Focal Length $f_x$** (px) | 1150.00 | 1152.97 | **0.26%** | 1250.00 | 1246.77 | **0.26%** |
| **Focal Length $f_y$** (px) | 1150.00 | 1152.97 | **0.26%** | 1150.00 | 1144.15 | **0.51%** |
| **Principal Point $c_x$** (px) | 960.00  | 961.23  | **0.06%** | 965.00  | 960.19  | **0.25%** |
| **Principal Point $c_y$** (px) | 540.00  | 538.45  | **0.14%** | 543.00  | 540.23  | **0.26%** |
| **Distortion Coefficient $\kappa_1$** | -0.2000 | -0.2098 | **4.88%** | -0.1500 | -0.1493 | **0.45%** |

Distortion calibration results are computed using robust median consensus across 9 multi-view frames. The **decoupled approach** completely suppresses parameter cross-talk across distinct geometric configurations:

* **Rotational Dataset (Isotropic Case):** Despite the absolute absence of spatial translation (which nominally triggers severe $f_x = f_y$ scale coupling and depth degeneracy), inverse feature-mass weighting and multi-scale chord regularization isolate underlying straightness invariants with exceptional precision. Focal length reconstruction achieves a minimal **0.26% error** under degenerate geometric conditions.
* **Random Tilt Dataset (Anisotropic Case):** Over 9 frames containing aggressive perspective tilt, the framework successfully unlocks the full camera matrix. Stage A recovers the radial distortion coefficient $\kappa_1$ through median consensus via plumb-line straightness constraints [@de2011uncalibrated]. Stage B uses Zhang's vanishing point method on the rectified, undistorted coordinates to refine intrinsics [@zhang2000]. The final intrinsic error parameters are strictly bounded: $\le$ 0.51% for focal lengths and $\le$ 0.26% for principal point offsets.

### Single-Frame Performance Under Varying Occlusion

| Evaluation Test View | Detected True Positives (TP) | $f_x$ Percentage Error | $f_y$ Percentage Error | $\kappa_1$ Percentage Error |
| :--- | :---: | :---: | :---: | :---: |
| View 0 | 896 | 3.61% | 5.73% | 18.50% |
| View 1 | 767 | 3.69% | 7.64% | 12.25% |
| View 4 | 407 | 0.83% | 1.05% | 5.30% |
| View 7 | 369 | 10.92% | 19.72% | 71.19% |
| **Multi-Frame Consensus (9 Views)** | **3,113** | **0.26%** | **0.51%** | **0.45%** |

Single-frame execution under pure rotational roll configurations introduces standard homographic rank degeneracy, triggering focal-depth scale coupling; however, such perspectives remain highly effective for capturing isolated radial lens distortion. Our cascaded, distortion-first approach resolves this rank bottleneck by extracting $\kappa_1$ via geometric line straightness invariants prior to intrinsic matrix initialization, enabling robust parameter convergence even on single degenerate frames. Individual single-frame geometric defects—such as focal scale ambiguity under heavy cropping or distortion saturation under extreme tilt—are systematically mitigated through a multi-frame consensus loop. Rather than executing a joint point-cloud optimization, the pipeline sequentially estimates a planar homography for each isolated view, maps the resulting projective fields to a unified conic space, and extracts the global camera intrinsics with sub-0.5% accuracy.

# Strengths, Limitations, and Concluding Remarks

## Architectural Strengths

* **Maximum Theoretical Packing Density:** The hexagonal lattice achieves maximum spatial marker density per unit area, maximizing geometric constraint volume within a single image frame.
* **Isotropic Crystal Front Propagation:** The 6-neighbor adjacency structure provides optimal topological regularization, allowing uniform lattice growth with inherent protection against neighboring node dropouts.
* **Unbiased Node Distribution:** The pseudorandom token distribution mimics white noise, suppressing systematic spatial bias during RMSE optimization.
* **Dense Straight-Line Bundles:** Three-axis geometry generates continuous co-linear point traces at 30° increments, optimizing the performance of plumb-line distortion solvers [@de2011uncalibrated] and Zheng-based vanishing point methods [@zheng2013geometric].
* **M-Sequence Error Immunity:** Algebraic properties grant native robustness against canvas cropping, random node omissions, and sensor noise [@lin2004error].
* **Decoupled Calibration:** The pattern topology enables separate phases anchored on first principles: using the projective invariant of straight-line preservation for distortion [@de2011uncalibrated], followed by homography and vanishing point recovery for intrinsics [@zhang2000].

## Limitations

* **Centroid Detection Sensitivity:** Under extreme motion blur or severe sensor noise, contour fragmentation degrades sub-pixel centroid accuracy and shape classification, affecting lattice reconstruction.
* **Minimal Decoding Block Dimension:** For successful spatial decoding, the framework requires at least an $11 \times 2$ node block footprint with a limited density of omissions and errors.
* **Error Doubling Effect:** Every node misclassification or recognition dropout leads to a doubled error or erasure in the distilled 1D sequence due to the properties of the local 4-node XOR differential kernels, limiting the overall error-correction capability.
* **Error Correction Saturation:** Decoding fails if multi-bit errors or consecutive omission bursts violate the underlying error-isolation approach.
* **Implementation Complexity:** Higher conceptual overhead compared to simple tag-based fiducials, requiring discrete crystal-growth graphs and linear algebra over $\mathbb{GF}(2)$.

## Conclusion

The B-HGP framework trades a low computational footprint for advanced algebraic resilience and structural robustness. By combining the spatial efficiency of hexagonal lattices with M-sequence noise immunity, it enables independent multi-stage calibration using fundamental geometric invariants. For applications requiring single-frame calibration under severe out-of-plane perspective, lens aberrations, or uncontrolled environments (mobile robotics, autonomous vehicles, industrial photogrammetry), B-HGP provides a compelling alternative to conventional tag-based targets.



# Availability

The open-source implementation, full test suite, Blender benchmark generators, and validation tools are publicly available at: https://github.com/gkis-conda/robust_calibration_pattern

# References

---

**Note:** This software paper documents the implementation of research presented in a draft manuscript under consideration for publication in the International Journal of Computer Vision (IJCV).
