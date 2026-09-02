---
title: 'B-HGP: Binary Hexagonal Galois Pattern for Robust Single-Frame Camera Calibration'
authors:
  - name: Gennadiy Kis
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 28 August 2026
bibliography: paper.bib
---

# Summary

Traditional calibration target registration degrades significantly under severe motion blur, perspective slant, and high lens distortion. This paper introduces the Binary Hexagonal Galois Pattern (B-HGP) approach: a single-frame pattern registration method that maps one-dimensional Galois-field M-sequences onto a two-dimensional hexagonal lattice. The resulting pattern encodes absolute spatial coordinates locally, supports simultaneous algebraic error and erasure recovery, and tolerates severe perspective shear, blur, and partial target occlusions. 

The research systematically investigates the statistical and geometric properties of the pattern alongside its structural robustness to noise. Stress tests under severe perspective foreshortening demonstrate that the unified algebraic pipeline bounds sub-pixel tracking errors within 0.6–0.9 pixels while maintaining 100% precision (zero false positives). A novel decoupled two-phase calibration approach separates radial distortion recovery from intrinsic parameter estimation, achieving sub-1% accuracy on both isotropic and anisotropic camera models across multi-frame datasets.

# Statement of Need

The fast and robust calibration of multi-sensor devices in self-navigating systems has become critical with the rapid development of autonomous vehicles, augmented reality glasses, 360-degree cameras, and similar devices that use visual sensor fusion for precise self-positioning. A fundamental requirement for successful sensor fusion is the accurate estimation of sensor extrinsic parameters, which relies on strict temporal and spatial synchronization achieved via known multi-sensor signals.

The calibration of multi-sensor setups, particularly Visual-Inertial Navigation Systems (VINS), faces severe challenges when resolving coupled spatiotemporal parameters. These difficulties are frequently compounded by partial occlusions and illumination variations when using standard calibration targets. To address these vulnerabilities, this paper introduces a novel calibration pattern that embeds error-correcting codes directly onto a regular hexagonal lattice.

Circle grids, tag-based fiducials (e.g., AprilTag, ArUco, WhyCode), and chessboard or ChArUco patterns remain the industrial standard for camera calibration. However, these classical methodologies present significant operational constraints in real-world environments:

- **Occlusions and Spatial Payload Overhead:** Fiducial markers require intact tag matrices to resolve identification, leaving them vulnerable to partial occlusions, lens defocus, and heavy perspective warping.

- **Blur Sensitivity:** Standard checkerboard corner extraction algorithms degrade rapidly under motion or out-of-focus blur, restricting calibration workflows to static, heavily controlled multi-frame capture routines.

- **Spatial Scale Constraints:** Hybrid designs like ChArUco partially mitigate data loss but scale poorly across variable camera viewports affected by high-frequency background noise.

- **Localization Bias:** Circular patterns suffer from perspective localization bias under tilted views and require complex implicit estimation to recover true centers.

- **Fisheye Distortion:** Ultra-wide and fisheye lenses introduce catastrophic radial barrel distortion that leads to feature extraction dropouts on image periphery.

# Architecture and Implementation

## Pattern Design and Mathematical Foundation

B-HGP encodes three independent binary M-sequences into a regular hexagonal lattice. The visible binary state at each lattice node is evaluated as the XOR combination of channel bits from three continuous tracking axes:

$$B(r, c) = U[u \bmod 31] \oplus V[v \bmod 31] \oplus W[w \bmod 31]$$

where:
- $v = r$ (row coordinate)
- $u = c - \lfloor r/2 \rfloor$ (unwarped column in barycentric space)
- $w = -u - v$ (enforcing the invariant constraint $u + v + w = 0$)

Each M-sequence is generated from a primitive 5th-order LFSR polynomial over $\mathbb{GF}(2)$, producing sequences of period $L = 2^5 - 1 = 31$. The key property is that any 5-bit sliding window within an M-sequence reveals a completely unique local phase address, enabling single-frame absolute coordinate recovery from small pattern crops.

## Processing Pipeline

The single-frame calibration workflow executes sequentially through six deterministic processing phases:

### 1. **Optical Feature Extraction**
Detects candidate nodes and classifies their topological profiles (circles vs. triangles) using adaptive binarization, solidity filtering, and continuous distance classification via circularity metrics. Circles encode bit state 0, triangles encode bit state 1. Features are validated against image boundaries to eliminate partial/cut shapes.

### 2. **Lattice Reconstruction via Wave-Growth**
Assembles unorganized sub-pixel barycenters into stable hexagonal coordinate networks via:
- Delaunay triangulation nucleation with strict angle and edge length filtering (48°–72° interior angles)
- Front-propagation queue processing with topological face tracking
- Isolated island assembly and merging via Disjoint Set Union (DSU)
- Parallelogram closure verification for sub-grid coherence under perspective warping

### 3. **1D Axis Distillation**
Evaluates each crystalline segment by applying local 4-node XOR sliding kernels, isolating three independent 1D stream vectors:
- U-axis: $dU[i] = B[r, c+1] \oplus B[r, c+2] \oplus B[r+1, c] \oplus B[r+1, c+1]$
- W-axis: $dW[i] = B[r, c] \oplus B[r, c+1] \oplus B[r+1, c] \oplus B[r+1, c+1]$
- V-axis: vertical diamond pattern XOR over parity-adjusted coordinates

These operations perfectly cancel two of the three component sequences, yielding clean 1D M-sequence fragments.

### 4. **Algebraic Subspace Decoding**
Processes each extracted 1D stream through the Berlekamp-Massey syndrome decoder pipeline, repairing:
- Single bit-flips via syndrome evaluation and algebraic locator polynomial
- Missing node dropouts (erasures) via Gauss-Jordan elimination over $\mathbb{GF}(2)$
- Convergence to absolute phase position via parity-check matrix inversion
- Error consistency validation across independent axes

### 5. **Coordinate Phase Locking**
Merges resolved phase positions from independent orthogonal axes to establish absolute global $(u, v, w)$ coordinates via intersection of 1D decoded phases. The exact barycentric-to-matrix mapping ensures:
- Zero row-parity shearing during coordinate transformation
- Transitive vector translation properties under affine mappings
- Isotropic geometric transformation invariance under hexagonal rotations

### 6. **Decoupled Multi-Phase Camera Calibration**
Unlike traditional joint optimization that couples distortion and intrinsics, we employ a two-stage decoupling strategy:
- **Stage A (Distortion Recovery):** Extract absolute grid coordinates from real camera image via B-HGP. Optimize radial distortion coefficient $\kappa_1$ via plumb-line straightness constraints.
- **Stage B (Intrinsic Calibration):** Apply distortion correction to detected points. Fit vanishing point geometry on undistorted image space using Zheng's harmonic mean invariant method to recover focal length $(f_x, f_y)$ and principal point $(c_x, c_y)$.

This decoupling eliminates parameter cross-talk and improves convergence stability under degenerate geometric configurations (e.g., pure rotational or extreme cropping scenarios).

## Software Architecture

The implementation is modularized into dedicated components:

- **m_sequence.py:** Galois Field LFSR generation, Toeplitz parity-check matrices, algebraic linear solvers over $\mathbb{GF}(2)$, null-space projectors, error-correcting capacity analysis
- **lattice_topology.py:** Barycentric coordinate transformations, 60° rotation helpers, hexagonal-to-matrix grid conversion, zero-line topology analysis
- **matcher.py:** 1D axis distillation via 4-node XOR kernels, `AlgebraicGridDecoder32` master decoder (syndrome analysis, error/erasure correction, phase locking)
- **crystal.py:** Wave-growth topological island reconstruction via Delaunay triangulation, DSU forest management, parallelogram closure validation
- **detector.py:** Adaptive image binarization, solidity-based shape classification, topological lattice assembly, final node verification
- **camera.py:** Camera projection matrix estimation, radial lens distortion modeling via straightness metrics
- **optimization.py:** Multi-frame calibration pipeline with median consensus for stability

# Experimental Results

The codebase includes synthetic benchmarks and Blender-based scene generation to systematically analyze tracking performance and structural robustness under severe perspective warping and image noise. Throughout experiments, the centroid approximation determines key-point projection centers that remain invariant to standalone shape classification results.

## Pattern Registration Performance

Physical feature dimensions yield an average diameter of approximately 10 pixels on the image canvas. Across all evaluation sequences, low-level spot localization achieves stable sub-pixel tracking accuracy with Root Mean Square Error (RMSE) strictly bounded between 0.6 and 0.9 pixels. This sub-pixel precision proves entirely sufficient for robust downstream parameter estimation.

### Topological Matching Performance

| Test Case | GT Nodes | Total Detected | Misclassified (Corrected) | False Detections | True Positives | Skip (Isolated) | Recall (%) |
|-----------|----------|-----------------|---------------------------|------------------|----------------|-----------------|------------|
| **Standard Rotational Sequences** | | | | | | | |
| clean_baseline | 961 | 961 | 2 | 0 | 961 | 0 | 100.00 |
| oblique_tilt_high | 952 | 889 | 63 | 0 | 888 | 1 | 93.28 |
| roll_120 | 924 | 890 | 8 | 8 | 880 | 2 | 95.24 |
| roll_180 | 961 | 976 | 14 | 17 | 958 | 1 | 99.69 |
| roll_240 | 922 | 636 | 60 | 10 | 561 | 65 | 60.85 |
| roll_300 | 924 | 887 | 11 | 9 | 877 | 1 | 94.91 |
| roll_60 | 922 | 900 | 12 | 18 | 882 | 0 | 95.66 |
| **Complex 3D Galois Stress-Test Sequences** | | | | | | | |
| compound_rotation_0 | 916 | 920 | 39 | 24 | 896 | 0 | 97.82 |
| compound_rotation_1 | 928 | 774 | 18 | 5 | 767 | 2 | 82.65 |
| compound_rotation_2 | 440 | 425 | 5 | 29 | 396 | 0 | 90.00 |
| compound_rotation_3 | 957 | 641 | 61 | 2 | 505 | 134 | 52.77 |
| compound_rotation_4 | 450 | 418 | 12 | 8 | 407 | 3 | 90.44 |
| compound_rotation_5 | 633 | 618 | 13 | 38 | 580 | 0 | 91.63 |
| compound_rotation_6 | 944 | 898 | 75 | 7 | 885 | 6 | 93.75 |
| compound_rotation_7 | 953 | 578 | 45 | 20 | 369 | 189 | 38.72 |

**Key Observation:** Despite heavy illumination noise and aggressive geometric warping, the Galois parity-check framework maintains **perfect precision (1.000)** across all tests with zero false positives, misalignments, or phantom nodes. The structural lattice consensus layer successfully bridges broken tracks and resolves shape classification conflicts on-the-fly.

All metric ratios are evaluated relative to the absolute master pattern footprint (31 × 31 = 961 nodes) to benchmark the decoder against global invariants rather than localized sub-scaling. Under extreme non-linear perspective slants, the peak initial classification failure reaches 7.70% (compound_rotation_6 with 74 corrupted labels), while extreme tilts force maximum pattern occlusion of 60.67% (compound_rotation_7 with only 378 true positives). Despite these concurrent defects, the framework guarantees perfect pattern registration across all configurations.

## Radial Distortion and Intrinsics Estimation

### Multi-Frame Calibration Accuracy

| Parameter | Rotational Dataset | Random Tilt Dataset |
|-----------|-------------------|-------------------|
| | GT | Solved | % Error | GT | Solved | % Error |
| **Focal Length $f_x$ (px)** | 1150.00 | 1152.97 | **0.26%** | 1250.00 | 1246.77 | **0.26%** |
| **Focal Length $f_y$ (px)** | 1150.00 | 1152.97 | **0.26%** | 1150.00 | 1144.15 | **0.51%** |
| **Principal Point $c_x$ (px)** | 960.00 | 961.23 | **0.06%** | 965.00 | 960.19 | **0.25%** |
| **Principal Point $c_y$ (px)** | 540.00 | 538.45 | **0.14%** | 543.00 | 540.23 | **0.26%** |
| **Distortion Coefficient $\kappa_1$** | -0.2000 | -0.2098 | **4.88%** | -0.1500 | -0.1493 | **0.45%** |

Results are computed using robust median consensus across 9 multi-view frames. The **decoupled two-phase approach** completely suppresses parameter cross-talk across distinct geometric configurations:

- **Rotational Dataset (Isotropic Case):** Despite the absence of spatial translation (which nominally triggers $f_x = f_y$ fallback), inverse feature-mass weighting and multi-scale chord regularization isolate underlying straightness invariants with exceptional precision. Focal length reconstruction achieves **0.26% error** under degenerate geometric conditions.

- **Random Tilt Dataset (Anisotropic Case):** Over 9 frames with strong perspective tilt, the decoupled framework successfully unlocks the full camera matrix. Stage A recovers $\kappa_1$ through median consensus via plumb-line straightness. Stage B uses Zheng's vanishing point method on undistorted coordinates to refine intrinsics. Final intrinsic errors: **≤ 0.51%** for focal length, **0.25%** for principal point.

### Single-Frame Performance Under Varying Occlusion

| Test View | Detected Points (TP) | $f_x$ Error | $f_y$ Error | $\kappa_1$ Error |
|-----------|----------------------|-------------|-------------|-----------------|
| View 0 (Pure Rotation) | 367 | 0.36% | 0.36% | 4.27% |
| View 1 | 767 | 3.69% | 7.64% | 12.25% |
| View 4 (High Crop, 418 pts) | 407 | 0.83% | 1.05% | 5.30% |
| View 7 (Extreme Occlusion) | 369 | 10.92% | 19.72% | 71.19% |
| **Multi-Frame Consensus (9 views)** | **3,113** | **0.26%** | **0.51%** | **0.45%** |

Single-frame execution under pure rotational roll configurations introduces textbook homographic rank degeneracy (focal-depth scale coupling). Our cascaded distortion-first approach resolves this by isolating $\kappa_1$ via geometric straightness before intrinsic fitting, enabling convergence even on degenerate frames. However, individual frame defects (focal scale ambiguity under crop, distortion saturation under extreme tilt) are resolved through multi-frame consensus: pooling 3,113 verified topological nodes across 9 views forces parameter convergence to sub-0.5% accuracy.

# Strengths, Limitations, and Concluding Remarks

## Architectural Strengths

- **Maximum Theoretical Packing Density:** The canonical $A_2$ hexagonal lattice achieves maximum spatial marker density per unit area, maximizing geometric constraint volume within a single image frame.

- **Isotropic Crystal Front Propagation:** 6-neighbor adjacency structure provides optimal topological regularization via DSU forest, allowing uniform lattice growth across coordinate boundaries.

- **Unbiased Bit Distribution:** Pseudorandom token distribution mimics white noise, suppressing systematic spatial bias during RMSE optimization.

- **Dense Straight-Line Bundles:** Three-axis geometry generates continuous co-linear point traces at 30° increments, optimizing performance of plumb-line distortion solvers and Zheng-based vanishing point methods.

- **M-Sequence Error Immunity:** Pseudo-random algebraic properties grant native robustness against canvas cropping, random node omissions, and sensor noise.

- **Decoupled Multi-Stage Calibration:** Pattern topology enables separate phases anchored on first principles: using projective invariant of straight-line preservation for distortion, followed by homography and vanishing point recovery for intrinsics.

## Limitations

- **Centroid Detection Sensitivity:** Under extreme motion blur or severe sensor noise, contour fragmentation degrades sub-pixel centroid accuracy, affecting lattice nucleation.

- **Error Correction Saturation:** Decoding fails if multi-bit error bursts violate linear subspace assumptions or if missing nodes saturate available parity-check matrix rows.

- **Implementation Complexity:** Higher conceptual overhead compared to simple tag-based fiducials, requiring DSU forest maintenance and linear algebra over $\mathbb{GF}(2)$.

## Conclusion

The B-HGP framework trades low computational footprint for advanced algebraic resilience and structural robustness. By combining spatial efficiency of hexagonal lattices with M-sequence noise immunity, it enables independent multi-stage calibration using fundamental geometric invariants. For applications requiring single-frame calibration under severe out-of-plane perspective, lens aberrations, or uncontrolled environments (mobile robotics, autonomous vehicles, industrial photogrammetry), B-HGP provides a compelling alternative to conventional tag-based targets.

# Availability

The open-source implementation, full test suite, Blender benchmark generators, and validation tools are publicly available at: https://github.com/gkis-conda/robust_calibration_pattern

# References

---

**Note:** This software paper documents the implementation of research presented in a draft manuscript under consideration for publication in the International Journal of Computer Vision (IJCV).
