# Binary Hexagonal Galois Pattern (B-HGP) Based on Galois Field Orthogonal Sequences

## Introduction

By encoding one-dimensional Galois Field M-sequences over a regular hexagonal lattice, this framework achieves absolute self-identification from almost any detected pattern fragment. The architecture maintains total geometric resilience under high perspective shear, severe lens distortion, and optical noise. This repository implements the B-HGP framework as a mathematically robust alternative to traditional fiducial targets (such as AprilTags, ChArUco, or standard checkerboards) that frequently fail under motion blur, aggressive geometric warping, or severe illumination gradients.

---

## Project Goals

* **Absolute Self-Identification:** Resolve absolute topological grid coordinate indices (row, col) from any localized crop without requiring visible center tags, perimeter anchors, or global reference landmarks.
* **Pattern Erasure Robustness:** Maintain structural tracking and successfully bridge up to 15% missing node dropouts or localized shadow occlusions.
* **Algebraic Error Correction:** Suppress and heal optical bit-flips directly at the algebraic 1D Galois Field syndrome evaluation layer before parameter estimation.
* **Absolute Rotational Invariance:** Any spatial direction maintains identical properties up to a control polynomial and is processed symmetrically, tracking exact coordinate alignments across rotations.
* **Computationally Robust Calibration:** Provide unbiased parameter estimations and enable highly accurate geometric calibration even when utilizing weak perspective images.

---

## Scientific Approach
The pattern's architecture merges 1D Linear Feedback Shift Register (LFSR) coding theory with discrete projectively warped hexagonal geometry:

1. **Lattice Topology:** The pattern is represented by a regular hexagonal grid layout, offering mathematically proven maximum node coverage, Delaunay triangulation robustness, and an unbiased spatial distribution of points. It provides high performance in nearest-neighbor search tasks and provides excellent resilience against response ambiguity caused by perspective distortion.
2. **Barycentric Closing Balance:** Three orthogonal 1D $M$-sequences slide across the $U$, $V$, and $W$ diagonal axes, rigidly locked by the global spatial coordinate invariant:
   $u + v + w = 0$
3. **Syndrome-Based Sequence Decoder:** Detects pattern view orientation and absolute location from an isolated pattern fragment, automatically correcting moderate detection noise and recognition dropouts via a greedy decoding pass.
4. **Decoupled Calibration:** Employs a split calibration approach to completely avoid the parameter-coupling problems inherent to the standard Brown-Conrady model. This mathematical separation enables highly accurate camera parameter recovery from a single planar view of the scene.

More detailed descriptions of the underlying algorithms and result samples can be found in the [docs](./docs/ALGORITHM.md) directory.

---

## Repository Script Directory

The calibration engine is implemented as a suite of command-line Python scripts. The codebase is fully compatible and tested with Python 3.6+.

### 1. Core Architecture Modules
* `m_sequence.py`: Manages the 5th-order binary LFSR primitive polynomial math loops over $GF(2)$ and generates the master cyclic tracking sequences.
* `lattice_topology.py`: Defines the staggered hexagonal mesh coordinates and implements coordinate transformations in the pure, unwarped $(u, v)$ algebraic domain.
* `camera.py`: Evaluates camera projection matrices ($\mathbf{K}, [\mathbf{R}\vert\mathbf{t}]$) and radial lens distortion models.
* `optimization.py`: Implements the non-linear radial distortion calibration solver.

### 2. Extraction & Mapping Framework
* `detector.py`: Extracts and classifies low-level feature nodes, acting as the primary entry point for the registration pipeline.
* `crystal.py`: Implements the wave-growth sub-graph algorithm. Clusters valid detected tracking nodes into continuous connected topological island patches.
* `matcher.py`: Houses core functions to populate the topological matrix and match it to the blueprint pattern.

### 3. Test Bench & Utilities
* `detector_factory.py`: Generates the physical vector mesh patterns.
* `pattern_decoder_test.py`: Automated verification suite processing perspective distortion, random errors, and erasure stress benchmarks.
* `blender_benchmark.py`: Generates photorealistic test frames using the automated Blender simulation environment.
* `run_validation.py`: Parser and metric estimator for simulated multi-frame test suites.

---

## Command-Line Usage

### 1. Generate Synthetic Calibration Target Sheets
To generate and render a printable vector SVG pattern configuration:
```bash
python detector_factory.py -e hgp -p <result_folder>
```

### 2. Run the Verification Bench Evaluation Suite
This test bench injects controllable distortions (including erasures and rotational skews) and checks the matching precision. To execute the automated suite:
```bash
python pattern_decoder_test.py -p <result_folder> --save-images
```
Produces diagnostic images and reports detailed summary metrics. For system conformance, all verification steps must achieve a 100% Precision score. See an example of the aggregated benchmark tracking matrix in the generated [summary_report.md](./docs/samples/summary_report.md).

### 3. Run a Live Pattern Extraction Pass
To parse an arbitrary image frame, execute a blind phase-lock capture pass, map the topological lattice, and perform camera distortion model calibration via Menger curvature straightness invariants (append the `-C` flag to trigger optimization):
```bash
python detector.py --input synthetic_shot_bitflips.png > synthetic_shot_bitflips.txt
```
If a ground-truth calibration JSON file is available in the target path, a metric evaluation and calibration analysis report is compiled automatically. See a sample output in [simple_rotation-calibration_summary.md](./docs/samples/simple_rotation-calibration_summary.md). This script natively supports directory paths for multi-frame bundle consensus calibration.

### 4. Simulator Benchmarks
To compare tracking performance under photorealistic conditions, use the Blender 2.92 scene generator to create images paired with absolute ground-truth metadata:
```bash
blender --background --python blender_benchmark.py -- --engine hgp --p <result_folder>
```
Generated photorealistic image example: [noisy compound rotation frame](./docs/samples/compound_rotation_7_0025.png), [camera parameters JSON](./docs/samples/compound_rotation_7_0025_camera.json) and [key-point ground truth](./docs/samples/compound_rotation_7_0025_gt.txt).

Evaluate the output folder via the validation suite:
```bash
python run_validation.py --engine hgp --p <input and result folder>
```
Diagnostic rendering layout example: [debug tool visualization](./docs/samples/compound_rotation_7_0025.png-debug.png) and ground truth validation [tracking performance status](./docs/samples/compound_rotation_7_0025.png-stat.png).

This tool automatically compiles and serializes a performance tracking and pattern matching statistics report. Review a complete runtime evaluation sheet in [summary.md](./docs/samples/summary.md). Note: The extra `--` separator is a standard constraint required to separate Blender's native CLI arguments from custom Python script flags.

---
## System Visualizations & Performance

### 1. Real-Time Lattice Sub-Graph Front Propagation & Syndrome Decoding Traces
Use the `--verbose` flag to collect and display low-level debugging telemetry streams during execution.
During front-propagation execution, the wave-growth sub-graph engine (`crystal.py`) outputs a real-time, text-based matrix trace directly to the terminal console at every consecutive wave ring iteration step. This allows developers to inspect the exact state of the DSU seed expansion:

```text
   -> Island 364 Shape: (2, 3)
[[ -1 364 402]
 [351 391 432]]
   -> Island 834 Shape: (4, 3)
[[ -1  -1  -1]
 [ -1 840  -1]
 [ -1 834 856]
 [828 849  -1]]
[Diag] Wave step complete: Added 305 structural ears.

Wave 2
   -> Island 279 Shape: (5, 4)
[[ -1  -1  -1  -1]
 [288 314 342  -1]
 [254 279 304 331]
 [245 269 294  -1]
 [ -1 237 260  -1]]
```

Also, decoding info for every extracted patch is collected as follows:

```text
row: 19, col 5, length: 13
valid_mask: [1 1 1 1 1 1 1 1 1 1 0]
patch:
 [[ 0  1  0  0  1  1  1  1  0  1  1  1  0]
 [ 1  0  0  1  1  1  1  1  0  0  1 -1  1]]
Hor:  {'origin_stream_phase': 4, 'recovered_sequence': [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1], 'errors_corrected': [4, 5], 'missing_gaps': [10], 'status': 'Corrected bit flip at index [4, 5] via syndrome projection', 'direction_key': 'W_forward'}
Vert:  {'origin_stream_phase': 17, 'recovered_sequence': [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], 'errors_corrected': [7, 8], 'missing_gaps': [10, 11], 'status': 'Corrected bit flip at index [7, 8] via syndrome projection', 'direction_key': 'V_reverse'}
[Warning] Consistency check failed. Skip
```

 * To review a complete, full-frame runtime console extraction log, see [**`compound_rotation_7_0025_full_log.txt`**](./docs/samples/compound_rotation_7_0025_full_log.txt), which tracks localized topological cell alignments, matrix patch outputs, and error-corrected phase outputs for the "compound_rotation_7_0025.png" image created by blender script.


### 2. High-Fidelity Parametric Stress Benchmarks & Tracking Results
Each synthetic stress configuration on our verification bench is paired with an automated tracking confirmation diagnostic plot (`*-diagnostic.png`). These overlays demonstrate the sub-pixel lattice extraction, node classifications, and 1D Galois Field sequence matching achieved by the wave decoder:

* **Pristine Reference Alignment:** [`synthetic_shot_clean_baseline.png`](./docs/samples/synthetic_shot_clean_baseline.png) $\to$ [Visual Tracking Output Overlay](./docs/samples/synthetic_shot_clean_baseline-diagnostic.png)
* **Severe 45-Degree Pitch Tilt Target:** [`synthetic_shot_severe_pitch_tilt_45deg.png`](./docs/samples/synthetic_shot_severe_pitch_tilt_45deg.png) $\to$ [Visual Tracking Output Overlay](./docs/samples/synthetic_shot_severe_pitch_tilt_45deg-diagnostic.png)
* **10% Random Recognition Dropouts Configuration:** [`synthetic_shot_erasures.png`](./docs/samples/synthetic_shot_erasures.png) $\to$ [Visual Tracking Output Overlay](./docs/samples/synthetic_shot_erasures-diagnostic.png)
* **Combined Multi-Fault Extreme Stress Frame:** [`synthetic_shot_extreme_stress.png`](./docs/samples/synthetic_shot_extreme_stress.png) $\to$ [Visual Tracking Output Overlay](./docs/samples/synthetic_shot_extreme_stress-diagnostic.png)

Test cases can be easily extended by editing camera intrinsics/extrinsics and running the automated code generation utilities to export paired diagnostic files.

---


## Licensing

This repository is dual-licensed to accommodate both open-source research and commercial deployments:

1. **Academic & Open-Source Use:** The project is strictly governed under the terms of the **GNU Affero General Public License v3.0 (AGPLv3)**. Any internal cloud modification, remote pipeline integration, or derivative deployment requires copyleft infrastructure open-sourcing. See the [LICENSE](LICENSE.txt) file for details.
2. **Commercial & Proprietary Use:** For closed-source integration into proprietary state-estimation software, industrial photogrammetry pipelines, or autonomous vehicle fleet deployments where AGPLv3 copyleft triggers are unacceptable, a private commercial license must be acquired.

We highly welcome industry partnerships, hardware bundling arrangements, and custom core integrations. Please contact the copyright holder directly to arrange mutually beneficial commercial terms.

