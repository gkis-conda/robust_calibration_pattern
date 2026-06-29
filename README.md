# Hexagonal Calibration Pattern based on Galois Field sequences

## Intro

An advanced, single-frame blind 2D coordinate tracking and camera-IMU calibration pattern. By wrapping one-dimensional Galois Field M-sequences over a staggered hexagonal rhomboid lattice mesh, this framework achieves absolute, 
self-identification from almost any localized window fragment. The system maintains total geometric resilience under high perspective shear and optical noise.
This repository serves as a mathematically alternative to traditional calibration patterns (like AprilTags, ChArUco, or checkerboards) that fail under motion blur, severe geometric warping, or illumination phase occlusions.

------------------------------
## Project Goals

* Absolute Self-Identification: Resolve absolute grid coordinate indices (row, col) from any localized crop without requiring visible center tags, perimeter anchors, or global reference points.
* Rank-Deficient Erasure Robustness: Leverage linear left null-space LUP projection matrices to handle up to 15% missing point sensor dropouts (shadow occlusions).
* Adjacent Error Detection Passes: Suppress thresholding noise and optical bit-flips directly at the algebraic 1D Galois Field syndrome evaluation layer.
* Hexagonal Parity Invariance: Maintain 100% verified geometric reversibility and true coordinate alignment across all 6 unique 60-degree rotational quadrants.

------------------------------
## Scientific Approach
The pattern's architecture merges 1D Linear Feedback Shift Register (LFSR) coding theory with discrete projectively warped hexagonal geometry:

   1. Lattice Topology: The workspace is packed as a staggered rhomboid grid where the continuous coordinates
   2. Barycentric Closing Balance: Three independent 1D M-sequences slide continuously across the $U$, $V$, and $W$ diagonal axes, tied globally by the spatial invariant:
   $$u + v + w = 0$$ 
   3. Temporal Symmetry Core: The forward and reversed chronological paths are structurally anchored around the invariant midpoint
   4. Subspace Null-Projector: When gaps (erasures) corrupt a 1D sequence window, an algebraic mask filters out unknown states, generating a left-null space matrix $T$ via LU decomposition to solve the clean underlying state syndrome with zero data insertion loss.

------------------------------
## System Visualizations & Performance
## 1. Lattice Topology & Sub-Graph Connections
The structural arrangement of the pattern layout showing the continuous three-axis coordinate tracking planes and the staggered rhomboid parity pairing rows.

* View Discrete Lattice Mesh Geometry Diagram
* View Wave-Growth Sub-Graph Extraction Sheet

## 2. High-Fidelity Parametric Test Targets
Generated target assets showcasing perspective distortion, lens distortion, and channel fault injections used on our verification bench:

* Pristine Perspective Calibration Frame (synthetic_calibration_shot.png)
* Severe 45-Degree Roll Rotation Target (synthetic_shot_rotated_45_roll.png)
* 15% Missing Node Dropouts (synthetic_shot_erasures.png)
* Extreme Multi-Fault Stress Frame (synthetic_shot_extreme_stress.png)

------------------------------
## Repository Script Directory
### 1. Core Architecture Modules

* m_sequence.py: Manages the 5th-order binary LFSR primitive polynomial math loops over $GF(2)$. Generates the master cyclic tracking sequences.
* lattice_topology.py: Defines the staggered hexagonal mesh coordinates and implements the coordinate transformations in the pure unwarped $(u, v)$ algebraic domain.
* camera.py: Evaluates camera projection matrices ($K, [R\vert t]$) and radial lens distortion equations.
* optimization.py Implements radial distorion calibration

### 2. Extraction & Mapping Framework

* detector.py: detects and classify pattern nodes, the entry for registration processing pipline.
* crystal.py: Implements the wave-growth sub-graph algorithm. Clusters valid detected tracking nodes into continuous connected topological island patches.
* matcher.py: House the core functions to fill a topological matrix and match it to the bluprint pattern. Contains the LUP matrix decomposition engine and left null-space projectors used to calculate algebraic syndromes across active sensor erasures.

### 3. Test Bench & Utilities

* generate.py: Generates the physical mesh patterns and renders high-fidelity camera viewports using customizable camera parameters.
* test_image.py: Orchestrates automated verification suites, processing strict multi-angle rotational sweeps and noise case benchmarks.

------------------------------
## Command-Line Usage
### 1. Generate Synthetic Calibration Target Sheets
To generate and render printable svg pattern

`python generate.py`

### 2. Run the Verification Bench Evaluation Suite
To run the automated verification suite including the exhaustive even-row grid sweeps and the strict 100% reversible multi-angle rotational tests run the main test matrix module:

`python test_image.py`

produces test and diagnostic images, reports prformance accuracy tests.

### 3. Run a Live Image Tracking Extraction Pass
To parse a single arbitrary image frame of the printed pattern, execute a blind phase-lock capture pass, match pattern and perform Menger curvature camera distortion calibration.

`python detector.py --input synthetic_shot_bitflips.png >synthetic_shot_bitflips.txt`

