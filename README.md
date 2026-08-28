# Binary Hexagonal Galois Pattern (B-HGP) based on Galois Field orthogonal sequences

## Intro

By wrapping one-dimensional Galois Field M-sequences over a hexagonal lattice, this framework achieves absolute, 
self-identification from almost any detected pattern fragment. The system maintains total geometric resilience under high perspective shear and optical noise.
This repository demonstrates B-HGP pattern as a mathematically alternative to traditional calibration patterns (like AprilTags, ChArUco, or checkerboards) 
that fail under motion blur, severe geometric distortion, or illumination .

------------------------------
## Project Goals

* Absolute Self-Identification: Resolve absolute grid coordinate indices (row, col) from any localized crop without requiring visible center tags, perimeter anchors, or global reference points.
* Pattern Erasure Robustness: handle up to 15% missing point sensor dropouts (shadow occlusions).
* Error Detection and Correction: Suppress optical bit-flips directly at the algebraic 1D Galois Field syndrome evaluation layer.
* Hexagonal Parity Invariance: Maintain geometric oordinate alignment across all 6 unique 60-degree rotatational directions.

------------------------------
## Scientific Approach
The pattern's architecture merges 1D Linear Feedback Shift Register (LFSR) coding theory with discrete projectively warped hexagonal geometry:

   1. Lattice Topology: The workspace is packed as a staggered rhomboid grid where the continuous coordinates
   2. Barycentric Closing Balance: Three independent 1D M-sequences slide continuously across the $U$, $V$, and $W$ diagonal axes, tied globally by the spatial invariant:
   $$u + v + w = 0$$ 
   3. Temporal Symmetry Core: The forward and reversed chronological paths are structurally anchored around the invariant midpoint
   4. Syndrome-based sequence encoder

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

* detector_factory.py: Generates the physical mesh patterns.
* pattern_decoder_test.py: automated verification suite, processing perspective distortion, random errors and erasures stress benchmarks.
* blender_benchmark.py: Blender simulation test suite generator.
* run_validation.py: Estimator for simulated test suites.
------------------------------
## Command-Line Usage
### 1. Generate Synthetic Calibration Target Sheets
To generate and render printable svg pattern

`python detector_factory.py -e hgp -p <result_folder>`

### 2. Run the Verification Bench Evaluation Suite
This test generates different controllable distortion including the erasures and rotational tests and checks the pattern matching accuracy.
To run the automated verification suite:

`python pattern_decoder_test.py -p <result_folder> --save-images`

produces test and diagnostic images, reports detailed and summary performance accuracy, all tests should be 100% passed.

### 3. Run a Live Image Tracking Extraction Pass
To parse a single arbitrary image frame of the printed pattern, execute a blind phase-lock capture pass, match pattern and perform Menger curvature camera distortion calibration.
and optionally calibrate distortion model (-C key)

`python detector.py --input synthetic_shot_bitflips.png > synthetic_shot_bitflips.txt`

If calibration parameters ground truth json is available then 
\<image name\>-calibration_summary.md file is generated and corresponding calibration json file is stored.
This scring supports a folder input for multiframe calibration. 

### 4. Benchmarks
To compare performance on more realistic simulation tests the Blender image generator in used. 
It creates set of test images with corresponding ground truth description.

`blender --background --python blender_benchmark.py -- --engine hgp --p <result folder>`

Evaluation is done with
`python run_validation.py --engine hgp --p <input and result folder>`
It generates summary.md with pattern matching statistic only.

To extend benchmark with your pattern use blender_factory.py.

*Note: The extra -- separator is a standard Blender python constraint required to separate Blender's native CLI args from custom Python script flags).*