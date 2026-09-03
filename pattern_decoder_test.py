import numpy as np
from generate import PhysicalMeshGenerator
from camera import compute_camera_projection_matrix, ProjectiveCamera
from detector import *
import json
import os
from run_validation import draw_topology_scene, \
    make_blueprint_dict, classify_topology_nodes, compute_topology_statistics, get_precision_threshold


def render_warped_grid_shapes(mesh_generator, cam: ProjectiveCamera, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Draws shapes onto the camera viewport by projectively transforming every
    explicit boundary contour vertex using the ProjectiveCamera abstraction layer.
    """
    # 1. Initialize a clean, white canvas matching your camera sensor array size
    canvas = np.full((cam.H_img, cam.W_img, 3), 255, dtype=np.uint8)

    # 2. Render Loop: Iterate through each active shape block inside the generator
    for i, j, shape_type, contour in mesh_generator:
        # Skip empty padding zones or dead tracking border background marks
        if shape_type < 0:
            continue

        world_pts = np.array(contour, dtype=np.float32)
        pixel_pts = cam.project_points(world_pts, R, t)
        if pixel_pts is None:
            continue

        # Convert to signed 32-bit integers required by OpenCV rendering tools
        pixel_pts_int = np.round(pixel_pts).astype(np.int32)
        cv2.fillPoly(canvas, [pixel_pts_int], 0, lineType=cv2.LINE_AA)

    return canvas


def inject_matrix_erasures(blueprint: np.ndarray, erasure_probability: float) -> np.ndarray:
    """
    Randomly drops active data nodes by replacing them with empty background flags (-1).
    Simulates dirt, physical print damage, or local camera sensor dead zones.
    """
    corrupted = np.copy(blueprint)
    np.random.seed(42)
    random_mask = np.random.rand(*blueprint.shape)
    corrupted[random_mask < erasure_probability] = -1
    return corrupted


def inject_matrix_bit_flips(blueprint: np.ndarray, flip_probability: float) -> np.ndarray:
    """
    Randomly flips the binary sequence bit states (0 to 1, or 1 to 0).
    Only corrupts valid tracking points (leaves -1 markers untouched).
    Simulates illumination glare or thresholding binarization noise.
    """
    corrupted = np.copy(blueprint)
    H, W = blueprint.shape
    np.random.seed(37)
    random_mask = np.random.rand(*blueprint.shape)
    corrupted[random_mask < flip_probability] ^= 1
    return corrupted


def apply_geometric_aperture_crop(blueprint: np.ndarray,
                                  center_row_pct: float = 0.5,
                                  center_col_pct: float = 0.5,
                                  radius_pct: float = 0.3) -> np.ndarray:
    """
    Crops the master lattice matrix into a custom bounded geometric island shape.
    Simulates partial camera frame visibility when tracking targets drift off-center.
    """
    cropped = np.copy(blueprint)
    H, W = blueprint.shape

    center_r = int(H * center_row_pct)
    center_c = int(W * center_col_pct)
    max_distance = max(H, W) * radius_pct

    for r in range(H):
        for c in range(W):
            distance = np.sqrt((r - center_r) ** 2 + (c - center_c) ** 2)
            if distance > max_distance:
                cropped[r, c] = -1
    return cropped


def apply_multi_island_mask(base_blueprint: np.ndarray) -> np.ndarray:
    """
    Applies a structured multi-island segment mask over a base hexagonal blueprint.
    Preserves the original cell identifiers completely, but carves isolation barriers
    of -1 tokens down the center lines to prevent wave-growth islands from connecting.

    Args:
        base_blueprint (np.ndarray): The source ground-truth matrix layout.

    Returns:
        np.ndarray: The modified blueprint with structural separation channels.
    """
    H, W = base_blueprint.shape
    modified_blueprint = np.copy(base_blueprint)

    # 1. Define the center coordinates for the dividing cross channels
    center_r = H // 2
    center_c = W // 2

    # 2. Carve a horizontal isolation channel (3 rows thick for safety gap)
    for r in range(center_r - 1, center_r + 2):
        if 0 <= r < H:
            modified_blueprint[r, :] = -1

    # 3. Carve a vertical isolation channel (3 columns thick for safety gap)
    for c in range(center_c - 1, center_c + 2):
        if 0 <= c < W:
            modified_blueprint[:, c] = -1

    return modified_blueprint


def evaluate_single_integration_case(base_blueprint: np.ndarray,
                                     detector:HexagonalTopologyDetector,
                                     case_name: str,
                                     case_payload: dict,
                                     save_dir: str = None,
                                     save_images: bool = False) -> dict:
    """
    Evaluates a single universal integration test case dataset configuration.
    Processes the case array buffer, decodes phases, and maps coordinates in-memory.

    """
    STEP_PX = 45.0

    blueprint = case_payload["blueprint"]
    cam_params = case_payload["camera"]
    cam_intrinsics = case_payload["intrinsics"]
    cam_obj = camera_io.deserialize_camera_from_dict(cam_intrinsics)

    # Compute camera projection extrinsics matrix [R|t] per case profile
    R, t = compute_camera_projection_matrix(
        roll_deg=cam_params["roll"],
        pitch_deg=cam_params["pitch"],
        yaw_deg=cam_params["yaw"],
        tx=cam_params["tx"],
        ty=cam_params["ty"],
        tz=cam_params["tz"]
    )

    # Simulate physics generation pass and build the local adaptive tilted patch
    generator = PhysicalMeshGenerator(blueprint, STEP_PX, STEP_PX / 5)

    # Render the frame cleanly using our short object component pipeline
    img = render_warped_grid_shapes(generator, cam_obj, R, t)

    if save_images:
        output_filename = f"synthetic_shot_{case_name}.png"
        cv2.imwrite(os.path.join(save_dir, output_filename), img)
        print(f" -> Export Complete: Saved original image to '{output_filename}'")

    # Adjust reference blueprint targets for out-of-frame boundary clipping
    visible_blueprint = compute_visible_blueprint(
        base_blueprint=blueprint,
        generator=generator,
        camera=cam_obj,
        R=R,
        t=t
    )
    result=detector.register_pattern(img, save_images)
    if result is None or result["status"] != "success":
        return {"status": "success",
            "case_name": case_name,
            "description": case_payload["description"]}

    print("original")
    print(visible_blueprint)
    print("restored")
    mapped_labels = map_matrix_indices(result["topological_matrix"], result["labels"])
    print(mapped_labels)

    gt_dict = make_blueprint_dict(base_blueprint, blueprint, generator, camera=cam_obj, R=R, t=t)
    node_status_dict = classify_topology_nodes(gt_dict, result["topological_matrix"], result["points"], result["labels"])

    if save_images:
        cv2.imwrite(os.path.join(save_dir, f"synthetic_shot_{case_name}-debug.png"), img)
        debug_overlay = draw_topology_scene(gt_dict, node_status_dict, img, legend_position="bottom_left")
        # Save the diagnostic visualization matrix directly to disk
        diagnostic_filename = f"synthetic_shot_{case_name}-diagnostic.png"
        cv2.imwrite(os.path.join(save_dir, diagnostic_filename), debug_overlay)
        print(f" -> Exported visual debug overlay to '{diagnostic_filename}'")

    metrics = compute_topology_statistics(gt_dict, node_status_dict)

    metrics["status"] = "success"
    metrics["case_name"] = case_name
    metrics["description"] = case_payload["description"]

    return metrics


def compute_visible_blueprint(base_blueprint: np.ndarray,
                              generator,
                              camera: ProjectiveCamera,
                              R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Evaluates node visibility on the sensor array using the ProjectiveCamera class objects.
    """
    H_NODES, W_NODES = base_blueprint.shape
    visible_blueprint = np.copy(base_blueprint)

    for r in range(H_NODES):
        for c in range(W_NODES):
            point_idx = base_blueprint[r, c]
            if point_idx < 0:
                continue

            # Retrieve node world position coordinates from the generator module
            node_world_center = generator.get_shape_center(r, c)
            pixel_pts = camera.project_points(node_world_center, R, t)

            if pixel_pts is None or not camera.is_visible(pixel_pts[0]):
                visible_blueprint[r, c] = -1

    return visible_blueprint


def save_test_case_markdown_report(case_name: str,
                                   case_payload: dict,
                                   metrics: dict,
                                   output_dir: str = ".") -> None:
    """
    Generates and saves a comprehensive performance report in Markdown format,
    documenting test case settings, camera configurations, and tracking yield scores.
    Also exports a raw JSON sibling file for automated parsing.

    Variables Description:
        case_name (str)    : Unique name string of the verified test scenario block.
        case_payload (dict): Ground-truth configurations dictionary containing camera params.
        metrics (dict)     : Raw output dictionary from calculate_reconstruction_metrics.
        output_dir (str)   : Target folder directory path to save the generated text report.

    Returns:
        None
    """
    # 1. Isolate the target paths cleanly
    report_filename = os.path.join(output_dir, f"report_{case_name.lower()}.md")
    json_filename = os.path.join(output_dir, f"report_{case_name.lower()}.json")

    # 2. Extract operational metrics parameters safely
    tp = metrics.get("true_positives", 0)
    visible = metrics.get("total_visible_targets", 0)
    misalignments = metrics.get("misalignments", 0) + metrics.get("fp", 0)
    skips = metrics.get("graph_skips", 0)
    misses = metrics.get("optical_misses", 0)
    ghosts = metrics.get("ghost_nodes", 0)
    erasures = metrics.get("expected_erasures", 0)
    leaks = metrics.get("erasure_leaks", 0)

    # 3. Retrieve descriptive hardware simulation states
    description = case_payload.get("description", "No scenario description provided.")
    cam_params = case_payload.get("camera", {})

    roll = cam_params.get("roll", 0.0)
    pitch = cam_params.get("pitch", 0.0)
    yaw = cam_params.get("yaw", 0.0)
    tx = cam_params.get("tx", 0.0)
    ty = cam_params.get("ty", 0.0)
    tz = cam_params.get("tz", 0.0)

    # Evaluate a clean visual indicator emoji matching the accuracy bounds
    status_indicator = "[PASS]" if metrics.get("is_passed", False) else "[FAIL]"

    # 4. Construct the complete Markdown layout string block
    md_content = []
    md_content.append(f"# Automated Test Verification Report: {case_name}")
    md_content.append(f"**Execution Status:** {status_indicator} | **Final Matching Precision:** {metrics['precision']:.2f}%\n")

    md_content.append("## Scenario Description")
    md_content.append(f"{description}\n")

    md_content.append("## Pattern Registration Performance Metrics")
    md_content.append("| Metric Parameter Name | Checked Count | Evaluation Analysis Notes |")
    md_content.append("| :--- | :--- | :--- |")
    md_content.append(
        f"| **True Positives (TP)** | {tp} | Successfully extracted, localized, and matches blueprint down to the single cell. |")
    md_content.append(
        f"| **Total Intended Visible Targets** | {visible} | Intended pattern grid targets visible within the active camera sensor boundaries. |")
    md_content.append(
        f"| **Index Alignment Drift (Misalignments)** | {misalignments} | Decoded matrix row/column cells that shifted away from ground-truth slots. |")
    md_content.append(
        f"| **Graph Traversal Skips** | {skips} | Geometric blobs extracted from video frames but skipped by wave-growth engine. |")
    md_content.append(
        f"| **Pure Optical Misses** | {misses} | Core blueprint dots inside view limits that failed the thresholding blob detector. |")
    md_content.append(
        f"| **Noise Artifacts (Ghosts)** | {ghosts + leaks} | Spurious noise blobs registered by camera that do not exist on the master template. |")
    md_content.append(
        f"| **Expected Erasures (True Negatives)** | {erasures - leaks} | Hardware-level missing spots or mask holes correctly bypassed by the tracker. |")
    md_content.append("")

    md_content.append("## Camera Simulation Extrinsics & Position Parameters")
    md_content.append("| Transformation Axis | Simulated Value Input | Geometric Spatial Unit |")
    md_content.append("| :--- | :--- | :--- |")
    md_content.append(f"| **Camera Roll Rotation** | {roll:.2f} | Degrees |")
    md_content.append(f"| **Camera Pitch Tilt** | {pitch:.2f} | Degrees |")
    md_content.append(f"| **Camera Yaw Angle** | {yaw:.2f} | Degrees |")
    md_content.append(f"| **Translation Vector X (tx)** | {tx:.2f} | mm (Horizontal Camera Sensor Offset) |")
    md_content.append(f"| **Translation Vector Y (ty)** | {ty:.2f} | mm (Vertical Camera Sensor Offset) |")
    md_content.append(f"| **Translation Vector Z (tz)** | {tz:.2f} | mm (Lens Distance focal height clearance) |")
    md_content.append(
        f"\n***\n*Report automatically compiled and serialized by {HexagonalTopologyDetector.ENGINE_FULL_NAME}.*")

    # 5. Write the compiled text report to disk safely
    with open(report_filename, 'w', encoding='ascii') as f:
        f.write("\n".join(md_content) + "\n")

    # 6. Sibling Output: Export structured JSON for database serialization or multi-frame chart logging
    structured_log = {
        "case_name": case_name,
        "description": description,
        "metrics": metrics,
        "camera_parameters": cam_params
    }
    with open(json_filename, 'w', encoding='ascii') as fj:
        json.dump(structured_log, fj, indent=4)

    print(f" -> [INFO]: Compiled Markdown report successfully saved to: {report_filename}")
    print(f" -> [INFO]: Automated JSON successfully saved to: {json_filename}")


def save_summary_markdown_report(results_dict: dict,
                                 output_dir: str = ".") -> None:
    """
    Compiles a structured dictionary of test case results into a unified
    master Markdown summary matrix table dashboard file.

    Variables Description:
        results_dict (dict) : Map of case profiles where the case name key tracks
                              the complete metrics configuration dictionary directly:
                              {
                                "CLEAN_BASELINE": {
                                    "description": "tracking run",
                                    "accuracy": 100.0,
                                    "true_positives": 841,
                                    ...
                                }, ...
                              }
        output_dir (str)    : Target folder path destination to save the summary file.

    Returns:
        None
    """
    summary_filename = os.path.join(output_dir, "summary_report.md")

    # 1. Build the formalized Markdown table header configuration
    md_content = []
    md_content.append("# Pattern Registration Performance Summary")

    md_content.append("| Case Name | Scenario Comment Description | Visible Targets | Recall | Precision | Status |")
    md_content.append("| :--- | :--- | :---: | :---: | :---: | :---: |")

    total_cases = len(results_dict)
    passed_cases = 0

    # 2. Iterate across the results dictionary keys to populate specific row slots
    for case_name, metrics in results_dict.items():
        # Dynamically extract description straight from the metrics array dictionary
        comment = metrics.get("description", "No description profile recorded.")

        visible = metrics.get("total_visible_targets", 0)
        recall = metrics.get("recall", 0)
        precision = metrics.get("precision", 0)

        # Evaluate status threshold signatures
        is_passed = metrics.get("is_passed", False)
        status_tag = "PASS" if is_passed else "FAIL"

        if is_passed:
            passed_cases += 1

        # Append row format block string data line directly
        md_content.append(
            f"| **{case_name}** | {comment} | {visible} | {recall:.2f}% | {precision:.2f}% | {status_tag} |"
        )

    # 3. Append high-level metrics
    md_content.append("\n## System Conformance Evaluation Analytics")
    md_content.append(f"- **Total Simulated Test Cases Checked:** {total_cases}")
    md_content.append(f"- **Total Successfully Passed Suites :** {passed_cases} / {total_cases}")
    md_content.append("\n### Definitions")
    md_content.append("To maintain mathematical consistency across all evaluation passes, metric tracking definitions follow standard confusion matrix guidelines:")
    md_content.append("1. **Recall Index:** Evaluates the target search coverage ratio relative to the active image pane boundary layout.")
    md_content.append("   $$\\text{Recall} = \\frac{TP}{\\text{Visible GT Nodes}} \\times 100.0$$")
    md_content.append("2. **Precision Index:** Quantifies the structural assignment reliability of the topological decoder, demonstrating its zero false-positive extraction rate.")
    md_content.append("   $$\\text{Precision} = \\frac{TP}{TP + \\text{Misalignments}} \\times 100.0$$")
    md_content.append(f"A test suite run is explicitly designated as **PASSED** if metching precision threshold is reached: `precision > {get_precision_threshold():.1f}%`")

    if total_cases > 0:
        yield_score = (passed_cases / total_cases) * 100.0
        md_content.append(f"- **Global Framework Compliance Index:** {yield_score:.2f}%")

    md_content.append(f"\n***\n*Generated automatically by {HexagonalTopologyDetector.ENGINE_FULL_NAME}*")

    # 5. Output the finalized text document stream to disk safely
    with open(summary_filename, 'w', encoding='ascii') as f:
        f.write("\n".join(md_content) + "\n")

    return summary_filename


if __name__ == "__main__":
    from detector_factory import parse_arguments, create_detector, pattern_blueprint
    args = parse_arguments(HexagonalTopologyDetector.ENGINE_FULL_NAME)
    H_NODES = args.rows
    W_NODES = args.cols
    detector = create_detector(args.engine, grid_rows=H_NODES, grid_cols=W_NODES)

    STEP_PX = 45.0
    Z_DISTANCE = -H_NODES * STEP_PX * 1.1
    DEFAULT_TX = 0
    DEFAULT_TY = 0
    IMG_SHAPE = (1920, 1080)
    K1 = -0.25
    INTRINSICS = {"fx": 1150.0, "fy": 1150.0, "cx": IMG_SHAPE[0]/2, "cy": IMG_SHAPE[1]/2, "k1": K1, "img_shape": IMG_SHAPE}
    base_blueprint = pattern_blueprint(args.engine, cols=W_NODES, rows=H_NODES, debug_output=args.verbose)
    # Define your centralized parametric evaluation dictionary matrix block
    cases = {
        "clean_baseline": {
            "description": "Pristine Baseline Frame (Standard Centered Orientation)",
            "blueprint": np.copy(base_blueprint),
            "camera": {"roll": 0.0, "pitch": 0.0, "yaw": -1.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE},
            "intrinsics": INTRINSICS
        },
        "erasures": {
            "description": "15% Missing Node Dropouts (Standard Centered Orientation)",
            "blueprint": inject_matrix_erasures(base_blueprint, erasure_probability=0.15),
            "camera": {"roll": 0.0, "pitch": 0.0, "yaw": -1.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE},
            "intrinsics": INTRINSICS
        },
        "bitflips": {
            "description": "5% Random Bit Flip Threshold Noise (Standard Centered Orientation)",
            "blueprint": inject_matrix_bit_flips(base_blueprint, flip_probability=0.05),
            "camera": {"roll": 0.0, "pitch": 0.0, "yaw": -1.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE},
            "intrinsics": INTRINSICS
        },
        "cropped": {
            "description": "Partial Viewport Aperture Geometric Crop (Standard Centered Orientation)",
            "blueprint": apply_geometric_aperture_crop(base_blueprint, center_row_pct=0.4, center_col_pct=0.4, radius_pct=0.25),
            "camera": {"roll": 0.0, "pitch": 0.0, "yaw": -1.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE},
            "intrinsics": INTRINSICS
        },
        "rotated_45_roll": {
            "description": "Severe 45-Degree Roll Rotation Around Optical Axis",
            "blueprint": np.copy(base_blueprint),
            # Severe roll skew applied around the optical Z-axis with an alternative translation offset
            "camera": {"roll": 45.0, "pitch": 0.0, "yaw": 0.0, "tx": DEFAULT_TX * 0.8, "ty": DEFAULT_TY * 0.8,
                       "tz": Z_DISTANCE * 1.2},
            "intrinsics": INTRINSICS

        },
        "extreme_stress": {
            "description": "Combined 10% Erasures + 45-Deg Roll Rotation + Camera Perspective Shift",
            "blueprint": inject_matrix_erasures(base_blueprint, erasure_probability=0.10),
            "camera": {"roll": 45.0, "pitch": 15.0, "yaw": -10.0, "tx": DEFAULT_TX * 0.9, "ty": DEFAULT_TY * 1.1,
                       "tz": Z_DISTANCE * 0.95},
            "intrinsics": INTRINSICS

        },
        "multi_island_stitch": {
            "description": "4 Separated Fragments",
            "blueprint": apply_multi_island_mask(base_blueprint),
            "camera": {
                "roll": 30.0, "pitch": 5.0, "yaw": -2.0,
                "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE
            },
            "intrinsics": INTRINSICS
        },
        "severe_pitch_tilt_45deg": {
            "description": "Severe 45-Degree Camera Pitch Test",
            "blueprint": np.copy(base_blueprint),
            "camera": {
                "roll": 0.0, "pitch": 45.0, "yaw": 0.0,
                "tx": DEFAULT_TX, "ty": H_NODES/2 * STEP_PX, "tz": Z_DISTANCE * 0.9
            },
            "intrinsics": INTRINSICS
        }
    }

    # Dynamically populate each of the 6 canonical 60-degree roll positions
    for step_idx in range(1, 6):
        target_roll = float(step_idx * 60)
        cases[f"roll_{int(target_roll)}"] = {
            "description": f"Strict {int(target_roll)}-Degree Roll Skew Around Optical Axis",
            "blueprint": np.copy(base_blueprint),
            "camera": {"roll": target_roll, "pitch": 0.0, "yaw": 0.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY,
                       "tz": Z_DISTANCE},
            "intrinsics": INTRINSICS
        }

    RESULT_DIR = args.path
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    accumulated_metrics_dictionary = {}

    def prints(stroke_len=40):
       print("=" * stroke_len)
    prints()
    print("Launching Pattern Evaluation Loop ...")
    print(f"Image Export: {'ENABLED' if args.save_images else 'DISABLED'}")
    prints()
    camera_io.save_dict_as_json(os.path.join(RESULT_DIR,"base_camera.json"), INTRINSICS)

    for case_name, case_payload in cases.items():
        #if case_name != "extreme_stress":
        #    continue
        print(f"\n[EVALUATING]: Case Module [{case_name.upper()}]")

        # Pass the command line flag directly down to the single-case evaluator
        result = evaluate_single_integration_case(
            base_blueprint=base_blueprint,
            detector=detector,
            case_name=case_name,
            case_payload=case_payload,
            save_dir=RESULT_DIR,
            save_images=args.save_images
        )

        # Export the detailed markdown file report automatically right at the finish line
        save_test_case_markdown_report(
            case_name=case_name,
            case_payload=case_payload,
            metrics=result,
            output_dir=RESULT_DIR
        )

        accumulated_metrics_dictionary[case_name] = result

        if result["status"] != "success":
            print(f" -> [WARNING] Decoder failed.")
            continue
        print(f" -> Metrics: precision={result['precision']:.2f}%, True Positives={result['true_positives']} from total visible {result['total_visible_targets']}")

    summary_filename = save_summary_markdown_report(results_dict=accumulated_metrics_dictionary, output_dir=RESULT_DIR)
    if len(summary_filename) > 0:
        print(f" -> [INFO]: Summary table successfully written to: {summary_filename}")
    else:
        print(f" -> [WARNING]: Summary table saving is failed")
    prints()
    print("Integration test for pattern decoder completed")
    prints()
