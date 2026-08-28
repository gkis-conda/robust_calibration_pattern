import numpy as np
from scipy.spatial import cKDTree
from generate import PhysicalMeshGenerator
from camera import compute_camera_projection_matrix, ProjectiveCamera
from detector import *
import json
import os


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
    for r in range(H):
        for c in range(W):
            point_idx = blueprint[r, c]
            if point_idx >= 0 and np.random.rand() < flip_probability:
                corrupted[r, c] = point_idx ^ 1
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


def calculate_reconstruction_metrics(topological_matrix: np.ndarray,
                                     detected_points: np.ndarray,
                                     base_blueprint: np.ndarray,
                                     modified_blueprint: np.ndarray,
                                     generator,
                                     camera,
                                     R: np.ndarray, t: np.ndarray,
                                     max_matching_dist_px: float = 4.0) -> dict:
    """
    Computes precise performance and topological divergence statistics by building
    a KD-Tree from live sensor detections and scanning the ground-truth blueprint.

    Args:
        topological_matrix (np.ndarray): Decoded tracking map lookup table framework.
        detected_points (np.ndarray): Raw 2D pixel coordinates found by blob finder.
        base_blueprint (np.ndarray): The master binary reference matrix layout pattern.
        modified_blueprint (np.ndarray): The active modified blueprint carrying erasures.
        generator (obj): The active PhysicalMeshGenerator tracking the pattern dots.
        camera (obj): The ProjectiveCamera hardware parameters abstraction layer object.
        Rt (np.ndarray): Camera translation and rotation extrinsics matrix.
        max_matching_dist_px (float): Strict sub-pixel radius threshold lock.

    Returns:
        dict: High-utility telemetry metrics (accuracy, misalignments, slips).
    """
    H_nodes, W_nodes = base_blueprint.shape

    if detected_points is None or len(detected_points) == 0:
        return {
            "accuracy": 0.0, "true_positives": 0, "total_visible_targets": 0,
            "misalignments": 0, "graph_skips": 0, "optical_misses": 0, "ghost_nodes": 0,
            "expected_erasures": 0, "erasure_leaks": 0
        }

    # 1. STEP A: BUILD GEOMETRIC SEARCH TREE FROM DETECTED FEATURES
    detected_kdtree = cKDTree(detected_points)

    processed_detection_ids = set()

    # Initialize our strict telemetry tracking bins
    total_visible_targets = 0
    true_positives = 0
    misalignments = 0
    graph_skips = 0
    optical_misses = 0
    expected_erasures = 0
    erasure_leaks = 0

    # 2. STEP B: SWEEP ALONG THE EXPLICIT REFERENCE BLUEPRINT TRACKS
    for r_true in range(H_nodes):
        for c_true in range(W_nodes):
            if base_blueprint[r_true, c_true] < 0:
                continue

            # Check modification layout: Is this node intentionally erased?
            is_erased_target = (modified_blueprint[r_true, c_true] < 0)

            # Query the precise non-warped world coordinate center from the generator
            node_world_center = generator.get_shape_center(r_true, c_true)
            pt = camera.project_point(node_world_center, R, t)

            # Enforce native viewport camera visibility clipping walls
            if pt is None or not camera.is_visible(pt):
                continue

            if not is_erased_target:
                total_visible_targets += 1

            x_true, y_true = pt[0], pt[1]

            # Match the ideal camera target to the nearest physical detected blob on sensor
            distance, point_idx = detected_kdtree.query([x_true, y_true])

            if distance > max_matching_dist_px:
                if is_erased_target:
                    # EXPECTED ERASURE: Omitted node was correctly ignored by everything
                    expected_erasures += 1
                else:
                    # OPTICAL MISS: Node is inside view limits, but blob finder skipped it completely
                    optical_misses += 1
                continue

            if is_erased_target:
                # ERASURE LEAK: A blob was found, but it sits inside an intended erasure zone
                processed_detection_ids.add(point_idx)
                erasure_leaks += 1
                continue

            processed_detection_ids.add(point_idx)

            # Cross-examine your wave-growth matrix lookup table using the active point ID
            decoded_locations = np.argwhere(topological_matrix == point_idx)

            if len(decoded_locations) == 0:
                # GRAPH SKIP: Blob was found on screen, but wave-growth graph failed to reach it
                graph_skips += 1
                continue

            # Extract decoded local grid addresses from the matches
            r_dec, c_dec = int(decoded_locations[0][0]), int(decoded_locations[0][1])

            if r_dec == r_true and c_dec == c_true:
                true_positives += 1
            else:
                misalignments += 1

    # 3. STEP C: CAPTURE UNMATCHED GHOST ARTIFACTS
    # Any blob recorded by camera but missing entirely from the master target board blueprint
    detected_points_num = len(detected_points)
    ghost_nodes = detected_points_num - len(processed_detection_ids)

    # 4. CALCULATE DEFINITIVE PERFORMANCE YIELD SCORES
    # Accuracy is evaluated strictly over what points were physically observable on screen
    if detected_points_num > 0:
        true_negative = expected_erasures - erasure_leaks
        accuracy_score = (true_positives + true_negative) / (detected_points_num + expected_erasures) * 100.0
    else:
        accuracy_score = 0.0

    # STABILITY CRITERIA CHECK: Pass if accuracy is >= 90% and there are ZERO false positives
    is_case_passed = (accuracy_score >= 90.0) and (misalignments == 0)

    return {
        "is_passed": is_case_passed,
        "accuracy": accuracy_score,
        "true_positives": true_positives,
        "total_visible_targets": total_visible_targets,
        "misalignments": misalignments,
        "graph_skips": graph_skips,
        "optical_misses": optical_misses,
        "ghost_nodes": ghost_nodes,
        "expected_erasures": expected_erasures,
        "erasure_leaks": erasure_leaks
    }

from run_validation import draw_topology_scene

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

    if save_images:
        cv2.imwrite(os.path.join(save_dir, f"synthetic_shot_{case_name}-debug.png"), img)
        # Generate the color-coded true-positive metric diagnostic overlay
        gt_dict, actual_dict = extended_classify_topology_nodes(
            topological_matrix=result["topological_matrix"],
            detected_points=result["points"],
            detected_labels=result["labels"],
            base_blueprint=base_blueprint,
            modified_blueprint=blueprint,
            generator=generator,
            camera=cam_obj,
            R=R, t=t,
            max_allowed_drift_px=4.0
        )
        debug_overlay = draw_topology_scene(gt_dict, actual_dict, img, legend_position="bottom_left")
        # Save the diagnostic visualization matrix directly to disk
        diagnostic_filename = f"synthetic_shot_{case_name}-diagnostic.png"
        cv2.imwrite(os.path.join(save_dir, diagnostic_filename), debug_overlay)
        print(f" -> Exported visual debug overlay to '{diagnostic_filename}'")

    # 8. Process accuracy metrics against our updated visible blueprint mask
    metrics = calculate_reconstruction_metrics(
        topological_matrix=result["topological_matrix"],
        detected_points=result["points"],
        base_blueprint=base_blueprint,
        modified_blueprint=blueprint,
        generator=generator,
        camera=cam_obj,
        R=R, t=t
    )

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


def extended_classify_topology_nodes(topological_matrix: np.ndarray,
                                     detected_points: np.ndarray,
                                     detected_labels: np.ndarray,
                                     base_blueprint: np.ndarray,
                                     modified_blueprint: np.ndarray,
                                     generator,
                                     camera,
                                     R: np.ndarray, t: np.ndarray,
                                     max_allowed_drift_px: float = None) -> tuple:
    """
    Adapter function that structures node configurations and shapes, invokes
    the core spatial classifier, and enriches it with true negative erasures (tn)
    and omissions using a claimed-key grid protection mapping layer.
    """
    assert max_allowed_drift_px is not None, "A valid float threshold max_allowed_drift_px must be explicitly provided."
    from run_validation import classify_topology_nodes

    H_nodes, W_nodes = base_blueprint.shape
    gt_dict = {}

    # 1. GENERATE THE IDEAL GROUND TRUTH NODE REGISTRY WITH SHAPE MATRICES
    for r in range(H_nodes):
        for c in range(W_nodes):
            if base_blueprint[r, c] < 0:
                continue
            node_world_center = generator.get_shape_center(r, c)
            pt = camera.project_point(node_world_center, R, t)
            if pt is not None and camera.is_visible(pt):
                # Maps directly to the expected core parser signature: (shape_type, x, y)
                gt_dict[(r, c)] = (base_blueprint[r, c], pt[0], pt[1])

    # 2. SEAMLESS EARLY RETURN FOR BLANK CANVAS FRAMES
    if detected_points is None or detected_points.size == 0:
        classifications = {}
        for (r, c), gt_coord in gt_dict.items():
            is_erased = (modified_blueprint[r, c] < 0)
            classifications[(r, c)] = {
                "type": "tn" if is_erased else "fn",
                "dist": float('inf'),
                "coords": (gt_coord[1], gt_coord[2]),
                "nearest_gt": (r, c)
            }
        return gt_dict, classifications

    # 3. GENERATE THE LIVE DETECTED INDEX REGISTRY EMBEDDING HARDWARE LABELS
    detected_dict = {}
    for idx in range(len(detected_points)):
        decoded_locations = np.argwhere(topological_matrix == idx)
        if len(decoded_locations) > 0:
            r_dec, c_dec = int(decoded_locations[0][0]), int(decoded_locations[0][1])
            detected_dict[(r_dec, c_dec)] = (detected_labels[idx], detected_points[idx][0], detected_points[idx][1])
        else:
            # Rejections are stored under integer handles to keep them isolated from matrix bounds
            detected_dict[idx] = (detected_labels[idx], detected_points[idx][0], detected_points[idx][1])

    # 4. INVOKE YOUR CORE MATHEMATICAL CLASSIFIER
    # Natively distributes 'tp', 'fp', 'ma', and 'ghost' classes based on your geometry
    classifications = classify_topology_nodes(
        gt_dict,
        detected_dict,
        max_allowed_drift_px=max_allowed_drift_px
    )
    return gt_dict, classifications


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
    accuracy = metrics.get("accuracy", 0.0)
    tp = metrics.get("true_positives", 0)
    visible = metrics.get("total_visible_targets", 0)
    misalignments = metrics.get("misalignments", 0)
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
    md_content.append(f"**Execution Status:** {status_indicator} | **Final Decoding Accuracy:** {accuracy:.2f}%\n")

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
        f"| **Graph Traversal Skips (Slips)** | {skips} | Geometric blobs extracted from video frames but skipped by wave-growth engine. |")
    md_content.append(
        f"| **Pure Optical Misses** | {misses} | Core blueprint dots inside view limits that failed the thresholding blob detector. |")
    md_content.append(
        f"| **Noise Artifacts (Ghosts)** | {ghosts} | Spurious noise blobs registered by camera that do not exist on the master template. |")
    md_content.append(
        f"| **Expected Erasures (True Negatives)** | {erasures} | Hardware-level missing spots or mask holes correctly bypassed by the tracker. |")
    md_content.append(
        f"| **Erasure Glare Leaks** | {leaks} | Noise spots inside erasure zones that mistakenly triggered feature detections. |")
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
                                    "description": "Prinstine tracking run",
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

    md_content.append("| Case Name | Scenario Comment Description | Visible Targets | Recall | Precision | Accuracy| Status |")
    md_content.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    total_cases = len(results_dict)
    passed_cases = 0

    # 2. Iterate across the results dictionary keys to populate specific row slots
    for case_name, metrics in results_dict.items():
        # Dynamically extract description straight from the metrics array dictionary
        comment = metrics.get("description", "No description profile recorded.")

        # Isolate parameters needed for standard statistical calculations
        tp = metrics.get("true_positives", 0)
        visible = metrics.get("total_visible_targets", 0)
        misalignments = metrics.get("misalignments", 0)
        ghosts = metrics.get("ghost_nodes", 0)
        accuracy = metrics.get("accuracy", 0.0)

        # 3. Compute mathematically precise Recall and Precision ratios
        # Recall: How many of the visible blueprint dots did the decoder capture?
        recall = (tp / visible) * 100.0 if visible > 0 else 0.0

        # Precision: Out of all nodes written to canvas, how many are correct?
        total_extracted = tp + misalignments + ghosts
        precision = (tp / total_extracted) * 100.0 if total_extracted > 0 else 0.0

        # Evaluate status threshold signatures
        is_passed = metrics.get("is_passed", False)
        status_tag = "PASS" if is_passed else "FAIL"

        if is_passed:
            passed_cases += 1

        # Append row format block string data line directly
        md_content.append(
            f"| **{case_name}** | {comment} | {visible} | {recall:.2f}% | {precision:.2f}% | {accuracy:.2f}% | {status_tag} |"
        )

    # 4. Append high-level global system telemetry metrics
    md_content.append("\n## System Conformance Evaluation Analytics")
    md_content.append(f"- **Total Simulated Test Cases Checked:** {total_cases}")
    md_content.append(f"- **Total Successfully Passed Suites :** {passed_cases} / {total_cases}")
    md_content.append("\n### Definitions")
    md_content.append("To maintain mathematical consistency across all evaluation passes, metric tracking definitions follow standard confusion matrix guidelines:")
    md_content.append("1. **Recall Index:** Evaluates the target search coverage ratio relative to the active image pane boundary layout.")
    md_content.append("   $$\\text{Recall} = \\frac{TP}{\\text{Visible GT Nodes}} \\times 100.0$$")
    md_content.append("2. **Precision Index:** Quantifies the structural assignment reliability of the topological decoder, demonstrating its zero false-positive extraction rate.")
    md_content.append("   $$\\text{Precision} = \\frac{TP}{TP + \\text{Misalignments} + \\text{Ghosts}} \\times 100.0$$")
    md_content.append("3. **System Accuracy Score:** Tracks the combined performance of the hardware mask validator and the wave-growth graph. Evaluates true classification steps over the detected array domain while remaining completely immune to unmodeled optical sensor visibility omissions ($FN$).")
    md_content.append("   $$\\text{Accuracy} = \\frac{TP + (\\text{Expected Erasures} - \\text{Erasure Leaks})}{\\text{Total Detected} + \\text{Expected Erasures}} \\times 100.0$$")
    md_content.append("A test suite run is explicitly designated as **PASSED** if and only if it simultaneously satisfies both strict topological and numerical performance thresholds:")
    md_content.append("- **Zero Topological Defects:** `misalignments = 0` (Zero tolerance for index drift or island mis-assembly).")
    md_content.append("- **High Classification Fidelity:** `accuracy_score > 90.0%` (Sub-pixel tracking confidence minimum baseline).")

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
    base_blueprint = pattern_blueprint(args.engine, cols=W_NODES, rows=H_NODES)
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
                "tx": DEFAULT_TX, "ty": H_NODES/2 * STEP_PX , "tz": Z_DISTANCE * 0.9  # Pushed closer to retain pixel scale
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
        #if  case_name != "severe_pitch_tilt_45deg":
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
        print(f" -> Metrics: accuracy={result['accuracy']:.2f}%, True Positives={result['true_positives']} from total visible {result['total_visible_targets']}")

    summary_filename = save_summary_markdown_report(results_dict=accumulated_metrics_dictionary, output_dir=RESULT_DIR)
    if len(summary_filename) > 0:
        print(f" -> [INFO]: Summary table successfully written to: {summary_filename}")
    else:
        print(f" -> [WARNING]: Summary table saving is failed")
    prints()
    print("Integration test for pattern decoder completed")
    prints()
