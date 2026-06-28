import cv2
import numpy as np
from scipy.spatial import KDTree
from generate import PhysicalMeshGenerator
from camera import compute_camera_projection_matrix, ProjectiveCamera
from detector import *

def render_warped_grid_shapes(mesh_generator, cam: ProjectiveCamera, Rt: np.ndarray) -> np.ndarray:
    """
    STAGE 3b: LOCAL WARP RENDERING WITH ACCURATE RADIAL LOOKUP FILTERING
    Draws shapes onto the camera viewport by projectively transforming every
    explicit boundary contour vertex using the ProjectiveCamera abstraction layer.

    """
    # 1. Initialize a clean, white canvas matching your camera sensor array size
    canvas = np.ones((cam.H_img, cam.W_img, 3), dtype=np.uint8) * 255

    # 2. Render Loop: Iterate through each active shape block inside the generator
    for i, j, shape_type, contour in mesh_generator:
        # Skip empty padding zones or dead tracking border background marks
        if shape_type < 0:
            continue

        world_pts = np.array(contour, dtype=np.float32)

        # RE-USE OBJECT COMPONENT: Offload all projection transformations,
        # homography math, and radius guardrail intercept checks to the class!
        pixel_pts = cam.project_points(world_pts, Rt)

        # If any single vertex broke past the stable limit radius, the camera
        # class safely returns None, and we drop this truncated edge node cleanly.
        if pixel_pts is None:
            continue

        # 3. Convert to signed 32-bit integers required by OpenCV rendering tools
        pixel_pts_int = np.round(pixel_pts).astype(np.int32)

        # Draw and fill the projected polygon directly on the primary canvas
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

    Fully ASCII-compliant implementation.

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
                                     Rt: np.ndarray,
                                     max_matching_dist_px: float = 4.0) -> dict:
    """
    Computes precise performance and topological divergence statistics by building
    a KD-Tree from live sensor detections and scanning the ground-truth blueprint.

    Fully ASCII-compliant implementation.

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

    # 1. STEP A: BUILD GEOMETRIC SEARCH TREE FROM DETECTED FEATURE BLOBS
    detected_kdtree = KDTree(detected_points)

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
            pixel_pts = camera.project_points(node_world_center, Rt)

            # Enforce native viewport camera visibility clipping walls
            if pixel_pts is None:
                continue
            pixel_pts = pixel_pts[0]
            if not camera.is_visible(pixel_pts):
                continue

            if not is_erased_target:
                total_visible_targets += 1

            x_true, y_true = float(pixel_pts[0]), float(pixel_pts[1])

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

            # THE ABSOLUTE TOPOLOGICAL CONFORMANCE AUDIT
            if r_dec == r_true and c_dec == c_true:
                true_positives += 1
            else:
                misalignments += 1

    # 3. STEP C: CAPTURE UNMATCHED GHOST ARTIFACTS
    # Any blob recorded by camera but missing entirely from the master target board blueprint
    ghost_nodes = len(detected_points) - len(processed_detection_ids)

    # 4. CALCULATE DEFINITIVE PERFORMANCE YIELD SCORES
    # Accuracy is evaluated strictly over what points were physically observable on screen
    accuracy_score = (true_positives / total_visible_targets) * 100.0 if total_visible_targets > 0 else 0.0

    return {
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


def evaluate_single_integration_case(base_blueprint: np.ndarray,
                                     case_name: str,
                                     case_payload: dict,
                                     save_images: bool = False) -> dict:
    """
    Evaluates a single universal integration test case dataset configuration.
    Processes the case array buffer, decodes phases, and maps coordinates in-memory.

    Natively initializes the ProjectiveCamera tracker completely via its constructor
    using parameters extracted straight from the case_payload['camera'] dictionary.

    """
    STEP_PX = 45.0

    blueprint = case_payload["blueprint"]
    cam_params = case_payload["camera"]
    cam_intrinsics = case_payload["intrinsics"]

    # 1. Extract hardware parameters directly from the dictionary block
    f_px = cam_intrinsics.get("f_px")
    width = cam_intrinsics.get("width_px")
    height = cam_intrinsics.get("height_px")
    cx = cam_intrinsics.get("cx", width/2)
    cy = cam_intrinsics.get("cy", height/2)
    k1 = cam_intrinsics.get("k1")

    cam_obj = ProjectiveCamera((width, height), f_px=f_px, cx=cx, cy=cy, k1=k1)

    # 3. Compute camera projection extrinsics matrix [R|t] per case profile
    Rt = compute_camera_projection_matrix(
        roll_deg=cam_params["roll"],
        pitch_deg=cam_params["pitch"],
        yaw_deg=cam_params["yaw"],
        tx=cam_params["tx"],
        ty=cam_params["ty"],
        tz=cam_params["tz"]
    )

    # 4. Simulate physics generation pass and build the local adaptive tilted patch
    generator = PhysicalMeshGenerator(blueprint, STEP_PX, STEP_PX / 5)

    # Render the frame cleanly using our short object component pipeline
    img = render_warped_grid_shapes(generator, cam_obj, Rt)

    # Optional image saving gate
    if save_images:
        output_filename = f"synthetic_shot_{case_name}.png"
        cv2.imwrite(output_filename, img)
        print(f" -> Export Complete: Saved output image array to '{output_filename}'")
        metrics = {"status": "success", "message": "save_only"}
    else:
        # 5. Adjust reference blueprint targets for out-of-frame boundary clipping
        visible_blueprint = compute_visible_blueprint(
            base_blueprint=blueprint,
            generator=generator,
            camera=cam_obj,
            Rt=Rt
        )

        pts, labels = detect_and_classify_grid_nodes(img)
        visualize_detections(img, pts, labels)
        topological_matrix = np.full(blueprint.shape, -1, dtype=np.int32)
        if len(pts) > 0:
            matches_islands = reconstruct_mesh(pts, labels)
            for island in matches_islands:
                island_label_map = map_matrix_indices(island, labels)
                match_result = localize_grid(island_label_map, base_blueprint.shape[1], base_blueprint.shape[0])
                if match_result is not None:
                    map_island_indices_to_blueprint(island, match_result, topological_matrix)
                    visualize_reconstructed_grid(img, island, pts)

            wiped_points_num = verify_and_cleanse_topological_matrix(
                topological_matrix,
                base_blueprint, labels)
            np.set_printoptions(threshold=np.inf, linewidth=200)
            print("original")
            print(visible_blueprint)
            print("restored")
            print(map_matrix_indices(topological_matrix, labels))

        output_filename = f"synthetic_shot_{case_name}_result.png"
        cv2.imwrite(output_filename, img)

        # Generate the color-coded true-positive metric diagnostic overlay
        debug_overlay = render_telemetry_grid_overlay(
            frame=None,
            topological_matrix=topological_matrix,
            detected_points = pts,
            base_blueprint=base_blueprint,  # Cross-reference against the visible layout mask
            modified_blueprint = blueprint,
            generator=generator,
            camera=cam_obj,
            Rt=Rt
        )

        # Save the diagnostic visualization matrix directly to disk
        diagnostic_filename = f"diagnostic_overlay_{case_name}.png"
        cv2.imwrite(diagnostic_filename, debug_overlay)
        print(f" -> Diagnostic Telemetry Complete: Exported visual debug overlay to '{diagnostic_filename}'")

        # 8. Process accuracy metrics against our updated visible blueprint mask
        metrics = calculate_reconstruction_metrics(
            topological_matrix=topological_matrix,
            detected_points=pts,  # Ensure your script extracts and forwards this array
            base_blueprint=base_blueprint,
            modified_blueprint=blueprint,
            generator=generator,
            camera=cam_obj,
            Rt=Rt
        )

        metrics["status"] = "success"
        metrics["case_name"] = case_name

    return metrics


def compute_visible_blueprint(base_blueprint: np.ndarray,
                              generator,
                              camera: ProjectiveCamera,
                              Rt: np.ndarray) -> np.ndarray:
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
            pixel_pts = camera.project_points(node_world_center, Rt)

            if pixel_pts is None or not camera.is_visible(pixel_pts[0]):
                visible_blueprint[r, c] = -1

    return visible_blueprint


def render_telemetry_grid_overlay(frame: np.ndarray,
                                  topological_matrix: np.ndarray,
                                  detected_points: np.ndarray,
                                  base_blueprint: np.ndarray,
                                  modified_blueprint: np.ndarray,
                                  generator,
                                  camera,
                                  Rt: np.ndarray,
                                  max_matching_dist_px: float = 4.0) -> np.ndarray:
    """
    Renders an inverted color-coded diagnostic geometric overlay by building a
    KD-Tree from raw detected points and sweeping across the ground-truth blueprint.

    Cross-checks against the active modified_blueprint array passed from your
    test runner configurations to seamlessly identify broken or omitted points.

    Color Guide:
      - Bright Green  (0, 255, 0)   : True Positive (Detected and decoded perfectly)
      - Bright Red    (0, 0, 255)   : False Positive / Misaligned Index Drift
      - Amber/Orange  (0, 165, 255) : Optical Lock Found, but Skipped by Wave-Growth
      - Bright Blue   (255, 0, 0)   : Pure Optical Miss (Hidden baseline point)
      - Bright Magenta(255, 0, 255) : True Negative / Expected Erasure (Broken Point)
    """
    H_nodes, W_nodes = base_blueprint.shape

    # 1. Canvas Frame Protection Initialization
    if frame is None:
        overlay_canvas = np.ones((camera.H_img, camera.W_img, 3), dtype=np.uint8) * 255
    elif len(frame.shape) == 2:
        overlay_canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        overlay_canvas = np.copy(frame)

    # If no points are found on screen, immediately paint any expected erasures and exit
    if detected_points is None or len(detected_points) == 0:
        print(" -> [WARNING] Visualizer received an empty detected_points array.")
        # Draw expected broken points even on a completely blank canvas frame
        for r_true in range(H_nodes):
            for c_true in range(W_nodes):
                if base_blueprint[r_true, c_true] >= 0 and modified_blueprint[r_true, c_true] < 0:
                    node_world_center = generator.get_shape_center(r_true, c_true)
                    pixel_pts = camera.project_points(node_world_center, Rt)
                    if pixel_pts is not None and camera.is_visible(pixel_pts):
                        cv2.circle(overlay_canvas, (int(np.round(pixel_pts[0])), int(np.round(pixel_pts[1]))),
                                   5, (255, 0, 255), -1, lineType=cv2.LINE_AA)
        return overlay_canvas

    # 2. STEP A: BUILD THE TREE DIRECTLY FROM LIVE SENSOR DETECTED POINTS
    detected_kdtree = KDTree(detected_points)
    processed_detection_ids = set()

    # 3. STEP B: TRAVERSE THE GROUND-TRUTH BLUEPRINT MATRIX SCHEMA
    for r_true in range(H_nodes):
        for c_true in range(W_nodes):
            # Skip unpopulated grid background voids entirely
            if base_blueprint[r_true, c_true] < 0:
                continue

            # Query the spatial model center point from the generator
            node_world_center = generator.get_shape_center(r_true, c_true)
            pixel_pts = camera.project_points(node_world_center, Rt)

            # Filter out points that fall outside your exact camera viewport boundary pass
            if pixel_pts is None:
                continue
            pixel_pts = pixel_pts[0]
            if not camera.is_visible(pixel_pts):
                continue
            # Check modification layout: Is this point known to be broken/erased?
            is_broken_point = (modified_blueprint[r_true, c_true] < 0)

            # Query the detected points tree to see if the blob finder captured anything here
            distance, point_idx = detected_kdtree.query(pixel_pts)

            if distance > max_matching_dist_px:
                if is_broken_point:
                    # TRUE NEGATIVE: Broken point was correctly ignored by the optical layer
                    cv2.circle(overlay_canvas, (int(np.round(pixel_pts[0])), int(np.round(pixel_pts[1]))),
                               5, (255, 0, 255), -1, lineType=cv2.LINE_AA)
                else:
                    # OPTICAL MISS: Baseline dot is inside view limits, but blob detector missed it!
                    cv2.circle(overlay_canvas, (int(np.round(pixel_pts[0])), int(np.round(pixel_pts[1]))),
                               4, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                continue

            # If the point was captured but it was supposed to be erased, flag it as an optical leak
            if is_broken_point:
                processed_detection_ids.add(point_idx)
                cv2.circle(overlay_canvas, (int(np.round(pixel_pts[0])), int(np.round(pixel_pts[1]))),
                           7, (0, 0, 255), 2, lineType=cv2.LINE_AA)  # Red hollow ring over broken leak
                continue

            processed_detection_ids.add(point_idx)

            # Search your topological matrix lookup table to cross-examine index placement
            decoded_locations = np.argwhere(topological_matrix == point_idx)

            if len(decoded_locations) == 0:
                # UNDECODED ELEMENT: Blob was physically found, but the wave-growth graph missed it!
                cv2.circle(overlay_canvas,
                           (int(np.round(detected_points[point_idx][0])), int(np.round(detected_points[point_idx][1]))),
                           5, (0, 165, 255), -1, lineType=cv2.LINE_AA)
                continue

            r_dec, c_dec = int(decoded_locations[0][0]), int(decoded_locations[0][1])

            # THE DEFINITIVE METRIC EQUALITY CHECKS
            if r_dec == r_true and c_dec == c_true:
                # TRUE POSITIVE: Geometric extraction and algebraic decoding match perfectly!
                node_color = (0, 255, 0)
                marker_radius = 6
            else:
                # FALSE POSITIVE ALIGNMENT DRIFT: Decoded to the wrong matrix index slot!
                node_color = (0, 0, 255)
                marker_radius = 8

            cv2.circle(overlay_canvas,
                       (int(np.round(detected_points[point_idx][0])), int(np.round(detected_points[point_idx][1]))),
                       marker_radius, node_color, -1, lineType=cv2.LINE_AA)

    # 4. STEP C: DETECT PHANTOM GHOST NOISE ANOMALIES
    # Any detected point left unmatched by the blueprint pass is drawn as a Red square
    for point_idx in range(len(detected_points)):
        if point_idx not in processed_detection_ids:
            cv2.rectangle(overlay_canvas,
                          (int(detected_points[point_idx][0]) - 4, int(detected_points[point_idx][1]) - 4),
                          (int(detected_points[point_idx][0]) + 4, int(detected_points[point_idx][1]) + 4),
                          (0, 0, 255), -1, lineType=cv2.LINE_AA)

    return overlay_canvas


if __name__ == "__main__":
    import argparse
    from generate import generate_triangular_gray_grid

    # Setup clean command line interface parameters parsing parsing
    parser = argparse.ArgumentParser(description="Universal End-to-End Core Integration Suite")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Export all high-fidelity rendered perspective-warped frames to PNG assets on disk."
    )
    args = parser.parse_args()

    W_NODES, H_NODES = 31, 31

    STEP_PX = 45.0
    Z_DISTANCE = H_NODES * STEP_PX * 1.1
    DEFAULT_TX = 0
    DEFAULT_TY = 0
    IMG_SHAPE = (1080, 1920)
    K1 = -1.5e-7

    base_blueprint = generate_triangular_gray_grid(width_nodes=W_NODES, height_nodes=H_NODES)
    # Define your centralized parametric evaluation dictionary matrix block
    cases = {
        "clean_baseline": {
            "description": "Pristine Baseline Frame (Standard Centered Orientation)",
            "blueprint": np.copy(base_blueprint),
            "camera": {"roll": 0.0, "pitch": 0.0, "yaw": -1.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE},
            "intrinsics": {"f_px": 1150.0, "k1": K1, "width_px": IMG_SHAPE[1], "height_px": IMG_SHAPE[0]}
        },
        "erasures": {
            "description": "15% Missing Node Dropouts (Standard Centered Orientation)",
            "blueprint": inject_matrix_erasures(base_blueprint, erasure_probability=0.15),
            "camera": {"roll": 0.0, "pitch": 0.0, "yaw": -1.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE},
            "intrinsics": {"f_px": 1150.0, "k1": K1, "width_px": IMG_SHAPE[1], "height_px": IMG_SHAPE[0]}
        },
        "bitflips": {
            "description": "5% Random Bit Flip Threshold Noise (Standard Centered Orientation)",
            "blueprint": inject_matrix_bit_flips(base_blueprint, flip_probability=0.05),
            "camera": {"roll": 0.0, "pitch": 0.0, "yaw": -1.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE},
            "intrinsics": {"f_px": 1150.0, "k1": K1, "width_px": IMG_SHAPE[1], "height_px": IMG_SHAPE[0]}
        },
        "cropped": {
            "description": "Partial Viewport Aperture Geometric Crop (Standard Centered Orientation)",
            "blueprint": apply_geometric_aperture_crop(base_blueprint, center_row_pct=0.4, center_col_pct=0.4, radius_pct=0.25),
            "camera": {"roll": 0.0, "pitch": 0.0, "yaw": -1.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE},
            "intrinsics": {"f_px": 1150.0, "k1": K1, "width_px": IMG_SHAPE[1], "height_px": IMG_SHAPE[0]}
        },
        "rotated_45_roll": {
            "description": "Severe 45-Degree Roll Rotation Around Optical Axis",
            "blueprint": np.copy(base_blueprint),
            # Severe roll skew applied around the optical Z-axis with an alternative translation offset
            "camera": {"roll": 45.0, "pitch": 0.0, "yaw": 0.0, "tx": DEFAULT_TX * 0.8, "ty": DEFAULT_TY * 0.8,
                       "tz": Z_DISTANCE * 1.2},
            "intrinsics": {"f_px": 1150.0, "k1": K1, "width_px": IMG_SHAPE[1], "height_px": IMG_SHAPE[0]}

        },
        "extreme_stress": {
            "description": "Combined 10% Erasures + 45-Deg Roll Rotation + Camera Perspective Shift",
            "blueprint": inject_matrix_erasures(base_blueprint, erasure_probability=0.10),
            "camera": {"roll": 45.0, "pitch": 15.0, "yaw": -10.0, "tx": DEFAULT_TX * 0.9, "ty": DEFAULT_TY * 1.1,
                       "tz": Z_DISTANCE * 0.95},
            "intrinsics": {"f_px": 1150.0, "k1": K1, "width_px": IMG_SHAPE[1], "height_px": IMG_SHAPE[0]}

        },
        "multi_island_stitch": {
            "description": "Stitching 3 Separated Occluded Fragment Patches in Memory",
            "blueprint": apply_multi_island_mask(base_blueprint),
            "camera": {
                "roll": 30.0, "pitch": 5.0, "yaw": -2.0,
                "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE
            },
            "intrinsics": {"f_px": 1150.0, "k1": K1, "width_px": IMG_SHAPE[1], "height_px": IMG_SHAPE[0]}
        },
        "severe_pitch_tilt_45deg": {
            "description": "Severe 45-Degree Camera Pitch Foreshortening Stress Test",
            "blueprint": np.copy(base_blueprint),
            "camera": {
                "roll": 0.0, "pitch": 45.0, "yaw": 0.0,
                "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE * 0.9  # Pushed closer to retain pixel scale
            },
            "intrinsics": {"f_px": 1150.0, "k1": K1, "width_px": IMG_SHAPE[1], "height_px": IMG_SHAPE[0]}
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
            "intrinsics":{ "f_px": 1150.0, "k1" : K1, "width_px": IMG_SHAPE[1],"height_px": IMG_SHAPE[0]}
        }

    print("=======================================================")
    print("Launching Universal Telemetry Evaluation Loop Sweep...")
    print(f"Image Export Policy: {'ENABLED' if args.save_images else 'DISABLED'}")
    print("=======================================================")

    for case_name, case_payload in cases.items():
        print(f"\n[EVALUATING]: Case Module [{case_name.upper()}]")

        # Pass the command line flag directly down to the single-case evaluator
        result = evaluate_single_integration_case(
            base_blueprint=base_blueprint,
            case_name=case_name,
            case_payload=case_payload,
            save_images=args.save_images
        )

        if result["status"] != "success":
            print(f" -> [WARNING] Subgraph decoder was unable to find phase lock consensus.")
            continue
        if not args.save_images:
            print(f" -> Metrics: accuracy={result['accuracy']:.2f}%, True Positives={result['true_positives']} from total visible {result['total_visible_targets']}")

    print("\n=======================================================")
    print("UNIVERSAL INTEGRATION METRICS SWEEP FULLY RUN!")
    print("=======================================================")
