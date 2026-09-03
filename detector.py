import numpy as np
from crystal import reconstruct_mesh
from matcher import AlgebraicGridDecoder32
from lattice_topology import *
from generate import generate_triangular_gray_grid
from optimization import *
import camera_io
import cv2


def map_matrix_indices(matrix, labels):
    labels_map = matrix.copy()
    h,w = labels_map.shape[:2]
    for r in range(h):
        for c in range(w):
            v = matrix[r, c]
            if v != -1:
                labels_map[r,c] = labels[v]
    return labels_map


def detect_and_classify_grid_nodes(img, min_area=15, max_area=5000, edge_margin_px=2):
    """
    Detects target dots from a real-world photo. Utilizes a continuous metric 
    distance classifier and filters out partial/cut shapes touching the image 
    edges to prevent false triangle classifications.
    """
    if img is None:
        raise ValueError("Detector error: Provided image matrix is empty or None.")
        
    if len(img.shape) == 3:
        H_img, W_img, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        H_img, W_img = img.shape
        gray = img.copy()
        
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=51, 
        C=7
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    points_list = []
    labels_list = []
    size_list = []
    # Ideal calibration references
    ideal_tri_circularity = 0.605
    ideal_circle_circularity = 1.000
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 1. Broad Area Gate
        if not (min_area < area < max_area):
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        # Reshape to easily evaluate point coordinate bounds
        pts = contour.reshape(-1, 2)
        
        # 2. Image Edge Proximity Filter
        # Skip shapes that are sliced/cut by the camera viewport boundaries
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        
        if (min_x <= edge_margin_px or max_x >= (W_img - 1 - edge_margin_px) or
            min_y <= edge_margin_px or max_y >= (H_img - 1 - edge_margin_px)):
            continue  # Safe skip: Discard cut edge shapes
            
        # 3. Extract Smooth Continuous Metrics
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 4. Dynamic Thresholding Noise Filter
        if solidity < 0.82 or circularity < ideal_tri_circularity * 0.5  or circularity > ideal_circle_circularity * 1.20:
            continue  
            
        # 5. Continuous Distance Classifier
        dist_to_triangle = abs(circularity - ideal_tri_circularity)
        dist_to_circle = abs(circularity - ideal_circle_circularity)
        
        # 6. Resolve Shape Identity
        if dist_to_triangle < dist_to_circle:
            shape_label = 1  # Triangle Profile
        else:
            shape_label = 0  # Circle Profile
            
        # Calculate fast barycenter for verified grid nodes
        cx, cy = np.mean(pts, axis=0)
        
        points_list.append([cx, cy])
        labels_list.append(shape_label)
        size_list.append(perimeter/np.pi)

    return np.array(points_list, dtype=np.float64), np.array(labels_list, dtype=np.int32), np.array(size_list, np.float64)


def visualize_detections(img, points, labels):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3  # Font size multiplier
    color = (0, 0, 0)
    thickness = 1  # Line thickness in pixels
    idx = 0
    radius = 7 # feature size both for circles and rects, rects should be changed to triangles
    for pt, label in zip(points, labels):
        ix, iy = int(np.round(pt[0])), int(np.round(pt[1])) # Fixed spatial indexing typo
        if label == 0:
            cv2.circle(img, (ix, iy), radius, (0, 255, 0), 1) # Green Circle
        else:
            cv2.rectangle(img, (ix - radius, iy - radius), (ix + radius, iy + radius), (255, 0, 0), 1) # Blue Square
        cv2.putText(img, str(idx),( ix + radius, iy + radius), font, font_scale, color, thickness, cv2.LINE_AA)
        idx += 1
            

def visualize_reconstructed_grid(canvas, grid_matrix, pts):
    """
    Draws structural wireframe edge connections directly onto the provided 
    canvas matrix in-place. Node shapes and file serialization are skipped.
    """
    H_matrix, W_matrix = grid_matrix.shape
    
    # Standard neighborhood shifts for a structured matrix representing a triangular layout.
    # To prevent rendering duplicate line overlays, we only shoot steps Forward and Down.
    # Looking at row parity (r % 2) determines whether columns shift on diagonal jumps.
    
    for r in range(H_matrix):
        row_parity = r % 2
        for c in range(W_matrix):
            curr_idx = grid_matrix[r, c]
            if curr_idx == -1:
                continue  # Skip unassigned grid holes
                
            p_curr = tuple(np.round(pts[curr_idx]).astype(np.int32))
            
            # Form standard structural neighbor edge directions
            # 1. Immediate Right: (r, c+1)
            # 2. Diagonal Down-Right
            # 3. Diagonal Down-Left
            neighbors = [
                (r, c + 1),
                (r + 1, c + row_parity),
                (r + 1, c - 1 + row_parity)
            ]
            
            for nr, nc in neighbors:
                if 0 <= nr < H_matrix and 0 <= nc < W_matrix:
                    neighbor_idx = grid_matrix[nr, nc]
                    if neighbor_idx != -1:
                        p_neighbor = tuple(np.round(pts[neighbor_idx]).astype(np.int32))
                        cv2.line(canvas, p_curr, p_neighbor, (100, 100, 100), 1, cv2.LINE_AA)


def map_island_indices_to_blueprint(
        island_patch: np.ndarray,
        registration_result: dict,
        topological_matrix: np.ndarray) -> None:
    """
    Transforms and maps real physical data indices from an isolated patch
    directly into an existing global topological tracking canvas (in-place).

    Performs all spatial translations inside the pure, unwarped linear barycentric
    domain inside the main loop to completely eliminate non-linear floor-division distortions.
    """
    # 1. Protection Gate: Abort immediately if the registration lock failed
    if registration_result is None or registration_result.get("status") != "success":
        return

    # 2. Extract decoded absolute target and source coordinates from the lock
    target_row = registration_result["row"]
    target_col = registration_result["col"]

    source_row_orig = registration_result["source_row"]
    source_col_orig = registration_result["source_col"]

    # 3. Resolve the rotational quadrant step count via externalized lookup helper
    k_steps = get_rotation_steps_from_axis(
        registration_result["horizontal_axis"],
        registration_result["direction"]
    )

    H_global, W_global = topological_matrix.shape

    # 4. Un-tilt the patch and extract the exact bounding box minimum shifts (min_r, min_c)
    flat_patch, (min_r, min_c) = rotate_barycentric_matrix_adaptive(island_patch, -k_steps)
    # switch to barycentric
    v_min = min_r
    u_min = min_c - (v_min // 2)

    # Map the absolute registered source cell coordinates into the un-tilted framework
    source_row_rot, source_col_rot = rotate_barycentric(source_row_orig, source_col_orig, -k_steps)
    v_src_abs = source_row_rot
    u_src_abs = source_col_rot - (v_src_abs // 2)

    # Perform the window translation subtraction inside the linear topological domain
    v_local = v_src_abs - v_min
    u_local = u_src_abs - u_min

    print("Target coords:", target_row, target_col)

    h, w = flat_patch.shape
    # Pre-unwarp global target anchors into the pure linear barycentric domain
    v_tgt_linear = target_row
    u_tgt_linear = target_col - (v_tgt_linear // 2)
    v_offset = v_tgt_linear - v_local
    u_offset = u_tgt_linear - u_local

    # 5. Single-pass mapping loop to merge data on the shared global matrix canvas
    for r in range(h):
        for c in range(w):
            point_idx = flat_patch[r, c]

            # Skip empty padding spaces or dead tracking border zones
            if point_idx == -1:
                continue

            # 1. Unwarp the current local patch coordinate into linear barycentric parameters
            v_pt_linear = r
            u_pt_linear = c - (v_pt_linear // 2)

            # 2. Perform flat continuous vector translation (100% stable and transitive!)
            v_global_linear = v_pt_linear + v_offset
            u_global_linear = u_pt_linear + u_offset

            # 3. Apply the absolute boundary normalization check directly on the linear coordinates
            # right before writing to the canvas matrix sheet
            global_r, global_c = get_coordinates_from_phase(u_global_linear, v_global_linear, 31)

            # Update elements in-place on the shared master tracking canvas
            if (0 <= global_r < H_global) and (0 <= global_c < W_global):
                topological_matrix[global_r, global_c] = point_idx


def verify_topological_matrix(topological_matrix: np.ndarray,
                              blueprint_matrix: np.ndarray,
                              labels: list):
    """
    Cross-checks the populated topological tracking matrix against the master
    blueprint matrix using real physical label bit assignments.

    Wipes out (sets to -1) any entry where the mapped index value's physical
    bit state conflicts with the expected blueprint bit at that exact coordinate.

    Args:
        topological_matrix (np.ndarray): The global in-place index canvas, shape (H, W).
        blueprint_matrix (np.ndarray): The pristine reference matrix containing correct 0/1 bits.
        labels: Dictionary mapping each discrete physical point index integer
                            to its true binary bit state token (0 or 1).

    Returns:
        int: Total number of mismatched false-positive points wiped out during the pass.
    """
    H_global, W_global = topological_matrix.shape
    wiped_count = 0
    label_status = labels.copy()
    # Walk cell-by-cell over the entire physical coordinate sheet viewport
    correct = set()

    for r in range(H_global):
        for c in range(W_global):
            point_idx = topological_matrix[r, c]
            # Skip unpopulated tracking slots
            if point_idx == -1:
                continue
            if labels[point_idx] == blueprint_matrix[r,c]:
                correct.add(point_idx)
            else:
                wiped_count += 1
                label_status[point_idx] = HexagonalTopologyDetector.MISCLASSIFIED

    mask = (label_status != HexagonalTopologyDetector.MISCLASSIFIED)
    mask[list(correct)] = 0
    label_status[mask] = HexagonalTopologyDetector.GHOST
    return wiped_count, label_status


class HexagonalTopologyDetector:
    MISCLASSIFIED = -1
    GHOST = -2
    ENGINE_FULL_NAME = "Hexagonal Galois Pattern Matching Engine"
    """
     Image decoder, performs full image matching processing pipeline
    """
    def __init__(self, grid_rows, grid_cols):
        self.grid_size = (grid_cols, grid_rows)
        self.decoder = AlgebraicGridDecoder32(grid_cols, grid_rows)

    def register_pattern(self, img, debug_overlay=True):
        """
        Returns:
            dict: {(row, col): [x_px, y_px]} containing indexed sub-pixel centers.
        """
        result = {}
        pts, labels, sizes = detect_and_classify_grid_nodes(img)
        result["points"] = pts
        result["labels"] = labels
        result["sizes"] = sizes
        width, height = self.grid_size
        topological_matrix = np.full((height, width), -1, dtype=np.int32)
        if len(pts) == 0:
            return result
        if debug_overlay:
            visualize_detections(img, pts, labels)

        matches_islands = reconstruct_mesh(pts)
        matches = []
        for island in matches_islands:
            if debug_overlay:
                visualize_reconstructed_grid(img, island, pts)
            island_label_map = map_matrix_indices(island, labels)
            if island_label_map.shape[0] < self.decoder.MIN_UNIQUE_SEQUENCE_LEN and \
                island_label_map.shape[1] < self.decoder.MIN_UNIQUE_SEQUENCE_LEN:
                continue # sort islands by size and use the sequence
            for k in range(6):
                rotated_map, _ = rotate_barycentric_matrix_adaptive(island_label_map, k)
                match_result = self.decoder.localize_grid(rotated_map)
                if match_result["status"] == "success":
                    matches.append(match_result)
                    island, _ = rotate_barycentric_matrix_adaptive(island, k)
                    map_island_indices_to_blueprint(island, match_result, topological_matrix)
                    break

        if len(matches) > 0:
            blueprint = generate_triangular_gray_grid(width, height)
            wiped_ghosts, label_status = verify_topological_matrix(
                topological_matrix, blueprint, labels)
            result["matches"] = matches
            result["label_status"] = label_status
            if wiped_ghosts > len(labels) / 2:
                result["status"] = "error"
                result["message"] = "Too many ghosts"
            else:
                result["status"] = "success"
                result["topological_matrix"] = topological_matrix
        else:
            result["status"] = "error"
            result["message"] = "Detected graph cannot be decoded"

        return result


# =====================================================================
# SYSTEM TERMINAL INTERFACE
# =====================================================================
if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    import glob
    import os


    def process_image(file_name, debug_overlay=False):
        img = cv2.imread(file_name)
        if img is None:
            print(f"[Error] Visualizer failed to open file at: '{file_name}'", file=sys.stderr)
            return None

        detector = HexagonalTopologyDetector(31, 31)
        result = detector.register_pattern(img, debug_overlay=debug_overlay)
        if result['status'] != 'success':
            print(f"[Error] Pattern registration failed for: '{file_name}'", file=sys.stderr)
            return None  # Changed exit(0) to return None to prevent full directory loop aborts!

        labels = result["labels"]
        print(f"Extraction Successful! Isolated {len(labels)} total pattern nodes from {Path(file_name).name}.")
        circles_num = np.sum(labels == 0)
        triangles_num = np.sum(labels == 1)
        total = circles_num + triangles_num
        if total > 0:
            print(f" -> Identified Circles: {circles_num} ({circles_num * 100/total:.2f}%)"
                  f" Triangles {triangles_num} ({triangles_num * 100/total:.2f}%)")
        else:
            print(f" -> No shapes detected")
        if debug_overlay:
            output = Path(file_name).stem + "-debug.png"
            success = cv2.imwrite(output, img)
            if success:
                print(f"Visualization overlay image with marked nodes saved successfully to '{output}'")
            else:
                print(f"Failed to write visualizer image to: '{output}'", file=sys.stderr)

        if "topological_matrix" in result:
            topological_matrix = result["topological_matrix"]
            np.set_printoptions(threshold=np.inf, linewidth=200)
            print("Final mapping")
            print(topological_matrix)
            mapped_labels = map_matrix_indices(topological_matrix, labels)
            print(mapped_labels)
            return result
        return None


    # Command line interface entry parameters
    parser = argparse.ArgumentParser(description="Grid Extraction Parser.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image file path OR dataset directory.")
    parser.add_argument("-o", "--output", type=str, default="", help="Output file path.")
    parser.add_argument("-C", "--calibrate", action='store_true', help="Perform focal distance and k1 calibration")
    parser.add_argument("--save-images", action="store_true", help="Save debug images")
    parser.add_argument("-V", "--verbose", action="store_true", help="Add debug output to console")

    args = parser.parse_args()
    set_debug_output(args.verbose)
    # Populate file-list for processing
    input_path = args.input
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    target_image_files = []

    if os.path.isdir(input_path):
        print(f" -> Input path identified as DIRECTORY: {input_path}")
        for file_name in glob.glob(os.path.join(input_path,'*')):
            # check popular processing filename suffix
            if file_name.lower().endswith(valid_extensions) and\
                    not Path(file_name).stem.endswith(("-debug", "-stat", "_result", "-diagnostic")):
                target_image_files.append(file_name)
        target_image_files.sort()  # Alphabetical sorting for deterministic tracking history
        print(f" -> Collected {len(target_image_files)} calibration graphics.")
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(valid_extensions):
            target_image_files.append(input_path)
        else:
            print(f"[Error] Standalone file extension format selection is not supported: '{input_path}'",
                  file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[Error] Target path destination does not exist on disk: '{input_path}'", file=sys.stderr)
        sys.exit(1)

    if len(target_image_files) == 0:
        print("[Error] Image queue is empty. Stop", file=sys.stderr)
        sys.exit(1)

    if args.calibrate:
        # Load standard config context template from your module profiles based on the first item profile
        baseline_cam = camera_io.find_camera_config(target_image_files[0], load=True)
        if baseline_cam is None:
            img = cv2.imread(target_image_files[0])
            h, w = img.shape[:2]
        else:
            w, h = baseline_cam.img_shape
        initial_cam = ProjectiveCamera((w, h), fx_px=w/2, fy_px=w/2, cx=w/2, cy=h/2, k1=-0.1)
        # Instantiate your new stateful multi-view accumulator container instance
        calibrator = MultiFrameCalibrator(camera_object=initial_cam, N=12, MIN_LEN=15)

        for file_path in target_image_files:
            print(f"\n--- Processing: {os.path.basename(file_path)} ---")
            frame_extraction_result = process_image(file_path, args.save_images)

            if frame_extraction_result is not None:
                calibrator.add_frame(
                    topological_matrix=frame_extraction_result["topological_matrix"],
                    detected_points=frame_extraction_result["points"],
                    point_weights=1./np.array(frame_extraction_result["sizes"])
                )

        # Trigger your smart polymorphic calibration routine (natively splits 1-view vs multi-view matrix models!)
        final_calibration_dict = calibrator.calibrate()
        print("\n--- FINAL CALIBRATION SUMMARY ---")
        print(final_calibration_dict)
        if final_calibration_dict["status"] == "success":
            result_cam = camera_io.deserialize_camera_from_dict(final_calibration_dict)
            if baseline_cam is not None:
                print("\n--- GROUND TRUTH INTRINSICS ---")
                print(camera_io.serialize_camera_to_dict(baseline_cam))
                camera_io.save_camera_comparison_md(input_path, result_cam, baseline_cam)
    else:
        # If calibration flags remain unchecked, execute pure standalone low-level topological mapping loops
        print(" -> Calibration flags deactivated (-C absent). Executing structural grid logging tracks only.")
        for file_path in target_image_files:
            print(f"\n--- Processing: {os.path.basename(file_path)} ---")
            process_image(file_path)
