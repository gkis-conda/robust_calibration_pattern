import cv2
import numpy as np
import argparse
import sys
from crystal import reconstruct_mesh
from matcher import localize_grid
from lattice_topology import *
from generate import generate_triangular_gray_grid
from optimization import *
from camera import ProjectiveCamera
from pathlib import Path


def map_matrix_indices(matrix, labels):
    labels_map = matrix.copy()
    h,w = labels_map.shape[:2]
    for r in range(h):
        for c in range(w):
            v = matrix[r, c]
            if v != -1:
                labels_map[r,c] = labels[v]
    return labels_map


def detect_and_classify_grid_nodes(img, min_area=15, max_area=5000, epsilon_coeff=0.04, edge_margin_px=2):
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
    
    # Ideal calibration references
    ideal_tri_circularity = 0.605
    ideal_circle_circularity = 1.000
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 1. Broad Area Gate
        if not (float(min_area) < area < float(max_area)):
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
        if solidity < 0.82 or circularity < 0.35 or circularity > 1.20:
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
                
    if len(points_list) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int32)
        
    return np.array(points_list, dtype=np.float64), np.array(labels_list, dtype=np.int32)


def visualize_detections(img, points, labels):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3  # Font size multiplier
    color = (0, 0, 0)
    thickness = 1  # Line thickness in pixels

    # 3. Add text to the image

    idx = 0
    radius = 7
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

    # Map the absolute registered source cell coordinates into the un-tilted framework
    source_row_rot, source_col_rot = rotate_barycentric(source_row_orig, source_col_orig, -k_steps)
    v_src_abs = source_row_rot
    u_src_abs = source_col_rot - (v_src_abs // 2)

    # Unwarp your discrete (r, c) bounding box corner into linear axes
    v_min = min_r
    u_min = min_c - (v_min // 2)

    # Perform the window translation subtraction inside the linear topological domain
    v_local = v_src_abs - v_min
    u_local = u_src_abs - u_min

    # Reconstruct local relative target coordinates to preserve the logging trace
    source_row = int(v_local)
    source_col = int(u_local + (source_row // 2))

    print("Source coords:", source_row, source_col)
    print("Target coords:", target_row, target_col)

    h, w = flat_patch.shape

    # Pre-unwarp global target anchors into the pure linear barycentric domain
    v_tgt_linear = target_row
    u_tgt_linear = target_col - (v_tgt_linear // 2)
    v_offset =  v_tgt_linear - v_local
    u_offset = u_tgt_linear - u_local

    # 5. Single-pass mapping loop to merge data on the shared global matrix canvas
    for r in range(h):
        for c in range(w):
            point_idx = flat_patch[r, c]

            # Skip empty padding spaces or dead tracking border zones
            if point_idx == -1:
                continue

            # --- PURE BARYCENTRIC TRANSLATION (NO COGNITIVE OVERHEAD) ---
            # 1. Unwarp the current local patch coordinate into linear barycentric parameters
            v_pt_linear = r
            u_pt_linear = c - (v_pt_linear // 2)

            # 2. Perform flat continuous vector translation (100% stable and transitive!)
            v_global_linear = v_pt_linear + v_offset
            u_global_linear = u_pt_linear + u_offset

            # 3. Apply the absolute boundary normalization check directly on the linear coordinates
            # right before writing to the canvas matrix sheet
            global_r, global_c = normalize_barycentric(u_global_linear, v_global_linear, 31)

            # Update elements in-place on the shared master tracking canvas
            if (0 <= global_r < H_global) and (0 <= global_c < W_global):
                topological_matrix[global_r, global_c] = point_idx


def verify_and_cleanse_topological_matrix(topological_matrix: np.ndarray,
                                          blueprint_matrix: np.ndarray,
                                          labels: list) -> int:
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

    # Walk cell-by-cell over the entire physical coordinate sheet viewport
    for r in range(H_global):
        for c in range(W_global):
            point_idx = topological_matrix[r, c]

            # Skip empty background locations or unpopulated tracking slots
            if point_idx == -1:
                continue

            if labels[point_idx] != blueprint_matrix[r,c]:
                # If an out-of-bounds or corrupt label key slips through, wipe it instantly
                topological_matrix[r, c] = -1
                wiped_count += 1

    return wiped_count



class HexagonalTopologyDetector:
    """
    """
    def __init__(self, grid_rows, grid_cols):
        # OpenCV grid patterns expect dimensions passed as (columns, rows)
        self.grid_size = (grid_cols, grid_rows)

    def register_pattern(self, img, overlay: True):
        """
        Returns:
            dict: {(row, col): [x_px, y_px]} containing indexed sub-pixel centers.
        """
        result = {}
        pts, labels = detect_and_classify_grid_nodes(img)
        result["points"] = pts
        result["labels"] = labels
        width, height = self.grid_size
        topological_matrix = np.full((height, width), -1, dtype=np.int32)
        if len(pts) == 0:
            return result
        if overlay:
            visualize_detections(img, pts, labels)

        matches_islands = reconstruct_mesh(pts, labels)
        matches = []
        for island in matches_islands:
            if overlay:
                visualize_reconstructed_grid(img, island, pts)
            island_label_map = map_matrix_indices(island, labels)
            match_result = localize_grid(island_label_map, width, height)
            if match_result is not None:
                matches.append(match_result)
                map_island_indices_to_blueprint(island, match_result, topological_matrix)

        wiped_ghosts = verify_and_cleanse_topological_matrix(
                topological_matrix, generate_triangular_gray_grid(width, height), labels)
        result["matches"] = matches
        if len(matches) > 0:
            result["topological_matrix"] = topological_matrix
        return result

# =====================================================================
# SYSTEM TERMINAL INTERFACE
# =====================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Extraction Parser.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input calibration frame image path.")
    parser.add_argument("-o", "--output", type=str, default="", help="Output file path.")
    parser.add_argument("-C", "--calibrate", action='store_true', help="Perform focal distance and k1 calibration")
    args = parser.parse_args()
    if args.output == "":
        output = Path(args.input).stem + "_result.png"
    else:
        output = args.output

    img = cv2.imread(args.input)
    if img is None:
        print(f"[Error] Visualizer failed to open file at: '{args.input}'", file=sys.stderr)
    else:
        detector = HexagonalTopologyDetector(31, 31)
        result = detector.register_pattern(img, overlay=True)
        labels = result["labels"]
        pts = result["points"]
        print(f"Extraction Successful! Isolated {len(pts)} total pattern nodes.")
        print(f" -> Circles identified: {np.sum(labels == 0)}")
        print(f" -> Triangles identified: {np.sum(labels == 1)}")
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
            H, W = img.shape[:2]
            if args.calibrate:
               cam = ProjectiveCamera((W, H), f_px=(W+H)/4, cx=W/2, cy=H/2, k1=-1.e-7)
               result = calibrate_single_frame_zhang_menger(topological_matrix, pts, cam)
               print(result)