import numpy as np
import cv2
from detector import detect_and_classify_grid_nodes
from scipy.spatial import KDTree
from typing import Any, Dict, List, Optional, Tuple


def generate_opencv_unary_blueprint(rows, cols):
    """
    OpenCV's cv2.findCirclesGrid demands a pristine array where every node is an identical
    circle tracking element (0). Missing cells or token variances crash the OpenCV graph solver.
    """
    return np.zeros((rows, cols), dtype=np.int32)

# ==============================================================================
# SECTION 1: BLENDER 2.92 GENERATOR INTERFACE (Plugs into rendering script)
# ==============================================================================
class OpenCVCirclesGridMeshGenerator:
    """
    Interface compatible with PhysicalMeshGenerator. 
    Generates a standard industry calibration target: a strict Asymmetric 
    Circles Grid matching OpenCV graph topology constraints.
    """
    def __init__(self, grid_matrix, grid_shape, step_mm, r_circ, circle_points_per_mm=2.0):
        """
        Initializes baseline geometry parameters.
        Note: grid_matrix acts as a structural mask. OpenCV asymmetric grids
        require rows/cols to alternate spacing, but we use your exact hex grid
        lattice setup for direct topology compatibility.
        """
        self.grid_matrix = np.array(grid_matrix)
        self.step_mm = float(step_mm)
        self.r_circ = float(r_circ)
        self.circle_points_per_mm = float(circle_points_per_mm)
        self.grid_shape = grid_shape

        # Center calibration offss: Center coordinate space relative to (0,0)
        self.center_x_offset = - (1.0 + float(self.grid_matrix.shape[1]//2)) / 2.0
        self.center_y_offset = (1.0 - float(self.grid_matrix.shape[0])) / 2.0

    def __iter__(self):
        """
        Iterates over the blueprint matrix to yield baseline dot primitives.
        """
        H_nodes, W_nodes = self.grid_matrix.shape
        for i in range(H_nodes):
            for j in range(W_nodes):
                shape_type = self.grid_matrix[i, j]
                # Enforce standard circles grid: even if blueprint contains triangles (1),
                # the baseline generator forces them to pure circles (0) for OpenCV compatibility.
                if shape_type >= 0: 
                    contour = self.get_shape_contour(i, j)
                    yield i, j, 0, contour  # Force shape_type to 0 (Circle)

    def get_shape_center(self, r, c):
        """
        Computes 2D physical world positions on the hexagonal row-staggered lattice.
        """
        if self.grid_shape == "circles":
            x_phys = (float(c) + self.center_x_offset) * self.step_mm
            y_phys = (float(r) + self.center_y_offset) * self.step_mm
        elif self.grid_shape == "asymmetric_circles":
            x_phys = (float(c) * 0.5 + self.center_x_offset) * self.step_mm
            y_phys = (float(r) + 0.5 * float(c % 2) + self.center_y_offset) * self.step_mm

        return [x_phys, y_phys]

    def get_shape_contour(self, r, c):
        """
        Generates continuous polygon vertices for standard circular tracking blobs.
        """
        x_phys, y_phys = self.get_shape_center(r, c)
        
        circle_perimeter = 2.0 * np.pi * self.r_circ
        num_circle_pts = max(8, int(round(circle_perimeter * self.circle_points_per_mm)))

        angles = np.linspace(0, 2.0 * np.pi, num_circle_pts, endpoint=False)
        circle_poly = []
        for angle in angles:
            cx = x_phys + self.r_circ * np.cos(angle)
            cy = y_phys + self.r_circ * np.sin(angle)
            circle_poly.append((cx, cy))
        return circle_poly

    def save_to_svg(self, filename: str) -> None:
        """
        Generates and exports the explicitly generated point boundaries into
        an SVG file by consuming the instance's grid iterator directly.
        """
        import svgwrite
        width_mm = (self.grid_matrix.shape[1] // 2 + 2) * self.step_mm
        height_mm = (self.grid_matrix.shape[0] + 1) * self.step_mm

        dwg = svgwrite.Drawing(
            filename,
            size=(f"{width_mm}mm", f"{height_mm}mm"),
            viewBox=f"{-width_mm / 2.0} {-height_mm / 2.0} {width_mm} {height_mm}"
        )

        for i, j, shape_type, contour in self:
            if contour:
                dwg.add(dwg.polygon(points=contour, fill='black'))

        dwg.save()

# ==============================================================================
# SECTION 2: PYTHON 3.6 DETECTOR INTERFACE (Plugs into validation framework)
# ==============================================================================
class ContourDetector(cv2.Feature2D):
   def __init__(self, min_area=15, max_area=5000, edge_margin_px=2):
        super().__init__()
        self.min_area= min_area
        self.max_area=max_area
        self.edge_margin_px=edge_margin_px

   def detect(self, img, mask = None):
        print("Call detect")
        pts, labels, sizes = detect_and_classify_grid_nodes(img, self.min_area, self.max_area, self.edge_margin_px)
        keypoints = []
        for pt, id, size in zip(pts, labels, sizes):
            keypoints.append(cv2.KeyPoint(float(pt[0]), float(pt[1]), float(size),-1.0, 0.0, 0, int(id)))
        print("Call success")
        return np.array(keypoints)


def filter_grid_outliers(pts, labels, sizes, threshold_sigma=2.5):
    if len(pts) < 5:
        return pts, labels, sizes

    points_array = np.array(pts, dtype=np.float32)
    tree = KDTree(points_array)

    # 1. Query 5 nearest neighbors
    distances, _ = tree.query(points_array, k=5)

    # 2. Compute mean distance to 4 real neighbors for each point
    local_distances = np.mean(distances[:, 1:5], axis=1).astype(np.float32)

    # 3. USE OPENCV TO CALCULATE MEAN AND STD DEV AT ONCE
    # cv2.meanStdDev expects a structured NumPy array
    global_mean_arr, global_std_arr = cv2.meanStdDev(local_distances)

    # Extract scalar values from OpenCV outputs
    global_mean = global_mean_arr[0][0]
    global_std = global_std_arr[0][0]

    # 4. Z-score filtering (immune to missing or extra target point count)
    z_scores = (local_distances - global_mean) / (global_std + 1e-6)
    valid_mask = z_scores < threshold_sigma

    # Recompile clean tracking arrays
    pts_filtered = [pts[i] for i, valid in enumerate(valid_mask) if valid]
    labels_filtered = [labels[i] for i, valid in enumerate(valid_mask) if valid]
    sizes_filtered = [sizes[i] for i, valid in enumerate(valid_mask) if valid]

    purged_count = len(pts) - len(pts_filtered)
    if purged_count > 0:
        print(f"[Stat Filter] OpenCV stats purged {purged_count} isolated noise points.")

    return pts_filtered, labels_filtered, sizes_filtered


class OpenCVGridDetector:
    """
    OpenCV-based standard calibration target grid detector.
    Supports either standard Chessboard layouts or Symmetric/Asymmetric Circle Grids.
    """

    def __init__(self, grid_rows, grid_cols, pattern_type="CHESSBOARD"):
        # OpenCV expects pattern dimensions passed as (columns, rows)
        self.grid_size = (grid_cols, grid_rows)
        self.pattern_type = pattern_type.upper()

    def register_pattern(self, img, debug_overlay=True):
        """
        Executes native OpenCV grid registration routines.
        Returns:
            dict: Consistent data payload mapping detected node topologies.
        """
        from detector import visualize_detections
        result = {
            "points": [],
            "labels": [],
            "matches": []
        }

        width, height = self.grid_size
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # === Configure Blob Detector ===

        # 1. Get coordinates and sizes from your custom detector
        from detector import detect_and_classify_grid_nodes
        pts, labels, sizes = detect_and_classify_grid_nodes(gray, min_area=15, max_area=500, edge_margin_px=2)
        print(f"Custom detector found {len(pts)} points. Generating ideal map...")
        pts, labels, sizes = filter_grid_outliers(pts, labels, sizes)

        # 2. Create an ideally white canvas matching the original image dimensions
        # Using a white background ensures compatibility with default blobColor=0
        canvas = np.full_like(gray, fill_value=255)
        # 3. Draw clean black circles strictly centered at your detected points
        for pt,size in zip(pts, sizes):
            center_x = int(round(pt[0]))
            center_y = int(round(pt[1]))
            # Draw an anti-aliased, filled black circle
            cv2.circle(canvas, (center_x, center_y), radius=8, color=0, thickness=-1, lineType=cv2.LINE_AA)
        cv2.imshow("Win", canvas)
        cv2.waitKey()
        # 4. Configure the native OpenCV blob detector for these generated circles
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 500
        params.filterByColor = True
        params.blobColor = 0  # Look for black blobs on a white background
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        # 1. Primary Feature Detection
        found = False
        corners = None

        if self.pattern_type == "CHESSBOARD":
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, self.grid_size, flags=flags)
            if found and corners is not None:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        elif self.pattern_type == "CIRCLES" or self.pattern_type == "SYMMETRIC_CIRCLES":
            flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
            found, corners = cv2.findCirclesGrid(canvas, self.grid_size, flags=flags, blobDetector=detector)

        elif self.pattern_type == "ASYMMETRIC_CIRCLES":
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
            found, corners = cv2.findCirclesGrid(canvas, (height, width), flags=flags, blobDetector=detector)

        # 2. Early exit if spatial configuration detection failed
        if not found or corners is None:
            return result

        # 3. Payload Extraction and Structuring
        # Flatten OpenCV output shape from (N, 1, 2) to standard (N, 2) array coordinates
        points_flattened = corners.reshape(-1, 2)
        result["points"] = points_flattened.tolist()

        # Standard sequential integer labels matching linear grid order index
        labels_generated = list(range(len(points_flattened)))
        result["labels"] = labels_generated

        # Optional debugging canvas projection step
        if debug_overlay:
            # We copy the source array to prevent mutating shared memory pipelines
            debug_canvas = img.copy()
            cv2.drawChessboardCorners(debug_canvas, self.grid_size, corners, found)
            # Invoke native call to your custom visualizer framework if required
            if 'visualize_detections' in globals():
                visualize_detections(img, points_flattened, labels_generated)

        # 4. Topological Layout Matrix Alignment
        # Instantiating an empty -1 base tracking layout grid mapping (height x width)
        topological_matrix = np.full((height, width), -1, dtype=np.int32)

        # OpenCV populates grids linearly: Row by Row, from Left to Right
        idx = 0
        for r in range(height):
            for c in range(width):
                if idx < len(labels_generated):
                    topological_matrix[r, c] = labels_generated[idx]
                    idx += 1

        # Replicating matching structures for downstream validation evaluators
        mock_match_metadata = {
            "origin_index": 0,
            "grid_width": width,
            "grid_height": height,
            "status": "OPENCV_NATIVE_RESOLVED"
        }

        result["matches"] = [mock_match_metadata]
        result["topological_matrix"] = topological_matrix

        return result


class OpenCVMultiFrameCalibrator:
    """OpenCV calibration wrapper matching the MultiFrameCalibrator interface."""

    def __init__(
        self,
        camera_object: Any,
        geometry_generator: Any,
        fix_k3: bool = True,
        fix_principal_point: bool = False,
        use_intrinsic_guess: bool = True,
    ) -> None:
        """Initialize OpenCV calibrator wrapper.

        Args:
            camera_object: Initial ProjectiveCamera instance providing image size and K guess.
            geometry_generator: Object implementing get_shape_center(r, c) for 3D mapping.
            fix_k3: Whether to lock k3 coefficient to zero during optimization.
            fix_principal_point: Whether to lock principal point (cx, cy) to initial guess.
            use_intrinsic_guess: Whether to initialize solver from camera_object intrinsics.
        """
        self.camera = camera_object
        self.generator = geometry_generator
        self.fix_k3 = fix_k3
        self.fix_principal_point = fix_principal_point
        self.use_intrinsic_guess = use_intrinsic_guess

        self.object_points_views: List[np.ndarray] = []
        self.image_points_views: List[np.ndarray] = []
        self.init_rvecs: List[np.ndarray] = []
        self.init_tvecs: List[np.ndarray] = []

    def add_frame(
        self,
        topological_matrix: np.ndarray,
        detected_points: np.ndarray,
        point_weights: Optional[np.ndarray] = None,
        initial_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Register a single frame correspondences from topological grid matrix.

        Args:
            topological_matrix: 2D array of shape (H_nodes, W_nodes) mapping grid (r, c)
                                to index in detected_points (< 0 indicates missing node).
            detected_points: 2D array of shape (N, 2) with detected pixel coordinates.
            point_weights: Unused in OpenCV backend; kept for strict interface parity.
            initial_pose: Optional (R, t) tuple to initialize extrinsics for this view.
        """
        h_nodes, w_nodes = topological_matrix.shape
        obj_pts: List[List[float]] = []
        img_pts: List[List[float]] = []

        for r in range(h_nodes):
            for c in range(w_nodes):
                pt_idx = int(topological_matrix[r, c])
                if pt_idx < 0:
                    continue

                x,y = self.generator.get_shape_center(r, c)
                obj_pts.append([x, y, 0])
                img_pts.append(detected_points[pt_idx])

        if len(obj_pts) < 4:
            return

        self.object_points_views.append(np.array(obj_pts, dtype=np.float32))
        self.image_points_views.append(np.array(img_pts, dtype=np.float32))

        if initial_pose is not None:
            R_init, t_init = initial_pose
            rvec, _ = cv2.Rodrigues(np.asarray(R_init, dtype=np.float32))
            self.init_rvecs.append(rvec.astype(np.float32))
            self.init_tvecs.append(np.asarray(t_init, dtype=np.float32).reshape(3, 1))

    def calibrate(self, max_valid_rms: float = 50.0) -> Dict[str, Any]:
        """Run OpenCV Levenberg-Marquardt optimization with convergence safeguards.

        Args:
            max_valid_rms: Threshold above which calibration is considered diverged.

        Returns:
            Dictionary formatted for camera_io.deserialize_camera_from_dict.
        """
        if len(self.object_points_views) < 2:
            return {"status": "failed", "reason": "Insufficient frames for OpenCV calibration"}

        img_size = (int(self.camera.width), int(self.camera.height))

        try:
            if self.use_intrinsic_guess:
                k_matrix = np.array(
                    [
                        [float(self.camera.fx_px), 0.0, float(self.camera.cx)],
                        [0.0, float(self.camera.fy_px), float(self.camera.cy)],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist_coeffs = np.zeros(5, dtype=np.float32)
            else:
                k_matrix = cv2.initCameraMatrix2D(
                    self.object_points_views, self.image_points_views, img_size, aspectRatio=1.0
                )
                dist_coeffs = np.zeros(5, dtype=np.float32)

            flags = 0
            if self.use_intrinsic_guess:
                flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            if self.fix_k3:
                flags |= cv2.CALIB_FIX_K3
            if self.fix_principal_point:
                flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-12)

            has_all_poses = len(self.init_rvecs) == len(self.object_points_views)
            rvecs_arg = self.init_rvecs if has_all_poses else None
            tvecs_arg = self.init_tvecs if has_all_poses else None

            rms, k_est, dist_est, _, _ = cv2.calibrateCamera(
                objectPoints=self.object_points_views,
                imagePoints=self.image_points_views,
                imageSize=img_size,
                cameraMatrix=k_matrix,
                distCoeffs=dist_coeffs,
                rvecs=rvecs_arg,
                tvecs=tvecs_arg,
                flags=flags,
                criteria=criteria,
            )
        except cv2.error as exc:
            return {"status": "failed", "reason": f"OpenCV solver exception: {str(exc)}"}

        fx_val = float(k_est[0,0])
        fy_val = float(k_est[1,1])

        # Check physical validity and convergence threshold
        if not np.isfinite(rms) or rms > max_valid_rms or fx_val <= 0.0 or fy_val <= 0.0:
            return {
                "status": "failed",
                "reason": f"OpenCV diverged (rms={rms:.2f}, fx={fx_val:.1f}, fy={fy_val:.1f})",
            }

        d_flat = dist_est.flatten()
        return {
            "status": "success",
            "width": img_size[0],
            "height": img_size[1],
            "fx": fx_val,
            "fy": fy_val,
            "cx": float(k_est[0,2]),
            "cy": float(k_est[1,2]),
            "rms": float(rms),
            "distortion_model": "BrownConradyFull",
            "k1": float(d_flat[0]),
            "k2": float(d_flat[1]),
            "p1": float(d_flat[2]),
            "p2": float(d_flat[3])
        }
