import numpy as np
import cv2

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
from detector import detect_and_classify_grid_nodes
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


import numpy as np
from scipy.spatial import KDTree


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

