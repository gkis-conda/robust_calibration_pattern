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
    def __init__(self, grid_matrix, step_mm, r_circ, circle_points_per_mm=2.0):
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

        # Center calibration offsets: Center coordinate space relative to (0,0)
        self.center_x_offset = (1.0 - float(self.grid_matrix.shape[1])) / 2.0
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
        x_phys = ((float(c) + 0.5 * float(r % 2)) + self.center_x_offset) * self.step_mm
        y_phys = (float(r) * np.sqrt(3.0) / 2.0 + self.center_y_offset) * self.step_mm
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


# ==============================================================================
# SECTION 2: PYTHON 3.6 DETECTOR INTERFACE (Plugs into validation framework)
# ==============================================================================
class OpenCVCirclesGridDetector:
    """
    Interface compatible with MockTopologyDetector. Runs native OpenCV 
    C++ routines to automatically find and index circles grid patterns.
    """
    def __init__(self, grid_rows, grid_cols):
        # OpenCV grid patterns expect dimensions passed as (columns, rows)
        self.grid_size = (grid_cols, grid_rows)

    def register_pattern(self, image_bgr):
        """
        Executes standard OpenCV asymmetric circles grid registration.
        Returns:
            dict: {(row, col): [x_px, y_px]} containing indexed sub-pixel centers.
        """
        # 1. Standard preprocessing (OpenCV blob analysis executes over grayscale)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. Configure detection flags. 
        # CALIB_CB_ASYMMETRIC_GRID handles row-staggered hexagonal grids.
        # If your layout spacing is perfectly rectangular, use CALIB_CB_SYMMETRIC_GRID.
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID
        
        # Native C++ processing loop execution
        found, corners = cv2.findCirclesGrid(gray, self.grid_size, flags=flags)
        
        registered_topology = {}
        
        # The ultimate weakness of the baseline: if extreme perspective tilt (>70) 
        # or dynamic motion blur destroys even 1 or 2 dots, findCirclesGrid 
        # completely fails to reconstruct the graph mapping arrays and returns False.
        if not found or corners is None:
            return registered_topology
            
        # 3. Reconstruct topological map coordinates matrix from OpenCV sequence array.
        # OpenCV output arrays are linearly ordered row-by-row, column-by-column.
        idx = 0
        for r in range(self.grid_size[1]):      # rows loop
            for c in range(self.grid_size[0]):  # columns loop
                if idx < len(corners):
                    pt = corners[idx][0]        # Extract [X, Y] sub-pixel position
                    registered_topology[(r, c)] = [float(pt[0]), float(pt[1])]
                    idx += 1
                    
        return registered_topology
