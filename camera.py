import numpy as np
import cv2

def apply_radial_distortion(x_px, y_px, cx_px, cy_px, fx_px, fy_px,  k1):
    """
    Applies the mathematical Brown-Conrady radial distortion model (K1 only)
    to a single 2D pixel coordinate point.
    """
    # 1. Move to normalized camera coordinates space (x, y)
    x_norm = (x_px - cx_px) / fx_px
    y_norm = (y_px - cy_px) / fy_px

    # Calculate square of radius from principal axis point center
    r2 = (x_norm ** 2) + (y_norm ** 2)

    # 2. Compute distortion scaling factor
    distortion_multiplier = 1.0 + k1 * r2

    # 3. Map back to absolute canvas screen pixels
    x_distorted = (x_norm * distortion_multiplier * fx_px) + cx_px
    y_distorted = (y_norm * distortion_multiplier * fy_px) + cy_px

    return x_distorted, y_distorted


class Distortion:
    def __init__(self, cx_px, cy_px, fx_px, fy_px, k1):
        self.cx = cx_px
        self.cy = cy_px
        self.fx = fx_px
        self.fy = fy_px
        self.k1 = k1

    def __call__(self, point):
        is_point = isinstance(point, tuple)
        if is_point:
            return apply_radial_distortion(point[0], point[1], self.cx, self.cy, self.fx, self.fy , self.k1)
        else:
            return [apply_radial_distortion(x_px, y_px, self.cx, self.cy, self.fx, self.fy, self.k1) for x_px, y_px in point]


class ReverseDistortion:
    def __init__(self, cx_px, cy_px, fx_px, fy_px, k1, aspect=1.0, step=0.001):
        """
        Initializes a 1D Look-Up Table for computing FORWARD mapping (r_ideal -> r_distorted).
        Symmetrical derivative limits used for both barrel and pincushion bounds.
        """
        self.cx = float(cx_px)
        self.cy = float(cy_px)
        self.fx = float(fx_px)
        self.fy = float(fy_px)
        self.aspect = float(aspect)  # New parameter for distance correction
        self.k1 = float(k1)
        self.step = float(step)

        # 1. Find the critical distorted radius using step-relative derivative constraint
        if np.abs(self.k1) > 1.e-8:
            # Symmetrical threshold: max_r = sqrt(1 / (3 * |k1|))
            self.max_r_distorted = np.sqrt(1.0 / (3.0 * abs(self.k1)))
        else:
            self.max_r_distorted = 3.0

        self.num_elements = int(np.ceil(self.max_r_distorted / self.step)) + 1
        r_distorted_grid = np.arange(self.num_elements, dtype=np.float32) * self.step

        self.lut_r_ideal = r_distorted_grid * (1.0 + self.k1 * (r_distorted_grid ** 2))
        self.max_r_ideal = self.lut_r_ideal[-1]

    def _process_single_point(self, x_px, y_px):
        """Internal function to process a single (x, y) pixel coordinate."""
        # 1. Convert incoming pixel coordinates to normalized camera space
        x_norm = (float(x_px) - self.cx) / self.fx
        y_norm = (float(y_px) - self.cy) / self.fy

        # 2. Compute the current ideal radius of the incoming point
        r_ideal = np.hypot(x_norm * self.aspect, y_norm)

        # Clip incoming radius to the maximum valid range of our LUT table
        r_ideal_clipped = max(0.0, min(r_ideal, self.max_r_ideal))

        # 3. Find the left neighbor index in the ordered lut_r_ideal array
        idx = np.searchsorted(self.lut_r_ideal, r_ideal_clipped, side='right')
        idx = min(idx, self.num_elements - 2)
        # 4. Extract boundary ideal radii for interpolation
        r_ideal_low = self.lut_r_ideal[idx]
        r_ideal_high = self.lut_r_ideal[idx + 1]

        # 5. Compute the fractional interpolation weight 't' between the grid cells
        denom = r_ideal_high - r_ideal_low
        t = 0.0
        if denom > 1e-8:
            t = (r_ideal_clipped - r_ideal_low) / denom

        # 6. Inverse interpolation: convert index and weight back to the distorted radius
        r_distorted = max(0, (idx + t) * self.step)

        # 7. Calculate the ray scaling factor: scale = r_distorted / r_ideal
        scale = 1.0
        if r_ideal > 1e-8:
            scale = r_distorted / r_ideal

        # 8. Apply the scaling factor to shift normalized coordinates along the radial vector
        x_norm_distorted = x_norm * scale
        y_norm_distorted = y_norm * scale

        # 9. Denormalize distorted coordinates back to pixel space
        x_px_distorted = x_norm_distorted * self.fx + self.cx
        y_px_distorted = y_norm_distorted * self.fy + self.cy

        return x_px_distorted, y_px_distorted

    def __call__(self, point):
        """
        Accepts either a single pixel tuple (x, y) OR a list of pixel tuples.
        Processes points using a clean, straightforward loop approach.
        """
        if isinstance(point, tuple):
            return self._process_single_point(point[0], point[1])
        else:
            return [self._process_single_point(x, y) for x, y in point]


# ==========================================
# FULL RESTORED UNIT TESTS
# ==========================================

def test_center_point_stability():
    """The center point (cx, cy) should remain invariant under transformation."""
    cx, cy = 960.0, 540.0
    model = ReverseDistortion(cx_px=cx, cy_px=cy, fx_px=1200, fy_px=1200, k1=-0.2, step=0.0001)
    x_c, y_c = model((cx, cy))
    assert abs(x_c - cx) < 1e-5 and abs(y_c - cy) < 1e-5, "Center shifted: {}, {}".format(x_c, y_c)
    print("-> test_center_point_stability passed.")


def test_directional_consistency_barrel():
    """For k1 < 0 (barrel), the physical distance from center must expand in forward mapping."""
    cx, cy = 960.0, 540.0
    model = ReverseDistortion(cx_px=cx, cy_px=cy, fx_px=1200, fy_px=1200, k1=-0.2, step=0.0001)

    x_ideal_px = cx + 200.0
    y_ideal_px = cy
    x_dist_px, y_dist_px = model((x_ideal_px, y_ideal_px))

    assert x_dist_px > x_ideal_px, "Expected x_dist_px ({}) > x_ideal_px ({}) for barrel".format(x_dist_px, x_ideal_px)
    assert abs(y_dist_px - cy) < 1e-5, "Alignment error on the primary axis line"
    print("-> test_directional_consistency_barrel passed.")


def test_mathematical_inversion_accuracy():
    """Checks the accuracy of the loop-based LUT inverse lookup against analytical expectations."""
    cx, cy = 960.0, 540.0
    fx, fy = 1200.0, 1200.0
    k1_barrel = -0.2
    model = ReverseDistortion(cx_px=cx, cy_px=cy, fx_px=fx, fy_px=fy, k1=k1_barrel, step=0.0001)

    r_dist_expected = 0.4
    r_ideal_input = r_dist_expected * (1.0 + k1_barrel * (r_dist_expected ** 2))

    x_pixel_ideal_input = r_ideal_input * fx + cx
    x_test_dist_px, _ = model((x_pixel_ideal_input, cy))
    x_pixel_dist_expected = r_dist_expected * fx + cx

    tolerance = 1.0  # Pixel-level accuracy tolerance threshold
    assert abs(x_test_dist_px - x_pixel_dist_expected) < tolerance, "Inversion failed: expected {}, got {}".format(
        x_pixel_dist_expected, x_test_dist_px)
    print("-> test_mathematical_inversion_accuracy passed.")


def test_numpy_batch_processing():
    """Ensures that mapping lists of tuples works seamlessly and equals scalar results."""
    cx, cy = 960.0, 540.0
    model = ReverseDistortion(cx_px=cx, cy_px=cy, fx_px=1200, fy_px=1200, k1=-0.2, step=0.0001)

    pixel_list = [
        (cx, cy),
        (cx + 200.0, cy + 200.0)
    ]
    output_list = model(pixel_list)

    assert isinstance(output_list, list), "Expected list output format"
    assert len(output_list) == 2, "List size altered during evaluation"

    single_ref = model((cx + 200.0, cy + 200.0))
    assert abs(output_list[1][0] - single_ref[0]) < 1e-5, "Batch discrepancy found on X"
    assert abs(output_list[1][1] - single_ref[1]) < 1e-5, "Batch discrepancy found on Y"
    print("-> test_numpy_batch_processing passed.")


def test_distortion_camera():
    """Validates distortion processing on an asymmetric or alternative camera sensor profile setup."""
    cx, cy = 640.0, 360.0
    fx, fy = 800.0, 600.0  # Different focal lengths (non-square pixel or sensor scaling simulation)
    model = ReverseDistortion(cx_px=cx, cy_px=cy, fx_px=fx, fy_px=fy, k1=0.15,
                              step=0.0001)  # Pincushion distortion test

    x_in, y_in = cx + 100.0, cy + 100.0
    x_out, y_out = model((x_in, y_in))

    # For k1 > 0 (pincushion), coordinates pull inward closer to the center point
    assert x_out < x_in, "Pincushion must reduce coordinates relative to principal axis on forward map"
    assert y_out < y_in, "Pincushion must reduce coordinates relative to principal axis on forward map"
    print("-> test_distortion_camera passed.")


class BlenderDistortion(ReverseDistortion):
    def __init__(self, width, height, k1, step=0.001):
        """
        Subclass designed to wrap ReverseDistortion and dynamically enforce
        Blender's lens distortion frame aspect metrics inside __call__.
        Compatible with Python 3.6+ and strictly ASCII-clean.
        """
        # Blender maps primary screen space relative to direct half-frame metrics
        cx_px = self.img_width / 2.0
        cy_px = self.img_height / 2.0
        fx_px = self.img_width / 2.0
        fy_px = self.img_height / 2.0

        # Initialize the base class with standard non-aspect focal configurations
        super(BlenderDistortion, self).__init__(
            cx_px=cx_px,
            cy_px=cy_px,
            fx_px=fx_px,
            fy_px=fy_px,
            k1=k1,
            aspect = self.img_width / self.img_height,
            step=step
        )

    def call(self, point):
        """
        Overrides the call sequence to pre-process coordinates before passing them
        to the superclass, and post-process them right after.
        """
        if isinstance(point, tuple):
            x, y = point
            # Pre-process: scale X relative to the center to force proper radial calculation
            x_mod = (x - self.cx) * self.img_aspect + self.cx

            # Execute base class single point sequence directly
            x_base_out, y_base_out = super(BlenderDistortion, self)._process_single_point(x_mod, y)

            # Post-process: revert the scaling multiplier on the transformed X coordinate
            x_final = (x_base_out - self.cx) / self.img_aspect + self.cx
            return (x_final, y_base_out)
        else:
            # Process lists of tuples using the exact same isolated logic path
            output_list = []
            for x, y in point:
                x_mod = (x - self.cx) * self.img_aspect + self.cx
                x_base_out, y_base_out = super(BlenderDistortion, self)._process_single_point(x_mod, y)
                x_final = (x_base_out - self.cx) / self.img_aspect + self.cx
                output_list.append((x_final, y_base_out))
            return output_list


class BlenderDistortion__:
    def __init__(self, width, height, k1, step=0.0001):
        """
        1D Look-Up Table matching Blender's CompositorNodeLensdist architecture.
        Solves the exact inverse cubic mapping (r_ideal -> r_distorted).
        Compatible with Python 3.6+ and strictly ASCII-clean.
        """
        self.width = float(width)
        self.height = float(height)
        self.k1 = float(k1)
        self.step = float(step)

        # Center coordinates matching Blender frame tracking
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.aspect = self.width / self.height

        # 1. Calculate boundaries using symmetrical derivative criteria
        if np.abs(self.k1) > 1.e-8:
            self.max_r_distorted = np.sqrt(1.0 / (3.0 * abs(self.k1)))
        else:
            self.max_r_distorted = 3.0

        # 2. Build the exact matching uniform / non-uniform lookup arrays
        self.num_elements = int(np.ceil(self.max_r_distorted / self.step)) + 1
        r_distorted_grid = np.arange(self.num_elements, dtype=np.float32) * self.step

        # Blender forward core representation: r_ideal = r_dist * (1 + k1 * r_dist^2)
        self.lut_r_ideal = r_distorted_grid * (1.0 + self.k1 * (r_distorted_grid ** 2))
        self.max_r_ideal = self.lut_r_ideal[-1]

    def _process_single_point(self, x_px, y_px):
        # 1. Convert pixel positions to Blender normalized window coordinates [-1.0, 1.0]
        x_norm = (float(x_px) - self.cx) / (self.width / 2.0)
        y_norm = (float(y_px) - self.cy) / (self.width / 2.0)

        # 2. Blender aspect adjustment happens strictly inside radius calculation
        r_ideal = np.sqrt((x_norm * self.aspect) ** 2 + y_norm ** 2)
        r_ideal_clipped = max(0.0, min(r_ideal, self.max_r_ideal))

        # 3. Perform accurate binary search in the ordered non-uniform grid
        idx = int(np.searchsorted(self.lut_r_ideal, r_ideal_clipped)) - 1
        idx = max(0, min(idx, self.num_elements - 2))

        r_ideal_low = self.lut_r_ideal[idx]
        r_ideal_high = self.lut_r_ideal[idx + 1]

        denom = r_ideal_high - r_ideal_low
        t = 0.0
        if denom > 1e-8:
            t = (r_ideal_clipped - r_ideal_low) / denom

        # Calculate true distorted radius value
        r_distorted = (idx + t) * self.step

        # 4. Ray shift scaling factor calculation
        scale = 1.0
        if r_ideal > 1e-8:
            scale = r_distorted / r_ideal

        # 5. Shift original normalized coordinates isotropically
        x_norm_distorted = x_norm * scale
        y_norm_distorted = y_norm * scale

        # 6. Re-project coordinates back into standard pixel frame bounds
        x_px_distorted = x_norm_distorted * (self.width / 2.0) + self.cx
        y_px_distorted = y_norm_distorted * (self.height / 2.0) + self.cy

        return (x_px_distorted, y_px_distorted)

    def __call__(self, point):
        """
        Accepts either a single point tuple (x, y) or a list of tuples.
        Processes points through a clean, standard loop setup.
        """
        if isinstance(point, tuple):
            return self._process_single_point(point[0], point[1])
        else:
            return [self._process_single_point(x, y) for x, y in point]


def compute_max_radius(distortion_model, cx, cy, img_shape=(1080, 1920)):
    """
    Calculates a strict global critical radius by scanning a dense radial array.
    Returns the exact radius right before the polynomial distortion inverts
    and starts pulling points back toward the center.
    """
    H_img, W_img = img_shape
    if distortion_model is None:
        return np.hypot(H_img, W_img) / 2

    # Calculate maximum possible radius to the image corners
    corners = np.array([[0, 0], [W_img, 0], [W_img, H_img], [0, H_img]], dtype=np.float32)
    max_frame_radius = np.ceil(np.max(np.hypot(corners[:, 0] - cx, corners[:, 1] - cy)))
    # Extend lookup slightly beyond corners to catch shapes entering the frame boundary
    extended_search_radius = max_frame_radius * 1.5
    # Create a dense radial sampling row from 0 up to the extended radius bound
    sample_radii = np.arange(0, extended_search_radius, 1.0, dtype=np.float32)
    # Project test points along a horizontal ray running outwards from the center
    test_pts = np.zeros((len(sample_radii), 2), dtype=np.float32)
    test_pts[:, 0] = cx + sample_radii
    test_pts[:, 1] = cy
    # Pass the dense evaluation line through the distortion profile
    distorted_test_pts = np.asarray(distortion_model(test_pts))
    # Measure distorted radial distances from center
    distorted_radii = np.sqrt((distorted_test_pts[:, 0] - cx) ** 2 + (distorted_test_pts[:, 1] - cy) ** 2)
    # Locate the exact index where a further point turns back and gets closer
    max_dist_idx = np.argmax(distorted_radii)

    return sample_radii[max_dist_idx]


from scipy.interpolate import RegularGridInterpolator


class ProjectiveCamera:
    """
    Manages camera intrinsic calibration, radial lens distortion pipelines,
    and spatial coordinates projection mapping using rigid PnP conventions.
    """

    def __init__(self,
                 img_shape: (int,int),
                 fx_px: float,
                 fy_px: float,
                 cx: float,
                 cy: float,
                 k1: float = 0.0,
                 mode = "perspective",
                 distortion_model = None):
        """
        Args:
            w_img (int): Total physical sensor pixel width (e.g., 1920).
            h_img (int): Total physical sensor pixel height (e.g., 1080).
            fx_px (float): Focal length in pixels.
            fy_px (float): Focal length in pixels.
            cx (float): Principal point X-coordinate (typically width / 2).
            cy (float): Principal point Y-coordinate (typically height / 2).
            k1 (float): Primary radial lens distortion coefficient.
        """
        self.W_img = img_shape[0]
        self.H_img = img_shape[1]
        self.img_shape = img_shape
        self.k1 = k1
        self.cx = float(cx)
        self.cy = float(cy)
        self.fx_px = fx_px
        self.fy_px = fy_px
        self.mode = mode
        # 1. Clean Constructor Initialization of the Intrinsic Camera Matrix (K)
        self.K = np.array([
            [self.fx_px, 0.0, self.cx],
            [0.0, self.fy_px, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        if mode == "affine":
            self.K[2,2] = 0

        # 2. Instantiate the Projective Lens Distortion Lambda Function
        self.distortion_model = None
        if abs(k1) > 1.e-3:
            if distortion_model == "reverse":
                self.distortion_model = ReverseDistortion(cx, cy, fx_px, fy_px, k1)
            elif distortion_model == "blender":
                self.distortion_model = BlenderDistortion(img_shape[0], img_shape[1], k1)
            else:
                self.distortion_model = Distortion(cx, cy, fx_px, fy_px, k1)

        # 3. Deterministic Structural Boundary Profiler
        # Always computed automatically based on the distortion coefficients
        self.max_stable_radius = compute_max_radius(self.distortion_model, self.cx, self.cy, img_shape=self.img_shape)

    def undistort(self, point):
        """
        :param point:
        :return:
        """
        if self.distortion_model is None:
            return point
        MAX_ITER = 30
        a = point
        ca = (a[0] - self.cx, a[1] - self.cy)
        b = (ca[0] * 2 + self.cx, ca[1] * 2 + self.cy)
        r_d = np.hypot(ca[0], ca[1])
        r_prev = 2 * r_d
        p = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        for k in range(MAX_ITER):
            p_d = self.distortion_model(p)
            r = np.hypot(p_d[0]-self.cx, p_d[1]-self.cy)
            if r < r_d:
                a = p
            else:
                b = p
            if np.abs(r - r_prev) < 0.5:
                break
            r_prev = r
            p = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        return p

    def undistort_points(self, points):
        return [self.undistort(point) for point in points]

    def compute_homography(self, R,t: np.ndarray) -> np.ndarray:
        """Computes a flat 3x3 planar Homography matrix from 3x4 extrinsics."""
        r0 = R[:, 0:1]
        r1 = R[:, 1:2]
        H_ext = np.hstack([r0, r1, t])
        return self.K @ H_ext

    def is_visible(self, point):
        """
        :param point: (x,y) pair
        :return: visible on the camera image
        """
        return (0 <= point[0] < self.W_img and 0 <= point[1] < self.H_img)

    def project_point(self, world_point: np.ndarray, rotation: np.ndarray, t: np.ndarray) -> np.ndarray:
        local_point = rotation @ world_point
        local_point[0] += t[0]
        local_point[1] += t[1]
        local_point[2] += t[2]
        local_point = self.K @ local_point
        x, y = local_point[0] /local_point[2], local_point[1]/local_point[2]
        r = np.hypot(x  - self.cx, y  - self.cy)
        if r >= self.max_stable_radius:
            return None

        if self.distortion_model:
            return self.distortion_model((x,y))

        return x,y

    def project_points(self, world_pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Projects 2D world space points into the distorted camera pixel plane."""
        pts = np.atleast_2d(np.array(world_pts, dtype=np.float32))
        num_pts = len(pts)

        h = self.compute_homography(R,t)
        homogeneous_world_pts = np.hstack([pts, np.ones((num_pts, 1), dtype=np.float32)])
        projected_pts = (h @ homogeneous_world_pts.T).T
        pixel_pts = projected_pts[:, :2] / projected_pts[:, 2:3]

        dx = pixel_pts[:, 0] - self.cx
        dy = pixel_pts[:, 1] - self.cy
        ideal_radii = np.hypot(dx, dy)

        # Guardrails strictly intercept runaway points before distortion loops execute
        if np.any(ideal_radii >= self.max_stable_radius):
            return None

        if self.distortion_model is not None:
            pixel_pts = self.distortion_model(pixel_pts)

        return pixel_pts


def compute_camera_projection_matrix(roll_deg, pitch_deg, yaw_deg, tx=0.0, ty=0.0, tz=1.0):
    """
    Computes the 3x4 Camera Projection Matrix P = K * [R | t] by combining
    intrinsic parameters and extrinsic 3D rotations/translations.
    
    Parameters:
        roll_deg (float): Rotation around the camera's Z-axis (forward).
        pitch_deg (float): Rotation around the camera's X-axis (sideways).
        yaw_deg (float): Rotation around the camera's Y-axis (up/down).
        tx, ty, tz (float): Camera translation relative to the world origin.
                            tz acts as the distance to the grid plane.
                            
    Returns:
        Rt (np.ndarray): 3x4 Extrinsic Matrix.
    """

    # 1. Convert angles to radians
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    
    # 2. Compute directional rotation matrices (Euler angles)
    R_x = np.array([
        [1.0,    0.0,     0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch),  np.cos(pitch)]
    ], dtype=np.float64)
    
    R_y = np.array([
        [ np.cos(yaw), 0.0, np.sin(yaw)],
        [       0.0, 1.0,       0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)]
    ], dtype=np.float64)
    
    R_z = np.array([
        [np.cos(roll), -np.sin(roll), 0.0],
        [np.sin(roll),  np.cos(roll), 0.0],
        [      0.0,        0.0, 1.0]
    ], dtype=np.float64)

    # Combine rotations YXZ  Cam to World
    R = R_y @ R_x @ R_z
    
    # 4. Construct Translation Vector t
    t = np.array([[tx], [ty], [tz]], dtype=np.float64)
    
    # 5. Assemble Extrinsic Matrix [R | t] (size 3x4)
    return R.T, -(R.T @ t)


def distort_image_via_undistort_grid(src_img, camera_inst: ProjectiveCamera):
    """
    Applies lens distortion simulation to an undistorted input image.
    Uses pure NumPy indexing and fast vectorized mask multiplication.
    No OpenCV required.
    """
    w, h = camera_inst.img_shape

    has_channels = len(src_img.shape) > 2
    src_h, src_w = src_img.shape[:2]
    # 1. Define a sparse regular grid
    grid_spacing = 32
    x_ticks = np.arange(0, w, grid_spacing)
    y_ticks = np.arange(0, h, grid_spacing)

    if x_ticks[-1] < w - 1:
        x_ticks = np.append(x_ticks, w - 1)
    if y_ticks[-1] < h - 1:
        y_ticks = np.append(y_ticks, h - 1)

    num_x = len(x_ticks)
    num_y = len(y_ticks)

    # 2. Allocate grid maps and the visibility mask
    grid_map_x = np.empty((num_y, num_x), dtype=np.float32)
    grid_map_y = np.empty((num_y, num_x), dtype=np.float32)

    offset_x = (src_w - w) / 2.0
    offset_y = (src_h - h) / 2.0

    # 3. Query camera object
    for i, y_dist in enumerate(y_ticks):
        for j, x_dist in enumerate(x_ticks):
            x_src, y_src = camera_inst.undistort((x_dist, y_dist))

            grid_map_x[i, j] = x_src + offset_x
            grid_map_y[i, j] = y_src + offset_y

    # 4. Upsample both maps and the mask
    interp_x = RegularGridInterpolator((y_ticks, x_ticks), grid_map_x, method='linear', bounds_error=False,
                                       fill_value=None)
    interp_y = RegularGridInterpolator((y_ticks, x_ticks), grid_map_y, method='linear', bounds_error=False,
                                       fill_value=None)

    # Generate dense execution grid coordinates
    dense_y = np.arange(h, dtype=np.float32)
    dense_x = np.arange(w, dtype=np.float32)
    dense_y_mesh, dense_x_mesh = np.meshgrid(dense_y, dense_x, indexing='ij')
    query_points = np.stack([dense_y_mesh.ravel(), dense_x_mesh.ravel()], axis=-1)

    # Evaluate the arrays and reshape back to full resolution
    map_x = interp_x(query_points).reshape((h, w))
    map_y = interp_y(query_points).reshape((h, w))

    dist_left = map_x
    dist_right = (src_w - 1) - map_x
    dist_top = map_y
    dist_bottom = (src_h - 1) - map_y

    min_dist_to_edge = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom))
    fade_width = 100
    smooth_mask = np.clip(min_dist_to_edge / fade_width, 0.0, 1.0)

    # Smoothstep
    smooth_mask = smooth_mask * smooth_mask * (3.0 - 2.0 * smooth_mask)

    # 5. Native NumPy Indexing Warp
    map_x_clipped = np.clip(np.round(map_x), 0, src_w - 1).astype(np.int32)
    map_y_clipped = np.clip(np.round(map_y), 0, src_h - 1).astype(np.int32)

    # Pull pixels natively from input image into their new positions
    distorted_img = src_img[map_y_clipped, map_x_clipped]

    # 6. Optimized Vectorized Masking (No per-pixel row looping)
    if has_channels:
        # Broadcast (H, W) mask to (H, W, 1) to multiply across all color planes simultaneously
        if src_img.shape[2] == 4:
            distorted_img[..., :3] = distorted_img[..., :3] * smooth_mask[..., None]
        else:
            distorted_img = distorted_img * smooth_mask[..., None]
    else:
        distorted_img = distorted_img * smooth_mask

    return distorted_img


def test_distortion_camera():
    camera = ProjectiveCamera((100,100), 50, 50, 50, 50, -0.1)
    assert camera.max_stable_radius > 40
    yaw, pitch, roll = 4, 5, 6
    R, t = compute_camera_projection_matrix(roll, pitch, yaw, 0, 0, -1)
    p = camera.project_point((1,2,0), R, t)
    R = np.eye(3,3)
    t = np.array([0,0,0])
    p_dist = camera.project_point((0.5,1,1), R, t)
    p = camera.undistort(p_dist)
    K = np.linalg.inv(camera.K)
    p = K @ np.array([p[0], p[1], 1])
    assert abs(p[0] - 0.5) < 0.05 and abs(p[1] - 1) < 0.05


def test_reverse_distortion_camera():
    camera = ProjectiveCamera((100,100), 50, 50, 50, 50, 0.1, distortion_model="reverse")
    assert camera.max_stable_radius > 40
    yaw, pitch, roll = 4, 5, 6
    R, t = compute_camera_projection_matrix(roll, pitch, yaw, 0, 0, -1000)
    camera.project_point((10,20,0), R, t)



if __name__ == "__main__":

    test_center_point_stability()
    test_directional_consistency_barrel()
    test_mathematical_inversion_accuracy()
    test_numpy_batch_processing()
    test_distortion_camera()
    test_reverse_distortion_camera()
