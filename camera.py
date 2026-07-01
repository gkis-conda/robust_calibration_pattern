import numpy as np

def apply_radial_distortion(x_px, y_px, cx_px, cy_px, f_px, k1):
    """
    Applies the mathematical Brown-Conrady radial distortion model (K1 only)
    to a single 2D pixel coordinate point.
    """
    # 1. Move to normalized camera coordinates space (x, y)
    x_norm = (x_px - cx_px) / f_px
    y_norm = (y_px - cy_px) / f_px

    # Calculate square of radius from principal axis point center
    r2 = (x_norm ** 2) + (y_norm ** 2)

    # 2. Compute distortion scaling factor
    distortion_multiplier = 1.0 + k1 * r2

    # 3. Map back to absolute canvas screen pixels
    x_distorted = (x_norm * distortion_multiplier * f_px) + cx_px
    y_distorted = (y_norm * distortion_multiplier * f_px) + cy_px

    return x_distorted, y_distorted


class Distortion:
    def __init__(self, cx_px, cy_px, f_px, k1):
        self.cx = cx_px
        self.cy = cy_px
        self.f = f_px
        self.k1 = k1

    def __call__(self, point):
        is_point = isinstance(point, tuple)
        if is_point:
            return apply_radial_distortion(point[0], point[1], self.cx, self.cy, self.f, self.k1)
        else:
            return [apply_radial_distortion(x_px, y_px, self.cx, self.cy, self.f, self.k1) for x_px, y_px in point]

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


class ProjectiveCamera:
    """
    Manages camera intrinsic calibration, radial lens distortion pipelines,
    and spatial coordinates projection mapping using rigid PnP conventions.
    """

    def __init__(self,
                 img_shape: (int,int),
                 f_px: float,
                 cx: float,
                 cy: float,
                 k1: float = -0.015,
                 mode = "perspective"):
        """
        Args:
            w_img (int): Total physical sensor pixel width (e.g., 1920).
            h_img (int): Total physical sensor pixel height (e.g., 1080).
            f_px (float): Focal length in pixels.
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
        self.f_px = f_px
        self.mode = mode
        # 1. Clean Constructor Initialization of the Intrinsic Camera Matrix (K)
        self.K = np.array([
            [f_px, 0.0, self.cx],
            [0.0, f_px, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        if mode == "affine":
            self.K[2,2] = 0

        # 2. Instantiate the Projective Lens Distortion Lambda Function
        self.distortion_model = Distortion(cx, cy, f_px, k1) if abs(k1) > 1.e-3 else None

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
        MAX_ITER = 20
        a = point
        b = point * 2
        r_d = np.hypot(a[0] - self.cx, a[1] - self.cy)
        r_prev = 2 * r_d
        for k in range(MAX_ITER):
            p = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            p_d = self.distortion_model(p)
            r = np.hypot(p_d[0]-self.cx, p_d[1]-self.cy)
            if r < r_d:
                a = p
            else:
                b = p
            if np.abs(r - r_prev) < 1:
                break
            r_prev = r
        return p


    def undistort_points(self, points):
        return [self.undistort(point) for point in points]

    def compute_homography(self, Rt: np.ndarray) -> np.ndarray:
        """Computes a flat 3x3 planar Homography matrix from 3x4 extrinsics."""
        r0 = Rt[:, 0:1]
        r1 = Rt[:, 1:2]
        t  = Rt[:, 3:4]
        H_ext = np.hstack([r0, r1, t])
        return self.K @ H_ext

    def is_visible(self, point):
        """
        :param point: (x,y) pair
        :return: visible on the camera image
        """
        return (0 <= point[0] < self.W_img and 0 <= point[1] < self.H_img)

    def project_points(self, world_pts: np.ndarray, Rt: np.ndarray) -> np.ndarray:
        """Projects 2D world space points into the distorted camera pixel plane."""
        pts = np.atleast_2d(np.array(world_pts, dtype=np.float32))
        num_pts = len(pts)

        H_global = self.compute_homography(Rt)
        homogeneous_world_pts = np.hstack([pts, np.ones((num_pts, 1), dtype=np.float32)])
        projected_pts = (H_global @ homogeneous_world_pts.T).T
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
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    
    # 2. Compute directional rotation matrices (Euler angles)
    R_x = np.array([
        [1.0,    0.0,     0.0],
        [0.0, np.cos(p), -np.sin(p)],
        [0.0, np.sin(p),  np.cos(p)]
    ], dtype=np.float64)
    
    R_y = np.array([
        [ np.cos(y), 0.0, np.sin(y)],
        [       0.0, 1.0,       0.0],
        [-np.sin(y), 0.0, np.cos(y)]
    ], dtype=np.float64)
    
    R_z = np.array([
        [np.cos(r), -np.sin(r), 0.0],
        [np.sin(r),  np.cos(r), 0.0],
        [      0.0,        0.0, 1.0]
    ], dtype=np.float64)

    # Combine rotations (Z * Y * X convention)
    R = R_z @ R_y @ R_x
    
    # 4. Construct Translation Vector t
    t = np.array([[tx], [ty], [tz]], dtype=np.float64)
    
    # 5. Assemble Extrinsic Matrix [R | t] (size 3x4)
    return np.hstack([R, t])

