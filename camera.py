import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ScaleDistortion:
    # Mock distortion profile
    name = "Scale"

    def __init__(self, cx, cy, scale):
        assert 0.1 < scale < 1. , "ScaleDistortion is a contractive function"
        self.scale = scale

        self.cx = cx
        self.cy = cy

    def __call__(self, point):
        pts = np.asanyarray(point, dtype=np.float32)
        center = np.array([self.cx, self.cy], dtype=np.float32)
        return (pts - center) * self.scale + center


def apply_radial_distortion(points, cx_px, cy_px, fx_px, fy_px, k1):
    """
    Applies the mathematical Brown-Conrady radial distortion model (K1 only)
    to a single 2D pixel coordinate point.
    """
    # Convert pixel coordinates to normalized camera plane
    x_norm = (points[:, 0] - cx_px) / fx_px
    y_norm = (points[:, 1] - cy_px) / fy_px

    # Use direct multiplication instead of power operator for speed
    x2 = x_norm * x_norm
    y2 = y_norm * y_norm
    r2 = x2 + y2

    # 2. Compute distortion scaling factor
    distortion_multiplier = 1.0 + k1 * r2

    # 3. Map back to absolute canvas screen pixels
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0] = cx_px + x_norm * (fx_px * distortion_multiplier)
    out[:, 1] = cy_px + y_norm * (fy_px * distortion_multiplier)

    return out


class Distortion:
    name = "BrownConrady"

    def __init__(self, cx_px, cy_px, fx_px, fy_px, k1):
        self.cx = cx_px
        self.cy = cy_px
        self.fx = fx_px
        self.fy = fy_px
        self.k1 = k1

    def __call__(self, point):
        is_point = isinstance(point, tuple)
        if is_point:
            return apply_radial_distortion(np.asarray([point]), self.cx, self.cy, self.fx, self.fy, self.k1)
        else:
            return apply_radial_distortion(np.asarray(point), self.cx, self.cy, self.fx, self.fy, self.k1)


def apply_brown_conrady_distortion( points: np.ndarray,
    cx_px: float, cy_px: float, fx_px: float, fy_px: float, k1: float, k2: float, p1: float,  p2: float) -> np.ndarray:
    """Apply full Brown-Conrady distortion model using vectorized NumPy operations.

    Args:
        points: Input 2D pixel coordinates array of shape (N, 2).
        cx_px: Principal point X coordinate in pixels.
        cy_px: Principal point Y coordinate in pixels.
        fx_px: Focal length X in pixels.
        fy_px: Focal length Y in pixels.
        k1: First radial distortion coefficient.
        k2: Second radial distortion coefficient.
        p1: First tangential distortion coefficient.
        p2: Second tangential distortion coefficient.

    Returns:
        Distorted 2D pixel coordinates array of shape (N, 2) in float64.
    """
    # Convert pixel coordinates to normalized camera plane
    x_norm = (points[:, 0] - cx_px) / fx_px
    y_norm = (points[:, 1] - cy_px) / fy_px

    # Use direct multiplication instead of power operator for speed
    x2 = x_norm * x_norm
    y2 = y_norm * y_norm
    r2 = x2 + y2
    r4 = r2 * r2

    # Compute radial distortion factor
    radial = 1.0 + k1 * r2 + k2 * r4

    # Compute tangential distortion components
    xy2 = 2.0 * x_norm * y_norm
    dx_tan = p1 * xy2 + p2 * (r2 + 2.0 * x2)
    dy_tan = p1 * (r2 + 2.0 * y2) + p2 * xy2

    # Apply radial and tangential distortions in normalized space
    x_dist_norm = x_norm * radial + dx_tan
    y_dist_norm = y_norm * radial + dy_tan

    # Map back to image sensor pixel coordinates
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0] = cx_px + fx_px * x_dist_norm
    out[:, 1] = cy_px + fy_px * y_dist_norm

    return out


class BrownConradyDistortion:
    """Brown-Conrady distortion model supporting single points and vectorized batches."""
    name: str = "BrownConradyFull"

    def __init__(
        self, cx_px: float, cy_px: float, fx_px: float, fy_px: float, k1: float,
        k2: float, p1: float = 0., p2: float = 0. ) -> None:
        self.cx = float(cx_px)
        self.cy = float(cy_px)
        self.fx = float(fx_px)
        self.fy = float(fy_px)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.p1 = float(p1)
        self.p2 = float(p2)

    def __call__(self, points) -> np.ndarray:
        """Apply distortion to a single 2D point or a batch of points (N, 2)."""
        pts_arr = np.asarray(points, dtype=np.float64)
        is_single_point = pts_arr.ndim == 1

        if is_single_point:
            if pts_arr.shape[0] != 2:
                raise ValueError(
                    f"Expected 2D point of length 2, got shape: {pts_arr.shape}"
                )
            pts_arr = pts_arr.reshape(1, 2)

        distorted = apply_brown_conrady_distortion( pts_arr, self.cx, self.cy,
                                                    self.fx, self.fy, self.k1, self.k2, self.p1, self.p2)
        if is_single_point:
            return float(distorted[0, 0]), float(distorted[0, 1])

        return distorted


class ReverseDistortion:
    name = "Reverse"

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


class BlenderDistortion(ReverseDistortion):
    name = "Blender"

    def __init__(self, width, height, k1, step=0.001):
        """
        Subclass designed to wrap ReverseDistortion and dynamically enforce
        Blender's lens distortion frame aspect metrics.
        ToDo: Should be fixed, not compatible with Brown-Condady
        """
        # Blender maps primary screen space relative to direct half-frame metrics

        cx_px = width / 2.0
        cy_px = height / 2.0
        fx_px = width / 2.0
        fy_px = height / 2.0

        # Initialize the base class with standard non-aspect focal configurations
        super(BlenderDistortion, self).__init__(
            cx_px=cx_px,
            cy_px=cy_px,
            fx_px=fx_px,
            fy_px=fy_px,
            k1=k1,
            aspect=self.img_width / self.img_height,
            step=step
        )

    def call(self, point):
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


class OpticalWedgeModel:
    """Physical assembly model linking true lens optics and wedge tilt to Brown-Conrady."""

    def __init__(
        self,
        tilt_deg: float = 0.0,
        azimuth_deg: float = 0.0,
        fx_lens: float = 1450.0,
        fy_lens: float = 1450.0,
        k1_lens: float = 0.0,
    ) -> None:
        self.tilt_deg = float(tilt_deg)
        self.azimuth_deg = float(azimuth_deg)
        self.fx_lens = float(fx_lens)
        self.fy_lens = float(fy_lens)
        self.k1_lens = float(k1_lens)

    @property
    def norm_offset(self) -> tuple:
        """Normalized angular tangent offsets (dx/fx, dy/fy)."""
        tan_tilt = float(np.tan(np.radians(self.tilt_deg)))
        az_rad = np.radians(self.azimuth_deg)
        return tan_tilt * float(np.cos(az_rad)), tan_tilt * float(np.sin(az_rad))

    @property
    def scale_shift(self) -> float:
        """Focal scale factor s = 1 + k1_lens * delta^2 induced by decentering."""
        ndx, ndy = self.norm_offset
        return 1.0 + self.k1_lens * (ndx * ndx + ndy * ndy)

    @property
    def effective_optics(self) -> tuple:
        """Apparent Brown-Conrady focal lengths and quadratic radial term (fx_eff, fy_eff, k1_eff)."""
        s = self.scale_shift
        return self.fx_lens * s, self.fy_lens * s, self.k1_lens / (s * s)

    @property
    def tangential_coefs(self) -> tuple:
        """Exact Brown-Conrady tangential coefficients (p1, p2) derived from true k1_lens."""
        ndx, ndy = self.norm_offset
        return -self.k1_lens * ndy, -self.k1_lens * ndx

    @property
    def pixel_offset(self) -> tuple:
        """Expected shift (cx_dist - cx_proj, cy_dist - cy_proj) in pixels."""
        ndx, ndy = self.norm_offset
        fx_eff, fy_eff, _ = self.effective_optics
        return fx_eff * ndx, fy_eff * ndy

    def _set_angles_from_norm(self, norm_dx: float, norm_dy: float) -> None:
        self.tilt_deg = float(np.degrees(np.arctan(np.hypot(norm_dx, norm_dy))))
        self.azimuth_deg = float(np.degrees(np.arctan2(norm_dy, norm_dx)))

    def update_from_brown_conrady(
        self, fx_eff: float, fy_eff: float, k1_eff: float, p1: float, p2: float
    ) -> None:
        """Reconstruct true physical lens and wedge state from calibrated Brown-Conrady params."""
        if abs(k1_eff) <= 1e-12:
            self.tilt_deg, self.azimuth_deg = 0.0, 0.0
            self.fx_lens, self.fy_lens, self.k1_lens = float(fx_eff), float(fy_eff), 0.0
            return

        # Solve coupled system: p1 = -k1_lens * ndy, p2 = -k1_lens * ndx, k1_lens = k1_eff * s^2
        k1_true = float(k1_eff)
        ndx, ndy = -p2 / k1_true, -p1 / k1_true
        s = 1.0
        for _ in range(4):
            s = 1.0 + k1_true * (ndx * ndx + ndy * ndy)
            k1_true = float(k1_eff) * (s * s)
            ndx = -p2 / k1_true
            ndy = -p1 / k1_true

        self._set_angles_from_norm(ndx, ndy)
        self.k1_lens = k1_true
        self.fx_lens = float(fx_eff) / s
        self.fy_lens = float(fy_eff) / s

    def update_from_centers(
        self,
        cx_proj: float,
        cy_proj: float,
        cx_dist: float,
        cy_dist: float,
        fx_eff: float,
        fy_eff: float,
        k1_eff: float,
    ) -> None:
        """Reconstruct true physical state directly from measured dual centers and effective optics."""
        ndx = (cx_dist - cx_proj) / float(fx_eff)
        ndy = (cy_dist - cy_proj) / float(fy_eff)
        self._set_angles_from_norm(ndx, ndy)

        # Recover true lens scale before decentering shift
        delta_sq = ndx * ndx + ndy * ndy
        # Solve k1_true / (1 + k1_true * delta_sq)^2 = k1_eff
        k1_true = float(k1_eff)
        s = 1.0
        for _ in range(4):
            s = 1.0 + k1_true * delta_sq
            k1_true = float(k1_eff) * (s * s)

        self.k1_lens = k1_true
        self.fx_lens = float(fx_eff) / s
        self.fy_lens = float(fy_eff) / s

    def to_brown_conrady_dict(self) -> dict:
        """Export apparent Brown-Conrady parameters for camera simulation or comparison."""
        fx_eff, fy_eff, k1_eff = self.effective_optics
        p1, p2 = self.tangential_coefs
        return {
            "fx": fx_eff,
            "fy": fy_eff,
            "k1": k1_eff,
            "p1": p1,
            "p2": p2,
        }


def get_parabola_extrema(k_prev, k_curr, k_next, f_prev, f_curr, f_next) -> tuple:
    """
    Computes both the exact coordinate (k_min) and the extreme value (f_min)
    of a parabola passing through 3 discrete points via fast finite differences.
    Returns: tuple -> (k_min, f_min)
    """
    dx_prev = float(k_curr - k_prev)
    dx_next = float(k_curr - k_next)

    df_prev = float(f_curr - f_next)
    df_next = float(f_curr - f_prev)

    num_x = (dx_prev ** 2) * df_prev - (dx_next ** 2) * df_next
    denom = dx_prev * df_prev - dx_next * df_next

    if abs(denom) < 1e-12:
        return float(k_curr), float(f_curr)

    # 1. Exact coordinate location of the extremum vertex
    k_min = k_curr - (0.5 * num_x / denom)

    geometric_scale = float(k_prev - k_curr) * float(k_curr - k_next) * float(k_prev - k_next)

    if abs(geometric_scale) < 1e-12:
        return k_min, float(f_curr)

    f_min = f_curr - (0.25 * (num_x ** 2) / (denom * geometric_scale))

    return k_min, f_min


def compute_max_radius(distortion_model, cx, cy, img_shape=(1080, 1920)):
    """
    Calculates a strict global critical radius by scanning a dense radial array.
    Returns the exact radius right before the polynomial distortion inverts
    and starts pulling points back toward the center.
    """
    H_img, W_img = img_shape
    if distortion_model is None:
        return np.hypot(H_img, W_img) / 2, np.hypot(H_img, W_img) / 2

    # Calculate maximum possible radius to the image corners
    corners = np.array([[0, 0], [W_img, 0], [W_img, H_img], [0, H_img]], dtype=np.float32)
    max_frame_radius = np.ceil(np.max(np.hypot(corners[:, 0] - cx, corners[:, 1] - cy)))
    # Extend lookup slightly beyond corners to catch shapes entering the frame boundary
    extended_search_radius = max_frame_radius * 1.5
    # Create a dense radial sampling row from 0 up to the extended radius bound
    step = 5
    sample_radii = np.arange(0, extended_search_radius, step, dtype=np.float32)
    samples_n = len(sample_radii)
    # Project test points along a horizontal ray running outwards from the center
    test_pts = np.zeros((samples_n, 2), dtype=np.float32)
    test_pts[:, 0] = cx + sample_radii
    test_pts[:, 1] = cy
    # Pass the dense evaluation line through the distortion profile
    distorted_test_pts = np.asarray(distortion_model(test_pts))
    # Measure distorted radial distances from center
    distorted_radii = np.hypot(distorted_test_pts[:, 0] - cx, distorted_test_pts[:, 1] - cy)
    # Locate the exact index where a further point turns back and gets closer
    max_dist_idx = np.argmax(distorted_radii)
    if max_dist_idx < samples_n - 1:
        loc, max_distorted_radius = get_parabola_extrema(-1, 0, 1,
                                                         distorted_radii[max_dist_idx - 1],
                                                         distorted_radii[max_dist_idx],
                                                         distorted_radii[max_dist_idx + 1])
        max_undistorted_radius = (loc + max_dist_idx) * step
    else:
        max_distorted_radius = distorted_radii[-1]
        max_undistorted_radius = sample_radii[-1]
    return max_undistorted_radius, max_distorted_radius


class ProjectiveCamera:
    """
    Manages camera intrinsic calibration, radial lens distortion pipelines,
    and spatial coordinates projection mapping using rigid PnP conventions.
    """

    def __init__(self,
                 img_shape: (int, int),
                 fx_px: float,
                 fy_px: float,
                 cx: float,
                 cy: float,
                 k1: float = None,
                 distortion_coefs={},
                 mode="perspective",
                 distortion_model=None):
        """
        Args:
            (w_img, h_img) (int): Total physical sensor pixel width and height e.g., (1920, 1080).
            fx_px (float): Focal length in pixels.
            fy_px (float): Focal length in pixels.
            cx (float): Principal point X-coordinate (typically width / 2).
            cy (float): Principal point Y-coordinate (typically height / 2).
            k1 (float): Primary radial lens distortion coefficient.
            distortion_coefs (dict): dictionary of any additional distortion coefficients, k1 can be addded as well
        """
        self.W_img = img_shape[0]
        self.H_img = img_shape[1]
        self.img_shape = img_shape
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
            self.K[2, 2] = 0

        self.distortion_model = None
        self.distortion_coefs = None
        self.max_stable_radius = self.W_img
        self.max_distorted_radius = self.W_img
        distortion_coefs_full = {}
        if k1 is not None and abs(k1) > 1.e-6:
            distortion_coefs_full["k1"] = k1
        if distortion_coefs:
            distortion_coefs_full.update(distortion_coefs)

        self.set_distortion(distortion_coefs_full, distortion_model)

    @property
    def width(self):
        return self.W_img

    @property
    def height(self):
        return self.H_img

    @property
    def k1(self):
        return self.distortion_coefs["k1"]

    def set_distortion(self, distortion_coefs, model_name=None, distortion_center=None) -> None:
        """
        Synchronized setter method to dynamically update distortion parameters.
        Delegates tangential coefficient and tilt recovery to OpticalWedgeModel.
        """
        if isinstance(distortion_coefs, dict):
            self.distortion_coefs = distortion_coefs.copy()
        else:
            self.distortion_coefs = {"k1": float(distortion_coefs)}

        epsilon = 1e-12
        k1_eff = float(self.distortion_coefs.get("k1", 0.0))

        if distortion_center is not None:
            wedge = OpticalWedgeModel()
            wedge.update_from_centers(
                cx_proj=self.cx,
                cy_proj=self.cy,
                cx_dist=float(distortion_center[0]),
                cy_dist=float(distortion_center[1]),
                fx_eff=self.fx_px,
                fy_eff=self.fy_px,
                k1_eff=k1_eff,
            )
            p1, p2 = wedge.tangential_coefs
            self.distortion_coefs["p1"] = p1
            self.distortion_coefs["p2"] = p2

        has_higher_order = (
            abs(self.distortion_coefs.get("k2", 0.0)) > epsilon
            or abs(self.distortion_coefs.get("p1", 0.0)) > epsilon
            or abs(self.distortion_coefs.get("p2", 0.0)) > epsilon
        )

        if model_name is None:
            if has_higher_order:
                model_name = "full"
            elif self.distortion_model is not None:
                model_name = getattr(self.distortion_model, "name", None)
            elif abs(k1_eff) > epsilon:
                model_name = "radial"

        if isinstance(model_name, str):
            if len(self.distortion_coefs) > 0:
                if model_name == "reverse":
                    self.distortion_model = ReverseDistortion(
                        self.cx,
                        self.cy,
                        self.fx_px,
                        self.fy_px,
                        self.distortion_coefs["k1"],
                    )
                elif model_name == "blender":
                    self.distortion_model = BlenderDistortion(
                        self.W_img, self.H_img, self.distortion_coefs["k1"]
                    )
                elif model_name in ("full", "BrownConradyFull"):
                    self.distortion_model = BrownConradyDistortion(
                        self.cx,
                        self.cy,
                        self.fx_px,
                        self.fy_px,
                        self.distortion_coefs.get("k1", 0.0),
                        self.distortion_coefs.get("k2", 0.0),
                        self.distortion_coefs.get("p1", 0.0),
                        self.distortion_coefs.get("p2", 0.0),
                    )
                else:
                    self.distortion_model = Distortion(
                        self.cx,
                        self.cy,
                        self.fx_px,
                        self.fy_px,
                        self.distortion_coefs.get("k1", 0.0),
                    )
        else:
            self.distortion_model = model_name

        max_radii = compute_max_radius(self.distortion_model, self.cx, self.cy, img_shape=self.img_shape)
        self.max_stable_radius = max_radii[0]
        self.max_distorted_radius = max_radii[1]

    def undistort(self, point):
        """
        :param point:
        :return:
        """
        if self.distortion_model is None:
            return point
        MAX_ITER = 20
        a = point
        ca = (a[0] - self.cx, a[1] - self.cy)
        b = (ca[0] * 2 + self.cx, ca[1] * 2 + self.cy)
        r_d = np.hypot(ca[0], ca[1])
        if r_d >= self.max_distorted_radius:
            # Extrapolation on just max radius
            unit_x = ca[0] / r_d
            unit_y = ca[1] / r_d
            return self.cx + self.max_stable_radius * unit_x, self.cy + self.max_stable_radius * unit_y

        r_prev = 2 * r_d
        p = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        for k in range(MAX_ITER):
            p_d = self.distortion_model(p)
            r = np.hypot(p_d[0] - self.cx, p_d[1] - self.cy)
            if r < r_d:
                a = p
            else:
                b = p
            if np.abs(r - r_prev) < 0.01:
                break
            r_prev = r
            p = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

        # 2D residual correction for tangential / lateral shift
        for _ in range(5):
            p_d = self.distortion_model(p)
            err = p_d - point
            if max(abs(err)) < 0.01:
                break
            p -= err
        return p

    def undistort_points(self, points):
        """
        Vectorized batch binary search undistortion over 2D points (N, 2).
        """
        if self.distortion_model is None:
            return points

        pts = np.asarray(points, dtype=np.float64)
        if pts.size == 0:
            return pts.copy()

        MAX_ITER = 20
        center = np.array([self.cx, self.cy], dtype=np.float64)
        ca = pts - center
        r_d = np.hypot(ca[:, 0], ca[:, 1])

        p = pts.copy()
        over_mask = r_d >= self.max_distorted_radius
        if np.any(over_mask):
            p[over_mask] = center + self.max_stable_radius * (ca[over_mask] / r_d[over_mask, None])
        interior = ~over_mask
        active = interior.copy()
        if not np.any(active):
            return p

        a = pts.copy()
        b = ca * 2.0 + center
        r_prev = 2.0 * r_d
        p[active] = (a[active] + b[active]) * 0.5

        for _ in range(MAX_ITER):
            p_d = np.asarray(self.distortion_model(p[active]), dtype=np.float64)
            r = np.hypot(p_d[:, 0] - self.cx, p_d[:, 1] - self.cy)

            less = r < r_d[active]
            a[active] = np.where(less[:, None], p[active], a[active])
            b[active] = np.where(less[:, None], b[active], p[active])

            # Exit check directly by 2D Euclidean distance to target distorted point
            err = p_d - pts[active]
            dist_err = np.hypot(err[:, 0], err[:, 1])

            use_2d = np.abs(r - r_prev[active]) < 1.
            r_prev[active] = r
            p_bisect = (a[active] + b[active]) * 0.5
            p_refine = p[active] - err
            p[active] = np.where(use_2d[:, None], p_refine, p_bisect)

            done = dist_err < 0.01
            active[active] = ~done
            if not np.any(active):
                break

        return p


    def compute_homography(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
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
        return 0 <= point[0] < self.W_img and 0 <= point[1] < self.H_img

    def project_point(self, world_point: np.ndarray, rotation: np.ndarray, t: np.ndarray) -> np.ndarray:
        if len(world_point) == 2:
            local_point = rotation @ [world_point[0], world_point[1], 0]  # assume the point is on pattern plane (Z=0)
        else:
            local_point = rotation @ world_point
        local_point += t.flatten()
        local_point = self.K @ local_point
        x, y = local_point[0] / local_point[2], local_point[1] / local_point[2]
        r = np.hypot(x - self.cx, y - self.cy)
        if r >= self.max_stable_radius:
            return None

        if self.distortion_model:
            return self.distortion_model((x, y))

        return x, y

    def project_points(self, world_pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Projects 2D world space points into the distorted camera pixel plane."""
        pts = np.atleast_2d(np.array(world_pts, dtype=np.float32))
        num_pts = len(pts)

        h = self.compute_homography(R, t)
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

    def aperture_radius(self):
        """
        Distance from (cx,cy) to the farthest image corner (pixels). Cached.
        """
        if hasattr(self, "_R_ap"):
            return self._R_ap
        self._R_ap = np.hypot(self.W_img, self.H_img) * 0.8
        return self._R_ap

    def _effective_focal(self):
        """
        Compute scalar effective focal length f_eff from fx, fy
        using Chebyshov formula: 1/f_eff^2 = 0.5*(1/fx^2 + 1/fy^2).
        """
        fx = self.fx_px
        fy = self.fy_px
        if fx <= 0 or fy <= 0:
            raise ValueError("fx and fy must be > 0 for effective focal computation")
        inv_fx2 = 1.0 / (fx * fx)
        inv_fy2 = 1.0 / (fy * fy)
        inv_fe2 = 0.5 * (inv_fx2 + inv_fy2)
        f_eff = inv_fe2 ** -0.5
        return f_eff

    @staticmethod
    def _scale_radial_coeffs(coeffs, scale_sq):
        """
        Scale even-power radial polynomial coefficients using sequential multiplication.
        """
        n = len(coeffs)
        out = [float(c) for c in coeffs]

        # Triangular pass: multiply suffix [i..n-1] by scale_sq at each step i
        for i in range(n):
            for j in range(i, n):
                out[j] *= scale_sq
        return out

    def convert_focal_to_aperture(self, coeffs_focal):
        """
        Convert polynomial coefficients given for focal-normalized radius r_f = r_px / f
        into aperture-normalized coefficients for r_ap = r_px / R_ap.

        Uses an effective focal f_eff computed from fx,fy:
            scale = R_ap / f_eff
            a_ap[i] = a_f[i] * scale^i

        Returns same container type as input.
        """
        f_eff = self._effective_focal()
        f_ap = self.aperture_radius()
        if f_eff == 0:
            raise ValueError("effective focal is zero")
        scale = f_ap / f_eff
        return self._scale_radial_coeffs(coeffs_focal, scale * scale)

    def convert_aperture_to_focal(self, coeffs_aperture):
        """
        Convert polynomial coefficients from aperture-normalized (r_ap = r_px / R_ap)
        into focal-normalized form (r_f = r_px / f_eff).
        """
        f_eff = self._effective_focal()
        R_ap = self.aperture_radius()
        if R_ap == 0:
            raise ValueError("aperture radius is zero")
        scale = f_eff / R_ap
        return self._scale_radial_coeffs(coeffs_aperture, scale * scale)

    def calculate_source_resolution(self, safety_margin: float = 1.05) -> tuple:
        """
        Calculates the required source image resolution (src_w, src_h) for
        distorting an image without producing black border gaps (holes),
        assuming intrinsic matrix K remains identical.

        Parameters:
        - target_w, target_h: Desired output (distorted) image dimensions.
        - safety_margin: Small multiplier (e.g., 1.05 for 5% safety) to prevent floating-point rounding gaps.

        Returns:
        - (src_w, src_h): Required dimensions for the undistorted source image.
        """
        target_w, target_h = self.W_img, self.H_img
        cx, cy = self.cx, self.cy

        test_points = np.array([
            [0, 0],  # Top-Left corner
            [target_w - 1, 0],  # Top-Right corner
            [0, target_h - 1],  # Bottom-Left corner
            [target_w - 1, target_h - 1],  # Bottom-Right corner
            [cx, 0],  # Top-Center
            [cx, target_h - 1],  # Bottom-Center
            [0, cy],  # Left-Center
            [target_w - 1, cy]  # Right-Center
        ], dtype=np.float32)

        src_pt = self.undistort_points(test_points)

        src_pt_array = np.array(src_pt)
        max_dx = np.max(np.abs(src_pt_array[:, 0] - cx))
        max_dy = np.max(abs(src_pt_array[:, 1] - cy))

        req_half_w = max_dx * safety_margin
        req_half_h = max_dy * safety_margin

        src_w = int(np.ceil(req_half_w * 2.0))
        src_h = int(np.ceil(req_half_h * 2.0))

        return src_w, src_h

    def distort_image(self, src_img):
        """
        Applies lens distortion simulation to an undistorted input image.
        Uses pure NumPy indexing and fast vectorized mask multiplication.
        No OpenCV required.
        """
        w, h = self.img_shape

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
                x_src, y_src = self.undistort((x_dist, y_dist))

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

        # Smooth step
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
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch), np.cos(pitch)]
    ], dtype=np.float64)

    R_y = np.array([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)]
    ], dtype=np.float64)

    R_z = np.array([
        [np.cos(roll), -np.sin(roll), 0.0],
        [np.sin(roll), np.cos(roll), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 3. Combine rotations YXZ  Cam to World
    R = R_y @ R_x @ R_z

    # 4. Construct Translation Vector t
    t = np.array([[tx], [ty], [tz]], dtype=np.float64)

    # 5. Assemble Extrinsic Matrix [R | t] (size 3x4)
    return R.T, -(R.T @ t)


#
# ===========================================================
# Camera Tests
# ===========================================================


def test_parabolic_approx():
    np.random.seed(42)

    # 1. Define our clean reference polynomial: F(k) = (k - 2.5)^2 + 10.0
    # The absolute true minimum is analytically locked at: k = 2.5, F(k) = 10.0
    def f_test(k):
        return (k - 2.5) ** 2 + 10.0

    TRUE_K_MIN = 2.5
    TRUE_F_MIN = 10.0
    NUMBER_OF_ITERATIONS = 50

    for iteration in range(NUMBER_OF_ITERATIONS):
        # 2. Generate a random asymmetric triplet around the target vertex
        # We sample one point to the left, one close to the middle, and one to the right
        k_prev = np.random.uniform(low=0, high=2)
        k_curr = np.random.uniform(low=2, high=4)
        k_next = np.random.uniform(low=3, high=5)

        # Evaluate the exact functional heights at these discrete locations
        f_prev = f_test(k_prev)
        f_curr = f_test(k_curr)
        f_next = f_test(k_next)

        k_sol, f_sol = get_parabola_extrema(k_prev, k_curr, k_next, f_prev, f_curr, f_next)

        assert abs(k_sol - TRUE_K_MIN) < 1e-6, f"Iteration {iteration}: Vertex X-coordinate split drifted."
        assert abs(f_sol - TRUE_F_MIN) < 1e-6, f"Iteration {iteration}: Vertex Y-height function value drifted."

    print("-> test_parabolic_approx passed.")


def test_distortion_camera():
    camera = ProjectiveCamera((100, 100), 50, 50, 50, 50, -0.1)
    assert camera.max_stable_radius > 40
    yaw, pitch, roll = 4, 5, 6
    R, t = compute_camera_projection_matrix(roll, pitch, yaw, 0, 0, -1)
    p = camera.project_point((1, 2, 0), R, t)
    R = np.eye(3, 3)
    t = np.array([0, 0, 0])
    p_dist = camera.project_point((0.5, 1, 1), R, t)
    p = camera.undistort(p_dist)
    K = np.linalg.inv(camera.K)
    p = K @ np.array([p[0], p[1], 1])
    assert abs(p[0] - 0.5) < 0.05 and abs(p[1] - 1) < 0.05


def test_reverse_distortion_camera():
    camera = ProjectiveCamera((100, 100), 50, 50, 50, 50, 0.1, distortion_model="reverse")
    assert camera.max_stable_radius > 40
    yaw, pitch, roll = 4, 5, 6
    R, t = compute_camera_projection_matrix(roll, pitch, yaw, 0, 0, -1000)
    camera.project_point((10, 20, 0), R, t)


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


def test_batch_processing():
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


def test_camera_radius():
    """
    Strict analytical verification of the camera deflection point.
    """
    # 1. Theoretical target derivation: r = sqrt(-1 / 3k1) * fx
    # For fx=1150, k1=-0.2: r_un = sqrt(1 / 0.6) * 1150 = 1484.6437 px
    k1_gt = -0.2
    fx_gt = 1150.0
    r_undistorted_gt = float(np.sqrt(-1.0 / (3.0 * k1_gt)) * fx_gt)
    r_distorted_gt = r_undistorted_gt * (1.0 + k1_gt * (-1.0 / (3.0 * k1_gt)))  # ~989.76 px

    # 2. Instantiate real camera (triggers compute_max_radius internally)
    cam = ProjectiveCamera(
        img_shape=(1080, 1920),
        fx_px=fx_gt, fy_px=1150.0,
        cx=960.0, cy=540.0,
        k1=k1_gt,
        mode="perspective",
        distortion_model="standard"
    )

    assert cam.max_stable_radius > 0, "Error: radius is negative or zero!"
    assert abs(cam.max_stable_radius - r_undistorted_gt) < 0.1, "Undistorted radius mismatch!"
    assert abs(cam.max_distorted_radius - r_distorted_gt) < 0.1, "Distorted radius mismatch!"
    print("-> Radius profiler passed.")


def test_calculate_required_source_resolution():
    """
    source resolution tests
    """

    target_w, target_h = 800, 600
    cx, cy = 405, 300
    scale = 0.9
    camera = ProjectiveCamera((target_w, target_h), cx=cx, cy=cy,
                              fx_px=300, fy_px=300, distortion_coefs={"scale": scale},
                              distortion_model=ScaleDistortion(cx, cy, scale))

    src_w, src_h = camera.calculate_source_resolution(safety_margin=1.0)

    max_half_w = max(abs(0 - cx), abs((target_w - 1) - cx))
    max_half_h = max(abs(0 - cy), abs((target_h - 1) - cy))

    expected_w = int(np.ceil(max_half_w / scale * 2.0))
    expected_h = int(np.ceil(max_half_h / scale * 2.0))

    assert (
            src_w >= target_w
    ), f"Source width {src_w} cannot be smaller than target {target_w}"
    assert (
            src_h >= target_h
    ), f"Source height {src_h} cannot be smaller than target {target_h}"
    assert (
            abs(src_w - expected_w) <= 2
    ), f"Width {src_w} deviates too much from expected {expected_w}"
    assert (
            abs(src_h - expected_h) <= 2
    ), f"Height {src_h} deviates too much from expected {expected_h}"

    margin = 1.10
    src_w_safe, src_h_safe = camera.calculate_source_resolution(
        safety_margin=margin
    )
    assert (
            src_w_safe >= src_w
    ), "Safety margin did not increase source width as expected"
    assert (
            src_h_safe >= src_h
    ), "Safety margin did not increase source height as expected"

    print("-> resolution tests passed.")


def test_wedge_model_center_offset_roundtrip() -> None:
    """Verify forward and inverse mapping via dual-center pixel offsets."""
    fx_lens, fy_lens, k1_lens = 1450.0, 1452.0, -0.22
    cx_proj, cy_proj = 974.5, 531.8
    expected_tilt = 0.85
    expected_azimuth = -33.7

    model = OpticalWedgeModel(
        tilt_deg=expected_tilt,
        azimuth_deg=expected_azimuth,
        fx_lens=fx_lens,
        fy_lens=fy_lens,
        k1_lens=k1_lens,
    )
    dx_px, dy_px = model.pixel_offset
    fx_eff, fy_eff, k1_eff = model.effective_optics

    recovered = OpticalWedgeModel()
    recovered.update_from_centers(
        cx_proj=cx_proj,
        cy_proj=cy_proj,
        cx_dist=cx_proj + dx_px,
        cy_dist=cy_proj + dy_px,
        fx_eff=fx_eff,
        fy_eff=fy_eff,
        k1_eff=k1_eff,
    )

    assert np.isclose(recovered.tilt_deg, expected_tilt, atol=1e-9)
    assert np.isclose(recovered.azimuth_deg, expected_azimuth, atol=1e-9)
    assert np.isclose(recovered.fx_lens, fx_lens, atol=1e-9)
    assert np.isclose(recovered.fy_lens, fy_lens, atol=1e-9)
    assert np.isclose(recovered.k1_lens, k1_lens, atol=1e-9)
    print("-> wedge distortion passed.")



def test_wedge_model_brown_conrady_roundtrip() -> None:
    """Verify forward and inverse mapping via Brown-Conrady parameters."""
    fx_lens, fy_lens, k1_lens = 1450.0, 1450.0, -0.22
    expected_tilt = 1.25
    expected_azimuth = 142.0

    model = OpticalWedgeModel(
        tilt_deg=expected_tilt,
        azimuth_deg=expected_azimuth,
        fx_lens=fx_lens,
        fy_lens=fy_lens,
        k1_lens=k1_lens,
    )
    bc_params = model.to_brown_conrady_dict()

    recovered = OpticalWedgeModel()
    recovered.update_from_brown_conrady(
        fx_eff=bc_params["fx"],
        fy_eff=bc_params["fy"],
        k1_eff=bc_params["k1"],
        p1=bc_params["p1"],
        p2=bc_params["p2"],
    )

    assert np.isclose(recovered.tilt_deg, expected_tilt, atol=1e-9)
    assert np.isclose(recovered.azimuth_deg, expected_azimuth, atol=1e-9)
    assert np.isclose(recovered.fx_lens, fx_lens, atol=1e-9)
    assert np.isclose(recovered.k1_lens, k1_lens, atol=1e-9)
    print("-> wedge Brown-Conrady distortion passed.")


def test_set_distortion_wedge_delegation() -> None:
    """Verify ProjectiveCamera.set_distortion tangential calculation and model upgrades."""
    cam = ProjectiveCamera(
        img_shape=(1920, 1080),
        fx_px=1450.0,
        fy_px=1450.0,
        cx=974.5,
        cy=531.8,
        k1=0.0,
    )

    # 1. Scalar radial k1 assignment
    cam.set_distortion(-0.22)
    assert np.isclose(cam.k1, -0.22, atol=1e-12)
    assert isinstance(cam.distortion_model, Distortion)

    # 2. Automatic upgrade to full Brown-Conrady via distortion_center offset
    cam.set_distortion({"k1": -0.22}, distortion_center=(994.3, 518.6))
    assert isinstance(cam.distortion_model, BrownConradyDistortion)
    assert "p1" in cam.distortion_coefs and "p2" in cam.distortion_coefs
    assert cam.distortion_coefs["p1"] < 0.0
    assert cam.distortion_coefs["p2"] > 0.0

    # 3. Direct initialization from explicit (p1, p2) without distortion_center
    cam.set_distortion({"k1": -0.22, "p1": -0.0020, "p2": 0.0030})
    assert isinstance(cam.distortion_model, BrownConradyDistortion)
    assert np.isclose(cam.distortion_coefs["p1"], -0.0020, atol=1e-12)
    assert np.isclose(cam.distortion_coefs["p2"], 0.0030, atol=1e-12)

    # 4. Resetting distortion to zero
    cam.set_distortion(0.0)
    assert np.isclose(cam.k1, 0.0, atol=1e-12)
    assert cam.distortion_coefs.get("p1", 0.0) == 0.0
    assert cam.distortion_coefs.get("p2", 0.0) == 0.0
    print("-> Camera radial distortion passed.")


def test_batch_undistort_accuracy() -> None:
    """Verify vectorized undistort_points against scalar undistort, roundtrip inversion, and boundary clamping."""
    cam = ProjectiveCamera(
        img_shape=(1920, 1080),
        fx_px=1450.0,
        fy_px=1450.0,
        cx=974.5,
        cy=531.8,
        distortion_coefs={
            "k1": -0.22,
            "k2": 0.015,
            "p1": -0.0020,
            "p2": 0.0030,
        },
        distortion_model="BrownConradyFull",
    )

    # 1. Generate a dense interior grid of ideal pixel points
    xs = np.linspace(200.0, 1720.0, 20)
    ys = np.linspace(150.0, 930.0, 15)
    grid_x, grid_y = np.meshgrid(xs, ys)
    ideal_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Apply forward full Brown-Conrady distortion
    distorted_pts = np.asarray(cam.distortion_model(ideal_pts), dtype=np.float64)

    # 2. Add extreme out-of-bounds points exceeding max_distorted_radius
    far_pts = np.array(
        [
            [cam.cx + cam.max_distorted_radius * 1.5, cam.cy],
            [cam.cx, cam.cy - cam.max_distorted_radius * 2.0],
        ],
        dtype=np.float64,
    )
    mixed_batch = np.vstack([distorted_pts, far_pts])

    # Run vectorized batch undistortion
    recovered_batch = np.asarray(cam.undistort_points(mixed_batch), dtype=np.float64)
    assert recovered_batch.shape == mixed_batch.shape
    assert not np.any(np.isnan(recovered_batch))

    # 3. Verify roundtrip inversion accuracy on interior points (binary search tol = 0.01 px)
    interior_recovered = recovered_batch[: len(ideal_pts)]
    max_roundtrip_err = np.max(np.linalg.norm(interior_recovered - ideal_pts, axis=1))
    assert max_roundtrip_err < 0.05, f"Batch roundtrip error too high: {max_roundtrip_err:.4f} px"

    # 4. Verify exact consistency between vectorized batch and scalar undistort
    scalar_sample = np.array(
        [cam.undistort(pt) for pt in mixed_batch[::15]], dtype=np.float64
    )
    batch_sample = recovered_batch[::15]
    max_diff = np.max(np.abs(scalar_sample - batch_sample))
    assert max_diff < 0.01, f"Mismatch between batch and scalar undistort: {max_diff}"

    # 5. Verify boundary clamping onto max_stable_radius for out-of-bounds points
    far_recovered = recovered_batch[len(ideal_pts) :]
    far_radii = np.hypot(far_recovered[:, 0] - cam.cx, far_recovered[:, 1] - cam.cy)
    assert np.allclose(far_radii, cam.max_stable_radius, atol=1e-6)


if __name__ == "__main__":
    test_parabolic_approx()
    test_camera_radius()
    test_center_point_stability()
    test_directional_consistency_barrel()
    test_mathematical_inversion_accuracy()
    test_batch_processing()
    test_distortion_camera()
    test_reverse_distortion_camera()
    test_calculate_required_source_resolution()
    test_wedge_model_brown_conrady_roundtrip()
    test_wedge_model_center_offset_roundtrip()
    test_set_distortion_wedge_delegation()
    test_batch_undistort_accuracy()