import numpy as np
import scipy.optimize
from camera import ProjectiveCamera


class FrameBundle:
    """
    Cached geometric structure for a single camera frame.
    Harvests line bundles once, filters for top high-integrity tracks,
    and pre-computes NumPy slices to eliminate overhead inside the loop.
    """

    def __init__(self, topological_matrix: np.ndarray, detected_points: np.ndarray, point_weights: np.array,
                 max_lines_per_axis: int = 8,
                 min_len: int = 15):
        self.detected_points = detected_points.copy()
        self.weights = point_weights
        u, v, w, h = _harvest_hexagonal_line_bundles(topological_matrix, N=max_lines_per_axis, MIN_LEN=min_len)
        self.lines = [u, v, w, h]
        self.selected_lines = u + v + w + h

    def undistort_selected(self, cam: ProjectiveCamera) -> np.ndarray:
        """
        Natively executes lazy, single-pass point undistortion strictly for
        the nodes present in the selected line bundles under the current cam state.
        """
        u_points = self.detected_points.copy()
        processed = np.zeros(len(self.detected_points), dtype=np.uint8)

        for line in self.selected_lines:
            n_pts = len(line)
            for i in range(n_pts):
                idx = line[i]
                if processed[idx]:
                    continue
                processed[idx] = 1
                u_points[idx] = cam.undistort(u_points[idx])

        return u_points

    def estimate_vp(self, cam:ProjectiveCamera)-> list:
        corrected_points = self.undistort_selected(cam)

        u_lines, v_lines, w_lines, h_lines = self.lines
        u_eqs = [compute_homogeneous_line(ln, corrected_points) for ln in u_lines]
        v_eqs = [compute_homogeneous_line(ln, corrected_points) for ln in v_lines]
        w_eqs = [compute_homogeneous_line(ln, corrected_points) for ln in w_lines]
        h_eqs = [compute_homogeneous_line(ln, corrected_points) for ln in h_lines]

        Vu = compute_homogeneous_vanishing_point(u_eqs)
        Vv = compute_homogeneous_vanishing_point(v_eqs)
        Vw = compute_homogeneous_vanishing_point(w_eqs)
        Vh = compute_homogeneous_vanishing_point(h_eqs)
        return [Vu, Vv, Vw, Vh]


def menger_curvature_loss(frame: FrameBundle,
                          cam: ProjectiveCamera, loss="menger", step=2) -> float:
    """
    Vectorized Menger curvature loss.
    loss: menger or inscribed radius - the radius is more soft and can be good for final adjustment
    return: total loss over all lines in the bundle

    """
    total_loss = 0.0
    u_points = frame.undistort_selected(cam)

    for line in frame.selected_lines:
        n_pts = len(line)

        # Extract node indices blocks for triplets using slices
        idx_p1 = line[0 : n_pts - 2 * step]
        idx_p2 = line[step : n_pts - step]
        idx_p3 = line[2 * step : n_pts]

        p1 = u_points[idx_p1]
        p2 = u_points[idx_p2]
        p3 = u_points[idx_p3]

        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p1 - p3

        cross_prod = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        abs_cross = np.abs(cross_prod)

        len_v1 = np.hypot(v1[:, 0], v1[:, 1])
        len_v2 = np.hypot(v2[:, 0], v2[:, 1])
        len_v3 = np.hypot(v3[:, 0], v3[:, 1])

        if loss == "menger":
            denom = len_v1 * len_v2 * len_v3
        else:
            denom = len_v1 + len_v2 + len_v3
        if frame.weights is None:
            total_loss += np.sum(abs_cross / denom)
        else:
            line_weights = frame.weights[idx_p1] + frame.weights[idx_p2] + frame.weights[idx_p3]
            total_loss += np.sum(line_weights * abs_cross / denom)

    return total_loss


def compute_homogeneous_line(line_indices: list, points_2d: np.ndarray) -> np.ndarray:
    """
    Fits a straight line to sub-pixel points inmathbb{P}^2 using SVD.
    Returns the line vector l = [A, B, C]^T such that l^T * x = 0.
    """
    pts = np.asarray(points_2d)[line_indices]
    if len(pts) < 2:
        return None
    # Construct homogeneous coordinates [x, y, 1] for all points
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts_hom = np.hstack((pts, ones))

    # The line vector is the right-singular vector matching the minimum singular value
    _, _, vh = np.linalg.svd(pts_hom)
    line_v = vh[-1]  # [A, B, C]
    return line_v / np.linalg.norm(line_v[:2])


def compute_homogeneous_vanishing_point(line_vectors: list, min_lines=3) -> np.ndarray:
    """
    Finds the intersection point of a line bundle in homogeneous space.
    Returns v = [x, y, w]^T. If lines are parallel, w converges to 0 cleanly.
    """
    valid_lines = [ln for ln in line_vectors if ln is not None]
    if len(valid_lines) < min_lines:
        return None

    # Stack line equations into an (M, 3) matrix L
    L_mat = np.vstack(valid_lines)

    # Solve L * v = 0 using SVD
    _, _, vh = np.linalg.svd(L_mat)
    v_point = vh[-1]  # [x, y, w]
    return v_point


def get_model_directions():
    # 3D theoretical model direction vectors on the hexagonal continuum sheet
    sqrt3_over2 = float(np.sqrt(3.0) / 2.0)
    return [
        [1.0, 0.0],  # Vu Axis
        [0.5, -sqrt3_over2],  # Vv Axis
        [0.5, sqrt3_over2],  # Vw Axis
        [0.0, -1.0]  # V_vert Axis
    ]


MIN_PERSPECTIVE_DET_MAGNITUDE = 1e-9
MIN_PIXEL_ASPECT_RATIO = 0.8
MAX_PIXEL_ASPECT_RATIO = 1/MIN_PIXEL_ASPECT_RATIO
MIN_PHYSICAL_FOCAL_PX = 400.0
MAX_PHYSICAL_FOCAL_PX = 4000.0


def solve_weak_perspectivity_matrix(vp: list, cx: float, cy: float) -> dict:
    """
    Zhang method for calibration estimation
    STAGE 1: Solves for general diagonal matrix (fx != fy) with 2 variables.
    STAGE 2: Isotropic fallback (fx == fy) if Stage 1 is unstable.
    """
    fail_registry = {
        "status": "failed",
        "message": "Structural configuration degraded or incomplete parameters."
    }

    if vp is None or len(vp) < 4:
        return fail_registry

    model_directions = get_model_directions()
    # 1. RUN STANDARD DLT PASS TO RESOLVE CENTERED LINE HOMOGRAPHY H
    A_rows = []
    for i in range(4):
        v = vp[i]
        if v is None:
            continue
        d = model_directions[i]

        x, y, w = v[0], v[1], v[2]
        x -= cx * w
        y -= cy * w
        dx, dy = d[0], d[1]

        # Cross product optimization mapping for 3x2 matrix elements
        # elements ordered as: h11, h12, h21, h22, h31, h32
        A_rows.append([w * dx, w * dy, 0.0, 0.0, -x * dx, -x * dy])
        A_rows.append([0.0, 0.0, w * dx, w * dy, -y * dx, -y * dy])

    if len(A_rows) < 5:
        return fail_registry

    A_dlt = np.array(A_rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(A_dlt)
    H_flat = vh[-1]
    h11, h12, h21, h22, h31, h32 = H_flat

    # =====================================================================
    # STAGE 1: GENERAL DIAGONAL MATRIX TESTING (2 Variables: gx=1/fx^2, gy=1/fy^2)
    # =====================================================================
    # Setting up 2x2 linear system: M_diag * [gx, gy]^T = N_diag
    # Eq 1 (Orthogonality):      (h11*h12)*gx       + (h21*h22)*gy       = -h31*h32
    # Eq 2 (Scale Equality):     (h11^2 - h12^2)*gx + (h21^2 - h22^2)*gy = h32^2 - h31^2
    M_diag = np.array([
        [h11 * h12, h21 * h22],
        [h11 ** 2 - h12 ** 2, h21 ** 2 - h22 ** 2]
    ], dtype=np.float64)

    N_diag = np.array([
        [-h31 * h32],
        [h32 ** 2 - h31 ** 2]
    ], dtype=np.float64)

    det_M = M_diag[0,0] * M_diag[1,1] - M_diag[0,1] * M_diag[1,0]

    if abs(det_M) > MIN_PERSPECTIVE_DET_MAGNITUDE:
        # Solve the 2x2 system analytically
        g_sol = np.linalg.solve(M_diag, N_diag)
        gx, gy = float(g_sol[0]), float(g_sol[1])

        if gx < 0 and gy < 0:
            gx = -gx
            gy = -gy

        # Physical constraint: inverse squares must be positive and within reasonable bounds
        G_MIN_BOUND = 1.0 / (MAX_PHYSICAL_FOCAL_PX ** 2)
        G_MAX_BOUND = 1.0 / (MIN_PHYSICAL_FOCAL_PX ** 2)

        if G_MIN_BOUND < gx < G_MAX_BOUND and G_MIN_BOUND < gy < G_MAX_BOUND:
            fx_final = np.sqrt(1.0 / gx)
            fy_final = np.sqrt(1.0 / gy)
            # Check for sudden heavy aspect ratio distortion (e.g. > 2.5x mismatch)
            # which usually indicates a noisy/degenerate mathematical artifact
            if MIN_PIXEL_ASPECT_RATIO < (fx_final / fy_final) < MAX_PIXEL_ASPECT_RATIO:
                print(f" -> [SUCCESS] Diagonal Matrix resolved. fx: {fx_final:.2f}, fy: {fy_final:.2f}")
                return {
                    "status": "success", "mode": "perspective",
                    "fx": fx_final, "fy": fy_final,
                    "message": "Full diagonal perspective solved successfully."
                }

    # =====================================================================
    # STAGE 2: FIXED ISOTROPIC FALLBACK (fx == fy = f)
    # =====================================================================
    print(" -> [FAILED]  Diagonal matrix unstable or non-physical. Activating Isotropic constraint (fx=fy)...")

    M_iso = np.array([
        [-h31 * h32],
        [h32 ** 2 - h31 ** 2]
    ], dtype=np.float64)

    dot_h1_h2 = h11 * h12 + h21 * h22
    norm_h1 = h11 ** 2 + h21 ** 2
    norm_h2 = h12 ** 2 + h22 ** 2
    N_iso = np.array([
        [dot_h1_h2],
        [norm_h1 - norm_h2]
    ], dtype=np.float64)

    L_solution, _, _, _ = np.linalg.lstsq(M_iso, N_iso, rcond=None)
    f2 = float(L_solution)
    if f2 < 0.0:
        L_solution, _, _, _ = np.linalg.lstsq(M_iso, -N_iso, rcond=None)
        f2 = float(L_solution.flatten())

    if MIN_PHYSICAL_FOCAL_PX**2 < f2 < MAX_PHYSICAL_FOCAL_PX **2:
        f_iso_final = np.sqrt(f2)
        print(f" -> [SUCCESS] Isotropic Perspective f: {f_iso_final:.2f} px.")
        return {
            "status": "success", "mode": "perspective",
            "fx": f_iso_final, "fy": f_iso_final,
            "message": "Isotropic constraint successfully extracted the focal scale."
        }

    # =====================================================================
    # STAGE 3: PURE INVARIANT INVERSE-AFFINE MODEL
    # =====================================================================
    print(" -> [FAILED] Pure planar degeneracy. Enforcing Affine Geometry fallback.")
    inv_fx_affine = float(np.sqrt(norm_h1))
    inv_fy_affine = float(np.sqrt(norm_h2))

    fx_final = 1.0 / inv_fx_affine if inv_fx_affine > 1e-6 else 0.0
    fy_final = 1.0 / inv_fy_affine if inv_fy_affine > 1e-6 else 0.0

    return {
        "status": "success", "mode": "affine",
        "fx": fx_final, "fy": fy_final,
        "message": "Weak perspective enforced due to degeneracy."
    }


def solve_multi_frame_zhang_matrix(multi_frame_vps: list, cx: float, cy: float) -> dict:
    """
    Multi-Frame Joint Zhang Engine. the same as weak estimator but process multimple frames

    1. Computes local line homographies H_k independently per frame via local SVD.
    2. Respects a purely diagonal Absolute Conic matrix due to pre-isolated (cx, cy).
    3. Leaves denominators on the right-hand side (N_diag) to guarantee total stability.
    """
    if multi_frame_vps is None or len(multi_frame_vps) < 1:
        fail_registry = {
            "status": "failed",
            "message": "Inbound multi-view tracking arrays contain no valid entries."
        }
        return fail_registry

    # Global overdetermined system containers for multi-frame pooling
    M_diag = []
    N_diag = []
    M_iso = []
    N_iso = []

    model_directions = get_model_directions()
    for frame_idx, vp in enumerate(multi_frame_vps):
        if vp is None:
            continue

        A_rows = []
        # Build the localized DLT matrix strictly for this independent frame pose
        for i in range(4):
            v = vp[i]
            if v is None:
                continue

            x, y, w = v[0], v[1], v[2]
            x_centered = x - cx * w
            y_centered = y - cy * w

            dx, dy = model_directions[i]

            A_rows.append([w * dx, w * dy, 0.0, 0.0, -x_centered * dx, -x_centered * dy])
            A_rows.append([0.0, 0.0, w * dx, w * dy, -y_centered * dx, -y_centered * dy])

        if len(A_rows) < 5:
            continue

        # Solve for the UNIQUE homography matrix H_k of this single viewpoint
        A_dlt = np.array(A_rows, dtype=np.float64)
        _, _, vh = np.linalg.svd(A_dlt)
        h11, h12, h21, h22, h31, h32 = vh[-1]

        # --- ZHANG CONIC EQUATION ---
        m11 = h11 * h12
        m12 = h21 * h22
        m21 = h11 ** 2 - h12 ** 2
        m22 = h21 ** 2 - h22 ** 2

        n1 = -h31 * h32
        n2 = (h32 ** 2 - h31 ** 2)

        # Only pool frames that carry a robust non-degenerate perspective rank.
        det_M = m11 * m22 - m12 * m21
        is_frame_perspective = False
        if abs(det_M) > MIN_PERSPECTIVE_DET_MAGNITUDE:
            M_local = np.array([[m11, m12], [m21, m22]], dtype=np.float64)
            N_local = np.array([n1, n2], dtype=np.float64)
            try:
                g_local = np.linalg.solve(M_local, N_local).flatten()
                gx, gy = float(g_local[0]), float(g_local[1])
                # Dynamic inverse squares check matching camera bounds
                G_MIN_BOUND = 1.0 / (MAX_PHYSICAL_FOCAL_PX ** 2)
                G_MAX_BOUND = 1.0 / (MIN_PHYSICAL_FOCAL_PX ** 2)

                if G_MIN_BOUND < gx < G_MAX_BOUND and G_MIN_BOUND < gy < G_MAX_BOUND:
                    fx = np.sqrt(1.0 / gx)
                    fy = np.sqrt(1.0 / gy)
                    # Ensure the frame yields a realistic pixel aspect ratio before pooling
                    if MIN_PIXEL_ASPECT_RATIO < (fx / fy) < MAX_PIXEL_ASPECT_RATIO:
                        is_frame_perspective = True
            except np.linalg.LinAlgError:
                pass

        frob_norm_M = np.sqrt(m11 ** 2 + m12 ** 2 + m21 ** 2 + m22 ** 2)
        scale = 1.0 / frob_norm_M
        dot_h1_h2 = h11 * h12 + h21 * h22
        norm_h1 = (h11 ** 2 + h21 ** 2)
        norm_h2 = (h12 ** 2 + h22 ** 2)

        M_iso.append([n1 * scale])
        M_iso.append([n2 * scale])
        N_iso.append(dot_h1_h2 * scale)
        N_iso.append((norm_h1 - norm_h2) * scale)

        if is_frame_perspective:
            # Isotropic equations are compiled STRICTLY for verified views to protect
            # the fallback least-squares matrix from un-filtered affine noise.
            M_diag.append([m11 * scale, m12 * scale])
            M_diag.append([m21 * scale, m22 * scale])
            N_diag.append(n1 * scale)
            N_diag.append(n2 * scale)

        else:
            print(f" -> [INFO] Frame {frame_idx} dropped from joint pool due to flat affine degeneracy.")


    if len(N_iso) < 2:
        fail_registry = {
            "status": "failed",
            "message": "Multi-view constraint matrix degraded."
        }
        return fail_registry

    # --- OVERDETERMINED ASYMMETRIC RECOVERY (gx, gy) ---
    if len(N_diag) > 2:
        try:
            M_diag_global = np.array(M_diag, dtype=np.float64)
            N_diag_global = np.array(N_diag, dtype=np.float64).flatten()
            g, _, _, _ = np.linalg.lstsq(M_diag_global, N_diag_global, rcond=None)
            g = g.flatten()
            gx, gy = g[0], g[1]

            G_MIN_BOUND = 1.0 / (MAX_PHYSICAL_FOCAL_PX ** 2)
            G_MAX_BOUND = 1.0 / (MIN_PHYSICAL_FOCAL_PX ** 2)

            if G_MIN_BOUND < gx < G_MAX_BOUND and G_MIN_BOUND < gy < G_MAX_BOUND:
                fx_final = np.sqrt(1.0 / gx)
                fy_final = np.sqrt(1.0 / gy)

                if MIN_PIXEL_ASPECT_RATIO < (fx_final / fy_final) < MAX_PIXEL_ASPECT_RATIO:
                    print(
                        f" -> [SUCCESS] Joint Zhang calibration fx: {fx_final:.2f}, fy: {fy_final:.2f} over {len(N_diag)//2} views.")
                    return {"status": "success", "mode": "perspective", "fx": fx_final, "fy": fy_final,
                            "message": "Multi-view diagonal solved successfully."}
                else:
                    print(
                        f" -> [WARNING] Joint Zhang calibration fx: {fx_final:.2f}, fy: {fy_final:.2f} fails pixel aspect ratio bounds.")
            else:
                print(f" -> [WARNING] Joint Zhang calibration focal length fails bounds.")
        except Exception:
            pass

    # --- ISOTROPIC FALLBACK ---
    print(f" -> fx<>fy case has unstable solution. Running Isotropic fx=fy=f on {len(N_iso)//2 } views...")
    M_iso_global = np.array(M_iso, dtype=np.float64)
    N_iso_global = np.array(N_iso, dtype=np.float64).flatten()
    try:
        g_iso, _, _, _ = np.linalg.lstsq(M_iso_global, N_iso_global, rcond=None)
        f2 = float(g_iso.flatten())

        if f2 < 0.0:
            g_iso_neg, _, _, _ = np.linalg.lstsq(M_iso_global, -N_iso_global, rcond=None)
            f2 = float(g_iso_neg.flatten())

        F2_MAX_BOUND = (MAX_PHYSICAL_FOCAL_PX ** 2)
        F2_MIN_BOUND = (MIN_PHYSICAL_FOCAL_PX ** 2)
        if F2_MIN_BOUND < f2 < F2_MAX_BOUND:
            f_iso_final = np.sqrt(f2)
            print(f" -> Uniform fx=fy calibration f: {f_iso_final:.2f} px.")
            return {"status": "success", "mode": "perspective", "fx": f_iso_final, "fy": f_iso_final,
                    "message": "Multi-view isotropic solved successfully."}
    except Exception:
        pass

    print(" -> Multi-view planar degeneracy encountered.")
    return {
        "status": "failed",
        "mode": "degenerate",
        "message": "Projective perspective matrix components degraded into absolute rank-deficiency."
    }
#
# ==========================================================================================================
# Radial distortion estimators
# 1. Nelder-Mead
# 2. cx,cy grid with 1d Brent optimizer (slow but gives high precision)
# 3. Coarse-fine Nelder-Mead estimator - apply twice, using internal input tetrahedron normalization property
# ===========================================================================================================


def strategy_nelder_mead(frame: FrameBundle, master_cam: ProjectiveCamera, coarse=True) -> tuple:
    """
    Preserved Classic Nelder-Mead Optimization Strategy.
    Returns: tuple -> (solved_k1_ap, solved_cx, solved_cy)
    """
    img_shape = master_cam.img_shape
    r_ap = master_cam.aperture_radius()

    # Anchor points map strictly to the exact physical mid-points of our search intervals.
    BARREL_K1_RANGE = 0.7
    k1_mid = -BARREL_K1_RANGE/2
    k1_radius = abs(k1_mid) - 1.e-6

    # Center tracking boundaries are locked to a fixed 20x20px envelope around the true image sensor center
    cx_mid = master_cam.cx
    cy_mid = master_cam.cy
    delta_c = 10.
    loss_type = "menger"
    step = 2
    if not coarse:
        k1_mid = master_cam.k1
        k1_radius *= 0.1
        delta_c = 3
        step = 3

    def aperture_menger_objective(params: list) -> float:
        k_norm, cx_norm, cy_norm = params

        k1_ap_cand = k1_mid + (k_norm * 2.0 * k1_radius)
        cx_cand = cx_mid + (cx_norm * 2.0 * delta_c)
        cy_cand = cy_mid + (cy_norm * 2.0 * delta_c)

        if k1_ap_cand > -1.e-6:
            return 1.e6
        modified_cam = ProjectiveCamera(
            img_shape=img_shape,
            fx_px=r_ap, fy_px=r_ap,
            cx=cx_cand, cy=cy_cand, k1=k1_ap_cand,
        )
        loss = menger_curvature_loss(frame, cam=modified_cam, loss=loss_type, step=step)
        print(f"loss={loss:.4f} cx={cx_cand:.2f} cy={cy_cand:.2f} k1={k1_ap_cand:.6f} k_norm={k_norm:.3f}]")
        return loss

    # Every parameter axis starts exactly aligned at 0.0, representing the interval centers
    x0 = [0.0, 0.0, 0.0]

    # 4. CONSTRUCT A PERFECTLY ISOTROPIC REGULAR TETRAHEDRON
    # Since all parameters have a matching scale weight, a uniform geometric step
    # creates a non-degenerated regular simplex in R^3 space.
    step_size = 0.1  # Moves parameters by exactly 20% of their allowed span bounds

    v0 = [x0[0] + step_size, x0[1] + step_size, x0[2] + step_size]
    v1 = [x0[0] + step_size, x0[1] - step_size, x0[2] - step_size]
    v2 = [x0[0] - step_size, x0[1] + step_size, x0[2] - step_size]
    v3 = [x0[0] - step_size, x0[1] - step_size, x0[2] + step_size]

    menger_simplex = np.array([v0, v1, v2, v3], dtype=np.float64)
    opt_res = scipy.optimize.minimize(
        fun=aperture_menger_objective,
        x0=x0,
        method='Nelder-Mead',
        options={ 'initial_simplex': menger_simplex,
            'xatol': 1e-3, 'fatol': 1e-6, 'maxiter': 100, 'disp': True
        }
    )
    if not opt_res.success:
        raise ValueError("Nelder-Mead loose convergence criteria reached on this frame pass.")

    # Unpack the raw optimized metrics from the result vector
    solved_k_norm, solved_cx_norm, solved_cy_norm = opt_res.x

    # Re-map the optimized normalized outputs to global master parameters
    solved_k1_ap = k1_mid + (solved_k_norm * 2.0 * k1_radius)
    solved_cx = cx_mid + (solved_cx_norm * 2.0 * delta_c)
    solved_cy = cy_mid + (solved_cy_norm * 2.0 * delta_c)
    return solved_k1_ap, solved_cx, solved_cy


def strategy_cascade_search(frame, master_cam) -> tuple:

    k1_coarse, cx_coarse, cy_coarse = strategy_nelder_mead(frame, master_cam)

    coarse_cam = ProjectiveCamera(
        img_shape=master_cam.img_shape,
        fx_px=master_cam.aperture_radius(), fy_px=master_cam.aperture_radius(),
        cx=cx_coarse, cy=cy_coarse,
        k1=k1_coarse)

    return strategy_nelder_mead(frame, coarse_cam, coarse=False)


def strategy_brent_grid(frame:FrameBundle, master_cam:ProjectiveCamera) -> tuple:
    BARREL_K1_RANGE = 0.7
    r_ap = master_cam.aperture_radius()
    nominal_cx = master_cam.cx
    nominal_cy = master_cam.cy

    # Global trackers to trap the absolute analytical minimum
    best_global_loss = float('inf')
    best_final_cx = nominal_cx
    best_final_cy = nominal_cy
    best_final_k1_ap = -BARREL_K1_RANGE/2  # Midpoint reference default

    GRID_RADIUS_PX = 10
    GRID_STRIDE_PX = 3

    last_k1 = np.full((2 * GRID_RADIUS_PX + 1), best_final_k1_ap, dtype=np.float)
    k1_grid = np.zeros((2 * GRID_RADIUS_PX + 1, 2 * GRID_RADIUS_PX + 1), dtype=np.float)

    row = 0
    for dy_px in range(-GRID_RADIUS_PX, GRID_RADIUS_PX + 1, GRID_STRIDE_PX):
        col = 0
        cy = nominal_cy + dy_px
        for dx_px in range(-GRID_RADIUS_PX, GRID_RADIUS_PX + 1, GRID_STRIDE_PX):
            cx = nominal_cx + dx_px
            def brent_objective(k1_ap_cand: float) -> float:
                virtual_aperture_cam = ProjectiveCamera(
                    img_shape=master_cam.img_shape,
                    fx_px=r_ap, fy_px=r_ap,
                    cx=cx, cy=cy,
                    k1=k1_ap_cand
                )
                return menger_curvature_loss(frame, cam=virtual_aperture_cam)

            RADIUS = BARREL_K1_RANGE/20
            k1 = (last_k1[col] + last_k1[col - 1]) * 0.5 if col > 0 else last_k1[col]
            lower_bound = max(-0.5, k1 - RADIUS)
            upper_bound = min(-1e-6, k1 + RADIUS)
            k1_grid[row,col] = k1
            # High-speed scalar minimization pass using inverse parabolic interpolation
            local_res = scipy.optimize.minimize_scalar(
                fun=brent_objective, bounds = (lower_bound, upper_bound),
                     method = 'bounded', options = {'xatol': 1e-3, 'maxiter': 10, 'disp': False}
            )

            # Cache this node's exact output so the next pixel neighbor inherits the solution instantly
            k1 = float(local_res.x)
            last_k1[col] = k1
            col += 1
            # Track if this specific grid node uncovers the absolute global minimum
            if local_res.fun < best_global_loss:
                best_global_loss = local_res.fun
                best_final_k1_ap = k1
                best_final_cx = cx
                best_final_cy = cy
        row += 1
    print(k1_grid)
    return best_final_k1_ap, best_final_cx, best_final_cy


class MultiFrameCalibrator:
    """
    Stateful calibration accumulator. Collects and processes frames
    sequentially, caching optimized aperture parameters and vanishing points,
    and dynamically selects single-frame or multi-frame math engines on-demand.
    """

    def __init__(self, camera_object: ProjectiveCamera, N: int = 12, MIN_LEN: int = 15):
        """
        camera_object: Baseline master camera instance providing sensor dimensions and state.
        N, MIN_LEN   : Structural extraction settings passed to the underlying FrameBundle.
        :return
        status dictionary compatible with camera_io
        """
        self.master_cam = camera_object
        self.img_shape = camera_object.img_shape
        self.orig_fx = camera_object.fx_px
        self.orig_fy = camera_object.fy_px
        self.orig_k1_ap = 0.
        self.N = N
        self.MIN_LEN = MIN_LEN

        # Stateful cache arrays to accumulate tracked metrics over time
        self.cached_k1_ap = []
        self.cached_cx = []
        self.cached_cy = []
        self.frames = []
        self.optimization_strategy = strategy_cascade_search

    def add_frame(self, topological_matrix: np.ndarray, detected_points: np.ndarray, point_weights: np.ndarray=None) -> dict:
        """
        Processes a single frame: solves localized [k1_ap, cx, cy] via isolated
        Menger curvature, rectifies line tracks, caches the resolved vanishing
        points, and immediately returns the localized tracking state.
        """
        frame = FrameBundle(topological_matrix, detected_points, point_weights, max_lines_per_axis=self.N, min_len=self.MIN_LEN)

        if len(frame.selected_lines) < 5:
            print(" -> [Warning] Skipping added frame: insufficient lines.")
            return {"status": "failed", "message": "Insufficient line density harvested."}

        try:
            solved_k1_ap, solved_cx, solved_cy = self.optimization_strategy(frame, self.master_cam)
        except Exception as e:
            print(f" -> [Warning] External optimization strategy failed. Error: {str(e)}")
            return {"status": "failed", "message": "External optimization engine exception."}

        # 5. Push extracted metrics straight into the accumulator cache state
        self.cached_k1_ap.append(solved_k1_ap)
        self.cached_cx.append(solved_cx)
        self.cached_cy.append(solved_cy)
        self.frames.append(frame)
        print(f" -> [SUCCESS] Radial distortion cx={solved_cx} cy={solved_cy} k1_ap={solved_k1_ap}")

        return {
            "status": "success",
            "local_cx": solved_cx,
            "local_cy": solved_cy,
            "local_k1_ap": solved_k1_ap
        }

    def calibrate(self) -> dict:
        """
        Finalizes camera calibration. Dynamically selects single-frame or multi-frame
        matrix engines on-demand based on the total accumulated frame count.
        """
        if len(self.frames) < 1:
            print(" -> [ERROR] Calibration aborted: The accumulator cache buffer contains zero valid frames.")
            return {
                "status": "failed",
                "message": "Calibration failed because no valid target tracking frames have been accumulated yet."
            }

        mean_cx = np.median(self.cached_cx)
        mean_cy = np.median(self.cached_cy)
        mean_k1_ap = np.median(self.cached_k1_ap)
        r_ap=self.master_cam.aperture_radius()
        aperture_cam = ProjectiveCamera(
            self.img_shape, r_ap, r_ap,
            mean_cx, mean_cy, mean_k1_ap,
            self.master_cam.mode
        )

        print(f"\n -> Finalizing Multi-Frame Calibration: Processing {len(self.frames)} pooled views...")
        global_multi_view_vps = []
        for frame in self.frames:
            vanishing_points = frame.estimate_vp(aperture_cam)
            print(vanishing_points)
            global_multi_view_vps.append(vanishing_points)

        if len(self.frames) == 1:
            zhang_res = solve_weak_perspectivity_matrix(global_multi_view_vps[0], cx=mean_cx, cy=mean_cy)
        else:
            print(" -> Run the joint Zhang multi-frame engine...")
            zhang_res = solve_multi_frame_zhang_matrix(global_multi_view_vps, cx=mean_cx, cy=mean_cy)

        if zhang_res["status"] != "success":
            print(f" -> [ERROR] Matrix solver failed to resolve structural constraints: {zhang_res['message']}")
            return {"status": "failed", "message": "Intrinsic focal matrix calculation pass failed."}

        # Handle projection mode and protect against weak-perspective / affine infinitum loops
        final_mode=zhang_res["mode"]
        if final_mode == "perspective":
            fx_px = zhang_res["fx"]
            fy_px = zhang_res["fy"]
        else:
            print(" -> [INFO] Weak-perspectivity identified. Retaining original baseline focal scales.")
            return {"status": "failed", "message": "Weak-perspectivity identified."}

        final_cam = ProjectiveCamera(
            self.img_shape, fx_px, fy_px,
            mean_cx, mean_cy
        )
        k1 = final_cam.convert_aperture_to_focal(mean_k1_ap)
        final_cam.set_distortion(k1)
        print("\n=======================================================")
        print("      GLOBAL HGP PIPELINE CALIBRATION SUCCESS          ")
        print("=======================================================")
        print(f" -> Mode Selected                : {final_mode.upper()} ENGINE")
        print(f" -> Total Processed Views        : {len(self.frames)}")
        print(f" -> Solved Focal Length (fx,fy)  : {final_cam.fx_px:.2f}, {final_cam.fy_px:.2f}")
        print(f" -> Solved Principal Point(cx,cy): {final_cam.cx:.2f}, {final_cam.cy:.2f}")
        print(f" -> Resolved Aperture k1         : {mean_k1_ap:.8f}")
        print(f" -> Resolved Brown-Conrady k1    : {final_cam.k1:.8f}")
        print("=======================================================\n")

        return {
            "status": "success",
            "mode": final_cam.mode,
            "img_shape" : final_cam.img_shape,
            "fx": final_cam.fx_px, "fy": final_cam.fy_px,
            "cx": final_cam.cx, "cy": final_cam.cy,
            "radial_k1_ap": mean_k1_ap, "k1": final_cam.k1
        }


def _extract_axis_lines(axis_map: dict,
                        all_harvested_lines: list,
                        min_len: int,
                        max_count: int) -> list:
    """
    Private helper that filters tracks by minimum length, stores them in the global
    harvest collection, and returns the top N longest lines sorted lexicographically.
    """
    axis_metadata = []

    # Initialize the tracking index once per axis execution step from the active list length
    current_line_idx = len(all_harvested_lines)

    for _, line_track in axis_map.items():
        line_len = len(line_track)
        if line_len >= min_len:
            all_harvested_lines.append(line_track)
            # Tuple structure is (line_len, original_index) for automatic native length sorting
            axis_metadata.append((line_len, current_line_idx))
            current_line_idx += 1

    # Native Python tuple sort (no lambda needed, keeps tuple row bounds intact)
    axis_metadata.sort()
    target_slice = axis_metadata[-max_count:] if len(axis_metadata) > max_count else axis_metadata

    axis_output_lines = []
    for _, idx in target_slice:
        # Fetch the raw tuple list from storage
        raw_line = all_harvested_lines[idx]
        # 1. Sort the tuple list in-place. Python automatically sorts by
        # the first element (your r or c sorting index)!
        raw_line.sort()
        # 2. Extract only the clean point_idx values from the sorted tuples list
        clean_sorted_line = [point_idx for _, point_idx in raw_line]
        axis_output_lines.append(clean_sorted_line)

    return axis_output_lines


def _harvest_hexagonal_line_bundles(topological_matrix: np.ndarray,
                                   N: int = 16,
                                   MIN_LEN: int = 5) -> tuple:
    """
    Scans a decoded hexagonal tracking canvas matrix and extracts the longest
    three-diagonal straight line index bundles separately for each axis.

    Variables Description:
        topological_matrix (np.ndarray): The final decoded canvas matrix frame, shape (H, W).
                                         Maps cell positions to sub-pixel point IDs.
                                         Empty tracking voids carry -1 tokens.
        N (int)                        : Maximum number of top longest lines to extract per axis.
        MIN_LEN (int)                  : Minimum number of aligned points required to qualify as a line.

    Returns:
        tuple: (u_lines, v_lines, w_lines) where each is a list of index trajectories:
               [[id1, id2, ...], [id3, id4, ...]]
    """
    H_nodes, W_nodes = topological_matrix.shape

    u_map = {}
    v_map = {}
    w_map = {}
    r_map = {}
    # Step 1: Map every active matrix node into its absolute barycentric tracking lane
    # Natural matrix sweep order automatically ensures points are sorted along lines
    for r in range(H_nodes):
        for c in range(W_nodes):
            point_idx = topological_matrix[r, c]
            if point_idx < 0:
                continue

            # Unwarp storage coordinates to the invariant linear domain
            u_linear = c - (r // 2)
            v_linear = r
            w_linear = -v_linear - u_linear

            if v_linear not in u_map:
                u_map[v_linear] = []
            u_map[v_linear].append((u_linear, point_idx))

            if w_linear not in v_map:
                v_map[w_linear] = []
            v_map[w_linear].append((v_linear, point_idx))

            if u_linear not in w_map:
                w_map[u_linear] = []
            w_map[u_linear].append((w_linear, point_idx))
            if r%2 == 0:
                if 2 * c not in r_map:
                    r_map[2 * c] = []
                r_map[2 * c].append((r, point_idx))
            else:
                if 2 * c + 1 not in r_map:
                    r_map[2 * c + 1] = []
                r_map[2 * c + 1].append((r, point_idx))

    all_harvested_lines = []
    # Process all three directional maps sequentially, length initializes internally
    u_lines = _extract_axis_lines(u_map, all_harvested_lines, MIN_LEN, N)
    v_lines = _extract_axis_lines(v_map, all_harvested_lines, MIN_LEN, N)
    w_lines = _extract_axis_lines(w_map, all_harvested_lines, MIN_LEN, N)
    r_lines = _extract_axis_lines(r_map, all_harvested_lines, MIN_LEN, N)

    return u_lines, v_lines, w_lines, r_lines


def _test_solve_perspectivity():
    """
    Validation test suite using strict assertions.
    Simulates physical projections through randomized 3D orientation spaces.
    """
    # 1. Setup Ground-Truth Intrinsics Matrix K (Asymmetric focal scales)
    gt_fx = 850.50
    gt_fy = 920.25
    gt_cx = 640.0
    gt_cy = 480.0

    K = np.array([
        [gt_fx, 0.0, gt_cx],
        [0.0, gt_fy, gt_cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 2. Setup synthetic orthogonal 3D rotation matrix via unique Euler steps
    alpha, beta, gamma = np.radians(25.0), np.radians(-15.0), np.radians(35.0)
    R_x = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    R_y = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    # Final unified SO(3) 3D camera rotation matrix
    R = R_z @ R_y @ R_x

    # 3. Reference Hexagonal Continuum Directions in 3D (Z = 0 plane)
    sqrt3_over2 = np.sqrt(3.0) / 2.0
    model_directions = [
        np.array([1.0, 0.0, 0.0]),  # 0 deg
        np.array([0.5, sqrt3_over2, 0.0]),  # 60 deg
        np.array([-0.5, sqrt3_over2, 0.0]),  # 120 (-60) deg
        np.array([0.0, 1.0, 0.0])  # 90 deg
    ]

    # 4. Generate homogenous un-normalized test points using projection models
    vp_list = []
    for D in model_directions:
        # Step A: Transform orientation vectors via 3D rotation space
        rotated_D = R @ D

        # Step B: Project vector coordinates directly to image camera matrix
        projected_vp = K @ rotated_D

        # Step C: Inject randomized non-zero scalars to test scale invariance
        random_scale = np.random.uniform(0.5, 5.0)
        vp_list.append(projected_vp * random_scale)

    # 5. Run the target intrinsics reconstruction pipeline
    output = solve_weak_perspectivity_matrix(vp_list, gt_cx, gt_cy)

    # 6. Execute strict verification assertions
    assert output["status"] == "success", "Focal execution failed to converge."
    assert output["mode"] == "perspective", "Incorrect geometry estimation model matched."

    # Verify sub-pixel structural alignment tolerance (atol = 0.01 px)
    assert np.isclose(output["fx"], gt_fx, atol=1e-2), f"Expected fx={gt_fx}, but got {output['fx']}"
    assert np.isclose(output["fy"], gt_fy, atol=1e-2), f"Expected fy={gt_fy}, but got {output['fy']}"

    print("Precision limits verified successfully. PASSED")


def _test_solve_multi_frame_perspectivity():
    """
    Validation test suite using strict multi-frame assertions.
    Simulates 7 independent physical viewpoints moving through a 3D spherical
    orientation space to verify the multi-view joint least-squares pipeline.
    """
    # 1. Setup Ground-Truth Intrinsics Matrix K (Asymmetric focal scales)
    gt_fx = 1250.0
    gt_fy = 1150.0
    gt_cx = 965.0
    gt_cy = 543.0

    K = np.array([
        [gt_fx, 0.0, gt_cx],
        [0.0, gt_fy, gt_cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Base reference hexagonal continuum axes directions in 3D (Z = 0 plane)
    sqrt3_over2 = np.sqrt(3.0) / 2.0
    model_directions = [
        np.array([1.0, 0.0, 0.0]),  # 0 deg
        np.array([0.5, -sqrt3_over2, 0.0]),  # 60 deg (synchronized configuration)
        np.array([0.5, sqrt3_over2, 0.0]),  # 120 deg
        np.array([0.0, -1.0, 0.0])  # 90 deg
    ]

    # Initialize the multi-view master accumulation database
    multi_frame_dataset = []

    # Generate 7 unique viewpoints with different perspective slope sweeps
    # Fix local random seed to ensure this synthetic data sweep remains deterministic
    np.random.seed(42)
    NUM_SYNTHETIC_VIEWS = 7

    for view_idx in range(NUM_SYNTHETIC_VIEWS):
        # Generate varied compound angle sweeps to force non-zero h31, h32 metrics
        # We ensure a minimum tilt angle to maintain high projective rank matrix constraints
        pitch = np.radians(np.random.uniform(15.0, 45.0))
        yaw = np.radians(np.random.uniform(5.0, 20.0))
        roll = np.radians(np.random.uniform(0.0, 360.0))

        # Explicitly build independent Euler orientation arrays
        R_x = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        R_z = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])

        # Combine rotation matrices matching your camera coordinate transform order
        R = R_z @ R_x @ R_y

        # Local frame point channel buffer
        current_frame_vps = []

        # Process each canonical track direction under this viewpoint
        for d in model_directions:
            # Map coordinates to pixels through the camera intrinsics tensor K
            projected_vp = K @ R @ d

            # Inject a random scale factor to test projective scale invariance (\lambda_k)
            random_scale = np.random.uniform(0.5, 5.0)
            current_frame_vps.append(projected_vp * random_scale)

        # Cash current viewpoint channels into the master pooled dataset
        multi_frame_dataset.append(current_frame_vps)

    # =====================================================================
    # EXECUTE TARGET JOINT SOLVER AND VERIFY METRICS
    # =====================================================================
    # Run the overdetermined multi-view least-squares reconstruction pipeline
    output = solve_multi_frame_zhang_matrix(multi_frame_dataset, gt_cx, gt_cy)

    # Execute strict photogrammetric verification assertions
    assert output["status"] == "success", "Multi-view joint matrix failed to converge."
    assert output["mode"] == "perspective", "Joint solver defaulted down to wrong projection model."

    # Verify sub-pixel calibration precision limits (atol = 0.01 pixels tolerance)
    assert np.isclose(output["fx"], gt_fx, atol=1e-2), f"Expected fx={gt_fx}, but got {output['fx']}"
    assert np.isclose(output["fy"], gt_fy, atol=1e-2), f"Expected fy={gt_fy}, but got {output['fy']}"

    print(f"Multi-frame precision verified across {NUM_SYNTHETIC_VIEWS} views. PASSED")


if __name__ == "__main__":
    _test_solve_perspectivity()
    _test_solve_multi_frame_perspectivity()
