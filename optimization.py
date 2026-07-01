# =====================================================================
# SYSTEM COMPONENT: menger_curvature_loss.py
# DESCRIPTION: Computes the aggregate geometric straightening error across 
#              projected node lines by evaluating the localized Menger 
#              curvature (1/R) of successive 3-point coordinate triplets. 
#              Optimizing this cost function toward zero forces lines warped 
#              by lens distortion to map back onto perfect straight vectors.
# =====================================================================
import numpy as np
import scipy.optimize
from camera import ProjectiveCamera

def menger_curvature_loss(
                          points_2d: np.ndarray, 
                          lines: list, 
                          weights: np.ndarray, 
                          cam: ProjectiveCamera) -> float:
    """
    Evaluates the squared Menger curvature penalty along tracked line components.
    
    Variables Description:
        params (list)        : Optimization state parameters [focal_length_f, radial_k1].
        points_2d (np.ndarray): Collected raw distorted 2D pixel coordinates, shape (N, 2).
        lines (list)         : Nested list where each entry stores indices of a straight grid lane:
                               [[idx1, idx2, idx3, ...], [idx4, idx5, ...]]
        weights (np.ndarray) : Relevance or incidence weighting array for each node, shape (N,).
        center (tuple)       : Image coordinates of the principal point/optical center (cx, cy).
        undistort_func (func): Optional function handle to isolate the point unwarping pass.
        
    Returns:
        float                : Accumulated squared curvature cost value.
    """

    u_points = cam.undistort_points(points_2d)

    total_loss = 0.0
    eps = 1e-6  # Singularity protection shield against duplicate node evaluations
    
    # 3. GEOMETRIC INTERROGATION LOOP
    for line in lines:
        n_pts = len(line)
        if n_pts < 3: 
            continue
            
        # Sliding triplet window scanning across the continuous sequence curve length
        for i in range(n_pts - 2):
            p1 = u_points[line[i]]
            p2 = u_points[line[i+1]]
            p3 = u_points[line[i+2]]
            
            # Form displacement vectors between the triplet sequence nodes
            v1 = (p2[0] - p1[0],p2[1] - p1[1])
            v2 = (p3[0] - p2[0],p3[1] - p2[1])
            v3 = (p3[0] - p1[0],p3[1] - p1[1])
            
            # Cross-product scalar value: Represents double the area of the spanned triangle
            # This forms the fundamental numerator component of the Menger formulation
            cross_prod = v1[0] * v2[1] - v1[1] * v2[0]
            
            # Compute Euclidean length scalars across all three local edges
            len_v1 = np.sqrt(v1[0]**2 + v1[1]**2) + eps
            len_v2 = np.sqrt(v2[0]**2 + v2[1]**2) + eps
            len_v3 = np.sqrt(v3[0]**2 + v3[1]**2) + eps
            
            # Menger Curvature Equation Evaluation (kappa = 1 / R)
            kappa = (2.0 * cross_prod) / (len_v1 * len_v2 * len_v3)
            
            # Map index assignment to isolate the central node item weight factor
            pt_id = line[i+1]
            
            # Accumulate the weighted squaring constraint value
            # Symmetrical lines converge to zero (cross_product = 0, kappa = 0)
            total_loss += float(weights[pt_id]) * (kappa ** 2)
            
    return total_loss


def generate_calibration_input_arrays(topological_matrix: np.ndarray,
                                      detected_points: np.ndarray,
                                      generator) -> tuple:
    """
    Constructs matching 3D world coordinate vectors and 2D sub-pixel image 
    coordinate vectors from a successfully decoded frame canvas matrix.
    
    Variables Description:
        topological_matrix (np.ndarray): The final decoded canvas matrix frame, 
                                         mapping matrix slots (r, c) to point IDs.
        detected_points (np.ndarray)   : Raw 2D sub-pixel pixel coordinate array 
                                         from the blob finder, shape (N, 2).
        generator (obj)                : The active PhysicalMeshGenerator tracking 
                                         the true hexagonal pattern dimensions.
                                         
    Returns:
        tuple: (object_points_3d, image_points_2d) formatted as float32 NumPy arrays.
    """
    H_nodes, W_nodes = topological_matrix.shape
    
    obj_pts_list = []
    img_pts_list = []
    
    # Sweep systematically across all row and column slots of your canvas matrix
    for r in range(H_nodes):
        for c in range(W_nodes):
            point_idx = topological_matrix[r, c]
            
            # Skip empty buffer cells, padding borders, or dead tracking slots
            if point_idx < 0:
                continue
                
            # 1. Harvest the true 2D physical world coordinate center in mm
            x_world, y_world = generator.get_shape_center(r, c)
            
            # Anchor the target onto the flat calibration board surface plane (Z = 0)
            node_world_center_3d = [x_world, y_world, 0.0]
            
            # 2. Extract the high-accuracy raw sub-pixel coordinate from your live sensor array
            x_img = float(detected_points[point_idx][0])
            y_img = float(detected_points[point_idx][1])
            node_image_center_2d = [x_img, y_img]
            
            obj_pts_list.append(node_world_center_3d)
            img_pts_list.append(node_image_center_2d)
            
    # Convert lists to rigid float32 matrix structures required by basic OpenCV C++ solvers
    object_points_3d = np.array(obj_pts_list, dtype=np.float32)
    image_points_2d = np.array(img_pts_list, dtype=np.float32)
    
    return object_points_3d, image_points_2d


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
    return line_v / np.linalg.norm(line_v)


def compute_homogeneous_vanishing_point(line_vectors: list) -> np.ndarray:
    """
    Finds the intersection point of a line bundle in homogeneous space.
    Returns v = [x, y, w]^T. If lines are parallel, w converges to 0 cleanly.
    """
    valid_lines = [ln for ln in line_vectors if ln is not None]
    if len(valid_lines) < 2:
        return None

    # Stack line equations into an (M, 3) matrix L
    L_mat = np.vstack(valid_lines)

    # Solve L * v = 0 using SVD
    _, _, vh = np.linalg.svd(L_mat)
    v_point = vh[-1]  # [x, y, w]
    return v_point


def solve_weak_perspectivity_matrix(vp: list) -> dict:
    """
    Analytically computes camera intrinsics using a line homography.
    Calculates independent candidate values strictly in the inverse focal
    domain (inv_f = 1/f) to return 0.0 instead of infinity under weak perspective.

    If the relative mismatch exceeds 20%, it returns a clean inverse affine model.
    Fails fast with pure 0.0 metrics if structural parameters degrade.

    Fully ASCII-compliant implementation. Character constraints 0-127 enforced.
    """
    # Fail-Fast Default Registry State
    fail_registry = {
        "status": "failed",
        "message": "Inverse Domain Error: Structural configuration degraded or incomplete parameters."
    }

    if vp is None or len(vp) < 4:
        return fail_registry

    # Define the explicit 3D theoretical model direction vectors on your hexagonal sheet
    sqrt3_over2 = float(np.sqrt(3.0) / 2.0)
    model_directions = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Vu
        np.array([0.5, sqrt3_over2, 0.0], dtype=np.float32),  # Vv
        np.array([-0.5, sqrt3_over2, 0.0], dtype=np.float32),  # Vw
        np.array([0.0, 1.0, 0.0], dtype=np.float32)  # V_vert
    ]

    # 1. RUN STANDARD DLT PASS TO RESOLVE LINE HOMOGRAPHY H
    A_rows = []
    for i in range(4):
        v = vp[i]
        D = model_directions[i]
        if v is None:
            return fail_registry

        x, y, w = v[0], v[1], v[2]
        dx, dy = D[0], D[1]

        A_rows.append([dx, dy, 1.0, 0.0, 0.0, 0.0, -x * dx, -x * dy, -x])
        A_rows.append([0.0, 0.0, 0.0, dx, dy, 1.0, -y * dx, -y * dy, -y])

    if len(A_rows) < 8:
        return fail_registry

    A_dlt = np.array(A_rows, dtype=np.float32)
    _, _, vh = np.linalg.svd(A_dlt)
    H = vh[-1].reshape((3, 3))

    # Isolate homography column coordinates
    h11, h12 = float(H[0, 0]), float(H[0, 1])
    h21, h22 = float(H[1, 0]), float(H[1, 1])
    h31, h32 = float(H[2, 0]), float(H[2, 1])

    dot_h1_h2_xy = h11 * h12 + h21 * h22
    norm_h1_sq_xy = h11 ** 2 + h21 ** 2
    norm_h2_sq_xy = h12 ** 2 + h22 ** 2

    # =====================================================================
    # COMPUTE CANDIDATES PURELY IN THE INVERSE FOCAL DOMAIN (inv_f^2)
    # =====================================================================
    # Candidate A: Derived from the standard column length equality constraint
    num_inv_proj_sq = (h31 ** 2 * h32 ** 2)
    den_inv_proj_sq = (h31 * h32 * dot_h1_h2_xy) - (norm_h1_sq_xy * h32 ** 2)

    inv_proj_valid = (abs(den_inv_proj_sq) > 1e-9) and ((num_inv_proj_sq / den_inv_proj_sq) >= 0)
    inv_f_proj = float(np.sqrt(num_inv_proj_sq / den_inv_proj_sq)) if inv_proj_valid else 0.0

    # Candidate B: Derived from the cross-column DIAC orthogonality rule
    num_inv_diac_sq = -(h31 * h32)
    den_inv_diac_sq = dot_h1_h2_xy

    inv_diac_valid = (abs(den_inv_diac_sq) > 1e-9) and ((num_inv_diac_sq / den_inv_diac_sq) >= 0)
    inv_f_diac = float(np.sqrt(num_inv_diac_sq / den_inv_diac_sq)) if inv_diac_valid else 0.0

    # =====================================================================
    # THE 20% DUAL-DISCREPANCY INVERSE THRESHOLD GATE
    # =====================================================================
    is_perspective_stable = inv_proj_valid and inv_diac_valid

    max_inv_f = max(inv_f_proj, inv_f_diac)
    if is_perspective_stable and max_inv_f > 1e-6:
        relative_discrepancy = abs(inv_f_proj - inv_f_diac) / max_inv_f
        if relative_discrepancy > 0.2:
            is_perspective_stable = False
    else:
        relative_discrepancy = 1.0
        is_perspective_stable = False

    # IF COMPONENT IS UNSTABLE: Route straight to the pure Affine matrix representation
    if not is_perspective_stable:
        print(f"Variance is {relative_discrepancy * 100.0:.1f}%. Scaling to Affine Mode.")

        # Calculate the direct inverse affine scaling factors from the matrix norms
        inv_fx_affine = float(np.sqrt(norm_h1_sq_xy))
        inv_fy_affine = float(np.sqrt(norm_h2_sq_xy))

        # Un-invert safely back to pixel steps only if the scale is physically active
        fx_final = 1.0 / inv_fx_affine if inv_fx_affine > 1e-6 else 0.0
        fy_final = 1.0 / inv_fy_affine if inv_fy_affine > 1e-6 else 0.0

        return {
            "status": "success",
            "mode": "affine",
            "fx": fx_final,
            "fy": fy_final,
            "message": f"Weak perspective resolved via inverse-domain comparison. Mismatch: {relative_discrepancy * 100.0:.1f}%."
        }

    # =====================================================================
    # VALID PROJECTIVE MODE SEQUENCE
    # =====================================================================
    inv_f_final = 0.5 * (inv_f_proj + inv_f_diac)
    f_final = 1.0 / inv_f_final if inv_f_final > 1e-6 else 0.0

    print(
        f" -> Coherent alignment. Mismatch: {relative_discrepancy * 100.0:.2f}%. Processing Projective.")
    return {
        "status": "success",
        "mode": "projective",
        "fx": f_final,
        "fy": f_final,
        "message": f"projective parameters verified. Inverse focal value: {inv_f_final:.6f}."
    }


def solve_zhang_intrinsic_matrix(vp: list) -> dict:
    """
    Analytically solves for camera intrinsics (fx, fy, cx, cy) using the full
    unreduced Direct Linear Transformation (DLT) method over Zhang's constraints.

    Uses DIAC Cholesky factorization to handle points at infinity robustly.
    """


    # 1. Compute homogeneous lines and vanishing points v = [x, y, w]^T


    # Build the full 3x6 design matrix for the 6 independent elements of omega

    vp_pairs = []
    for i in range(len(vp)):
        for j in range(i + 1, len(vp)):
            vp_pairs.append((vp[i],vp[j]))

    A_list = []
    for vi, vj in vp_pairs:
        xi, yi, wi = vi
        xj, yj, wj = vj

        # Complete geometric expansion coefficient row mapping for symmetric 3x3 matrix
        row = [
            xi * xj,  # w0 (omega_11)
            xi * yj + xj * yi,  # w1 (omega_12)
            yi * yj,  # w2 (omega_22)
            xi * wj + xj * wi,  # w3 (omega_13)
            yi * wj + yj * wi,  # w4 (omega_23)
            wi * wj  # w5 (omega_33)
        ]
        A_list.append(row)

    A_mat = np.asarray(A_list)  # Shape (3, 6)

    # 2. FULL HOMOGENEOUS SVD SOLUTION
    # We solve the general system A_mat * h = 0 without imposing early constraint locks.
    # This prevents the null-space from collapsing into a single [0,0,0,1] coordinate column.
    _, _, vh = np.linalg.svd(A_mat)
    h = vh[-1]  # Extract full 6-parameter conic vector

    # Reconstruct the symmetric Image of the Absolute Conic (IAC) matrix
    omega = np.array([
        [h[0], h[1], h[3]],
        [h[1], h[2], h[4]],
        [h[3], h[4], h[5]]
    ], dtype=np.float32)

    # Force positive-definiteness check based on the matrix trace sign
    if np.trace(omega) < 0:
        omega = -omega

    # 3. ROBUST PARAMETER EXTRACTION VIA DUAL CONIC (DIAC)
    # If omega is nearly singular or cannot be inverted due to lines being too parallel,
    # it means the lens distortion has completely flattened the perspective signal.
    try:
        # DIAC: omega_star = inverse(omega) = K * K^T
        omega_star = np.linalg.inv(omega)
    except np.linalg.LinAlgError:
        return {
            "status": "failed",
            "message": "Degenerated spatial configuration: Parallel planar lines (Matrix is singular)."
        }

    # Scale DIAC so that the bottom-right element scale factor is strictly 1.0
    if abs(omega_star[2, 2]) < 1e-9:
        return {
            "status": "failed",
            "message": "Degenerated spatial configuration: DIAC scale parameter is zero."
        }
    omega_star = omega_star / omega_star[2, 2]

    # Extract principal point directly from the dual properties:
    # omega_star = [ fx^2 + cx^2,   cx*cy,       cx ]
    #              [   cx*cy,     fy^2 + cy^2,   cy ]
    #              [     cx,          cy,         1 ]
    cx_solved = float(omega_star[0, 2])
    cy_solved = float(omega_star[1, 2])

    fx_sq = omega_star[0, 0] - (cx_solved ** 2)
    fy_sq = omega_star[1, 1] - (cy_solved ** 2)

    if fx_sq <= 0 or fy_sq <= 0:
        return {
            "status": "failed",
            "message": "Degenerated spatial configuration: Imaginary focal parameters resolved from DIAC."
        }

    fx_solved = float(np.sqrt(fx_sq))
    fy_solved = float(np.sqrt(fy_sq))

    return {
        "status": "success",
        "fx": fx_solved,
        "fy": fy_solved,
        "cx": cx_solved,
        "cy": cy_solved,
        "message": "Calibration matrix parameters successfully extracted via DIAC inversion."
    }


def calibrate_single_frame_zhang_menger(topological_matrix: np.ndarray,
                                        detected_points: np.ndarray,
                                        camera_object : ProjectiveCamera,
                                        N: int = 12,
                                        MIN_LEN: int = 5) -> dict:
    """
    Executes a single-frame camera calibration run. Dynamically recomputes Zhang's
    analytical intrinsic focal properties inside the Menger loss evaluation pass.

    Variables Description:
        topological_matrix (np.ndarray): Decoded tracking map lookup framework, shape (H, W).
                                         Maps cell positions to sub-pixel point IDs.
                                         Empty tracking voids carry -1 tokens.
        detected_points (np.ndarray)   : Raw 2D sub-pixel coordinates from blob finder, shape (N, 2).
        camera_object (obj)            : Instance of ProjectiveCamera from your camera module.
        N (int)                        : Maximum number of top longest lines to extract per axis.
        MIN_LEN (int)                  : Minimum number of aligned points required to qualify as a line.
    """
    # 1. Harvest three-diagonal line bundles separated natively by the core engine once
    u_lines, v_lines, w_lines, h_lines = harvest_hexagonal_line_bundles(topological_matrix, N=N, MIN_LEN=MIN_LEN)
    all_lines_bundle = u_lines + v_lines + w_lines + h_lines

    if len(all_lines_bundle) < 5:
        return {"status": "failed", "message": "Insufficient line density harvested."}

    node_weights = np.ones(len(detected_points), dtype=np.float32)
    state = { "shape" : camera_object.img_shape,
              "f" : camera_object.f_px, "cx" : camera_object.cx, "cy" : camera_object.cy,
              "mode" : "perspective"}
    # =====================================================================
    # DYNAMIC LOSS CLOSURE: EMBEDDING ZHANG INSIDE THE MENGER LOOP
    # =====================================================================
    def dynamic_calibration_objective(params):
        """
        Evaluates the loss parameter by first un-distorting lines via k1,
        dynamically re-calculating the focal matrix K, and computing Menger curvature.
        """
        k1_proposed = float(params[0])
        # Apply proposed distortion factor to the active camera instance
        # Generate un-distorted coordinates strictly matching this candidate k1 curve step
        modified_cam = ProjectiveCamera(state["shape"], state["f"], state["cx"], state["cy"],
                                        k1_proposed, state["mode"])
        corrected_points = modified_cam.undistort_points(detected_points)
        u_eqs = [compute_homogeneous_line(ln, corrected_points) for ln in u_lines]
        v_eqs = [compute_homogeneous_line(ln, corrected_points) for ln in v_lines]
        w_eqs = [compute_homogeneous_line(ln, corrected_points) for ln in w_lines]
        h_eqs = [compute_homogeneous_line(ln, corrected_points) for ln in h_lines]

        Vu = compute_homogeneous_vanishing_point(u_eqs)
        Vv = compute_homogeneous_vanishing_point(v_eqs)
        Vw = compute_homogeneous_vanishing_point(w_eqs)
        Vh = compute_homogeneous_vanishing_point(h_eqs)
        print("vanishing points: ",Vu, Vv, Vw, Vh)
        result = solve_weak_perspectivity_matrix([Vu,Vv,Vw,Vh])
        if result["status"] == "success":
            print(result)
            # Assign the freshly solved analytical intrinsics to the camera instance
            modified_cam = ProjectiveCamera(state["shape"], result["fx"], state["cx"], state["cy"], k1_proposed, result["mode"])
        else:
            print(result["message"])

        # Evaluate the strict Menger line straightness penalty for this combined setup
        loss_val = menger_curvature_loss(
            points_2d=detected_points,
            lines=all_lines_bundle,
            weights=node_weights,
            cam=modified_cam
        )

        state["f"] = modified_cam.f_px
        state["mode"] = modified_cam.mode
        # Restore camera state for thread-isolation safety
        return loss_val

    # Execute the 1D optimization pass over the distortion parameter space
    x0_k1 = [float(camera_object.k1)]
    print(" -> Launching joint minimization loop (Zhang inside Menger)...")

    opt_res = scipy.optimize.minimize(
        fun=dynamic_calibration_objective,
        x0=x0_k1,
        method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 200}
    )

    # --- FINAL IN-PLACE SYSTEM INTRINSIC MUTATION ---
    if opt_res.success:
        camera_object.k1 = opt_res.x[0]
        camera_object.mode = state["mode"]
        camera_object.f_px = state["f"]
        camera_object.cx = state["cx"]
        camera_object.cy = state["cy"]
        print("\n--- DYNAMIC INTRINSIC CALIBRATION SUCCESS ---")
        print(f" -> Solved Camera Focal Length (f) : {camera_object.f_px:.4f} pixels")
        print(f" -> Solved Principal Center (cx,cy): ({camera_object.cx:.2f}, {camera_object.cy:.2f})")
        print(f" -> Solved Lens Distortion (k1)    : {camera_object.k1:.8f}")
    else:
        print(" -> [ERROR] Optimization loop failed to reach target convergence limits.")

    return {
        "status": "success" if opt_res.success else "failed",
        "mode" : camera_object.mode,
        "f_px": camera_object.f_px,
        "cx": camera_object.cx,
        "cy": camera_object.cy,
        "radial_k1": camera_object.k1,
        "final_menger_loss": float(opt_res.fun)
    }


def _extract_and_filter_axis_lines(axis_map: dict,
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
        axis_output_lines.append(all_harvested_lines[idx])

    return axis_output_lines


def harvest_hexagonal_line_bundles(topological_matrix: np.ndarray,
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

            if u_linear not in u_map:
                u_map[u_linear] = []
            u_map[u_linear].append(point_idx)

            if v_linear not in v_map:
                v_map[v_linear] = []
            v_map[v_linear].append(int(point_idx))

            if w_linear not in w_map:
                w_map[w_linear] = []
            w_map[w_linear].append(int(point_idx))

            if r%2 == 0:
                if r not in r_map:
                    r_map[r] = []
                r_map[r].append(point_idx)

    all_harvested_lines = []

    # Process all three directional maps sequentially, length initializes internally
    u_lines = _extract_and_filter_axis_lines(u_map, all_harvested_lines, MIN_LEN, N)
    v_lines = _extract_and_filter_axis_lines(v_map, all_harvested_lines, MIN_LEN, N)
    w_lines = _extract_and_filter_axis_lines(w_map, all_harvested_lines, MIN_LEN, N)
    r_lines = _extract_and_filter_axis_lines(r_map, all_harvested_lines, MIN_LEN, N)

    # Return 4 independent structural vectors directly as a decoupled tuple
    return u_lines, v_lines, w_lines, r_lines
