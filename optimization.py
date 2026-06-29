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

def menger_curvature_loss(params: list, 
                          points_2d: np.ndarray, 
                          lines: list, 
                          weights: np.ndarray, 
                          center: tuple,
                          undistort_func=None) -> float:
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
    f, k1 = float(params[0]), float(params[1])
    
    # 1. OPTIMIZER SAFETY WALL: Enforce absolute minimal physical constraints
    if f < 100.0: 
        return 1e10
    
    if undistort_func is not None:
        u_points = undistort_func(points_2d, f, k1, center)
    else:
        cx, cy = float(center[0]), float(center[1])
        u_points = np.zeros_like(points_2d, dtype=np.float32)
        
        # Center coordinates relative to the optical principal point
        dx = points_2d[:, 0] - cx
        dy = points_2d[:, 1] - cy
        
        # Squared normalized radial displacements
        r2 = (dx * dx + dy * dy) / (f * f)
        
        # Apply standard low-overhead first-order radial correction multiplier
        radial_scale = 1.0 + k1 * r2
        u_points[:, 0] = cx + dx * radial_scale
        u_points[:, 1] = cy + dy * radial_scale
    
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


def harvest_hexagonal_line_bundles(topological_matrix: np.ndarray, N:int = 16, MIN_LEN : int = 5) -> list:
    """
    Scans a decoded hexagonal tracking canvas matrix and extracts three 
    independent nested index bundles (Rows, Main Diagonals, Transverse Diagonals)
    representing straight line tracks to feed your Menger curvature optimizer.
    
    Variables Description:
        topological_matrix (np.ndarray): The final decoded canvas matrix frame, shape (H, W).
                                         Maps cell positions to sub-pixel point IDs.
                                         Empty tracking voids carry -1 tokens.
                                         
    Returns:
        list: Nested collection of straight line index trajectories:
              [[id1, id2, id3, ...], [id4, id5, id6, ...]]
    """
    H_nodes, W_nodes = topological_matrix.shape

    # Map every matrix node into a dictionary keyed by its absolute linear u identifier
    u_map = {}
    v_map = {}
    w_map = {}

    for r in range(H_nodes):
        for c in range(W_nodes):
            point_idx = topological_matrix[r, c]
            if point_idx < 0:
                continue
                
            # Unwarp storage coordinates to the invariant linear domain
            u_linear = c - (r // 2)
            v_linear = r
            w_linear = -v_linear-u_linear

            if u_linear not in u_map:
                u_map[u_linear] = []
            u_map[u_linear].append(point_idx)
            if v_linear not in v_map:
                v_map[v_linear] = []
            v_map[v_linear].append(point_idx)
            if w_linear not in w_map:
                w_map[w_linear] = []
            w_map[w_linear].append(point_idx)

    # Sort nodes along each tracked diagonal path by their row indices to ensure proper line order
    line_idx = 0
    all_harvested_lines = []
    u_lines = []
    for _, line_track in u_map.items():
        line_len = len(line_track)
        if line_len >= MIN_LEN:
            all_harvested_lines.append(line_track)
            u_lines.append((line_len, line_idx))
            line_idx += 1

    v_lines = []
    for _, line_track in v_map.items():
        line_len = len(line_track)
        if line_len >= MIN_LEN:
            all_harvested_lines.append(line_track)
            v_lines.append((line_len, line_idx))
            line_idx += 1

    w_lines = []
    for _, line_track in w_map.items():
        line_len = len(line_track)
        if line_len >= MIN_LEN:
            all_harvested_lines.append(line_track)
            w_lines.append((line_len, line_idx))
            line_idx += 1

    longest_lines = []

    if len(u_lines) > N:
        u_lines = np.sort(u_lines)
        for _, idx in u_lines[-N:]:
            longest_lines.append(all_harvested_lines[idx])
    else:
        for _, idx in u_lines:
            longest_lines.append(all_harvested_lines[idx])

    if len(v_lines) > N:
        v_lines = np.sort(v_lines)
        for _, idx in v_lines[-N:]:
            longest_lines.append(all_harvested_lines[idx])
    else:
        for _, idx in v_lines:
            longest_lines.append(all_harvested_lines[idx])

    if len(w_lines) > N:
        w_lines = np.sort(w_lines)
        for _, idx in w_lines[-N:]:
            longest_lines.append(all_harvested_lines[idx])
    else:
        for _, idx in w_lines:
            longest_lines.append(all_harvested_lines[idx])

    return longest_lines


def calibrate_camera(topological_matrix: np.ndarray,
                                     detected_points: np.ndarray,
                                     camera_object,
                                     options_dict: dict = None) -> dict:
    """
    Orchestrates the entire lens calibration routine by re-using standalone
    harvest_hexagonal_line_bundles and menger_curvature_loss functions.
    Directly optimizes the provided ProjectiveCamera object's parameters
    and modifies its internal state fields in-place upon convergence.

    Variables Description:
        topological_matrix (np.ndarray): Decoded tracking map lookup framework, shape (H, W).
                                         Maps matrix slots to sub-pixel point IDs.
                                         Empty tracking voids carry -1 tokens.
        detected_points (np.ndarray)   : Raw 2D sub-pixel coordinates from blob finder, shape (N, 2).
        camera_object (obj)            : Instance of ProjectiveCamera from your camera module.
                                         Must have attributes .f, .k1, .cx, .cy and
                                         a method .undistort_points() or equivalent.
        options_dict (dict)            : Optional parameters for the SciPy optimization simplex.

    Returns:
        dict: Solved parameters, final loss metric, and optimizer convergence metadata.
    """
    grid_lines_bundle = harvest_hexagonal_line_bundles(topological_matrix)

    num_detected_blobs = len(detected_points)
    node_weights = np.ones(num_detected_blobs, dtype=np.float32)
    center_cx_cy = (camera_object.cx, camera_object.cy)

    print("=== INITIATING DRY-REUSED CAMERA MENGER OPTIMIZATION ===")
    print(f" -> Total Straight Line Bundles Harvested: {len(grid_lines_bundle)}")
    print(f" -> Current Camera Focal Length (f)     : {camera_object.f_px:.2f}")
    print(f" -> Current Camera Distortion (k1)       : {camera_object.k1:.6f}")

    if len(grid_lines_bundle) < 5:
        return {
            "status": "failed",
            "message": "Insufficient straight lines harvested to reliably guide optimization.",
            "final_f": camera_object.f_px, "final_k1": camera_object.k1
        }

    # 2. DEFINE THE ADAPTER WRAPPER CLOSURE FOR OBJECT MUTATION
    def camera_objective_adapter(params):
        """Bridge closure mapping optimization parameters back onto the active camera object."""
        f_proposed, k1_proposed = float(params[0]), float(params[1])

        # Guard wall: Enforce absolute minimal physical constraints on focal length
        if f_proposed < 100.0:
            return 1e10

        # Temporarily back up original parameters for thread isolation hygiene
        orig_f, orig_k1 = camera_object.f_px, camera_object.k1

        # Assign proposed parameters to object fields to feed native .undistort_points()
        camera_object.f = f_proposed
        camera_object.k1 = k1_proposed

        # Invoke the re-used core loss function pass
        # Instead of writing custom inline math, it triggers the standalone loss engine
        loss_val = menger_curvature_loss(
            params=[f_proposed, k1_proposed],
            points_2d=detected_points,
            lines=grid_lines_bundle,
            weights=node_weights,
            center=center_cx_cy,
            undistort_func=lambda pts, f, k, c: camera_object.undistort_points(pts)
        )

        # Restore camera state safely until the final convergence checkpoint
        camera_object.f = orig_f
        camera_object.k1 = orig_k1

        return loss_val

    # Evaluate the starting geometric error footprint before optimization triggers
    x0_initial = np.array([camera_object.f_px, camera_object.k1], np.float32)
    initial_loss = camera_objective_adapter(x0_initial)

    print(f" -> Initial Menger Curvature Residual Cost : {initial_loss:.4e}")

    # =====================================================================
    # STAGE 3: EXECUTE THE CLOSED-LOOP NUMERICAL MINIMIZER
    # =====================================================================
    opt_settings = {'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500}
    if options_dict is not None:
        opt_settings.update(options_dict)

    print(" -> Executing Nelder-Mead optimization simplex over camera object parameter space...")
    opt_res = scipy.optimize.minimize(
        fun=camera_objective_adapter,
        x0=x0_initial,
        method='Nelder-Mead',
        options=opt_settings
    )

    f_optimized, k1_optimized = opt_res.x
    final_loss = opt_res.fun

    # Apply the final optimized parameters back onto your working camera instance
    if opt_res.success or final_loss < initial_loss:
        camera_object.f_px = float(f_optimized)
        camera_object.k1 = float(k1_optimized)
        print(" -> In-place state update: ProjectiveCamera parameters modified successfully.")

    print(f" -> Optimization Convergence Match : {'SUCCESS' if opt_res.success else 'FAILED'}")
    print(f" -> Final Curvature Residual Cost  : {final_loss:.4e} (Delta: {initial_loss - final_loss:.4e})")
    print(f" -> Solved Camera Focal Length (f) : {camera_object.f_px:.4f} pixels")
    print(f" -> Solved Camera Distortion (k1)  : {camera_object.k1:.8f}")

    return {
        "status": "success" if opt_res.success else "converged_with_warnings",
        "optimizer_message": str(opt_res.message),
        "final_f": camera_object.f,
        "final_k1": camera_object.k1,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "iterations": int(opt_res.nit)
    }
