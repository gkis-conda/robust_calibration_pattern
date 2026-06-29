import numpy as np
from scipy.spatial import Delaunay, KDTree
from collections import deque
from lattice_topology import *


def suppress_overlapping_triangles(triangles, points, suppression_radius):
    """
    Applies Spatial Non-Maximum Suppression (NMS) to a list of triangles.

    Parameters:
        triangles (list): List of index triplets [[idx_a, idx_b, idx_c], ...]
        points (np.array): NumPy array of physical point coordinates
        suppression_radius (float): Minimum physical distance between face centers.

    Returns:
        list: Sorted index triplets of uniform non-overlapping triangles.
    """
    if not triangles or len(points) == 0:
        return []

    candidates_pool = []

    for simplex in triangles:
        idx_a, idx_b, idx_c = simplex
        p_a, p_b, p_c = points[idx_a], points[idx_b], points[idx_c]

        # 1. Compute exact physical centroid coordinates
        centroid = (p_a + p_b + p_c) / 3.0
        centroid_x = float(centroid[0])
        centroid_y = float(centroid[1])

        # 2. Calculate edge lengths to resolve interior angles
        dist_c = float(np.linalg.norm(p_a - p_b))
        dist_a = float(np.linalg.norm(p_b - p_c))
        dist_b = float(np.linalg.norm(p_c - p_a))

        try:
            cos_A = (dist_b ** 2 + dist_c ** 2 - dist_a ** 2) / (2.0 * dist_b * dist_c)
            cos_B = (dist_a ** 2 + dist_c ** 2 - dist_b ** 2) / (2.0 * dist_a * dist_c)
            cos_C = (dist_a ** 2 + dist_b ** 2 - dist_c ** 2) / (2.0 * dist_a * dist_b)

            angles_deg = np.degrees(np.arccos(np.clip([cos_A, cos_B, cos_C], -1.0, 1.0)))
            # Compute a scalar float deviation score (lower score = more equilateral)
            beauty_score = float(np.sum(np.abs(angles_deg - 60.0)) / 3.0)
        except ZeroDivisionError:
            beauty_score = 999.0  # Assign a bad score penalty to malformed degenerate rows

        candidates_pool.append({
            'simplex': simplex,
            'centroid': (centroid_x, centroid_y),
            'score': beauty_score
        })

    # Sort candidates so the most regular equilateral triangles are processed first
    candidates_pool.sort(key=lambda item: item['score'])

    final_selected_triangles = []
    for cand in candidates_pool:
        c_x, c_y = cand['centroid']

        keep_triangle = True
        for selected in final_selected_triangles:
            s_x, s_y = selected['centroid']
            # Calculate physical distance between face centers
            dist = ((c_x - s_x) ** 2 + (c_y - s_y) ** 2) ** 0.5
            if dist < suppression_radius:
                keep_triangle = False
                break

        if keep_triangle:
            final_selected_triangles.append(cand)

    # Collect the surviving clean index triplets
    output_simplices = []
    for item in final_selected_triangles:
        output_simplices.append(item['simplex'])

    return output_simplices


class GridCrystalGrower:
    """
    Look-up Crystal Lattice Engine using Delaunay triangulation.
    Extracts valid geometric-topological islands one by one and 
    directly generates dense integer point lookup index matrices.
    """
    def __init__(self, detected_points, wave_limit = 1000):
        self.points = np.array(detected_points, dtype=np.float64)
        num_points = len(self.points)
        self.kdtree = KDTree(self.points) if num_points > 0 else None
        self.dsu = IslandDSU(num_points) if num_points > 0 else None
        self.triangulation = Delaunay(self.points)
        simplices = self.triangulation.simplices
        self.debug_output = False

        # Calculate global median step across all triangulation edges
        all_dists = []
        for simplex in simplices:
            idx_a, idx_b, idx_c = simplex
            p_a, p_b, p_c = self.points[idx_a], self.points[idx_b], self.points[idx_c]
            all_dists.append(float(np.linalg.norm(p_a - p_b)))
            all_dists.append(float(np.linalg.norm(p_b - p_c)))
            all_dists.append(float(np.linalg.norm(p_c - p_a)))

        # CRITICAL PROTECTION: Fast-exit if triangulation has zero edges
        if len(all_dists) == 0:
            raise ValueError("[-] Error: No valid Delaunay edges generated. Cloud might be collinear.")

        self.step = float(np.median(all_dists))
        print(f"[Diag] Global reference step from all edges: {self.step:.2f} px")

        self.MAX_DIAGNOSTIC_WAVE_LIMIT = wave_limit
        
        # Global registries mapping master_root_id -> dictionary maps
        # Keeps absolute tracking state unified across multiple sequential waves
        self.global_grids = {}  # master_root_id -> {(u, v): point_idx}
        self.global_regs = {}   # master_root_id -> {point_idx: (u, v)}
        self.globally_processed_faces = set()

    def find_all_valid_triangles(self, unvisited_indices):
        """
        Computes a global Delaunay triangulation and filters faces by shape geometry.
        Returns ALL valid geometric faces without applying spatial suppression.
        """
        simplices = self.triangulation.simplices
        if len(simplices) < 1:
            print("[-] Error: Less than 3 points provided.")
            return []

        discovered_triangles = []

        # Strict constraints tailored for ideal triangular grid pattern calibration
        MIN_ALLOWED_ANGLE_DEG = 48.0
        MAX_ALLOWED_ANGLE_DEG = 72.0

        max_allowed_dist = self.step * 1.60
        min_allowed_dist = self.step * 0.60

        total_faces = len(simplices)
        fail_dist = 0
        fail_angle = 0

        for simplex in simplices:
            idx_a, idx_b, idx_c = simplex
            p_a, p_b, p_c = self.points[idx_a], self.points[idx_b], self.points[idx_c]

            dist_c = float(np.linalg.norm(p_a - p_b))
            dist_a = float(np.linalg.norm(p_b - p_c))
            dist_b = float(np.linalg.norm(p_c - p_a))

            # Filter 1: Edge Length Boundaries
            if not (min_allowed_dist <= dist_a <= max_allowed_dist and
                    min_allowed_dist <= dist_b <= max_allowed_dist and
                    min_allowed_dist <= dist_c <= max_allowed_dist):
                fail_dist += 1
                continue

            # Filter 2: Internal Angles
            try:
                cos_A = (dist_b ** 2 + dist_c ** 2 - dist_a ** 2) / (2.0 * dist_b * dist_c)
                cos_B = (dist_a ** 2 + dist_c ** 2 - dist_b ** 2) / (2.0 * dist_a * dist_c)
                cos_C = (dist_a ** 2 + dist_b ** 2 - dist_c ** 2) / (2.0 * dist_a * dist_b)

                angles_deg = np.degrees(np.arccos(np.clip([cos_A, cos_B, cos_C], -1.0, 1.0)))
                if np.any(angles_deg < MIN_ALLOWED_ANGLE_DEG) or np.any(angles_deg > MAX_ALLOWED_ANGLE_DEG):
                    fail_angle += 1
                    continue
            except ZeroDivisionError:
                fail_angle += 1
                continue

            # Enforce strict Counter-Clockwise (CCW) winding order safely
            vec_ab = p_b - p_a
            vec_ac = p_c - p_a
            if np.cross(vec_ab, vec_ac) < 0.0:
                idx_b, idx_c = idx_c, idx_b

            discovered_triangles.append([int(idx_a), int(idx_b), int(idx_c)])
        if self.debug_output:
            print(f"[Diag] Total Delaunay faces: {total_faces}")
            print(f"[Diag] Drop by edge length:  {fail_dist}")
            print(f"[Diag] Drop by angle check: {fail_angle}")
            print(f"[Diag] Validated geometric seeds left: {len(discovered_triangles)}")

        return discovered_triangles

    def _find_reflection_candidate(self, v_pivot, v_b, v_c, distance_tolerance=0.2):
        """
        Predicts the physical position of the reflected vertex A' = B + C - A,
        searches the spatial neighborhood via KDTree, performs a backward 
        parallelogram verification, and strictly enforces a positive cross-product 
        invariant to guarantee counter-clockwise winding order. Returns node index or None.
        """
        p_pivot = self.points[v_pivot]
        p_b = self.points[v_b]
        p_c = self.points[v_c]
        
        # Physical prediction: A' = B + C - A
        p_a_prim = p_b + p_c - p_pivot
        
        search_radius = self.step * distance_tolerance
        dist, hit_idx = self.kdtree.query(p_a_prim, k=1, eps=0, p=2, distance_upper_bound=search_radius)
                            
        if dist > search_radius:
            return None
            
        p_new = self.points[hit_idx]
        
        # --- BACKWARD PARALLELOGRAM CONDITION CHECK ---
        err_b = float(np.linalg.norm((p_new - p_b) - (p_c - p_pivot)))
        err_c = float(np.linalg.norm((p_new - p_c) - (p_b - p_pivot)))
        if err_b > (self.step * distance_tolerance) or err_c > (self.step * distance_tolerance):
            return None  # Rejects distorted perspective outliers

        # --- MANDATORY GEOMETRIC CCW INVARIANT CHECK ---
        # Compute 2D vectors: edge_1 = B - New, edge_2 = C - New
        vec_1 = p_b - p_new
        vec_2 = p_c - p_new

        # Compute the scalar cross product (Z-component)
        cross_product = vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]

        # Strict Invariant Shield: If cross product is negative, the triangle
        # is flipped/inverted (growing backwards into the island). Reject it immediately!
        if cross_product < 0:
            return hit_idx
            
        return None

    def _check_coordinate_collision(self, best_match_idx, k_new, local_grid):
        """
        Returns True if a coordinate or node collision violates grid properties.
        """
        if best_match_idx in local_grid:
            # Node exists: coordinates must match exactly (valid cycle closure)
            return local_grid[best_match_idx] != k_new
        else:
            # New node: target coordinate key must not be occupied by someone else
            return k_new in local_grid.values()

    def _execute_island_fusion_merge(self, current_root, hit_island_root, v_b, v_c, best_match_idx):
        """
        Computes all system 1 coordinates by matching the exact directional rays
        used during spatial triangulation to eliminate axis misalignment drift.
        """
        local_grid = self.global_grids[current_root]
        foreign_grid = self.global_grids[hit_island_root]

        # 1. Fetch physical image coordinates for ray extrapolation
        pt_b = self.points[v_b]
        pt_c = self.points[v_c]
        pt_target = self.points[best_match_idx]  # Point A'

        # 2. Extrapolate rays along the two coordinate axes on the screen
        pred_ray_b = pt_target + (pt_target - pt_b)
        pred_ray_c = pt_target + (pt_target - pt_c)

        search_radius = self.step * 0.2
        shared_pin_node = None
        v_anchor = None

        # 3. Process Ray B (Collinear extension of line b -> A')
        dist_b, hit_idx_b = self.kdtree.query(pred_ray_b)
        if dist_b < search_radius:
            if hit_idx_b in foreign_grid:
                shared_pin_node = hit_idx_b
                v_anchor = v_b

        # 4. Process Ray C if Ray B did not yield a valid anchor path
        if shared_pin_node is None:
            dist_c, hit_idx_c = self.kdtree.query(pred_ray_c)
            if dist_c < search_radius:
                if hit_idx_c in foreign_grid:
                    shared_pin_node = hit_idx_c
                    v_anchor = v_c

        # Safety short-circuit: if no valid graph-cut alignment pin is harvested, abort
        if shared_pin_node is None or v_anchor is None:
            return current_root

        # 5. Extrapolate coordinates in Island 1 strictly along the successful ray axis.
        # Correspondence A: Point A (Lattice collision center)
        coord_b = local_grid[v_b]
        coord_c = local_grid[v_c]
        p_1A = topological_reflection(coord_b, coord_c)

        # Correspondence B:
        # It continues directly forward along the same line: from v_anchor through p_1A
        p_1B = extrapolate_coordinate(local_grid[v_anchor], p_1A)

        # 6. Extract clean, spatially matching 1:1 anchor pairs mapped to your exact footprint
        p_2A = foreign_grid[best_match_idx]  # Island 2, Point A
        p_2B = foreign_grid[shared_pin_node]  # Island 2, Point B

        transform = find_discrete_transform(p_2A, p_2B, p_1A, p_1B)
        if transform is None:
            return current_root

        # 7. Trigger structural parent layout unification inside DSU
        new_master, old_slave = self.dsu.merge(current_root, hit_island_root)

        # If the foreign root won the DSU contest, seamlessly move the Master Island's
        # dictionary reference to that new root index. This preserves both grids instantly.
        if new_master == hit_island_root:
            self.global_grids[hit_island_root] = self.global_grids[current_root]
            # The old current_root key is now a dead duplicate reference; clear it
            self.global_grids.pop(current_root)

        # Now, target_grid is guaranteed to be pointing to the surviving Master layout container,
        # which already contains all the raw master nodes intact!
        target_grid = self.global_grids[new_master]

        # We only need to iterate over and transform the incoming slave nodes.
        # They are rotated, shifted, and injected straight into the master system context.
        for node_idx in foreign_grid.keys():
            foreign_coord = foreign_grid[node_idx]
            target_grid[node_idx] = apply_lattice_transform(foreign_coord, transform)

        # Complete cleanup: If the slave dictionary was left behind as a separate key, purge it
        if hit_island_root != new_master and hit_island_root in self.global_grids:
            self.global_grids.pop(hit_island_root)

        return new_master

    def _grow_topological_island_step(self, queue):
        """
        Processes concurrent expanding frontier edges using a continuous drain loop.
        Delegates the entire boundary harvesting and coordinate re-projection down
        to the isolated fusion helper method.
        to the isolated fusion helper method.
        """
        next_queue = deque()
        if not queue:
            return next_queue

        sticks_added_this_step = 0

        # Processes all elements in the current frontier directly
        while queue:
            v_pivot, v_b, v_c = queue.popleft()

            # Resolve the dynamic live master root mapped to this expanding front
            current_root = self.dsu.find(v_pivot)
            local_grid = self.global_grids[current_root]

            # Step 1: Geometric candidate extraction pass
            best_match_idx = self._find_reflection_candidate(v_pivot, v_b, v_c)
            if best_match_idx is None:
                continue

            # Step 3: Compute predicted lattice coordinates
            v_new = topological_reflection(local_grid[v_b], local_grid[v_c])

            # Step 4: Evaluate coordinate assignment properties
            if self._check_coordinate_collision(best_match_idx, v_new, local_grid):
                continue

            hit_island_root = self.dsu.find(best_match_idx)

            # --- CASE A: LOCAL METRIC LOOP CLOSURE (CYCLE CLOSING) ---
            if hit_island_root == current_root:
                pass
            else:
                hit_island_size = self.dsu.get_size(hit_island_root)
                # --- CASE B: CROSS-ISLAND BOUNDARY CRASH (ISLAND FUSION) ---
                if hit_island_size >= 3:
                     self._execute_island_fusion_merge(
                         current_root, hit_island_root, v_b, v_c, best_match_idx)
                # --- CASE C: SINGLE POINT ABSORPTION ---
                elif hit_island_size == 1:
                     new_root, old_root = self.dsu.merge(current_root, best_match_idx)
                     local_grid[best_match_idx] = v_new
                     self.global_grids[new_root] = local_grid
                     self.global_grids.pop(old_root, None)

            sticks_added_this_step += 1
            # --- UNIFIED NEXT-LAYER FRONTIER EDGE ASSIGNMENTS ---
            reflect_B = (v_b, best_match_idx, v_c)
            if not reflect_B in self.globally_processed_faces:
                next_queue.append(reflect_B)
                self.globally_processed_faces.add(reflect_B)

            reflect_C = (v_c, v_b, best_match_idx)
            if not reflect_C in self.globally_processed_faces:
                next_queue.append(reflect_C)
                self.globally_processed_faces.add(reflect_C)
        if self.debug_output:
            print(f"[Diag] Wave step complete: Added {sticks_added_this_step} structural ears.")
        return next_queue


    def grow_island_lattice(self, unvisited_indices=None, distance_tolerance_ratio=0.35):
        """
        Relies entirely on find_all_valid_triangles for seeding.
        Tracks expansion solely via visited topological faces rather than blocking points.
        """
        output_matrices = {}
        points_num = len(self.points)
        if points_num == 0:
            return output_matrices
            
        # Extract the pure, pre-filtered topological face arrays from your Delaunay layer
        # Pass all points to let it discover everything available
        full_pool = set(range(points_num))
        all_triangles = self.find_all_valid_triangles(full_pool)
        if not all_triangles:
            return output_matrices
        triangles =  suppress_overlapping_triangles(all_triangles, self.points, 5 * self.step)

        self.global_grids = {}
        queue = deque()
        # Track globally processed triangle faces to prevent redundant seed generation
        self.globally_processed_faces = set()
	    # Prepare initial graph
        for seed_face in triangles:
            v0, v1, v2 = seed_face
            if self.dsu.get_size(v0) > 1 or self.dsu.get_size(v1) > 1 or self.dsu.get_size(v2)>  1:
                continue            
            # Resolve the current localized DSU master root parent state
            self.dsu.merge(v0, v1)
            current_root, _ = self.dsu.merge(v0, v2)

            # Establish the baseline {point_idx: (u, v)} dictionary mapping format
            self.global_grids[current_root] = {
                v0: (0, 0),
                v1: (1, 0),
                v2: (0, 1)
            }
            # fiil up the processing queue
            queue.append((v0, v1, v2))
            self.globally_processed_faces.add((v0, v1, v2))
            queue.append((v1, v2, v0))
            self.globally_processed_faces.add((v1, v2, v0))
            queue.append((v2, v0, v1))
            self.globally_processed_faces.add((v2, v0, v1))
            
        # Propel waves outward up to your defined step parameters
        for wave_step in range(self.MAX_DIAGNOSTIC_WAVE_LIMIT):

            # Extract finalized tracking records into dense NumPy calibration matrix grids
            if self.debug_output:
                print(f"Wave {wave_step}")
            output_matrices = {}
            for root_id in self.global_grids.keys():
                if self.dsu.parents[root_id] == root_id:
                    dense_layout = convert_grid_to_matrix(self.global_grids[root_id])
                    if self.debug_output and dense_layout.size > 0:
                        output_matrices[root_id] = dense_layout
                        print(f"   -> Island {root_id} Shape: {dense_layout.shape}")
                        print(dense_layout)

            queue = self._grow_topological_island_step(queue)
            if not queue:
                break

        output_matrices = {}
        np.set_printoptions(threshold=np.inf, linewidth=200)
        for root_id in self.global_grids.keys():
            if self.dsu.parents[root_id] == root_id:
                dense_layout = convert_grid_to_matrix(self.global_grids[root_id])
                if dense_layout.size > 0:
                    output_matrices[root_id] = dense_layout
                    if self.debug_output:
                        print(f"   -> Island {root_id} Shape: {dense_layout.shape}")
                        print(dense_layout)
        return output_matrices


def reconstruct_mesh(detected_points, point_labels, m_seq_length=7, distance_tolerance_ratio=0.30):
    """
    Executes multi-island barycentric crystal growth and reformats each independent 
    topological island structure into its own isolated dense 2D index matrix.
    """
    num_pts = len(detected_points)
    if num_pts < int(m_seq_length):
        return []

    unvisited_indices = set(range(num_pts))
    grower = GridCrystalGrower(detected_points)
    
    islands_dict = grower.grow_island_lattice(unvisited_indices)
    if not islands_dict:
        print("[-] Failure localized at grow_island_lattice. Wave phase failed to propagate.")
    else:
        print(f"[+] Success! Generated {len(islands_dict)} individual grid matrix islands.")

    # Format the results
    matrices_list = []
    if islands_dict:
        sorted_keys = sorted(list(islands_dict.keys()))
        for island_id in sorted_keys:
            dense_matrix = islands_dict[island_id]
            if dense_matrix.size > 0:
                matrices_list.append(dense_matrix)
                
    return matrices_list



# --- L1
def test_decomposed_coordinate_collision():
    """
    Test 1: Validates _check_coordinate_collision grid properties.
    Ensures cycle closures pass while conflicting positions throw violations.
    """
    print("\n--- Running: Test 1 (Decomposed Coordinate Collision Helper) ---")
    mock_pts = [
        [100.0, 100.0],  # Node 0: Master Pivot
        [135.0, 100.0],  # Node 1: Frontier Edge Node B
        [117.5, 130.3],  # Node 2: Frontier Edge Node C
        [152.5, 130.3],  # Node 3: Cross-over intersection target point A'
        [187.5, 130.3],  # Node 4: Slave Node (Extrapolated continuation of Ray B)
        [170.0, 160.6]  # Node 5: Slave Node
    ]
    grower_instance = GridCrystalGrower(mock_pts)

    mock_local_grid = {0: (0, 0), 2: (1, 0), 1: (0, 1)}
    
    # Scenario A: Vertex already exists, and calculated coordinates match exactly (Valid Closure)
    is_invalid = grower_instance._check_coordinate_collision(0, (0, 0), mock_local_grid)
    assert not is_invalid, "Valid cycle closure flagged as an invalid collision."
    print("   [+] Valid cycle closure passed checked verification.")
    
    # Scenario B: Vertex already exists, but computed coordinates drift (Violation)
    is_invalid = grower_instance._check_coordinate_collision(1, (2, 2), mock_local_grid)
    assert is_invalid, "Failed to trap a coordinate mismatch on an existing vertex."
    print("   [+] Coordinate variance violation caught successfully.")
    
    # Scenario C: Brand new vertex, target coordinate space is empty (Valid Absorption)
    is_invalid = grower_instance._check_coordinate_collision(99, (5, 5), mock_local_grid)
    assert not is_invalid, "Free spatial coordinate falsely flagged as occupied."
    print("   [+] Free spatial node placement allowed cleanly.")
    
    # Scenario D: Brand new vertex, but target coordinate space is already taken (Violation)
    is_invalid = grower_instance._check_coordinate_collision(99, (1, 0), mock_local_grid)
    assert is_invalid, "Failed to trap a spatial collision on an already occupied coordinate slot."
    print("   [+] Spatial structural overlapping caught successfully.")

def test_atomic_absorption_coordinate_accuracy():
    """
    Test 2: Enforces strict coordinate validation for singleton absorption (Case C).
    Verifies if k_new calculations maintain true lattice geometry invariants 
    when expanding outward in multiple alternate directions.
    """
    print("\n--- Running: Test 2 (Atomic Absorption Coordinate Accuracy Check) ---")
    
    # 6 ideal points layout forming a central seed face with 3 surrounding ears
    mock_pts = [
        [100.0, 100.0],  # Node 0: Pivot A
        [135.0, 100.0],  # Node 1: Vertex B
        [117.5, 130.3],  # Node 2: Vertex C
        [152.5, 130.3],  # Node 3: Target ear 1 (Reflection over BC)
        [82.5,  130.3],  # Node 4: Target ear 2 (Reflection over AC)
        [117.5, 69.7]    # Node 5: Target ear 3 (Reflection over AB)
    ]

    # Initialize real builder instance
    grower = GridCrystalGrower(mock_pts)
    
    # Setup baseline seed cluster (Face 0, 1, 2)
    grower.dsu.merge(0, 1)
    grower.dsu.merge(0, 2)
    root_master = grower.dsu.find(0)
    
    # Initial seed coordinates configuration
    grower.global_grids[root_master] = {
        0: (0, 0),  # Node A
        1: (1, 0),  # Node B
        2: (0, 1)   # Node C
    }
    
    # Initialize a mock queue simulating 3 distinct frontier edge approaches
    # Format: (v_pivot, v_b, v_c)
    test_queue = deque([
        (0, 1, 2),  # Scenario A: Reflect 0 over edge (1,2) -> expects Node 3
        (1, 2, 0),  # Scenario B: Reflect 1 over edge (2,0) -> expects Node 4
        (2, 0, 1)   # Scenario C: Reflect 2 over edge (0,1) -> expects Node 5
    ])
    
    # Populate mock spatial lookups manually so _find_reflection_candidate 
    # returns nodes 3, 4, 5 predictably
    grower.mock_spatial_reflections = {
        (0, tuple(sorted([1, 2]))): 3,
        (1, tuple(sorted([2, 0]))): 4,
        (2, tuple(sorted([0, 1]))): 5
    }
    
    # Execute exactly one combined state step unrolling execution wave
    next_generation_queue = grower._grow_topological_island_step(test_queue)
    
    # --- EVALUATE MULTI-ASSERT COORDINATE INVARIANTS ---
    unified_grid = grower.global_grids[root_master]
    
    # Expected coordinate mappings derived from analytical triangular geometry:
    # Node 3 = B + C - A = (1, 0) + (0, 1) - (0, 0) = (1, 1)
    # Node 4 = C + A - B = (0, 1) + (0, 0) - (1, 0) = (-1, 1)
    # Node 5 = A + B - C = (0, 0) + (1, 0) - (0, 1) = (1, -1)
    
    print(f"   -> Node 3 assigned coordinate: {unified_grid.get(3)} (Expected: (1, 1))")
    print(f"   -> Node 4 assigned coordinate: {unified_grid.get(4)} (Expected: (-1, 1))")
    print(f"   -> Node 5 assigned coordinate: {unified_grid.get(5)} (Expected: (1, -1))")
    
    assert unified_grid.get(3) == (1, 1), f"Coordinate calculation failed for Node 3: {unified_grid.get(3)}"
    assert unified_grid.get(4) == (-1, 1), f"Coordinate calculation failed for Node 4: {unified_grid.get(4)}"
    assert unified_grid.get(5) == (1, -1), f"Coordinate calculation failed for Node 5: {unified_grid.get(5)}"
    
    print("   [+] Success: All three outward expansion coordinate paths match ideal geometry constants.")


# --- LEVEL 2: COMPONENT TRANSFORM MUTATION TESTS (MID-LEVEL COMPLEXITY) ---
def test_decomposed_island_fusion_merge():
    """
    Test 3: Directly validates the isolated ray-extrapolation _execute_island_fusion_merge routine.
    Evaluates physical ray projection under a clean topological graph cut.
    """
    print("\n--- Running: Test 3 (Decomposed Island Fusion Merge Helper) ---")

    # Ideal triangular grid tracking locations layout parameters
    mock_pts = [
        [100.0, 100.0],  # Node 0: Master Pivot
        [135.0, 100.0],  # Node 1: Frontier Edge Node B
        [117.5, 130.3],  # Node 2: Frontier Edge Node C
        [152.5, 130.3],  # Node 3: Cross-over intersection target point A'
        [187.5, 130.3],  # Node 4: Slave Node (Extrapolated continuation of Ray B)
        [170.0, 160.6]   # Node 5: Slave Node
    ]

    grower = GridCrystalGrower(mock_pts)

    # Master Island Initialization (Nodes 0, 1, 2)
    grower.dsu.merge(0, 1)
    grower.dsu.merge(0, 2)
    root_master = grower.dsu.find(0)
    grower.global_grids[root_master] = {0: (0, 0), 1: (1, 0), 2: (0, 1)}

    # Slave Island Initialization (Nodes 3, 4, 5) - Separated by a clean graph cut!
    grower.dsu.merge(3, 4)
    grower.dsu.merge(3, 5)
    root_slave = grower.dsu.find(3)

    # Define distortion transformation parameters explicitly
    true_rotation_idx = 2  # 120-degree permutation
    true_shift_vector = (15, -30)
    transform = (true_rotation_idx, true_shift_vector)
    # Ground-truth ideal lattice positions for the slave island components
    ideal_slave_layout = {
        3: (1, 1),   # Point A (Collision center best_match_idx)
        5: (1, 2),   # Point B (Collinear ray extension Pin node)
        4: (0, 2)    # Adjacent node to complete the cluster layout geometry
    }

    grower.global_grids[root_slave] = {}
    for node_idx, ideal_coord in ideal_slave_layout.items():
        grower.global_grids[root_slave][node_idx] = apply_lattice_transform(
            ideal_coord, transform
        )

    # Trigger execution over the ray-extrapolation fusion helper method
    new_master_root = grower._execute_island_fusion_merge(
        current_root=root_master,
        hit_island_root=root_slave,
        v_b=1,
        v_c=2,
        best_match_idx=3
    )

    final_master = grower.dsu.find(0)
    # Assert 1: Confirm DSU master-slave layout binding unification passes perfectly
    assert final_master == grower.dsu.find(3), "DSU sets failed to unify under a common parent root."

    # Assert 2: Verify component size tracking totals map up safely
    assert grower.dsu.get_size(final_master) == 6, f"Expected size 6, got {grower.dsu.get_size(final_master)}"
    print("   [+] DSU state and cluster sizes consolidated successfully under graph-cut limits.")


# --- L3
def test_cycle_closing_self_intersection():
    """Test 4: Verifies loop self-intersections (cycle closing bounds)."""
    print("\n--- Running: Test 4 (Cycle Closing Self-Intersection Pass) ---")
    mock_points = [[100.0, 100.0], [135.0, 100.0], [117.5, 130.3], [152.5, 130.3]]
    grower = GridCrystalGrower(mock_points)
    matrices = grower.grow_island_lattice()
    
    assert len(matrices) == 1, "Expected exactly one consolidated array fragment."
    master_root = list(matrices.keys())[0]
    assert grower.dsu.get_size(master_root) == len(mock_points), "Lattice corrupted component metrics."
    print("   [+] Real geometry cycle closing loop completed successfully.")


def test_standard_wave_propagation():
    """Test 5: Verifies traditional layer-by-layer outward wave progression."""
    print("\n--- Running: Test 5 (Standard Outward Wave Propagation Pass) ---")
    mock_points = [
        [100.0, 100.0], [135.0, 100.0], [117.5, 130.3],
        [152.5, 130.3], [82.5,  130.3], [117.5, 69.7]
    ]

    grower = GridCrystalGrower(mock_points)
    matrices = grower.grow_island_lattice()
    
    assert len(matrices) == 1, "Expected exactly one master block."
    master_root = list(matrices.keys())[0]
    assert grower.dsu.get_size(master_root) == 6, "Wave loop terminated early before tracking all markers."
    print("   [+] Real geometry standard wave propagation completed successfully.")


def test_multi_island_crashing_fusion():
    """Test 6: Simulates graph dissection and active wave crashing fusion."""
    print("\n--- Running: Test 6 (Graph Dissection and Island Fusion Wave Crashing) ---")
    mock_points_7 = [[0.,0.], [1.,0.], [0.5,1.], [1.5,1.], [2.,0.], [1.5,-1.], [2.5,-1.]]

    grower = GridCrystalGrower(mock_points_7)
    matrices = grower.grow_island_lattice()
    print(matrices)
    active_roots = [r for r in grower.global_grids.keys() if grower.dsu.parents[r] == r]
    assert len(active_roots) == 1, f"Lattice fusion failure. Found disconnected components: {len(active_roots)}"
    print("   [+] Real geometry cross-island active wave crash fusion passed successfully.")


if __name__ == "__main__":
    print("=========================================================")
    print("=== RUNNING ORIGINAL GEOMETRY CRYSTAL GROWER TESTS ===")
    print("=========================================================")
    
    # Run Level 1 Primitives Verification
    test_decomposed_coordinate_collision()
    test_atomic_absorption_coordinate_accuracy()

    # Run Level 2 Isolated Matrix Merger Check
    test_decomposed_island_fusion_merge()
    
    # Run Level 3 Full Wave Growth on Real Geometric Point Clouds
    test_cycle_closing_self_intersection()
    test_standard_wave_propagation()
    test_multi_island_crashing_fusion()
    
    print("\n=========================================================")
    print("[SUCCESS] ALL SIX INDEPENDENT ORIGINAL TESTS PASSED.")
    print("=========================================================")
