import numpy as np
# =========================================================================
# CORE TOPOLOGICAL TRANSFORMATION MODULE
# =========================================================================
def normalize_barycentric(u_linear: int, v_linear: int, lfsr_period: int = 31) -> tuple:
    """
    Normalizes continuous linear barycentric coordinates (u, v) into the canonical
    discrete matrix coordinate viewport (row, col) inside the 31x31 target field.

    Enforces a strict two-sided algebraic residue wrapper to prevent bottom-left
    corner clipping and index compression under extreme rotation shifts.
    """
    # 1. Bring both independent linear parameters into their positive residue rings
    # This completely insulates the boundary tracking loops from sign inversions
    v_canonical = int(v_linear) % lfsr_period
    u_canonical = int(u_linear) % lfsr_period

    # 2. THE AIRTIGHT TWO-SIDED CHESS GATE
    # Map the combined rhomboid footprint properties symmetrically across both edges
    combined_envelope = u_canonical + (v_canonical // 2)

    if combined_envelope >= lfsr_period:
        # Resolve rightward and bottom-right overflow leaks
        u_canonical -= lfsr_period
    elif combined_envelope < 0:
        # Resolve leftward and bottom-left underflow drops
        u_canonical += lfsr_period

    # 3. Reconstruct definitive absolute matrix indices safely
    r_matrix = v_canonical
    c_matrix = u_canonical + (v_canonical // 2)

    # Double-check safety limits before passing out to the mapping canvas sheet
    # If a extreme warp still leaks past constraints, force a safe wrap
    c_matrix = c_matrix % lfsr_period

    return r_matrix, c_matrix


def get_phase_from_coordinates(r: int, c: int, lfsr_period: int = 31) -> tuple:
    """
    FORWARD PATH: Translates positive discrete matrix row and column indices (r, c)
    directly into pure, unwarped linear algebraic phase parameters (u, v).

    Fully wrapped inside the positive residue class pool.
    """
    v_phase = r % lfsr_period
    # Since c and r are positive, unwarping via floor division is completely stable
    u_phase = (c - (v_phase // 2)) % lfsr_period
    return u_phase, v_phase


def get_coordinates_from_phase(pu: int, pv: int, lfsr_period: int = 31) -> tuple:

    # Rule 1: Row coordinate v is aprioristically positive because v = r and r >= 0.
    # If a rotation or window tracking offset drops v below zero, we apply
    # a positive modulo period shift to pull it back into the visible semi-space.
    assert pv>=0 and pu >=0
    r = pv
    s = r//2
    c = pu + s

    # If the candidate column leaks past the physical matrix boundaries,
    # it confirms that u belongs in the negative coordinate space (r > c condition)
    c = c%lfsr_period

    return r, c


def add_rhomboid_coordinates(coord: tuple, delta: tuple) -> tuple:
    """
    Performs precise coordinate translation over a staggered rhomboid/hexagonal grid.
    Accepts and returns unified (r, c) tuples to eliminate tracking drift.

    Args:
        coord (tuple): Original source coordinate pair (r, c).
        delta (tuple): Translation offset step pair (dr, dc).

    Returns:
        tuple: (global_r, global_c) raw absolute destination coordinates.
    """
    r, c = coord
    v = r
    u = c - r // 2

    dr, dc = delta
    dv = dr
    du = dc - dr // 2

    v += dv
    u += du

    # Row tracking maps forward linearly
    global_r = v
    global_c = u + (v // 2)

    return global_r, global_c


def sub_rhomboid_coordinates(coord: tuple, delta: tuple) -> tuple:
    """
    Performs precise coordinate translation over a staggered rhomboid/hexagonal grid.
    Accepts and returns unified (r, c) tuples to eliminate tracking drift.

    Args:
        coord (tuple): Original source coordinate pair (r, c).
        delta (tuple): Translation offset step pair (dr, dc).

    Returns:
        tuple: (global_r, global_c) raw absolute destination coordinates.
    """
    r, c = coord
    v = r
    u = c - r // 2

    dr, dc = delta
    dv = dr
    du = dc - dr // 2

    v -= dv
    u -= du

    # Row tracking maps forward linearly
    global_r = v
    global_c = u + (v // 2)

    return global_r, global_c


def convert_grid_to_matrix(point_grid):
    """
    Transforms a sparse topological {point_id: (u, v)} dictionary
    into a dense layout matrix mapped to row-parity neighborhood steps.

    Dynamically injects an empty padding row at index 0 if the initial
    minimum V coordinate bound has an odd parity to maintain alignment phase.

    Compatible with Python 3.2 and old NumPy versions.
    """
    if not point_grid:
        return np.array([[]], dtype=np.int32)

    # Phase 1: Scan bounds to isolate the minimum V coordinate
    min_v = None
    for pid, (u, v) in point_grid.items():
        if min_v is None or v < min_v:
            min_v = v

    # Determine parity of the first row bound.
    # If the starting V row is odd, we force a padding row injection (shift = 1)
    # to preserve the relative row-alternating matrix column layout.
    inject_padding_row = (min_v % 2 != 0)
    row_offset_modifier = 1 if inject_padding_row else 0

    matrix_mappings = {}
    min_r, max_r = None, None
    min_c, max_c = None, None

    # Phase 2: Translate all (u, v) pairs using your updated +1 column formula
    for pid, (u, v) in point_grid.items():
        # Row is determined by V, shifted up by the padding modifier if active
        r = v + row_offset_modifier
        c = u + (v // 2)

        matrix_mappings[pid] = (r, c)

        if min_r is None or r < min_r: min_r = r
        if max_r is None or r > max_r: max_r = r
        if min_c is None or c < min_c: min_c = c
        if max_c is None or c > max_c: max_c = c

    # If padding was injected, force min_r down to 0 to preserve the empty top row slot
    if inject_padding_row:
        # Shift the bounding box calculation down to encompass the forced empty top row
        min_r = min_v

    rows_count = max_r - min_r + 1
    cols_count = max_c - min_c + 1

    # Phase 3: Allocate dense array pre-filled with -1
    dense_matrix = np.full((rows_count, cols_count), -1, dtype=np.int32)

    # Phase 4: Populate matrix elements
    for pid, (r, c) in matrix_mappings.items():
        target_row = r - min_r
        target_col = c - min_c
        dense_matrix[target_row, target_col] = pid

    return dense_matrix


def convert_matrix_to_grid(dense_matrix, min_u_anchor=0, min_v_anchor=0):
    """
    Reconstructs the sparse topological {point_id: (u, v)} dictionary
    from a dense layout matrix based on row-parity neighborhood steps.

    Ignores -1 padding elements (representing empty space).
    Fully optimized and compatible with Python 3.2.
    """
    point_grid = {}
    if dense_matrix is None or dense_matrix.size == 0:
        return point_grid

    rows_count, cols_count = dense_matrix.shape

    # Iterate through all grid row and column matrix indexes
    for row_idx in range(rows_count):
        for col_idx in range(cols_count):
            point_id = dense_matrix[row_idx, col_idx]

            # Skip unallocated spaces padded with -1
            if point_id == -1:
                continue

            # Restore relative offsets back onto absolute system axes
            r = row_idx + min_v_anchor
            c = col_idx + min_u_anchor

            # Apply the exact algebraic inverse of the parity shift formula
            v = r
            u = c - (v // 2)

            point_grid[int(point_id)] = (u, v)

    return point_grid


def extrapolate_coordinate(coord_a, coord_b):
    """
    For an oriented triplet ABC, given the integer coordinates of points A and B,
    computes the coordinate of point C lying on the direct ray extension.
    Formula: C = B + (B - A)

    Parameters:
      coord_a: Tuple (u, v) representing the starting node A.
      coord_b: Tuple (u, v) representing the intermediate node B.

    Returns:
      Tuple: (u_c, v_c) representing the extrapolated node C.
    """
    u_a, v_a = coord_a
    u_b, v_b = coord_b

    # Calculate the vector step from A to B
    du = u_b - u_a
    dv = v_b - v_a

    # Project forward by adding the delta to B
    return u_b + du, v_b + dv


def get_rotation_steps_from_axis(horizontal_axis: str, direction: str) -> int:
    """
    Determines the precise hexagonal rotational step count k (0 to 5)
    based strictly on the decoded horizontal axis identity and direction string.
    """
    # Map out the exact 6-step loop wheel permutations
    if horizontal_axis == "U":
        return 0 if direction == "forward" else 3
    elif horizontal_axis == "W":
        return 1 if direction == "reverse" else 4
    elif horizontal_axis == "V":
        return 2 if direction == "forward" else 5

    # Safe default fallback boundary
    return 0


def rotate_barycentric(r: int, c: int, k: int) -> tuple:
    """
    Performs a strict 60-degree parameterized rotation of a single discrete
    grid cell index over a hexagonal matrix layout using the algebraic domain.

    Fully synchronized with your Cartesian physical projection verification table.
    Fully ASCII-compliant implementation.
    """
    steps = k % 6
    if steps == 0:
        return r, c

    v_curr = r
    s = r // 2
    u_curr = c - s
    w_curr = -v_curr - u_curr

    for _ in range(steps):
        u_next = -w_curr
        v_next = -u_curr
        w_next = -v_curr

        u_curr = u_next
        v_curr = v_next
        w_curr = w_next

    r_rotated = v_curr
    s = r_rotated // 2
    c_rotated = u_curr + s

    return r_rotated, c_rotated


def rotate_barycentric_matrix_adaptive(original_matrix: np.ndarray, k: int) -> tuple:
    """
    Rotates a barycentric matrix by (k * 60) degrees CCW with adaptive canvas resizing.
    Evaluates bounding box limits strictly within the discrete (r, c) storage matrix space,
    but performs the local window translations inside the linear topological domain.

    Fully ASCII-compliant implementation.
    """
    H_orig, W_orig = original_matrix.shape
    steps = k % 6

    # Short-circuit optimize: k=0 represents flat identity map pass
    if steps == 0:
        return np.copy(original_matrix), (0, 0)

    transformed_rows = []
    transformed_cols = []

    # Step 1: Pre-calculate the bounding box boundaries strictly within the discrete (r, c) space
    for r in range(H_orig):
        for c in range(W_orig):
            if original_matrix[r, c] < 0:
                continue

            # Invoke the modular scalar rotation helper
            r_dest, c_dest = rotate_barycentric(r, c, steps)

            transformed_rows.append(r_dest)
            transformed_cols.append(c_dest)

    # Protection Guard: If the patch contains no valid points, return a blank sheet
    if not transformed_rows:
        return np.full((1, 1), -1, dtype=original_matrix.dtype), (0, 0)

    # Isolate absolute coordinate boundaries inside (r, c) space to anchor the target size
    min_r = min(transformed_rows)
    max_r = max(transformed_rows)
    min_c = min(transformed_cols)
    max_c = max(transformed_cols)

    # Calculate adaptive target dimensions strictly from the (r, c) limits
    H_target = max_r - min_r + 1
    W_target = max_c - min_c + 1

    rotated_matrix = np.full((H_target, W_target), -1, dtype=original_matrix.dtype)

    # 2. Unwarp your discrete (r, c) bounding box corner into linear axes
    v_min = min_r
    u_min = min_c - (v_min // 2)

    # Step 2: Loop forward over the original source matrix elements
    for r_source in range(H_orig):
        for c_source in range(W_orig):
            val = original_matrix[r_source, c_source]
            if val < 0:
                continue

            # Retrieve perfectly invariant absolute rotated coordinates
            r_dest_abs, c_dest_abs = rotate_barycentric(r_source, c_source, steps)

            v_abs = r_dest_abs
            u_abs = c_dest_abs - (v_abs // 2)

            # 3. Perform the window translation subtraction inside the linear topological domain
            v_local = v_abs - v_min
            u_local = u_abs - u_min

            # 4. Pack back into local target array coordinates
            r_dest_local = v_local
            c_dest_local = u_local + (v_local // 2)

            # Assign elements safely within your pre-allocated canvas boundaries
            if (0 <= r_dest_local < H_target) and (0 <= c_dest_local < W_target):
                rotated_matrix[r_dest_local, c_dest_local] = val

    # =====================================================================
    # STAGE 3: CLEAN TRAILING COLUMN TRIM
    # =====================================================================
    # If the last column consists entirely of -1, slice it off cleanly
    if W_target > 1 and np.all(rotated_matrix[:, -1] == -1):
        rotated_matrix = rotated_matrix[:, :-1]
    return rotated_matrix, (min_r, min_c)


# The 6 directional vectors ordered as a single continuous counter-clockwise cycle.
# Each element at index i naturally maps to the next element at index (i + 1) % 6.
CYCLE_VECTORS = (
    (1, 0),  # Right  Axis U forward
    (0, 1),  # Down-Right Axis V forward
    (-1, 1), # Down-Left
    (-1, 0), # Left Axis U inverted
    (0, -1), # Up-Left Axis V inverted
    (1, -1)  # Up-Right
)


def topological_reflection(coord_b, coord_c):
    """
    Finds the vertex a' of a rhombus aba'c sharing the edge bc.
    Valid for a universally CCW oriented space where cross(b-a, c-a) > 0.
    The input triangle abc is oriented counter-clockwise (positive signed area).
    The coordinate system is defined as:
    - Axis U: goes horizontally to the right.
    - Axis V: goes diagonally downwards and to the right.

    Example:
        b=(1,0), c=(0,1) -> du=-1, dv=1 (Index 2 in CYCLE_VECTORS)
        Offset picked from Index (2 - 1) % 6 = 1 -> (0, 1)
        Result: a' = (1+0, 0+1) = (1, 1)
    """
    b_u, b_v = coord_b
    c_u, c_v = coord_c

    du = c_u - b_u
    dv = c_v - b_v

    try:
        # Find where our edge vector lies in the orientation cycle
        idx = CYCLE_VECTORS.index((du, dv))
    except ValueError:
        raise RuntimeError("Invalid or non-minimal edge direction for this lattice")

    # Symmetrical shift formula under your custom cross-product layout
    r_u, r_v = CYCLE_VECTORS[(idx - 1) % 6]

    return (b_u + r_u, b_v + r_v)


def apply_single_rotation(u, v, rot_idx):
    """
    Computes a discrete 60-degree rotation step in O(1) time.
    Sequential executions form a closed cyclic group of order 6.

    Valid for CCW-positive coordinate configurations.
    """
    # Forward cyclic stepping over the basis mapping:
    # Axis U (index 0) advances forward to index (rot_idx % 6)
    u_base_u, u_base_v = CYCLE_VECTORS[rot_idx % 6]

    # Axis V (index 1) advances forward to index ((1 + rot_idx) % 6)
    v_base_u, v_base_v = CYCLE_VECTORS[(1 + rot_idx) % 6]

    # Combine components via standard linear combination matrix multiply
    return u * u_base_u + v * v_base_u, u * u_base_v + v * v_base_v


def apply_lattice_transform(coord, transform):
    """
    Applies the forward affine mapping: Rotate then Translate.
    """
    rot_idx, shift = transform
    ru, rv = apply_single_rotation(coord[0], coord[1], rot_idx)
    return ru + shift[0], rv + shift[1]


def find_discrete_transform(p_src_A, p_src_B, p_dst_A, p_dst_B):
    """
    Fully O(1), completely bijective, and supports macro-vectors of any length.

    Formula: DST = Rotate(SRC, rot) + Shift
    """
    # 1. Compute delta macro-vectors for both regions
    du_src = p_src_B[0] - p_src_A[0]
    dv_src = p_src_B[1] - p_src_A[1]

    du_dst = p_dst_B[0] - p_dst_A[0]
    dv_dst = p_dst_B[1] - p_dst_A[1]

    # 2. Vector Length Validation using oblique metric: L^2 = u^2 + v^2 - u*v
    len_src = du_src * du_src + dv_src * dv_src + du_src * dv_src
    len_dst = du_dst * du_dst + dv_dst * dv_dst + du_dst * dv_dst
    if len_src != len_dst or len_src == 0:
        return None

    # 3. Generate all 6 possible structural rotations of the source vector.
    # This acts as an explicit, branchless geometric match filter.
    rot_idx = None
    for i in range(6):
        if (du_dst, dv_dst) == apply_single_rotation(du_src, dv_src, i):
            rot_idx = i
            break
    if rot_idx is None:
        return None
    # 5. Calculate translation shift vector anchored to the chosen spaces
    r_pivot = apply_single_rotation(p_src_A[0], p_src_A[1], rot_idx)
    tu = p_dst_A[0] - r_pivot[0]
    tv = p_dst_A[1] - r_pivot[1]

    return rot_idx, (tu, tv)


class IslandDSU:
    """
    ASCII-compliant Disjoint Set Union (DSU) tracker for grid islands.
    Manages point cluster indices and tracks structural sizes for optimization.
    """
    def __init__(self, num_points):
        # Each point starts as its own independent master root
        self.parents = list(range(num_points))
        # Maps root index to the explicit set of point indices contained within
        self.island_sets = {i: {i} for i in range(num_points)}

    def find(self, point_idx):
        """
        Finds the root representative of a point cluster with path compression.
        """
        path = []
        while self.parents[point_idx] != point_idx:
            path.append(point_idx)
            point_idx = self.parents[point_idx]
        for node in path:
            self.parents[node] = point_idx
        return point_idx

    def get_size(self, point_idx):
        """
        Returns the number of elements inside the point's current island.
        """
        root = self.find(point_idx)
        return len(self.island_sets[root])

    def merge(self, point_idx_1, point_idx_2):
        """
        Merges two independent sets using union-by-size optimization.
        Returns a tuple: (master_root, slave_root) if a merge happened, 
        or (master_root, None) if they were already unified.
        """
        root1 = self.find(point_idx_1)
        root2 = self.find(point_idx_2)
        
        if root1 == root2:
            return root1, None
            
        # Optimization: Pour the smaller set elements into the larger set
        if len(self.island_sets[root1]) < len(self.island_sets[root2]):
            root1, root2 = root2, root1
            
        # root1 absorbs root2
        self.parents[root2] = root1
        self.island_sets[root1].update(self.island_sets[root2])
        del self.island_sets[root2]
        
        return root1, root2

    def disjunction(self, root_id):
        """
        Clears and breaks down an entire island tree knowing only its master parent root ID.
        Resets all member nodes back to isolated singletons with a size of 1.
        """
        # Scan the flat parent array. Any node pointing directly to root_id
        # (or the root_id itself) belongs to the corrupt island.
        for idx in range(len(self.parents)):
            if self.parents[idx] == root_id:
                self.parents[idx] = idx
                self.sizes[idx] = 1

# =========================================================================
# ISOLATED UNIT TESTS SECTION
# =========================================================================
def test_dsu_basic_operations():
    """
    Validates DSU element size tracking, find path compression, 
    and multi-node union integrity.
    """
    print("\n--- Running: test_dsu_basic_operations ---")
    
    # Initialize 5 independent tracking elements (0 to 4)
    dsu = IslandDSU(5)
    
    # Verify unmerged single element conditions
    assert dsu.get_size(0) == 1
    assert dsu.get_size(3) == 1
    print("   [+] Initial isolated element state validated.")
    
    # Perform a single pair cluster union operation
    master, slave = dsu.merge(0, 1)
    assert dsu.find(0) == dsu.find(1)
    assert dsu.get_size(0) == 2
    assert dsu.get_size(1) == 2
    print(f"   [+] Core union verified. Master: {master}, Absorbed: {slave}")
    
    # Expand the cluster to three elements
    master, slave = dsu.merge(0, 2)
    assert dsu.get_size(master) == 3
    assert 2 in dsu.island_sets[master]
    print(f"   [+] Cluster growth verified. New size: {dsu.get_size(master)}")
    
    # Execute redundant bypass validation
    master_retry, slave_retry = dsu.merge(1, 2)
    assert master_retry == master
    assert slave_retry is None
    assert dsu.get_size(master) == 3
    print("   [+] Redundant merge operations bypassed safely.")


def test_reflection_invariant():
    """
    Verifies the linear parallelogram reflection formula mapping.
    """
    print("\n--- Running: test_reflection_invariant ---")
    coord_a = (0, 0)
    coord_b = (1, 0)
    coord_c = (0, 1)
    
    expected_d = (1, 1)
    calculated_d = topological_reflection(coord_b, coord_c)
    
    assert calculated_d == expected_d, f"Expected {expected_d}, got {calculated_d}"
    print(f"   [+] Parallel vector reflection verified: {calculated_d}")


def test_transform_discovery():
    """
    Simulates transform matrix discovery via common baseline match 
    and maps single points between mismatched coordinate spaces.
    """
    print("\n--- Running: test_transform_discovery_and_merger ---")
    
    # Island B is rotated 120 deg (idx 2) and shifted by (10, -5) relative to A
    p1_A = (0, 0)
    p1_B = (3, 1)    # (3, 1) rotated 120deg + shifted maps exactly to (11, -9)
    shift = (10,-5)
    rot = 2
    p2_A = apply_lattice_transform(p1_A, (rot, shift))
    p2_B = apply_lattice_transform(p1_B, (rot, shift))

    # Resolve transformation
    transform_result = find_discrete_transform(p1_A, p1_B, p2_A, p2_B)
    assert transform_result is not None, "Failed to resolve transform metrics."
    
    rot_idx, shift_result = transform_result
    assert rot_idx == rot, f"Incorrect rotation index found: {rot_idx}"
    assert shift == shift_result, f"Incorrect shift found: {shift_result}"
    print(f"   [+] Solved Matrix parameters: Rotation Idx={rot_idx}, Shift={shift}")


def test_graph_dissection_and_fusion():
    """
    Simulates complex multi-island segmentation, wave propagation, 
    collision discovery, and global lattice system unification using
    pure geometric extrapolation over a universally CCW oriented triangle abc.
    """
    print("\n--- Running: test_graph_dissection_and_fusion ---")

    # Ground truth reference space where all triangles are oriented CCW
    ideal_lattice = {
        0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1),
        4: (2, 0), 5: (1, -1), 6: (2, 1)
    }

    # Dissect layout into two disjoint sub-lattices
    island1_nodes = [0, 1, 2, 5]
    island2_nodes = [3, 4, 6]

    # Target transform parameters mapping Island 2 coordinates away
    rot_idx_target = 2
    shift_target = (15, -30)

    # Initialize separate coordinate spaces for each island
    island_grids = [{}, {}]

    # Populate coordinates for Island 1 (remains in ideal space)
    for node in island1_nodes:
        island_grids[0][node] = ideal_lattice[node]

    # Distort and populate coordinates for Island 2
    for node in island2_nodes:
        island_grids[1][node] = apply_lattice_transform(
            ideal_lattice[node], (rot_idx_target, shift_target)
        )

    # Track grouping with DSU structure
    dsu = IslandDSU(len(ideal_lattice))

    # Build initial disjoint sets for both clusters
    for n in island1_nodes:
        dsu.merge(island1_nodes[0], n)

    for n in island2_nodes:
        dsu.merge(island2_nodes[0], n)

    root_1 = dsu.find(island1_nodes[0])
    root_2 = dsu.find(island2_nodes[0])

    assert dsu.get_size(root_1) == len(island1_nodes)
    assert dsu.get_size(root_2) == len(island2_nodes)
    print("   [+] Topology dissection phase checked out cleanly.")

    # Simulated wave propagation frontier locates a boundary collision.
    # Under universal CCW rules:
    # Triangle (0, 1, 2) is CCW. Reflecting node 0 over edge (1, 2)
    # yields node 3 (1, 1). Ray extension from 0 through 1 yields node 4 (2, 0).
    target_hit_node_a = 3  # Matches p_1A geometrically (the reflected a')
    target_hit_node_b = 4  # Matches p_1B geometrically (the extrapolated point)

    hit_island_root = dsu.find(target_hit_node_a)

    if hit_island_root == root_2:
        print("   [!] Intersection located on foreign cluster. Re-aligning matrices...")

        # Source triangle vertices residing strictly inside Island 1
        a = island_grids[0][0]  # (0, 0)
        b = island_grids[0][1]  # (1, 0)
        c = island_grids[0][2]  # (0, 1)

        # Dynamic geometric extrapolation in Island 1 space using your functions.
        # To get the true external vertex a' across edge bc, we pass the edge
        # in the correct topological direction matching your cycle table:
        p_1A = topological_reflection(b, c)  # Properly yields node 3: (1, 1)
        assert p_1A == (1,1)
        p_1B = extrapolate_coordinate(a, b)  # Properly yields node 4: (2, 0)
        assert p_1B == (2,0)

        # Get the local positions of these identical boundary markers inside Island 2
        p_2A = island_grids[1][target_hit_node_a]
        p_2B = island_grids[1][target_hit_node_b]

        # Discover the transformation required to merge Island 2 into Island 1
        discovered_transform = find_discrete_transform(p_1A, p_1B, p_2A, p_2B)
        assert discovered_transform is not None, "Failed to resolve alignment transformation matrix."

        disc_rot, disc_shift = discovered_transform
        assert disc_rot == rot_idx_target, f"Mismatched rotation index {disc_rot} , expected {rot_idx_target}"
        assert disc_shift == shift_target, f"Mismatched shift vector {disc_shift}, expected {shift_target}"

        # Verify that transforming local p_2A yields its correct global coordinate (1, 1)
        test_coord = apply_lattice_transform((1,1), discovered_transform)
        assert test_coord == p_2A, f"Transform failed validation check. Got: {test_coord}"

        print(f"   [+] Extracted transformation safely resolved system drift.")
        print(f"       Resolved Rotation: {disc_rot}, Shift: {disc_shift}")

        # Complete full structural fusion inside DSU
        dsu.merge(root_1, hit_island_root)

    final_root = dsu.find(0)
    assert dsu.get_size(final_root) == len(island1_nodes) + len(island2_nodes)
    print(f"   [+] Post-fusion check passed successfully.")


def test_matrix_extraction():
    """Validates structural extraction matrix matching your coordinate system."""
    sample_grid = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0)
    }

    dense_matrix = convert_grid_to_matrix(sample_grid)

    # CORRECTED: Node 1 (0, 1) goes to row 1, Node 2 (1, 0) goes to row 0
    expected_matrix = np.array([
        [0, 2],
        [1, -1]
    ], dtype=np.int32)

    assert np.array_equal(dense_matrix, expected_matrix), "Matrix elements mismatch ground truth configuration."
    print("   [+] Matrix extraction test passed cleanly!")


def test_matrix_grid_roundtrip():
    print("\n--- Running: test_matrix_grid_roundtrip ---")

    # 1. Prepare sample topological grid matching your universal CCW space
    original_grid = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1)
    }

    # 2. Transform to dense row-parity neighbor matrix layout
    matrix_layout = convert_grid_to_matrix(original_grid)
    print("   [+] Constructed Matrix Layout:")
    print(matrix_layout)

    # 3. Reconstruct back to sparse dictionary via inverse mapping
    reconstructed_grid = convert_matrix_to_grid(matrix_layout)

    # 4. Assert total state equality
    assert len(original_grid) == len(reconstructed_grid), "Mismatched element volume."
    for pid, coords in original_grid.items():
        assert reconstructed_grid[pid] == coords, f"Node {pid} drifted: {reconstructed_grid[pid]} != {coords}"

    print("   [+] Bidirectional matrix-grid conversions validated successfully!")


def test_barycentric_rotation_reversibility():
    """
    Exhaustively sweeps a broad coordinate plane to verify the strict
    mathematical reversibility of the rotate_barycentric operator.
    Ensures that for any coordinate, rotating by k and then by (6 - k)
    collapses back to the original index with zero pixel drift.
    """
    print("\n=======================================================")
    print("Launching Bounded Barycentric Inversibility Audit...")
    print("=======================================================")

    total_checks = 0
    passed_checks = 0

    # Define an expansive coordinate window spanning from deep negative
    # tracking zones out to the upper limits of a large calibration plate
    min_test_coord = -50
    max_test_coord = 50

    # 1. Sweep across the entire simulated spatial plane
    for orig_r in range(min_test_coord, max_test_coord):
        for orig_c in range(min_test_coord, max_test_coord):

            # 2. Iterate through all possible discrete angular steps (0 to 5)
            for k in range(6):
                total_checks += 1
                r_rotated, c_rotated = rotate_barycentric(orig_r, orig_c, k)
                r_restored, c_restored = rotate_barycentric(r_rotated, c_rotated, -k)
                r_restored_full, c_restored_full = rotate_barycentric(r_rotated, c_rotated, 6-k)

                try:
                    assert orig_r == r_restored, (
                        f"Row inversion failure at k={k}: "
                        f"Started with row {orig_r}, but restored row {r_restored}"
                    )
                    assert orig_c == c_restored, (
                        f"Col inversion failure at k={k}: "
                        f"Started with col {orig_c}, but restored col {c_restored}"
                    )
                    assert orig_r == r_restored_full, (
                        f"Row inversion failure at k={k}: "
                        f"Started with row {orig_r}, but restored row {r_restored_full}"
                    )
                    assert orig_c == c_restored_full, (
                        f"Col inversion failure at k={k}: "
                        f"Started with col {orig_c}, but restored col {c_restored_full}"
                    )

                    passed_checks += 1

                except AssertionError as err:
                    print(f"\n -> [CRASH] Geometric Reversibility Broken!")
                    print(f"    Original Vector:  Row {orig_r}, Col {orig_c}")
                    print(f"    Rotated State:    k = {k} ({k * 60} deg CCW) -> Row {r_rotated}, Col {c_rotated}")
                    print(f"    Restored State:   Row {r_restored}, Col {c_restored}")
                    print(f"    Details: {err}")
                    raise err

    print("\n=======================================================")
    print("BARYCENTRIC INVERSIBILITY AUDIT PASSED PERFECTLY!")
    print(f"Verified Checks: {passed_checks} / {total_checks} Coordinate Permutations")
    print("THE 60-DEGREE ROTARY RING IS PROVEN FULLY INVARIANT!")
    print("=======================================================")


def test_hexagonal_grid_parity_invariance():
    """
    Explicitly validates if the rotate_barycentric operator preserves the
    fundamental grid tracking invariant (c = u + r // 2) across both
    positive and negative coordinate spaces.
    """
    print("\n=======================================================")
    print("Launching Hexagonal Grid Parity Invariance Test...")
    print("=======================================================")

    # 1. Choose a valid baseline target cell (Row 0, Col 24)
    r_orig, c_orig = -10, 24

    # Calculate its unwarped algebraic coordinates
    v_orig = r_orig
    u_orig = c_orig - v_orig // 2
    w_orig = -v_orig - u_orig

    # 2. Apply a 60-degree CCW rotation (k = 1)
    # This mathematically permutes the algebraic vectors:
    # u_new = -w_orig = 24
    # v_new = -u_orig = -24  <-- NOTICE THIS BECOMES NEGATIVE!
    # w_new = -v_orig = 0
    expected_u_rot = -w_orig
    expected_v_rot = -u_orig
    r_expected = expected_v_rot
    c_expected = expected_u_rot + r_expected // 2

    # 3. Invoke your physical coordinate rotation function
    r_rot, c_rot = rotate_barycentric(r_orig, c_orig, k=1)

    print(f" -> Source Cell:   Row {r_orig}, Col {c_orig}")
    print(f" -> Rotated Array: Row {r_rot}, Col {c_rot}")
    print(f" -> Expected Array:Row {r_expected}, Col {c_expected}")
    print(f" -> Expected Alg:  u = {expected_u_rot}, v = {expected_v_rot}")

    v_recovered = r_rot
    u_recovered = c_rot - (v_recovered // 2)

    try:
        assert v_recovered == expected_v_rot, (
            f"Row assignment desynchronized! Expected {expected_v_rot}, got {v_recovered}"
        )
        assert u_recovered == expected_u_rot, (
            f"Lattice Tearing Detected! Bounded grid parity rule broken in negative space. "
            f"Expected algebraic u = {expected_u_rot}, but recovered u = {u_recovered}"
        )
        print("\n -> [PASS] Grid structural invariants perfectly preserved!")
        print("=======================================================")

    except AssertionError as err:
        print("\n -> [CRASH] Geometric Grid Bug Successfully Isolated!")
        print(f"    Details: {err}")
        print("=======================================================")
        raise err


def test_matrix_rotation(H, W):
    # 1. Initialize a 4x4 array with consecutive numbers (0 to 15)
    matrix_orig = np.arange(H * W).reshape(H, W)

    print(f"--- ORIGINAL {H}x{W} CONSECUTIVE GRID ---")
    for r in range(H):
        print(f"Row {r}: {matrix_orig[r, :]}")

    # 2. Track where coordinates map under 180-degree rotation (k=3)
    matrix_rotated, (min_r, min_c) = rotate_barycentric_matrix_adaptive(matrix_orig, 3)

    print("\n--- 180-DEGREE ROTATED RECONSTRUCTED MATRIX ---")
    print(min_r, min_c)
    for r in range(matrix_rotated.shape[0]):
        print(f"Local Row {r}: {matrix_rotated[r, :]}")

def test_hexagonal_phase_bijection_pure():
    """Sweeps the complete 31x31 space using raw inline assert constraints."""
    period = 31
    hit_map = np.zeros((period, period), dtype=np.int32)

    for r_start in range(period):
        for c_start in range(period):
            # 1. Forward transform
            u_phase, v_phase = get_phase_from_coordinates(r_start, c_start, period)

            # Assert phase parameters are strictly positive primitives within Galois ring bounds
            assert u_phase >= 0, f"Negative u at ({r_start}, {c_start})"
            assert v_phase >= 0, f"Negative v at ({r_start}, {c_start})"
            assert u_phase < period, f"u overflow at ({r_start}, {c_start})"
            assert v_phase < period, f"v overflow at ({r_start}, {c_start})"

            # 2. Reverse transform
            r_rec, c_rec = get_coordinates_from_phase(u_phase, v_phase, period)

            # Assert perfect 1-to-1 inversion loop recovery
            assert r_start == r_rec, f"Row error: expected {r_start}, got {r_rec}"
            assert c_start == c_rec, f"Col error: expected {c_start}, got {c_rec}"

            hit_map[r_rec, c_rec] += 1

    # Assert zero holes and zero duplicated coordinate space collisions
    assert np.all(hit_map == 1), "Topology broken: found empty holes or duplicate collisions!"

    print(f" -> Success! All {period * period} nodes verified. 100% Bijective.")


# =========================================================================
# AUTOMATED TEST SUITE RUNNER
# =========================================================================

if __name__ == "__main__":
    print("=========================================================")

    test_matrix_rotation(3,4)
    test_matrix_rotation(4,4)
    test_matrix_rotation(4,3)
    test_hexagonal_phase_bijection_pure()
    test_dsu_basic_operations()
    test_reflection_invariant()
    test_transform_discovery()
    test_graph_dissection_and_fusion()
    test_matrix_extraction()
    test_matrix_grid_roundtrip()
    test_barycentric_rotation_reversibility()
    test_hexagonal_grid_parity_invariance()
    print("\n=========================================================")
    print("[SUCCESS] ALL ISOLATED UNIT TESTS EXECUTED AND PASSED.")
    print("=========================================================")
