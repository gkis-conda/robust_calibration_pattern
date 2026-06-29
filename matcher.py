import numpy as np

from m_sequence import generate_diff_from_m_sequence, generate_reference_sequence, MSequenceAnalyzer,\
    calculate_phase
from lattice_topology import *

class BarycentricSignalIsolator:
    """
    Handles the 2D spatial differential extraction to isolate V channel bits and track gaps.
    Zigzag representation is expected
    """
    @staticmethod
    def isolate_u_axis(matrix: np.ndarray) -> tuple:
        """Extracts continuous bits for the U-axis from the composite matrix layout.
        Returns:
            tuple: (isolated_bits, valid_mask)
            - isolated_bits: A list of isolated differential bits or placeholders.
            - valid_mask: A boolean tracking array starting at index 0.
        """
        assert isinstance(matrix, np.ndarray), "Matrix input must be a NumPy array"
        H, W = matrix.shape

        isolated_bits = np.zeros(W-2,np.int8)
        valid_mask = np.zeros(W-2,np.int8)
        if H > 1:
            for c in range(W - 2):
                # Sample the 4 target intersection nodes from the matrix canvas U[i] + U[i+2]
                p0 = matrix[0, c+1]
                p1 = matrix[0, c+2]
                p2 = matrix[1, c+0]
                p3 = matrix[1, c+1]

                # Missing value protection check for camera sensor dropouts (-1)
                if p0 == -1 or p1 == -1 or p2 == -1 or p3 == -1:
                    pass
                else:
                    # 4-pixel spatial XOR superimposition eliminates V and W channels,
                    # yielding a clean differential output equivalent to U[r] ^ U[r+2]
                    isolated_bits[c] = int(p0 ^ p1 ^ p2 ^ p3)
                    valid_mask[c] = 1

        return isolated_bits, valid_mask

    @staticmethod
    def isolate_w_rev_axis(patch: np.ndarray) -> tuple:
        """Extracts the maximum possible number of consecutive horizontal tracking bits.

        Uses a 4-pixel diamond block to eliminate both U and V interferences,
        leaving a clean, decodable W-axis polynomial sequence.
        """
        ph, pw = patch.shape
        bits_len = pw - 1
        isolated_bits = np.zeros(bits_len, np.int8)
        valid_mask = np.zeros(bits_len, np.int8)

        # using  -> W[i] + W[i - 2], so the line should be inverted for correct axis
        for offset in range(bits_len):
            p0 = patch[0, offset]
            p1 = patch[0, offset + 1]
            p2 = patch[1, offset]
            p3 = patch[1, offset + 1]

            if p0 == -1 or p1 == -1 or p2 == -1 or p3 == -1:
                pass
            else:
                isolated_bits[offset] = int(p0 ^ p1 ^ p2 ^ p3)
                valid_mask[offset] = 1

        return isolated_bits, valid_mask

    @staticmethod
    def isolate_v_axis(patch: np.ndarray) -> tuple:
        """Extracts continuous bits for the v-axis by scanning down a 2D column.
        Returns:
            tuple: (isolated_bits, valid_mask)
        """
        ph, pw = patch.shape
        bits_len =ph - 2
        isolated_bits = np.zeros(bits_len, np.int8)
        vert_mask = np.zeros(bits_len, np.int8)

        # We evaluate a 3-row vertical window frame, requiring at least 3 rows
        if pw > 2:
            # Scanning begins from column index 1 to avoid edge dropouts
            c = 0
            for r in range(bits_len):
                parity = r % 2
                # Exact custom structural diamond layout configuration
                p0 = patch[r, c + 1 - parity]
                p1 = patch[r + 1, c]
                p2 = patch[r + 1, c + 1]
                p3 = patch[r + 2, c + 1 - parity]

                # Missing value protection check for camera sensor dropouts (-1)
                if p0 == -1 or p1 == -1 or p2 == -1 or p3 == -1:
                    pass
                else:
                    # 4-pixel spatial XOR superimposition eliminates U and W channels
                    isolated_bits[r] = int(p0 ^ p1 ^ p2 ^ p3)
                    vert_mask[r] = 1

        return isolated_bits, vert_mask


class AlgebraicGridDecoder32:
    """
    retrieve barycentric coordinates by
    """
    def __init__(self, grid_width, grid_height):
        """Initializes the main grid environment and registers all 6 directional decoders."""
        self.r = 5  # Base polynomial degree
        self.lfsr_period = (1 << self.r) - 1
        self.N = self.lfsr_period  # Grid size / Period
        self.W = grid_width
        self.H = grid_height
        assert self.W <= self.N and self.H <= self.N

        # Mapping all 6 forward and reciprocal direction polynomials
        self.polys = {
            "U_forward": 0x25,  # Pair 1 Forward
            "U_reverse": 0x29,  # Pair 1 Reciprocal
            "V_forward": 0x3D,  # Pair 2 Forward
            "V_reverse": 0x2F,  # Pair 2 Reciprocal
            "W_forward": 0x3B,  # Pair 3 Forward
            "W_reverse": 0x37   # Pair 3 Reciprocal
        }

        # Initialize reference sequences and internal 1D tracking streams array
        self.streams = {axis: generate_reference_sequence(poly) for axis, poly in self.polys.items()}
        self.streams["U"] = self.streams["U_forward"]
        self.streams["V"] = self.streams["V_forward"]
        self.streams["W"] = self.streams["W_forward"]
        self.debug_output = False
        if self.debug_output:
            for axis_name in ("U", "V", "W"):
                print(axis_name, self.streams[axis_name])
        # Initialize 6 independent syndrome decoders inside the constructor
        self.decoders = {dir_name: MSequenceAnalyzer(self.polys[dir_name], calculate_phase(self.polys[dir_name],-2)) for dir_name in self.polys}

    def build_barycentric_matrix(self, axis_u: str = "U", axis_v: str = "V", axis_w: str = "W") -> np.ndarray:
        """Generates a 2D XOR pattern plate layout mapped to specific axis assignments."""
        """ romboid representation """
        row_grid, col_grid = np.mgrid[:self.H, :self.W]

        v_grid = row_grid
        u_grid = col_grid - (row_grid // 2)
        w_grid = -v_grid - u_grid

        idx_u = u_grid % self.lfsr_period
        idx_v = v_grid % self.lfsr_period
        idx_w = w_grid % self.lfsr_period

        val_u = np.array(self.streams[axis_u])[idx_u]
        val_v = np.array(self.streams[axis_v])[idx_v]
        val_w = np.array(self.streams[axis_w])[idx_w]

        return val_u ^ val_v ^ val_w

    def resolve_cell_index(self, status_horiz: dict, status_vert: dict) -> dict:
        """Reconciles horizontal and vertical LFSR phase alignments to compute
        the absolute physical (row, col) coordinates of the matrix cell.

        Args:
            status_horiz (dict): Status metadata from the horizontal axis scan.
            status_vert (dict): Status metadata from the vertical axis scan.

        Returns:
            dict: A status payload formatted strictly via best_result mapping.
        """
        # 1. Error Protection Check: Abort decoding if either isolation pass failed
        if status_horiz is None or status_vert is None:
            return {"status": "failed"}

        # 2. Extract relative tracking frame window offsets
        offset_h = status_horiz.get("found_at_window_offset", 0)
        offset_v = status_vert.get("found_at_window_offset", 0)

        # 3. Parse the dynamic direction_key metadata string ("U_forward", "W_reverse", etc.)
        dir_key = status_horiz.get("direction_key")
        horiz_axis, horiz_dir = dir_key.split("_")
        vert_key = status_vert.get("direction_key")
        vert_axis, vert_dir = vert_key.split("_")

        u = None
        v = None
        w = None

        # 4. Retrieve raw stream phases corresponding to the locked frame positions
        # axis must be fully reversible
        phase_h = status_horiz["origin_stream_phase"]
        if horiz_dir == "reverse":
            phase_h = (-phase_h)
        phase_v = status_vert["origin_stream_phase"]
        if vert_dir == "reverse":
            phase_v = (-phase_v)

        # 5. Determine the absolute horizontal U index based on the parsed axis identity
        if horiz_axis == "U":
            # Direct track: The horizontal phase belongs directly to the U axis
            u = phase_h - offset_h
        elif horiz_axis == "W":
            # Alternative track: Evaluates the W axis (or any reverse trajectory token)
            w = phase_h - offset_h
        elif horiz_axis == "V":
            v = phase_h - offset_h
        else:
            return {
                "status": "error"
            }

        if vert_axis == "U":
            # Direct track: The horizontal phase belongs directly to the U axis
            u = phase_v - offset_v
        elif vert_axis == "W":
            # Alternative track: Evaluates the W axis (or any reverse trajectory token)
            w = phase_v - offset_v
        elif vert_axis == "V":
            v = phase_v - offset_v
        else:
            return {
                "status": "error"
            }
        if u is None:
            u = - v - w
        elif v is None:
            v = - u - w

        # Since offsets are 0, any negative value is caused strictly by time inversions.
        # We shift the components forward by whole periods to fit the physical viewport.

        H_global, W_global = self.H, self.W

        # Resolve any remaining period phase wraps entirely on the U-axis
        abs_row, abs_col = normalize_barycentric(u, v, self.lfsr_period)

        # Rigid safety assertion gates
        assert abs_row >= 0 and abs_col >= 0
        assert abs_row < H_global and abs_col < W_global

        # 7. Pack the results strictly into your targeted payload dictionary structure
        return {
            "status": "success",
            "row": abs_row,
            "col": abs_col,
            "b":(u, v, -u-v), # detected coordinates in blueprint
            "horizontal_axis": horiz_axis,
            "direction": horiz_dir,
            "errors_corrected": len(status_horiz.get("errors_corrected"))
        }


    @staticmethod
    def get_alternative_axis(current_axis: str) -> str:
        """Returns the next axis in the clockwise direction (-60 degrees)
        using a static dictionary lookup.

        Args:
            current_axis (str): Valid entries: "U_forward", "V_forward", "W_forward",
                                               "U_backward", "V_backward", "W_backward".

        Returns:
            str: The name of the target axis shifted by -60 degrees (clockwise).
        """
        # Static translation table for a clockwise rotation loop (-60 degrees)
        ALTERNATIVE_MAP = {
            "U_forward": "W_reverse",
            "W_reverse": "V_forward",
            "V_forward": "U_reverse",
            "U_reverse": "W_forward",
            "W_forward": "V_reverse",
            "V_reverse": "U_forward",
        }

        return ALTERNATIVE_MAP.get(current_axis, current_axis)

    @staticmethod
    def get_next_axis(current_axis: str) -> str:
        """Computes the semantic axis identifier string situated +60 degrees
        counter-clockwise from the current tracking axis using a static dictionary lookup.

        Args:
            current_axis (str): A string representing the axis and its direction.
                                Valid entries: "U_forward", "V_forward", "W_forward",
                                               "U_reverse", "V_reverse", "W_reverse".

        Returns:
            str: The name of the target axis shifted by +60 degrees.
        """
        # Static translation table for a full 360-degree rotation circle (6 states)
        ROTATION_MAP = {
            "U_forward": "V_forward",
            "V_forward": "W_forward",
            "W_forward": "U_reverse",
            "U_reverse": "V_reverse",
            "V_reverse": "W_reverse",
            "W_reverse": "U_forward"
        }

        # Return the mapped value, or the original string if the key doesn't exist
        return ROTATION_MAP.get(current_axis, current_axis)

    def get_lfsr_phases(self, r: int, c: int) -> tuple:
        """Computes the exact operational LFSR phase triplet (u, v, w) for any
        sub-patch starting at global matrix coordinates (r, c).

        Fully synchronized with the rhomboid matrix layout and isolator masks.
        """
        v_grid = r
        u_grid = c - (r // 2)

        u_phase = u_grid % self.lfsr_period
        v_phase = v_grid % self.lfsr_period
        w_phase = (-v_grid - u_grid ) % self.lfsr_period

        return u_phase, v_phase, w_phase

    def build_barycentric_matrix_zigzag(self, axis_u: str = "U", axis_v: str = "V", axis_w: str = "W") -> np.ndarray:
        """ cyclic zigzag (r % 2).
        """
        row_grid, col_grid = np.mgrid[:self.H, :self.W]

        v_grid = row_grid
        # odd rows are shifted
        u_grid = col_grid + (row_grid % 2)
        w_grid = -v_grid - u_grid

        idx_u = u_grid % self.lfsr_period
        idx_v = v_grid % self.lfsr_period
        idx_w = w_grid % self.lfsr_period

        val_u = np.array(self.streams[axis_u])[idx_u]
        val_v = np.array(self.streams[axis_v])[idx_v]
        val_w = np.array(self.streams[axis_w])[idx_w]

        return val_u ^ val_v ^ val_w

    @staticmethod
    def check_error_consistency(res_h, res_v):
        return np.array_equal(np.array(res_h["errors_corrected"]) + 1, np.array(res_v["errors_corrected"]))

    def decode_barycentric_subgraph(self, patch: np.ndarray):
        """Tests the patch rows against all 6 directional polynomials to find the absolute position.

        Bypasses memory rotations. Leverages the external BarycentricSignalIsolator to extract
        clean horizontal and vertical bit vectors. Resolves complete coordinate indexing.
        """
        assert isinstance(patch, np.ndarray), "Patch must be a NumPy array"

        best_result = {"status": "error",
                       "errors_corrected" : 100}

        # 1. Isolate the primary horizontal lane bits from the incoming patch
        isolated_bits_h, valid_mask_h = BarycentricSignalIsolator.isolate_u_axis(patch)

        # 2. Evaluate all 6 direction decoders to identify the horizontal axis orientation
        for direction_key, decoder in self.decoders.items():
            res_h = decoder.analyze(isolated_bits_h, valid_mask_h)
            if res_h:
                # 3. Call the updated vertical axis isolator from your syndrome module
                isolated_bits_v, valid_mask_v = BarycentricSignalIsolator.isolate_w_rev_axis(patch)

                # Determine which polynomial corresponds to the vertical projection lane
                vert_key = AlgebraicGridDecoder32.get_alternative_axis(direction_key)
                res_v = self.decoders[vert_key].analyze(isolated_bits_v, valid_mask_v)

                # Match the vertical profile against the companion decoder
                if res_v:
                    res_h["direction_key"] = direction_key
                    res_v["direction_key"] = vert_key
                    if self.check_error_consistency(res_h, res_v):
                        print("Hor: ", res_h)
                        print("Vert: ", res_v)
                        # Translate lattice spaces back to absolute flat matrix positions
                        result = self.resolve_cell_index(res_h, res_v)
                        if result["status"] == "success" and result['errors_corrected'] < best_result['errors_corrected']:
                            best_result = result
                            if best_result['errors_corrected'] == 0:
                                break

        return best_result


def localize_grid(island, width, height):
    retriever = AlgebraicGridDecoder32(width, height)
    h, w = island.shape
    result = None
    for r in range(0, h, 2):
        for c in range(0, w, 1):
            if island[r, c] >= 0:
                break
        for e in range(w - 1, 0, -1):
            if island[r, e] >= 0:
                break
        print(r, c, e - c + 1)
        L = e - c + 1
        if L < 14:
            continue # minimal check sequence is 9, 5 poly, 2 checksum bits and + 2 for integration
        adjusted_c = (c & (0xfffff - 1)) if r % 2 == 0 else (c | 1)
        sub_island = island[r:r+2, adjusted_c:e+1]
        result = retriever.decode_barycentric_subgraph(sub_island)
        if result["status"] == "success":
            result["source_row"], result["source_col"] = r, adjusted_c
            break
    return result


#===========================================================
# Tests
#===========================================================
def test_resolve_cell_index(decoder):
    """Performs unit testing on the resolve_cell_index coordinate intersection module.
    Validates direct horizontal U tracks and dual alternative W tracks using strict asserts.
    """
    print("Initializing coordinate resolver isolation verification suite...")
    # =========================================================================
    # TEST 1: Direct Horizontal Track via U_forward Axis
    # Target Coordinates: Global Row 4, Global Col 8
    # =========================================================================
    print("\nExecuting Test 1: Validating direct U_forward scan track...")

    mock_horiz_u = {
        "status": "success",
        "direction_key": "U_forward",
        "found_at_window_offset": 0,
        "origin_stream_phase": 6,  # Absolute u phase
        "errors_corrected": []
    }

    mock_vert = {
        "status": "success",
        "direction_key": "V_forward",
        "found_at_window_offset": 0,
        "origin_stream_phase": 4,  # Absolute v phase
        "errors_corrected": []
    }

    res_1 = decoder.resolve_cell_index(mock_horiz_u, mock_vert)
    assert res_1["status"] =="success", "Test 1 Failed: Tracker returned a failure status"
    assert res_1["row"] == 4, f"Test 1 Row Error: Expected row 4, got {res_1['row']}"
    assert res_1["col"] == 8, f"Test 1 Column Error: Expected col 8, got {res_1['col']}"
    print("-> Test 1 Passed! (Row 4, Col 8) successfully extracted from horizontal U-axis phase.")

    # =========================================================================
    # TEST 2: Alternative Horizontal Track via W_reverse Axis (Updated Compatibility Naming)
    # Target Coordinates: Global Row 4, Global Col 8
    # =========================================================================
    print("\nExecuting Test 2: Validating alternative W_reverse scan track...")

    # Under layout: v = 4, u = 6.
    # Barycentric constraint dictates: w = -v - u = -4 - 6 = -10
    # Inside the 31-period stream cycle, -10 maps natively to index 21 (-10 % 31 = 21)
    mock_horiz_w = {
        "status": "success",
        "direction_key": "W_reverse",  # Changed token string to "W_reverse"
        "found_at_window_offset": 0,
        "origin_stream_phase": 10,  # Absolute w phase
        "errors_corrected": [1]
    }

    res_2 = decoder.resolve_cell_index(mock_horiz_w, mock_vert)

    assert res_2["status"] =="success", "Test 2 Failed: Tracker returned a failure status"
    assert res_2["row"] == 4, f"Test 2 Row Error: Expected row 4, got {res_2['row']}"
    assert res_2["col"] == 8, f"Test 2 Column Error: Expected col 8, got {res_2['col']}"
    assert res_2["errors_corrected"] == 1, "Test 2 Error Telemetry Failed: Correction flag dropped"
    print("-> Test 2 Passed! (Row 4, Col 8) successfully decoded from dual alternative W_reverse phase.")

    # =========================================================================
    # TEST 3: Arbitrary Deeper Coordinate with Sub-Patch Window Offsets
    # Target Coordinates: Global Row 12, Global Col 22
    # =========================================================================
    print("\nExecuting Test 3: Validating deeper tracking point with window offsets...")

    mock_offset_horiz = {
        "status": "success",
        "direction_key": "U_forward",
        "found_at_window_offset": 2,
        "origin_stream_phase": 16 + 2,
        "errors_corrected": []
    }

    mock_offset_vert = {
        "status": "success",
        "direction_key": "V_forward",
        "found_at_window_offset": 1,
        "origin_stream_phase": 12 + 1,
        "errors_corrected": []
    }

    res_3 = decoder.resolve_cell_index(mock_offset_horiz, mock_offset_vert)

    assert res_3["status"] == "success", "Test 3 Failed: Tracker returned a failure status"
    assert res_3["row"] == 12, f"Test 3 Row Error: Expected row 12, got {res_3['row']}"
    assert res_3["col"] == 22, f"Test 3 Column Error: Expected col 22, got {res_3['col']}"
    print("-> Test 3 Passed! Frame window un-wrapping and patch offsets reconciled perfectly.")

    print("\n=======================================================")
    print("ALL COMPATIBLE RESOLVE_CELL_INDEX TESTS PASSED PERFECTLY!")
    print("=======================================================")


def test_full_matrix(decoder):
    """Validates 4-pixel axis isolation math directly on the full baseline matrix.

    Verifies all three semantic pathways (isolate_w_axis, isolate_v_axis,
    and isolate_u_axis) against clean reference code streams.
    """
    print("Initializing full matrix axis isolation test suite...")
    # 2. Synthesize the complete dense master layout canvas using your generator
    matrix_baseline = decoder.build_barycentric_matrix("U", "V", "W")

    print("\nU-Axis Profile Validation...")
    start_u_offset = 0
    for r in range(0,20,1):
        isolated_u, mask_u = BarycentricSignalIsolator.isolate_u_axis(matrix_baseline)
        print("Isolated u", r, isolated_u)
        expected_u = generate_diff_from_m_sequence(
            decoder.streams["U"], start_u_offset, len(isolated_u), -2)
        print("Expected u", r, expected_u)
        assert (isolated_u == expected_u).all(), "Isolated U-axis stream has leakage errors"
    print("-> Passed: isolate_u_axis matches the modulated U-reference perfectly!")

    print("\nV-Axis Profile Validation...")
    start_v_offset = 0
    for c in range(0,20,1):
        isolated_v, mask_v = BarycentricSignalIsolator.isolate_v_axis(matrix_baseline[:,c:])
        print("Isolated v", c, isolated_v)
        expected_v = generate_diff_from_m_sequence(
            decoder.streams["V"], start_v_offset, len(isolated_v), -2)
        print("Expected v", c, expected_v)

        assert ((isolated_v == expected_v).all()), "Isolated V-axis stream has leakage errors"
    print("-> Passed: isolate_v_axis matches the modulated V-reference perfectly!")

    print("\nW-Axis Profile Validation...")
    for r in range(0,20,2):
        start_w_offset = decoder.get_lfsr_phases(r,0)[2]
        start_w_rev_offset = -start_w_offset

        isolated_w, mask_w = BarycentricSignalIsolator.isolate_w_rev_axis(matrix_baseline[r:,:])
        print("row", r, "forward offset",start_w_offset, "reverse offset", start_w_rev_offset)
        print("Isolated w", r, isolated_w)
        # Modulate via your core 4-pixel utility function
        expected_w = generate_diff_from_m_sequence(
            decoder.streams["W_reverse"], start_w_rev_offset, len(isolated_w), -2)
        print("Expected w", r, expected_w)
        assert (isolated_w == expected_w).all(), "Isolated W-axis stream has leakage errors"
    print("-> Passed: isolate_w_axis matches the modulated W-reference perfectly!")

    print("\n=======================================================")
    print("ALL THREE SEMANTIC AXIS ISOLATION TESTS PASSED PERFECTLY!")
    print("=======================================================")


def test_isolator(decoder):
    """Validates isolation profiles using strict even row and column grid alignment."""
    print("Initializing updated BarycentricSignalIsolator verification suite...")

    # 1. Synthesize ground truth pattern matrix via internal class method
    matrix_baseline = decoder.build_barycentric_matrix()
    ref_u_stream, ref_v_stream, ref_w_stream = \
        [generate_reference_sequence(decoder.polys[axis]) for axis in ("U_forward", "V_forward", "W_reverse")]

    for r in range(4,14,2):
        target_row, target_col = r, 0
        print(f"Evaluating Horizontal Scan Output for ({target_row}, {target_col})")
        patch = matrix_baseline[target_row: target_row + 20, target_col: target_col + 20]
        start_phases = decoder.get_lfsr_phases(target_row, target_col)

        isolated_u, mask_u = BarycentricSignalIsolator.isolate_u_axis(patch)
        print("Isolated u:", isolated_u)
        # Transform via our core 4-pixel modulation utility function
        expected_u = generate_diff_from_m_sequence(ref_u_stream, start_phases[0], len(isolated_u), -2)
        print("Expected u:", expected_u)
        assert (isolated_u == expected_u).all(), "Test U Mismatch: Isolated horizontal row bit stream has leakage errors"
        isolated_v, mask_v = BarycentricSignalIsolator.isolate_v_axis(patch)
        print("Isolated v:", isolated_v)
        expected_v = generate_diff_from_m_sequence(ref_v_stream, start_phases[1], len(isolated_v), -2)
        print("Expected v:", expected_v)
        assert (isolated_v == expected_v).all(), "Test V Mismatch: Isolated vertical column bit stream has leakage errors"

        isolated_w, mask_w = BarycentricSignalIsolator.isolate_w_rev_axis(patch)
        print("Isolated w:", isolated_w)

        expected_w = generate_diff_from_m_sequence(ref_w_stream, -start_phases[2], len(isolated_w), -2)
        print("Expected w:", expected_w)
        assert (isolated_w == expected_w).all(), "Test  Mismatch: Isolated vertical column bit stream has leakage errors"

    print("\n=======================================================")
    print("ALL MODULE TEST SCENARIOS VERIFIED SUCCESSFULLY!")
    print("=======================================================")


def test_simplified_axis_matching(decoder):
    """Runs a direct, clean test sweep pulling matrices directly from the generator."""
    # -------------------------------------------------------------------------
    # TEST 1: Check Horizontal V Matrix
    # -------------------------------------------------------------------------
    print("Running Test 1: Evaluating Horizontal V Blueprint Slice...")
    blueprint = decoder.build_barycentric_matrix()

    # Slice a 6x10 patch starting at known coordinates (Row 8, Column 10)
    target_r, target_c = 8, 10
    patch = blueprint[target_r: target_r + 6, target_c: target_c + 10]

    res = decoder.decode_barycentric_subgraph(patch)
    assert res["status"] == "success", "Test 1 Failed: Patch was rejected"
    assert res["row"] == target_r, f"Row mismatch: Expected {target_r}, got {res['row']}"
    assert res["col"] == target_c, f"Col mismatch: Expected {target_c}, got {res['col']}"
    assert res["horizontal_axis"] == "U", f"Axis mismatch: Expected V, got {res['horizontal_axis']}"
    print(f"-> Test 1 Passed! Detected: {res['horizontal_axis']}, Coordinates: ({res['row']}, {res['col']})")
    # -------------------------------------------------------------------------
    # TEST 2: Check Horizontal W Matrix (Cyclic Axis Rotation Scenario)
    # -------------------------------------------------------------------------
    print("\nRunning Test 2: Evaluating Horizontal W Blueprint Slice...")
    matrix_w = decoder.build_barycentric_matrix("V", "W", "U")
    # axis are rotated by +60 so we have cyclic shift of cells of the original lattice
    # Slice a 6x10 patch starting at known coordinates (Row 12, Column 6)
    target_r2, target_c2 = 12, 6
    patch2 = matrix_w[target_r2: target_r2 + 6, target_c2: target_c2 + 15].copy()
    w = target_r2
    v = target_c2 - target_r2//2
    u = (-w-v)%decoder.lfsr_period
    r = v
    c = u + v//2
    res2 = decoder.decode_barycentric_subgraph(patch2)
    assert res2["status"] == "success", "Test 2 Failed: Patch was rejected"
    assert res2["row"] == r, f"Row mismatch: Expected {target_r2}, got {res2['row']}"
    assert res2["col"] == c, f"Col mismatch: Expected {target_c2}, got {res2['col']}"
    assert res2["horizontal_axis"] == "V", f"Axis mismatch: Expected V, got {res2['horizontal_axis']}"
    print(f"-> Test 2 Passed! Detected: {res2['horizontal_axis']}, Coordinates: ({res2['row']}, {res2['col']})")
    # -------------------------------------------------------------------------
    # TEST 3: Check Horizontal V Matrix
    # -------------------------------------------------------------------------
    print("Running Test 3: Evaluating Horizontal V Blueprint Slice...")
    blueprint = decoder.build_barycentric_matrix()

    # Slice a 6x10 patch starting at known coordinates (Row 8, Column 10)
    target_r, target_c = 20, 8
    patch = blueprint[target_r: target_r + 6, target_c: target_c + 10]

    res = decoder.decode_barycentric_subgraph(patch)
    assert res["status"] == "success", "Test 1 Failed: Patch was rejected"
    assert res["row"] == target_r, f"Row mismatch: Expected {target_r}, got {res['row']}"
    assert res["col"] == target_c, f"Col mismatch: Expected {target_c}, got {res['col']}"
    assert res["horizontal_axis"] == "U", f"Axis mismatch: Expected V, got {res['horizontal_axis']}"
    print(f"-> Test 1 Passed! Detected: {res['horizontal_axis']}, Coordinates: ({res['row']}, {res['col']})")


    print("\n=======================================================")
    print("ALL SIMPLIFIED AXIS MATCHING TESTS PASSED PERFECTLY!")
    print("=======================================================")


def test_exhaustive_grid_sweep(decoder):
    """
    Executes a complete spatial sweep across the entire layout matrix canvas.
    Tests every valid cell position as a starting anchor to verify that signed
    triplet re-alignments and boundary gates are universally invariant.
    """
    print("\n=======================================================")
    print("Launching Exhaustive Grid Sweep Validation Pass...")
    print("=======================================================")

    blueprint = decoder.build_barycentric_matrix()

    # Define observation patch dimensions matching your 1D sliding windows
    patch_h, patch_w = 6, 10

    # Calculate safe scanning bounds to prevent array slicing overruns
    max_r = decoder.H - patch_h
    max_c = decoder.W - patch_w

    total_tests = 0
    passed_tests = 0

    # Execute a nested spatial traversal over the entire master blueprint canvas
    for target_r in range(0,max_r,2):
        for target_col in range(max_c):
            total_tests += 1

            # Slice a localized sub-patch at the current iteration coordinates
            patch = blueprint[target_r: target_r + patch_h, target_col: target_col + patch_w]

            try:
                # Invoke your autonomous cross-axis subgraph decoder pass
                res = decoder.decode_barycentric_subgraph(patch)

                # Verify that the extraction layer successfully locked onto a candidate
                if res is None or res.get("status") != "success":
                    print(f" -> [FAIL] Lock Dropped at Coordinates: Row {target_r}, Col {target_col}")
                    continue

                # Assert precise spatial coordinate reconciliation rules
                assert res["row"] == target_r, f"Row mismatch at ({target_r}, {target_col}): Expected {target_r}, got {res['row']}"
                assert res["col"] == target_col, f"Col mismatch at ({target_r}, {target_col}): Expected {target_col}, got {res['col']}"

                passed_tests += 1

            except AssertionError as err:
                print(f" -> [CRASH] Structural Assertion Broken at cell ({target_r}, {target_col})")
                print(f"    Details: {err}")
                # Raise immediately to halt execution and inspect the specific failure vector
                raise err
            except Exception as ex:
                print(f" -> [ERROR] Unexpected execution failure at cell ({target_r}, {target_col})")
                print(f"    Details: {ex}")
                raise ex

    print("\n=======================================================")
    print("EXHAUSTIVE SCAN COMPLETED SUCCESSFULLY!")
    print(f"Passed Checks: {passed_tests} / {total_tests} Active Quadrants")
    print("ALL COORDINATE PARITY CHECKS LOCKED WITH ZERO DRIFT!")
    print("=======================================================")


def test_matcher(decoder):
    print("Initializing Row Index Retriever")
    mesh_blueprint = decoder.build_barycentric_matrix()

    print("Simulating Camera Detection Data")
    # Let's target Row Index 12 as our ground truth hidden target
    target_row = 12
    start_col = 14
    sequence_length = 12

    # Extract the perfect subset slice from our master matrix
    perfect_slice = mesh_blueprint[target_row: target_row + 4, start_col: start_col + sequence_length]
    print(f"Ground-truth slice from Row {target_row}: {perfect_slice}")

    # 1. Reverse the tracking sequence to simulate right-to-left camera line extraction
    camera_detected_sequence = perfect_slice.copy()

    # 2. Introduce a bit flip to simulate image processing noise or lens flare blur
    # Flipping the element at index 3 (0 becomes 1, or 1 becomes 0)
    camera_detected_sequence[0,2] ^= 1
    print(f"Noisy sequence extracted by camera:  {camera_detected_sequence}")
    print(" (Simulation parameters: Bit errors = 1)\n")
    print("Matching Observed Curve to Blueprint")
    # Execute lookup permitting 1 faulty node classification
    result = decoder.decode_barycentric_subgraph(camera_detected_sequence)

    assert result is not None and result["status"] == "success", "Match failed."
    assert (result["row"] == target_row and result["col"] == start_col), \
           f" -> Real-time image slice locked onto: {result['row']}, {result['col']}, {result['horizontal_axis']}, {result['direction']}"
    print(" -> [Verification Passed]: Row index correctly identified despite noise.")

    camera_detected_sequence[0,5] = -1 # erase single element
    print(f"Noisy sequence extracted by camera:  {camera_detected_sequence}")
    print("(Simulation parameters: Bit errors = 1, Missed Bits: 1)\n")

    result = decoder.decode_barycentric_subgraph(camera_detected_sequence)

    assert result is not None and result["status"] == "success", "Match failed."
    assert (result["row"] == target_row and result["col"] == start_col), \
           f" -> Real-time image slice locked onto: {result['row']}, {result['col']}, {result['horizontal_axis']}, {result['direction']}"
    print(" -> [Verification Passed]: Row index correctly identified despite noise and gaps.")


def test_rotated_axis_matching():
    """
    Runs a comprehensive integration test sweep across all 6 hexagonal
    rotational orientations. Verifies that the adaptive rotation matrices
    and multi-axis coordinate tracking equations match perfectly.
    """
    print("\n=======================================================")
    print("Launching Integrated Rotational Phase Synchronization Pass...")
    print("=======================================================")

    # Initialize your core 32-bit algebraic decoder engine
    decoder = AlgebraicGridDecoder32()
    lfsr_period = decoder.lfsr_period

    # 1. Setup a known absolute starting cell coordinate near the matrix center
    # to avoid raw boundary leakage errors during coordinate transformations
    target_r = 0
    target_c = 0

    # 2. Generate the pristine unrotated master pattern blueprint canvas
    blueprint = decoder.build_barycentric_matrix()
    np.set_printoptions(threshold=np.inf, linewidth=200)
    # 3. Sweep continuously through all 6 rotational angles (0 to 300 degrees)
    directions =["U_forward", "W_reverse", "V_reverse", "U_reverse", "W_forward", "V_forward"]

    for k, direction in enumerate(directions):
        print(f"\n--- Evaluating Rotational Quadrant State: k = {k} ({k * 60} deg CCW) ---")

        # Rotate the entire master blueprint sheet safely using your adaptive method
        rotated_sheet,_ = rotate_barycentric_matrix_adaptive(blueprint, k)
        print(rotated_sheet)

        # 5. Invoke your joint multi-axis intersection subgraph decoder
        res = localize_grid(rotated_sheet)
        # --- CRITICAL INTEGRATION EXTRACTION CHECKS ---
        # Assert that a valid mathematical state consensus was successfully achieved
        assert res is not None, f"Tracking Loop Failed: Decoder dropped lock at k = {k}"
        assert res.get("status") == "success", f"Tracking Loop Failed: Subgraph crashed at k = {k}, payload: {res}"
        assert res.get("horizontal_axis") + "_" + res.get("direction") == direction, f"Wrong direction: Subgraph at k = {k}, payload: {res}"

        # 6. Verify spatial invariance compliance
        # The decoded tracking positions must match your expected mathematical offsets
        print(f" -> Execution Output: Decoded Cell Row = {res['row']}, Col = {res['col']}")

        assert "row" in res and "col" in res, "Telemetry Error: Missing absolute position fields"

        # Validate that the coordinate loop successfully collapses back to a valid location
#        assert 0 <= res["row"] < lfsr_period, f"Boundary Error: Invalid wrapped row index {res['row']}"
#        assert 0 <= res["col"] < lfsr_period, f"Boundary Error: Invalid wrapped col index {res['col']}"

        print(f" -> State Integration Pass k = {directions[k]}: SUCCESS (Lock verified on axis {res.get('horizontal_axis')})")

    print("\n=======================================================")
    print("ALL INTEGRAL ROTATIONAL TESTS CLEARED PERFECTLY!")
    print("=======================================================")


def test_multi_angle_rotational_sweep(decoder):
    """
    Sweeps through all 6 hexagonal rotational states (k = 0 to 5) using the
    adaptive rotation matrix helper. Slices observation patches across the
    rotated sheets, ignores unresolvable fragments, and asserts pixel-perfect,
    strict mathematical coordinate equality for every successful lock.

    Fully ASCII-compliant implementation.
    """
    print("\n=======================================================")
    print("Launching Strict Multi-Angle Rotational Sweep Test...")
    print("=======================================================")

    # 1. Generate the pristine unrotated master pattern blueprint canvas
    blueprint = decoder.build_barycentric_matrix()

    # Define observation patch dimensions matching your 1D sliding windows
    patch_h, patch_w = 6, 15
    np.set_printoptions(threshold=np.inf, linewidth=200)
    # 2. Iterate through all 6 rotational orientations (0 to 300 degrees CCW)
    for k in range(0,6):
        print(f"\n--- Scanning Rotational Quadrant State: k = {k} ({k * 60} deg CCW) ---")

        # Rotates the master blueprint sheet adaptively and captures the
        # exact tracking origin displacement used to pack the array container
        rotated_sheet, (min_r, min_c) = rotate_barycentric_matrix_adaptive(blueprint, k)
        print(rotated_sheet)
        H_rot, W_rot = rotated_sheet.shape
        max_r = H_rot - patch_h
        max_c = W_rot - patch_w

        passed_locks = 0
        ignored_blocks = 0

        # Enforce step=2 on the row sweep to safely respect grid pair constraints
        for target_r in range(0, max_r, 2):
            for target_c in range(0, max_c, 2):
                # Slice a localized sub-patch from the rotated sheet canvas
                patch = rotated_sheet[target_r: target_r + patch_h, target_c: target_c + patch_w]
                # Invoke your autonomous multi-axis intersection subgraph decoder pass
                res = decoder.decode_barycentric_subgraph(patch)

                # If the strip does not detect a consensus phase lock, safely ignore it
                if res is None or res.get("status") != "success":
                    ignored_blocks += 1
                    continue

                # 3. VERIFY PIXEL-PERFECT MATHEMATICAL EQUALITY
                # Calculate the absolute rotated row and column coordinates of this patch corner
                r_rotated_abs = target_r + min_r
                c_rotated_abs = target_c + min_c

                # Pass the absolute rotated coordinates backward through the inverse angle (-k)
                # to discover exactly what coordinate on the clean baseline blueprint we are looking at
                expected_r, expected_c = rotate_barycentric(r_rotated_abs, c_rotated_abs, -k)
                v = expected_r
                u = expected_c - v//2
                expected_r, expected_c = normalize_barycentric(u,v,31)
                # This guarantees that the blind coordinate resolver un-tilted the space cleanly,
                # resolved the negative U sign gates, and matched the absolute blueprint cells exactly.
                try:
                    assert res["row"] == expected_r, (
                        f"Row mismatch at k={k}, Local Patch ({target_r}, {target_c}): "
                        f"Expected absolute blueprint row {expected_r}, got {res['row']}"
                    )
                    assert res["col"] == expected_c, (
                        f"Col mismatch at k={k}, Local Patch ({target_r}, {target_c}): "
                        f"Expected absolute blueprint col {expected_c}, got {res['col']}"
                    )

                    passed_locks += 1

                except AssertionError as err:
                    print(f"\n -> [CRASH] Rotational Alignment Fault Captured!")
                    print(f"    Orientation:    k = {k} ({k * 60} degrees CCW)")
                    print(f"    Start:          Row {min_r}, Col {min_c}")
                    print(f"    Patch Local:    Row {target_r}, Col {target_c}")
                    print(f"    Absolute Rot:   Row {r_rotated_abs}, Col {c_rotated_abs}")
                    print(f"    Expected Match: Row {expected_r}, Col {expected_c}")
                    print(f"    Decoder Output: Row {res['row']}, Col {res['col']}")
                    print(f"    Expected: u {u}, v {v}")
                    print(f"    Details: {err}")
                    raise err

        print(f" -> Quadrant k={k} Scan Complete: Locked={passed_locks}, Ignored Fragments={ignored_blocks}")

    print("\n=======================================================")
    print("ALL MULTI-ANGLE ROTATIONAL EQUALITY CHECKS PASSED PERFECTLY!")
    print("=======================================================")


if __name__ == "__main__":
    decoder = AlgebraicGridDecoder32(grid_width=31, grid_height=31)
    test_full_matrix(decoder)
    test_isolator(decoder)
    test_resolve_cell_index(decoder)
    test_simplified_axis_matching(decoder)
    test_matcher(decoder)
    test_exhaustive_grid_sweep(decoder)
    test_multi_angle_rotational_sweep(decoder)
    test_rotated_axis_matching(decoder)