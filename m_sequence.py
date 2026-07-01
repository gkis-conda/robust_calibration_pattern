import numpy as np


def calculate_reverse_alignment_phase(forward_hex, reverse_hex):
    """
    Dynamically computes the zero-alignment phase constant between
    a forward polynomial and its reciprocal (reverse) partner.
    Automatically scales to any polynomial degree.
    """
    # Automatically find polynomial degree n and max period length dynamically
    n = forward_hex.bit_length() - 1
    period = (1 << n) - 1

    # 1. Generate full baseline periods using YOUR reference loop
    fwd_seq = generate_reference_sequence(forward_hex)
    rev_seq = generate_reference_sequence(reverse_hex)

    # 2. Find the exact cyclic offset where the time-reversed fwd_seq
    # aligns perfectly with the natively generated rev_seq
    fwd_reversed = fwd_seq[::-1]

    alignment_offset = -1
    for shift in range(period):
        shifted_fwd = fwd_reversed[shift:] + fwd_reversed[:shift]
        if shifted_fwd == rev_seq:
            alignment_offset = shift
            break

    if alignment_offset == -1:
        raise ValueError("Polynomials are not a valid reciprocal pair.")

    # 3. Derive the final mathematical constant for the phase mirror equation
    phase = (period - 1 + alignment_offset) % period
    return phase


def build_parity_check_matrix(L, coeffs, debug=False):
    """
    Builds the global parity check matrix H of shape (num_eqs, L)
    for a standard M-sequence based on the inner taps [c1, c2, ..., cn-1].

    L:      Total length of the sequence fragment.
    coeffs: List of inner feedback coefficients (length n-1).
    debug:  If True, prints a visual grid layout of the matrix.
    """
    n = len(coeffs) + 1
    num_eqs = L - n

    if num_eqs <= 0:
        raise ValueError("Sequence length L must be greater than polynomial degree n.")

    H = np.zeros((num_eqs, L), dtype=int)

    for eq_idx in range(num_eqs):
        t = eq_idx + n
        H[eq_idx, t] = 1

        for i in range(1, n):
            if coeffs[i - 1] == 1:
                H[eq_idx, t - i] = 1

        H[eq_idx, t - n] = 1

    if debug:
        print(f"\n--- Full M-Sequence Matrix H Layout (Shape: {H.shape}) ---")
        print("      " + " ".join([f"{i:2d}" for i in range(L)]))
        print("      " + "-" * (3 * L))
        for row_idx, row in enumerate(H):
            row_str = "  ".join([str(bit) for bit in row])
            print(f"Eq {row_idx:2d}: {row_str}")
        print("-" * (3 * L + 6) + "\n")

    return H


def find_error_via_syndrome_projection__(seq, mask, coeffs):
    """
    Algebraically isolates the error syndrome vector while totally
    ignoring the unknown bits using subspace projection.

    Strictly compatible with the decoder_instance.gauss_jordan_gf2 signature.

    Corrects paired, strictly adjacent bit-flip errors (k, k+1)
    caused by a single matrix pixel error bleeding into consecutive
    spatial distillation steps.

    Returns: A tuple of (status_string, list_of_error_indices)
    """
    n = len(coeffs) + 1
    L = len(seq)
    num_eqs = L - n

    unknown_indices = np.where(mask == 0)[0]
    known_indices = np.where(mask == 1)[0]

    # Build the master parity check matrix for the LFSR sequence
    H = build_parity_check_matrix(L, coeffs, debug=False)
    S = (H @ seq.reshape(-1, 1)) % 2
    if np.all(S == 0):
        return "clean", []
    # Stack the matrices together to build the complete elimination canvas
    num_gaps = unknown_indices.size


    # --- ROUTING LOGIC BASED ON GAP PRESENTATION ---
    if num_gaps > 0:
        # Gaps are present: Stack only H_mask with H_clear and S to eliminate unknowns
        H_mask = H[:, unknown_indices]

        # Invoke our mathematically certified LU null-space solver
        T_matrix = extract_null_space_projector_via_lu(H_mask)

        # Guard case: if gaps consume all degrees of freedom, projection is impossible
        if T_matrix.shape == 0:
            return "uncorrectable", []

        # Compress the known clear reference space and syndrome into the clean null-space
        H_clear = H[:, known_indices]
        H_projected_clear = (T_matrix @ H_clear) % 2
        S_projected = (T_matrix @ S.reshape(-1, 1)) % 2
    else:
        # No gaps present: The projected space is identical to the raw unreduced space!
        H_projected_clear = H
        S_projected = S

    # If the active syndrome projection evaluates to zero here, it means it's clean
    if np.all(S_projected == 0):
        return "clean", []

    # --- Step 4: Locate propagated ADJACENT errors (k and k+1) ---
    # Create a fast local index lookup dictionary for known valid indices
    global_to_local = {int(g_idx): l_idx for l_idx, g_idx in enumerate(known_indices)}

    for col_local_idx in range(known_indices.size):
        global_k = int(known_indices[col_local_idx])
        global_k_plus_1 = global_k + 1  # Strictly adjacent error location

        # Check if the consecutive neighbor index is also available in the known mask footprint
        if global_k_plus_1 in global_to_local:
            col_local_idx_paired = global_to_local[global_k_plus_1]

            # Combine the two consecutive projected columns via GF(2) linear XOR subtraction
            combined_column = (H_projected_clear[:, [col_local_idx]] ^
                               H_projected_clear[:, [col_local_idx_paired]])

            # If the joint signature matches the remaining syndrome, the error is localized
            if np.array_equal(combined_column, S_projected):
                return "corrected", [global_k, global_k_plus_1]

    return "uncorrectable", []


def find_error_via_syndrome_projection(seq, mask, coeffs):
    """
    Algebraically isolates the error syndrome vector while totally
    ignoring the unknown bits using a Pure Algebraic LU Subspace projection.

    Features a Residual Validation Pass to verify the true error weight
    and protect the pipeline against multi-error syndrome aliasing traps.
    """
    n = len(coeffs) + 1
    L = len(seq)

    unknown_indices = np.where(mask == 0)[0]
    known_indices = np.where(mask == 1)[0]

    # Build the master parity check matrix for the LFSR sequence
    H = build_parity_check_matrix(L, coeffs, debug=False)

    # Compute the raw syndrome vector before any reduction
    S = (H @ seq.reshape(-1, 1)) % 2

    if np.all(S == 0):
        return "clean", []

    num_gaps = unknown_indices.size

    # --- THE MODULAR LU SUBSPACE PROJECTION ROUTER ---
    if num_gaps > 0:
        H_mask = H[:, unknown_indices]
        T_matrix = extract_null_space_projector_via_lu(H_mask)

        if T_matrix.shape == 0:
            return "uncorrectable", []

        H_clear = H[:, known_indices]
        H_projected_clear = (T_matrix @ H_clear) % 2
        S_projected = (T_matrix @ S.reshape(-1, 1)) % 2
    else:
        H_projected_clear = H[:, known_indices]
        S_projected = S.reshape(-1, 1)

    if np.all(S_projected == 0):
        return "clean", []

    # --- Helper function to perform the Residual Validation Check ---
    def verify_correction_validity(candidate_indices):
        """
        Applies a candidate fix to a copy of the stream and re-evaluates
        the master syndrome to verify if the error weight is fully cleared.
        """
        test_seq = np.copy(seq)
        for idx in candidate_indices:
            test_seq[idx] ^= 1

        # Re-compute the global unreduced master syndrome vector
        new_S = (H @ test_seq.reshape(-1, 1)) % 2

        if num_gaps > 0:
            # Project the new syndrome to check if the valid subspace is fully zeroed out
            new_S_projected = (T_matrix @ new_S.reshape(-1, 1)) % 2
            return np.all(new_S_projected == 0)
        else:
            return np.all(new_S == 0)

    # --- Step 1: Evaluate the Propagated ADJACENT Double-Error Hypothesis ---
    global_to_local = {int(g_idx): l_idx for l_idx, g_idx in enumerate(known_indices)}

    for col_local_idx in range(known_indices.size):
        global_k = int(known_indices[col_local_idx])
        global_k_plus_1 = global_k + 1

        if global_k_plus_1 in global_to_local:
            col_local_idx_paired = global_to_local[global_k_plus_1]

            combined_column = (H_projected_clear[:, [col_local_idx]] ^
                               H_projected_clear[:, [col_local_idx_paired]])

            if np.array_equal(combined_column, S_projected):
                candidate = [global_k, global_k_plus_1]
                # Validate the true error weight before executing the return
                if verify_correction_validity(candidate):
                    return "corrected", candidate

    # --- Step 2: Evaluate the Single Isolated Bit-Error Hypothesis ---
    for col_local_idx in range(known_indices.size):
        if np.array_equal(H_projected_clear[:, [col_local_idx]], S_projected):
            candidate = [int(known_indices[col_local_idx])]
            # Validate the true error weight before executing the return
            if verify_correction_validity(candidate):
                return "corrected", candidate

    # If matches were found but failed the residual check, it means the window
    # contains combined mixed errors (e.g., 1 double + 1 single), which are uncorrectable.
    return "uncorrectable", []


def generate_diff_from_m_sequence(b_ref_seq, phase, num_bits, diff_delay):
    """
    Generates a differential sequence where modular arithmetic strictly holds.
    d[i] = b[i] ^ b[i - diff_delay]

    Both diff_delay = -2 and diff_delay = 29 will produce the EXACT same output.
    """
    b_ref = np.array(b_ref_seq, dtype=np.int8)
    period = len(b_ref)

    # Strictly force diff_delay into a positive residue (e.g., -2 % 31 = 29)
    diff_delay = diff_delay % period

    # Generate output bits using pure modular index arithmetic
    diff_seq = np.zeros(num_bits, dtype=np.int8)
    for i in range(num_bits):
        # Current bit index in the reference timeline
        idx_now = (phase + i) % period
        # Historical bit index strictly following modular lookback
        idx_past = (idx_now - diff_delay + period) % period

        # d[i] = b[idx_now] ^ b[idx_past]
        diff_seq[i] = b_ref[idx_now] ^ b_ref[idx_past]

    return diff_seq


def calculate_phase(poly, delay):
    """
    Calculates the required initial phase of the original M-sequence (b)
    to achieve the desired modulation shift in the final output (d).
    Reuses generate_from_m_sequence to solve for calibration offset k.
    """
    if delay == 0:
        return 0
    n = poly.bit_length() - 1  # Automatically find polynomial degree n
    period = (1 << n) - 1  # Compute maximum period length (e.g., 31)

    # 1. Generate one baseline full period using YOUR Galois right-shift LFSR
    ref_seq = generate_reference_sequence(poly)

    output_seq = generate_diff_from_m_sequence(
        b_ref_seq=ref_seq,
        phase=0,
        num_bits=period,
        diff_delay=delay
    )

    # 3. Algebraically locate the shift constant k:
    # Match the first n bits of the unshifted output stream against
    # all possible sliding windows of the base reference sequence
    target_state = ref_seq[:n]

    k = -1
    for step in range(period):
        cyclic_indices = np.arange(step, step + n) % period
        if np.array_equal(output_seq[cyclic_indices], target_state):
            k = step
            break

    if k == -1:
        raise ValueError("Could not resolve linear dependency alignment for this configuration.")
    return k


def resolve_transmitter_seed_phase(modulation_shift, hex_poly, delay, forward_poly_pair=None):
    """
    Unified master solution to compute the precise initial seed phase for your transmitter loop.
    Extracts the lookback factor k globally and evaluates the forward channel configuration first.

    modulation_shift:  The requested temporal phase offset for the final output signal (d).
    hex_poly:          The current operational polynomial HEX code (e.g., 0x25 or 0x29).
    delay:             The differential lookback parameter (e.g., 2 or -2).
    forward_poly_pair: The matching forward HEX companion. Set to None for standard forward channels.
    """
    n = hex_poly.bit_length() - 1
    period = (1 << n) - 1

    # Extract the internal lookback tracking constant k globally for the current polynomial layout
    k = calculate_phase(hex_poly, delay)

    if forward_poly_pair is None:
        # STAGE A: Handle Standard Forward timeline tracking
        # Execute direct causal calculation layout: phase_b = (shift - k) % period
        required_b_phase = (modulation_shift - k) % period
    else:
        # STAGE B: Handle Time-Reversed Galois memory layout mirroring
        # 1. Dynamically compute the absolute clock zero alignment phase offset between the pair
        mirror_constant = calculate_reverse_alignment_phase(forward_poly_pair, hex_poly)

        # 2. Project the target modulation phase back across the mirror boundary coordinates
        # to ensure that reading backward returns the exact same baseline phase value
        raw_phase_fwd = (mirror_constant - modulation_shift) % period

        # 3. Compensate for the specific internal lookback tap properties extracted globally above
        required_b_phase = (raw_phase_fwd - k) % period

    return required_b_phase


def compute_conjugate_channel_phase(forward_offset, hex_poly, delay, forward_poly_pair=None):
    """
    Computes the calibrated initial phase tracking coordinate for a conjugate polynomial.
    Maintains a completely unchanged delay parameter sign everywhere.
    """
    n = hex_poly.bit_length() - 1
    period = (1 << n) - 1

    # Calculate k using the native operational delay parameter (-2)
    k = calculate_phase(hex_poly, delay)

    if forward_poly_pair is None:
        # Standard Forward Route
        resolved_phase = (forward_offset - k) % period
    else:
        # Reverse/Conjugate Route
        modulation_shift = (forward_offset + 1) % period

        # Universal hardware zero-alignment phase constant (3)
        mirror_constant = calculate_reverse_alignment_phase(forward_poly_pair, hex_poly)

        # SPATIAL ALIGNMENT ADJUSTMENT:
        # To strictly preserve modular arithmetic when running delay=-2 on a time-reversed
        # channel without changing flags, we compensate for the 2-bit physical window index shift.
        raw_phase_fwd = (mirror_constant - modulation_shift + delay) % period
        resolved_phase = (raw_phase_fwd - k) % period

    return resolved_phase


def gauss_jordan_gf2(M):
    """
    Standalone helper function to solve a binary linear system in GF(2).
    The number of unknown variables is automatically inferred from the matrix shape.

    M: Augmented matrix [A | B] of shape (num_eqs, num_vars + 1)
    Returns: (Solved matrix M, pivot columns list) or (None, None) if inconsistent.
    """
    num_eqs, num_cols = M.shape
    num_vars = num_cols - 1

    r = 0
    pivot_cols = []

    for c in range(num_vars):
        if r >= num_eqs:
            break
        # Find pivot in column c
        pivot = r + np.argmax(M[r:, c])
        if M[pivot, c] == 0:
            continue
        # Swap rows if necessary
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]

        pivot_cols.append(c)
        # Eliminate entries below and above the pivot using XOR
        rows_to_xor = np.where(M[:, c] == 1)[0]
        for row in rows_to_xor:
            if row != r:
                M[row] ^= M[r]
        r += 1


    # Check for inconsistencies (rows looking like [0 0 ... 0 | 1])
    for row_idx in range(num_eqs):
        if np.all(M[row_idx, :-1] == 0) and M[row_idx, -1] == 1:
            return None, None

    # Check if we have enough independent equations for all variables
    if len(pivot_cols) < num_vars:
        return None, None

    return M, pivot_cols



def lup_decomposition_gf2(A: np.ndarray) -> tuple:
    """
    Performs LUP Decomposition over Galois Field 2 (GF2).
    A represents the augmented matrix canvas of shape (M, N).

    Returns:
        tuple: (P, L, U) matrices over GF(2) such that (P @ A) % 2 == (L @ U) % 2.
               P is a permutation matrix of shape (M, M).
               L is lower triangular with 1s on diagonal, shape (M, M).
               U is upper triangular (row echelon form), shape (M, N).
    """
    M, N = A.shape

    # Initialize P as identity, L as identity, and U as a copy of A
    P = np.eye(M, dtype=np.int8)
    L = np.eye(M, dtype=np.int8)
    U = np.copy(A)

    r_ptr = 0
    for c_idx in range(N):
        if r_ptr >= M:
            break

        # 1. Search for a pivot bit down the current column
        pivot_row = r_ptr + np.argmax(U[r_ptr:, c_idx])
        if U[pivot_row, c_idx] == 0:
            continue  # No pivot available in this column, skip

        # 2. Synchronously swap rows across U and P matrices
        if pivot_row != r_ptr:
            U[[r_ptr, pivot_row]] = U[[pivot_row, r_ptr]]
            P[[r_ptr, pivot_row]] = P[[pivot_row, r_ptr]]

            # Swap historical multiplier entries inside L below the diagonal
            if r_ptr > 0:
                L[[r_ptr, pivot_row], :r_ptr] = L[[pivot_row, r_ptr], :r_ptr]

        # 3. Eliminate downstream ones using GF(2) row operations
        for target_row in range(r_ptr + 1, M):
            if U[target_row, c_idx] == 1:
                # Record the elimination multiplier bit inside L
                L[target_row, r_ptr] = 1
                # Subtract (XOR) the active pivot row out of U
                U[target_row] ^= U[r_ptr]

        r_ptr += 1

    return P, L, U


def extract_null_space_projector_via_lu(A: np.ndarray) -> np.ndarray:
    """
    Uses the LUP decomposition matrices to cleanly extract the left null-space
    projector matrix T. Guarantees a minimal shape of (M - rank, M).
    """
    M, N = A.shape
    P, L, U = lup_decomposition_gf2(A)

    # Identify the rank by counting non-zero rows in the row-echelon matrix U
    rank = 0
    for row_idx in range(M):
        if np.any(U[row_idx] != 0):
            rank += 1

    # The inverse row-operation mapping is given by: Inverse(L) @ P
    # But since we only need to solve for rows below the rank boundary,
    # we can compute the forward row multiplier combination matrix directly:
    L_inv = np.eye(M, dtype=np.int8)
    for i in range(M):
        for j in range(i):
            if L[i, j] == 1:
                L_inv[i] ^= L_inv[j]

    # T captures the exact linear combinations that produced the zero rows in U
    # Formula: T = L_inv[rank:, :] @ P
    T_full = (L_inv @ P) % 2
    T = T_full[rank:, :]

    return T


def polynomial_degree(poly: int) -> int:
    """Finds the index of the highest set bit, which dictates the LFSR polynomial degree."""
    return poly.bit_length() - 1


def generate_reference_sequence(poly: int) -> list:
    """Generates a raw bit sequence matching the feedback polynomial recurrence loop."""
    state = 12 # mirror symmetrical state S[t] == S'[-t]
    seq = []
    length = (1 << polynomial_degree(poly)) - 1
    for _ in range(length):
        seq.append(state & 1)
        feedback = state & 1
        state >>= 1
        if feedback:
            state ^= (poly >> 1)
    return seq


def poly_coeffs(poly: int) -> list:
    """
    Extracts feedback coefficients [c1, c2, ..., cn-1] from left to right.
    Uses the polynomial degree to find the starting bit position.
    """
    n = polynomial_degree(poly)
    coeffs = []
    for i in range(1, n):
        poly >>= 1
        coeffs.append(poly & 1)

    return coeffs


class MSequenceAnalyzer:
    def __init__(self, poly: int, shift_k: int = 0):
        """
        hex_poly_dict: HEX integers (e.g. 0x25, 0x29)
        phase: phase shift values k for each polynomial
        """
        self.shift_k = shift_k
        self.poly = poly
        self.n = polynomial_degree(poly)
        
        # Automatically unpack HEX representations into binary lists [c1, c2, c3, c4, c5]
        self.c = poly_coeffs(poly)

    def build_augmented_matrix(self, seq, mask):
        """
        Constructs the system of equations in GF(2) matrix format.
        Perfectly synchronized with the global parity matrix layout.
        """
        n = len(self.c) + 1
        L = len(seq)
        num_eqs = L - n

        unknown_indices = np.where(mask == 0)[0]

        # 1. Build the base full parity matrix
        H = build_parity_check_matrix(L, self.c, debug=False)

        # 2. Separate columns belonging to unknown masks
        H_mask = H[:, unknown_indices]

        # 3. Compute the constant terms vector based on visible bits
        # Wherever the mask is 0, we treat that sequence value as 0
        # so that it doesn't leak into the constant column
        clean_seq = np.array(seq, copy=True)
        clean_seq[unknown_indices] = 0
        B = (H @ clean_seq.reshape(-1, 1)) % 2

        # 4. Merge them into an augmented format [A | B]
        M = np.hstack([H_mask, B])

        return M, unknown_indices

    def calculate_initial_phase(self, full_d_seq):
        """Calculates the absolute initial phase shift based on YOUR Galois baseline."""
        # Using 5 because your polynomials are all degree-5
        L = len(full_d_seq)
        if L < self.n:
            return None
        # Locate the first valid n-bit state window in recovered data
        first_state = full_d_seq[:self.n]

        # 1. Generate one full period baseline matching your exact function
        ref_sequence = generate_reference_sequence(self.poly)

        period = (1 << self.n) - 1
        ref_shift_d = -1
        # 2. Extract every possible n-bit state window from your baseline sequence
        for step in range(period):
            cyclic_indices = np.arange(step, step + self.n) % period
            window = np.array(ref_sequence)[cyclic_indices]
            if np.array_equal(window, first_state):
                ref_shift_d = step
                break

        if ref_shift_d == -1:
            return None

        # 4. Compensate for the custom delay offset k and layout window padding
        initial_phase = (ref_shift_d + self.shift_k) % period
        return initial_phase

    def analyze(self, diff_seq, mask):
        """
        Unified loop: dynamically isolates noise via standalone syndrome projection,
        corrects data, then solves the missing mask variables using Gauss elimination.
        """
        diff_seq = np.array(diff_seq, dtype=int)
        mask = np.array(mask, dtype=int)

        working_seq = np.array(diff_seq, copy=True)

        # Step 1: Run the fast standalone algebraic syndrome locator
        status, error_idxs = find_error_via_syndrome_projection(working_seq, mask, self.c)

        if status == "uncorrectable":
            return None  # Multiple corruptions, polynomial cannot fit cleanly

        if status == "corrected":
            # Extract index and flip the corrupted bit back natively
            working_seq[error_idxs] ^= 1
            status_msg = f"Corrected bit flip at index {error_idxs} via syndrome projection"
        else:
            status_msg = "Clean known data (No bit flip errors detected)"

        # Step 2: Build the core matrix using the noise-corrected stream
        M, unknown_indices = self.build_augmented_matrix(working_seq, mask)
        if M is None:
            return None

        if len(unknown_indices):
            # Step 3: Solve remaining missing gaps
            M_solved, pivot_cols = gauss_jordan_gf2(M)
            if M_solved is None:
                return None  # Contradiction remaining, mismatch polynomial

            sol = np.zeros(unknown_indices.size, dtype=int)
            for i, c in enumerate(pivot_cols):
                sol[c] = M_solved[i, -1]

            working_seq[unknown_indices] = sol

        initial_phase = self.calculate_initial_phase(working_seq)
        if initial_phase is None:
            return None

        return {
            "origin_stream_phase": initial_phase,
            "recovered_sequence": working_seq.tolist(),
            "errors_corrected": error_idxs,
            "missing_gaps": unknown_indices,
            "status": status_msg
        }


# =====================================================================
# ASSERTION TESTS
# =====================================================================

def test_qr():
    print("=======================================================")
    print("Initializing Multi-Scenario Binary QR Test Suite...")
    print("=======================================================")

    # Select the target function provider (either a class instance method or global helper)
    solver = extract_null_space_projector_via_lu

    # -----------------------------------------------------------------
    # CASE 1: OVER-CONSTRAINED SYSTEM (5 Equations, 2 Gaps)
    # Target profile: Fewer missing tracking gaps than row equations.
    # Expected: Q shape (5,2), R shape (2,2), T shape (3,5).
    # -----------------------------------------------------------------
    print("\n--- CASE 1: Over-Constrained (5 Equations, 2 Gaps) ---")
    # Flat representation containing all 10 bit elements explicitly allocated
    A1_flat = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.int8)
    A1 = A1_flat.reshape(5, 2)

    T1 = solver(A1)

    print("T1: ", T1)

    # Assert fundamental algebraic identities
    assert T1.shape == (3, 5), f"Case 1 Shape Error: Expected T to have 3 rows, got {T1.shape[0]}"
    assert np.all((T1 @ A1) % 2 == 0), "Case 1 Failure: Left null-space T failed to annihilate A"
    print("-> Case 1 Passed! Symmetrically retained exactly 3 error-checking equations.")

    # -----------------------------------------------------------------
    # CASE 2: FULL-RANK VARIABLE SYSTEM (3 Equations, 3 Gaps)
    # Target profile: Gaps completely saturate all available row equations.
    # Expected: Q shape (3,3), R shape (3,3), T shape (0,3) [Empty placeholder].
    # -----------------------------------------------------------------
    print("\n--- CASE 2: Full-Rank Erasure (3 Equations, 3 Gaps) ---")
    # Flat representation containing all 9 bit elements explicitly allocated
    A2_flat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8)
    A2 = A2_flat.reshape(3, 3)

    T2 = solver(A2)

    print("T2: ", T2)

    # Assert fundamental algebraic identities
    assert T2.shape == (0, 3), f"Case 2 Shape Error: Expected empty T with 0 rows, got {T2.shape[0]}"
    print("-> Case 2 Passed! T successfully collapsed to a safe 0-row subspace buffer.")

    # -----------------------------------------------------------------
    # CASE 3: NULL-SPACE DOMINANT SYSTEM (4 Equations, 2 Gaps)
    # Target profile: Erasure stream matrix contains columns of all zeros.
    # Expected: Q shape (4,0), R shape (0,2), T shape (4,4) [Full Identity preserved].
    # -----------------------------------------------------------------
    print("\n--- CASE 3: Null-Space Dominant (4 Equations, 2 All-Zero Gaps) ---")
    # Flat representation containing all 8 zero bit elements explicitly allocated
    A3_flat = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    A3 = A3_flat.reshape(4, 2)

    T3 = solver(A3)

    print("T3:\n", T3)

    # Assert fundamental algebraic identities
    assert T3.shape == (4, 4), f"Case 3 Shape Error: Expected full T identity with 4 rows, got {T3.shape[0]}"
    assert np.all((T3 @ A3) % 2 == 0), "Case 3 Failure: Left null-space T failed to annihilate A"
    print("-> Case 3 Passed! Pristine grid system redundancy fully preserved.")

    print("\n=======================================================")
    print("ALL COMPREHENSIVE QR TEST SUITE ASSERTS CLEARED PERFECTLY!")
    print("=======================================================")


def test_gauss_jordan_gf2_valid():
    # Matrix represents equations:
    # row 0: 1*x0 + 0*x1 = 0
    # row 1: 0*x0 + 1*x1 = 1
    row0 = [1, 0, 0]
    row1 = [0, 1, 1]
    M = np.array([row0, row1], dtype=int)

    M_solved, pivots = gauss_jordan_gf2(M)
    assert M_solved is not None, "Valid system returned None"

    expected_pivots = [0, 1]
    assert pivots == expected_pivots, f"Expected pivots, got {pivots}"
    assert M_solved[0, -1] == 0, f"Expected x0=0, got {M_solved[0, -1]}"
    assert M_solved[1, -1] == 1, f"Expected x1=1, got {M_solved[1, -1]}"
    print("test_gauss_jordan_gf2_valid: PASSED")


def test_gauss_jordan_gf2_inconsistent():
    # Matrix represents contradictory equations:
    # row 0: 1*x0 = 0
    # row 1: 1*x0 = 1
    row0 = [1, 0]
    row1 = [1, 1]
    M = np.array([row0, row1], dtype=int)

    M_solved, pivots = gauss_jordan_gf2(M)
    assert M_solved is None, "Inconsistent system did not return None"
    print("test_gauss_jordan_gf2_inconsistent: PASSED")


def test_build_parity_check_matrix():
    """
    Dedicated unit test for the build_parity_check_matrix function.
    Validates structural dimensions and exact bit positions for degree-5 loops.
    """

    # 1. Feedback taps vector matching U_forward (0x25) ordered [c1, c2, c3, c4]
    coeffs_u = [0, 1, 0, 0]

    # 2. Define test fragment length (L = 10 bits)
    L = 10
    n = len(coeffs_u) + 1  # Degree n = 5
    expected_rows = L - n  # 10 - 5 = 5 rows expected

    # Generate the test matrix
    H = build_parity_check_matrix(L, coeffs_u, debug=False)

    # Assert dimension bounds match matrix geometric requirements
    assert H.shape == (expected_rows, L), f"Expected shape {(expected_rows, L)}, got {H.shape}"

    # 3. Structural validation of sliding window equations (mod 2 checks)
    # Every row in U_forward configuration must contain exactly three 1s:
    # - One at the newest target index (t = eq_idx + 5)
    # - One at the internal tap index (t - 2 = eq_idx + 3)
    # - One at the oldest lookback bound (t - 5 = eq_idx)
    for eq_idx in range(expected_rows):
        row = H[eq_idx]

        # Count total active links per row
        total_ones = int(row.sum())
        assert total_ones == 3, f"Row {eq_idx} should have exactly 3 ones, found {total_ones}"

        # Verify specific structural index flags match expected positions
        target_bit = eq_idx + 5
        tap_bit = eq_idx + 3
        oldest_bit = eq_idx

        assert row[target_bit] == 1, f"Missing target bit at index {target_bit} on row {eq_idx}"
        assert row[tap_bit] == 1, f"Missing loop feedback tap at index {tap_bit} on row {eq_idx}"
        assert row[oldest_bit] == 1, f"Missing tracking baseline bit at index {oldest_bit} on row {eq_idx}"

        # Confirm remainder indices are entirely clear (zeros padding)
        for i in range(L):
            if i not in (target_bit, tap_bit, oldest_bit):
                assert row[i] == 0, f"Unexpected active entry at column {i} on row {eq_idx}"

    print("test_build_parity_check_matrix: PASSED\n")


def test_find_error_via_syndrome_projection():
    """
    Dedicated test suite for the find_error_via_syndrome_projection helper.
    Validates three fundamental channel scenarios.
    """
    print("Running test_find_error_via_syndrome_projection...")

    poly = 0x25
    L = 20
    # 1. Inner feedback coefficients for U_forward (0x25) ordered [c1, c2, c3, c4]

    coeffs = poly_coeffs(poly)

    # 2. Perfect 20-bit differential sequence matching U_forward, phase=12
    perfect_seq = generate_reference_sequence(poly)[:L]

    # Define a mask that hides indexes 4 and 12 (zeros mark unknown gaps)
    broken_bits = [4,12]
    base_mask = np.ones(L, dtype=int)
    base_mask[broken_bits] = 0

    # -------------------------------------------------------------------------
    # CASE A: Clean stream modulo missing mask variables
    # -------------------------------------------------------------------------
    clean_input = np.array(perfect_seq, copy=True)
    clean_input[broken_bits] = 9

    status, error_idxs = find_error_via_syndrome_projection(clean_input, base_mask, coeffs)

    assert status == "clean", f"Expected 'clean', got '{status}'"
    assert error_idxs == [], f"Expected empty list, got {error_idxs}"
    print("-> Case A (Clean sequence with missing gaps): PASSED")

    # -------------------------------------------------------------------------
    # CASE B: Missing mask gaps + a single corrupted unmasked bit flip
    # -------------------------------------------------------------------------
    expected_list = [7]
    noisy_input = np.array(clean_input, copy=True)
    noisy_input[expected_list] ^= 1

    status, error_idxs = find_error_via_syndrome_projection(noisy_input, base_mask, coeffs)

    assert status == "corrected", f"Expected 'corrected', got '{status}'"

    assert error_idxs == expected_list, f"Expected error at index {expected_list}, got {error_idxs}"
    print("-> Case B (Missing gaps + single bit-flip error at index 7): PASSED")

    # -------------------------------------------------------------------------
    # CASE C: Missing mask gaps + multiple uncorrectable errors (double flip)
    # -------------------------------------------------------------------------
    expected_list = [7,15]
    broken_input = np.array(clean_input, copy=True)
    broken_input[expected_list] ^= 1

    status, error_idxs = find_error_via_syndrome_projection(broken_input, base_mask, coeffs)
    assert status == "uncorrectable", f"Expected 'uncorrectable', got '{status}'"
    assert error_idxs == [], f"Expected empty list, got {error_idxs}"
    print("-> Case C (Missing gaps + double bit-flip uncorrectable): PASSED")

    print("All find_error_via_syndrome_projection checks: PASSED\n")


def test_m_sequence_decoding_and_matching():
    chosen_delay = 2  # This matches our earlier fixed-delay calculations
    my_hex_polys = {
        "U_forward": 0x25, "U_reverse": 0x29,
        "V_forward": 0x3D, "V_reverse": 0x2F,
        "W_forward": 0x3B, "W_reverse": 0x37
    }
    my_shifts = { name : calculate_phase(poly,chosen_delay) for name, poly in my_hex_polys.items()}

    sequence = "U_forward"
    analyzer = MSequenceAnalyzer(my_hex_polys[sequence], my_shifts[sequence])

    # Valid differential M-sequence fragment for U_forward (0x25) with initial phase b = 10
    base_m_period = generate_reference_sequence(my_hex_polys[sequence])
    # One full period (31 bits) of an M-sequence matching U_forward starting from state

    target_phase = 14
    requested_length = 25

    # Generate streams dynamically using the new generalized logic
    perfect_diff_seq = generate_diff_from_m_sequence(
        base_m_period, phase=target_phase, num_bits=requested_length, diff_delay=chosen_delay
    )

    # Check shape integrity
    assert len(perfect_diff_seq) == requested_length

    # Inject synthetic missing masks at index 3 and index 10 (values hidden inside sequence)
    test_seq = np.array(perfect_diff_seq, copy=True)
    test_seq[3] = 9
    test_seq[10] = 9

    test_mask = np.ones(requested_length, dtype=int)
    test_mask[3] = 0
    test_mask[10] = 0

    # Validate with the original solver class configured for a delay of 2
    result = analyzer.analyze(test_seq, test_mask)

    assert result is not None, "Analyzer returned None for generalized sequence parsing"
    assert result["recovered_sequence"] == perfect_diff_seq.tolist(), "Data sequence correction failed!"
    assert result["origin_stream_phase"] == target_phase

    print("test_m_sequence_decoding_with_generalized_delay: PASSED")



def test_sequence():
    base_m_period = generate_reference_sequence(0x25)
    rev_m_period = generate_reference_sequence(0x29)
    mark = base_m_period[:5][::-1]
    count = 0
    phase = -1
    for k in range(31):
        if rev_m_period[k:k+5] == mark:
            assert count == 0
            phase = k
    print(phase)
    phase0 = calculate_reverse_alignment_phase(0x29, 0x25)
    phase1 = calculate_reverse_alignment_phase(0x25, 0x29)
    assert phase1 == phase0
    phase0 = calculate_reverse_alignment_phase(0x3D, 0x2F)
    phase1 = calculate_reverse_alignment_phase(0x2F, 0x3D)
    assert phase1 == phase0
    phase0 = calculate_reverse_alignment_phase(0x3B, 0x37)
    phase1 = calculate_reverse_alignment_phase(0x37, 0x3B)
    assert phase1 == phase0

if __name__ == "__main__":
    # Execute assertion tests
    print("Running assertion-based test suite...")
    test_qr()
    test_sequence()
    test_build_parity_check_matrix()
    test_gauss_jordan_gf2_valid()
    test_gauss_jordan_gf2_inconsistent()
    test_find_error_via_syndrome_projection()
    test_m_sequence_decoding_and_matching()
    print("All tests completed successfully!")
