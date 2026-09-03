import numpy as np


def calculate_reverse_alignment(forward):
    """
    Dynamically computes the zero-alignment phase constant between
    a forward polynomial and its reciprocal (reverse) partner.
    Automatically scales to any polynomial degree.
    """
    # Automatically find polynomial degree n and max period length dynamically
    poly = Polynomial(forward)
    fwd_seq = poly.generate_reference_sequence()
    rev_seq = poly.reciprocal().generate_reference_sequence()

    # 2. Find the exact cyclic offset where the time-reversed fwd_seq
    # aligns perfectly with the natively generated rev_seq
    fwd_reversed = fwd_seq[::-1]

    alignment_offset = -1
    for shift in range(poly.period()):
        shifted_fwd = fwd_reversed[shift:] + fwd_reversed[:shift]
        if shifted_fwd == rev_seq:
            alignment_offset = shift
            break

    if alignment_offset == -1:
        raise ValueError("Polynomials are not a valid reciprocal pair.")

    return (alignment_offset + 1) % poly.period()


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


def find_error_via_syndrome_projection(seq, mask, coeffs):
    """
    Hybrid Joint Error-Erasure Decoder over GF(2).
    """
    n = len(coeffs) + 1
    L = len(seq)
    num_rows = L - n

    # Build the standard unreduced global parity check matrix H
    H = build_parity_check_matrix(L, coeffs, debug=False)

    # Compute the raw, global syndrome vector based on the received sequence
    S_working = ((H @ seq.reshape(-1, 1)) % 2).flatten()

    # If the global raw syndrome is completely zero, the stream is natively clean
    if np.all(S_working == 0):
        return "corrected", []

    unknown_indices = np.where(mask == 0)[0]

    # Precompute the row mask to shield Stage 1 from erasure noise floors
    row_mask = np.ones(num_rows, dtype=np.uint8)
    if unknown_indices.size > 0:
        for gap_idx in unknown_indices:
            row_mask[np.where(H[:, gap_idx] != 0)[0]] = 0

    collected_errors = []
    paired = False
    # =========================================================================
    # STAGE 0: REVERSE GREEDY PRE-CLEANUP (ISOLATED FAULT REMOVAL)
    # =========================================================================
    for k in range(L - 1, 1, -1):
        if paired:
            paired = False
            continue
        if mask[k] == 1:
            continue

        start_row = max(0, k - n)
        end_row = min(num_rows, k + 1)

        # Hypothesis 1: Priority Adjacent Double-Error Check
        k_minus_1 = k - 1
        if k_minus_1 >= 0 and mask[k] == mask[k_minus_1]:
            paired_start_row = max(0, k_minus_1 - n)
            w = np.sum(S_working[paired_start_row:end_row])
            if w == 0:
                continue
            paired_column = (H[paired_start_row:end_row, k] ^ H[paired_start_row:end_row, k_minus_1])
            paired_column_norm = np.sum(paired_column ^ S_working[paired_start_row:end_row])
            if paired_column_norm < w:
                collected_errors.extend([k_minus_1, k])
                S_working[paired_start_row:end_row] ^= paired_column  # Global footprint flush
                paired = True
                continue

        # Hypothesis 2: Single Isolated Bit-Flip Check
        single_column = H[start_row:end_row, k]
        w = np.sum(S_working[start_row:end_row])
        if  w == 0:
            continue
        single_column_norm = np.sum(single_column ^ S_working[start_row:end_row])
        if single_column_norm < w:
            collected_errors.append(k)
            S_working[start_row:end_row] ^= single_column  # Global footprint flush


    # =========================================================================
    # STAGE 1: REVERSE GREEDY PRE-CLEANUP (ISOLATED FAULT REMOVAL)
    # =========================================================================
    paired = False
    for k in range(L - 1, 1, -1):
        if paired:
            paired = False
            continue
        if mask[k] == 0:
            continue

        start_row = max(0, k - n)
        end_row = min(num_rows, k + 1)

        active_row_mask = row_mask[start_row:end_row]

        # Hypothesis 1: Priority Adjacent Double-Error Check
        k_minus_1 = k - 1
        if k_minus_1 >= 0 and mask[k] == mask[k_minus_1]:
            paired_start_row = max(0, k_minus_1 - n)
            paired_row_mask = row_mask[paired_start_row:end_row]
            if np.sum(S_working[paired_start_row:end_row]) == 0:
                continue
            paired_column = (H[paired_start_row:end_row, k] ^ H[paired_start_row:end_row, k_minus_1])
            paired_column_norm = (paired_column ^ S_working[paired_start_row:end_row])@ paired_row_mask

            if paired_column_norm == 0:
                collected_errors.extend([k_minus_1, k])
                S_working[paired_start_row:end_row] ^= paired_column  # Global footprint flush
                paired = True
                continue

        # Hypothesis 2: Single Isolated Bit-Flip Check
        single_column = H[start_row:end_row, k]
        if np.sum(S_working[start_row:end_row]) == 0:
            continue
        single_column_norm = (single_column ^ S_working[start_row:end_row]) @ active_row_mask

        if single_column_norm == 0:
            collected_errors.append(k)
            S_working[start_row:end_row] ^= single_column  # Global footprint flush

    # =========================================================================
    # STAGE 2: TOPOLOGICAL VARIABLE HARVESTING FROM ACTIVE SYNDROMES
    # =========================================================================
    # If the greedy reverse pass successfully zeroed out the entire syndrome,
    # we bypass the Gaussian stage completely and jump straight to success.
    collected_errors =[idx for idx in collected_errors if mask[idx] == 1]
    if np.sum(S_working) == 0:
        return "corrected", sorted(collected_errors)

    # Initialize our uncertainty pool with all erasure gaps (they are mandatory variables)
    unresolved_variables = set(unknown_indices)
    for idx in collected_errors:
        unresolved_variables.add(idx)

    unresolved_variables = sorted(list(unresolved_variables))

    # If no variables could be harvested but the syndrome is still dirty, it's a structural collapse
    if len(unresolved_variables) == 0:
        return "uncorrectable", []

    clean_seq = np.array(seq, copy=True)
    b = (H @ clean_seq.reshape(-1, 1)) % 2
    H_b = H[:, unresolved_variables]
    # Merge them into an augmented format [A | B]
    M = np.hstack([H_b, b])
    M_solved, pivot_cols = gauss_jordan_gf2(M)

    if M_solved is None:
        return "uncorrectable", []

    # Extract additional verified corrections pinpointed by the row reduction pass
    collected_errors = []
    for i, col_idx in enumerate(pivot_cols):
        idx = unresolved_variables[col_idx]
        # We only flip bits that belong to our verified known sensor positions
        if mask[idx] == 1:
            if M_solved[i, -1] == 1:
                collected_errors.append(idx)

    # =========================================================================
    # FINAL REGISTRATION AND CLOSED-LOOP CONVERGENCE PROOF
    # =========================================================================
    test_seq = np.copy(seq)
    test_seq[collected_errors] ^= 1

    S = ((H @ test_seq.reshape(-1, 1)) % 2).flatten()
    final_weight = row_mask @ S
    if final_weight == 0:
        return "corrected", sorted(collected_errors)

    return "uncorrectable", []


def generate_diff_from_m_sequence(b_ref_seq, phase, num_bits, diff_delay):
    """
    Generates a differential sequence where modular arithmetic strictly holds.
    d[i] = b[i] ^ b[i - diff_delay]
    phase - the original sequence can be shifted due to geometry needs (for example in some sliding window
    b_ref_seq - should be full M-sequence
    Both diff_delay = -2 and diff_delay = 29 will produce the same output.
    """
    b_ref = np.array(b_ref_seq, dtype=np.int8)
    period = len(b_ref)

    # Strictly force diff_delay into a positive residue (e.g., -2 % 31 = 29)
    diff_delay = diff_delay % period

    # Generate output bits using pure modular index arithmetic
    diff_seq = np.zeros(num_bits, dtype=np.int8)
    for i in range(num_bits):
        # Current bit index in the reference timeline
        idx_now = (i + phase) % period
        # Historical bit index strictly following modular lookback
        idx_past = (idx_now - diff_delay + period) % period
        # d[i] = b[idx_now] ^ b[idx_past]
        diff_seq[i] = b_ref[idx_now] ^ b_ref[idx_past]

    return diff_seq


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

    return L, U, P


def gf2_matrix_pseudo_inverse(H):
    """
    Computes the binary Moore-Penrose pseudo-inverse H^+ of an unreduced
    parity-check matrix over GF(2) using the algebraic Gramian formulation:
    H^+ = (H^T @ H)^-1 @ H^T (mod 2)

    Returns:
        np.ndarray: The pseudo-inverse matrix of shape (L, L-n) over GF(2).
        None: If the Gramian matrix is singular (under-determined boundary traps).
    """
    # 1. Compute the Gramian matrix over GF(2) using native matrix multiplication modulo 2
    Gram = (H.T @ H) % 2

    # 2. Invert the Gramian matrix utilizing our square inversion helper
    Gram_inv = gf2_matrix_inverse_via_lup(Gram)

    if Gram_inv is None:
        return None  # Gramian is singular, pseudo-inverse cannot be established

    # 3. Construct the final pseudo-inverse matrix: H^+ = Gram_inv @ H^T (mod 2)
    H_pseudo_inv = (Gram_inv @ H.T) % 2
    return H_pseudo_inv


def gf2_matrix_inverse_via_lup(A):
    """
    Computes the matrix inverse over GF(2) by solving P @ A = L @ U
    column-by-column using fast forward and backward substitution steps.
    """
    decomp = lup_decomposition_gf2(A)
    if decomp is None:
        return None

    L, U, P = decomp
    n = A.shape[0]
    A_inv = np.zeros((n, n), dtype=np.uint8)

    # Solve A @ x_j = e_j for each canonical column vector e_j
    # This is equivalent to solving L @ U @ x_j = P @ e_j
    for j in range(n):
        # Extract the j-th column of the permutation matrix P (the shifted target vector)
        b = P[:, j].copy()

        # 1. Forward Substitution: Solve L @ y = b (mod 2)
        y = np.zeros(n, dtype=np.uint8)
        for r in range(n):
            # y[r] = b[r] ^ (L[r, :r] @ y[:r] % 2)
            y[r] = b[r] ^ (np.bitwise_xor.reduce(L[r, :r] & y[:r]) if r > 0 else 0)

        # 2. Backward Substitution: Solve U @ x = y (mod 2)
        x = np.zeros(n, dtype=np.uint8)
        for r in range(n - 1, -1, -1):
            # x[r] = y[r] ^ (U[r, r+1:] @ x[r+1:] % 2)
            x[r] = y[r] ^ (np.bitwise_xor.reduce(U[r, r + 1:] & x[r + 1:]) if r < n - 1 else 0)

        A_inv[:, j] = x

    return A_inv


def extract_null_space_projector(A: np.ndarray) -> np.ndarray:
    """
    Uses the LUP decomposition matrices to cleanly extract the left null-space
    projector matrix T. Guarantees a minimal shape of (M - rank, M).
    """
    L, U, P = lup_decomposition_gf2(A)

    # Identify the rank by counting non-zero rows in the row-echelon matrix U
    rank = 0
    M, N = A.shape
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


class Polynomial:
    def __init__(self, poly:int):
        self.poly = poly
        self.sequence = None

    def degree(self) -> int:
        """Finds the index of the highest set bit, which dictates the LFSR polynomial degree."""
        return self.poly.bit_length() - 1

    def period(self) -> int:
        return (1 << self.degree()) - 1

    def generate_reference_sequence(self) -> list:
        """Generates a raw bit sequence matching the feedback polynomial recurrence loop."""
        if self.sequence is not None:
            return self.sequence
        # lazy initialization
        state = 12 # mirror symmetrical state S[t] == S'[-t]
        seq = []
        poly = self.poly >> 1
        for _ in range(self.period()):
            feedback = state & 1
            seq.append(feedback)
            state >>= 1
            if feedback:
                state ^= poly
        self.sequence = seq
        return seq

    def coeffs(self) -> list:
        """
        Extracts feedback coefficients [c1, c2, ..., cn-1] from left to right.
        """
        n = self.degree()
        coeffs = []
        poly = self.poly
        for i in range(1, n):
            poly >>= 1
            coeffs.append(poly & 1)

        return coeffs

    def reciprocal(self):
        """
        Compute the reciprocal (dual) polynomial r(x) = x^n * p(1/x)
        for a polynomial encoded as an integer `poly` where the bit at position
        n is the x^n coefficient (leading 1) and bit 0 is the constant term.

        Returns integer encoding of the reciprocal polynomial.
        """
        # degree n = bit_length - 1  (because bit n is the leading 1)
        n = self.degree()
        rev = 0
        # reverse n+1 bits: for i in [0..n], map bit i -> bit (n-i)
        for i in range(n + 1):
            if (self.poly >> i) & 1:
                rev |= 1 << (n - i)
        return Polynomial(rev)

    def get_delayed_xor_phase(self, delay: int) -> int:
        """
        Calculates the required initial phase of the original M-sequence (b)
        to achieve the desired modulation shift in the final output (d) of xor sum with delayed sequence.
        """
        if delay == 0:
            return 0
        n = self.degree()
        period = self.period()

        # 1. Generate one baseline full period
        ref_seq = np.array(self.generate_reference_sequence())

        target_state = generate_diff_from_m_sequence(
            b_ref_seq=ref_seq,
            phase = 0,
            num_bits=n,
            diff_delay=delay
        )

        # 3. Algebraically locate the shift constant k:
        # Match the first n bits of the unshifted output stream against
        # all possible sliding windows of the base reference sequence
        k = -1
        for step in range(period):
            cyclic_indices = np.arange(step, step + n) % period
            if np.array_equal(ref_seq[cyclic_indices], target_state):
                k = step
                break

        if k == -1:
            raise ValueError("Could not resolve linear dependency alignment for this configuration.")
        return period - k


class MSequenceAnalyzer:
    def __init__(self, poly: int, phase: int = 0):
        """
        poly: integers (e.g. 0x25, 0x29)
        phase: phase shift values k for each polynomial
        """
        self.phase = phase
        poly = Polynomial(poly)
        self.n = poly.degree()
        self.period = poly.period()
        self.c = poly.coeffs()
        self.ref_sequence = poly.generate_reference_sequence()
        ref_arr = np.asarray(self.ref_sequence, dtype=np.uint8)
        self.extended_ref_bytes = bytes(np.concatenate([ref_arr, ref_arr[:self.n]]))

    def build_augmented_matrix(self, seq, mask):
        """
        Constructs the system of equations in GF(2) matrix format.
        """
        L = len(seq)
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

    def calculate_phase(self, full_d_seq):
        """
        Calculates the absolute initial phase shift using fast substring matching.
        Executes at pure C-level speed via Python's native string search engine.
        """
        L = len(full_d_seq)
        if L < self.n:
            return None

        # Extract the first n-bit state window (e.g., 5 elements)
        first_state = full_d_seq[:self.n]

        # Convert the localized target n-bit window into a byte pattern
        pattern_bytes = bytes(first_state.astype(np.uint8))

        # Fast substring index lookup - executing via highly optimized Boyer-Moore-Horspool
        ref_shift_d = self.extended_ref_bytes.find(pattern_bytes)

        if ref_shift_d == -1:
            return None

        # Compensate for the custom delay offset k and layout window padding
        initial_phase = (ref_shift_d + self.phase) % self.period
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

        initial_phase = self.calculate_phase(working_seq)
        if initial_phase is None:
            return None

        return {
            "origin_stream_phase": initial_phase,
            "recovered_sequence": working_seq.tolist(),
            "errors_corrected": error_idxs,
            "missing_gaps": unknown_indices.tolist(),
            "status": status_msg
        }


# =====================================================================
# TESTS
# =====================================================================
def test_qr():
    print("=======================================================")
    print("Initializing Multi-Scenario Binary QR Test Suite...")
    print("=======================================================")

    # Select the target function provider (either a class instance method or global helper)
    solver = extract_null_space_projector

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

    poly = Polynomial(0x25)
    L = 20
    # 1. Inner feedback coefficients for U_forward (0x25) ordered [c1, c2, c3, c4]

    coeffs = poly.coeffs()

    # 2. Perfect 20-bit differential sequence matching U_forward, phase=12
    perfect_seq = poly.generate_reference_sequence()[:L]

    # Define a mask that hides indexes 4 and 12 (zeros mark unknown gaps)
    broken_bits = [12]
    base_mask = np.ones(L, dtype=int)
    # -------------------------------------------------------------------------
    # CASE 0: Clean stream modulo missing mask variables
    # -------------------------------------------------------------------------
    broken_input = np.array(perfect_seq, copy=True)
    broken_input[broken_bits] ^= 1

    status, error_idxs = find_error_via_syndrome_projection(broken_input, base_mask, coeffs)
    assert status == "corrected", f"Expected 'corrected', got '{status}'"
    assert error_idxs == broken_bits, f"Expected error at index {broken_bits}, got {error_idxs}"
    print("-> Case 0 (isolated bit-flip error at index 4): PASSED")

    broken_bits = [4,12]
    base_mask = np.ones(L, dtype=int)
    # -------------------------------------------------------------------------
    # CASE 1: Clean stream modulo missing mask variables
    # -------------------------------------------------------------------------
    broken_input = np.array(perfect_seq, copy=True)
    broken_input[broken_bits] ^= 1

    status, error_idxs = find_error_via_syndrome_projection(broken_input, base_mask, coeffs)
    assert status == "corrected", f"Expected 'corrected', got '{status}'"
    assert error_idxs == broken_bits, f"Expected error at index {broken_bits}, got {error_idxs}"
    print("-> Case 1 (isolated bit-flip error at index 4, 12): PASSED")

    # -------------------------------------------------------------------------
    # CASE A: Clean stream modulo missing mask variables
    # -------------------------------------------------------------------------
    clean_input = np.array(perfect_seq, copy=True)
    clean_input[broken_bits] ^= 1
    base_mask[broken_bits] = 0

    status, error_idxs = find_error_via_syndrome_projection(clean_input, base_mask, coeffs)

    assert status == "corrected", f"Expected 'corrected', got '{status}'"
    assert error_idxs == [], f"Expected empty list, got {error_idxs}"
    print("-> Case A (Clean sequence with missing gaps): PASSED")

    # -------------------------------------------------------------------------
    # CASE B: Missing mask gaps + a single corrupted unmasked bit flip
    # -------------------------------------------------------------------------
    expected_list = [8]
    noisy_input = np.array(perfect_seq, copy=True)
    noisy_input[expected_list] ^= 1
    noisy_input[broken_bits] = 0
    base_mask[broken_bits] = 0

    status, error_idxs = find_error_via_syndrome_projection(noisy_input, base_mask, coeffs)
    assert status == "corrected", f"Expected 'corrected', got '{status}'"
    assert error_idxs == expected_list, f"Expected error at index {expected_list}, got {error_idxs}"
    print("-> Case B (Missing gaps + single bit-flip error at index 7): PASSED")


    # -------------------------------------------------------------------------
    # CASE B: Missing mask gaps + a paired bit flip
    # -------------------------------------------------------------------------
    expected_list = [7,8]
    noisy_input = np.array(perfect_seq, copy=True)
    noisy_input[expected_list] ^= 1
    noisy_input[broken_bits] = 0
    base_mask[broken_bits] = 0

    status, error_idxs = find_error_via_syndrome_projection(noisy_input, base_mask, coeffs)
    assert status == "corrected", f"Expected 'corrected', got '{status}'"
    assert error_idxs == expected_list, f"Expected error at index {expected_list}, got {error_idxs}"
    print("-> Case B paired (Missing gaps + paired bit-flip error at index 7): PASSED")

    # -------------------------------------------------------------------------
    # CASE C: Missing mask gaps + multiple uncorrectable errors (double flip)
    # -------------------------------------------------------------------------
    expected_list = [7,15]
    broken_input = np.array(perfect_seq, copy=True)
    broken_input[expected_list] ^= 1
    noisy_input[broken_bits] = 0
    base_mask[broken_bits] = 0

    status, error_idxs = find_error_via_syndrome_projection(broken_input, base_mask, coeffs)
    assert status == "corrected", f"Expected 'corrected', got '{status}'"
    assert error_idxs == expected_list, f"Expected empty list, got {error_idxs}"
    print("-> Case C (Missing gaps + double bit-flip): PASSED")

    # -------------------------------------------------------------------------
    # CASE D: Missing mask gaps + double flip near the edge
    # -------------------------------------------------------------------------
    expected_list = [2,3]
    broken_input = np.array(perfect_seq, copy=True)
    broken_input[expected_list] ^= 1
    base_mask = np.ones(L, dtype=int)
    broken_bits =[4,5]
    base_mask[broken_bits] = 0
    broken_input[broken_bits] = 0
    status, error_idxs = find_error_via_syndrome_projection(broken_input, base_mask, coeffs)
    assert status == "corrected", f"Expected 'corrected', got '{status}'"
    assert error_idxs == expected_list, f"Expected empty list, got {error_idxs}"
    print("-> Case C (Missing gaps + double bit-flip): PASSED")




    print("find_error_via_syndrome_projection checks: PASSED")


def test_m_sequence_decoding_and_matching():
    chosen_delay = 2
    my_hex_polys = {
        "U_forward": 0x25, "U_reverse": 0x29,
        "V_forward": 0x3D, "V_reverse": 0x2F,
        "W_forward": 0x3B, "W_reverse": 0x37
    }
    my_shifts = { name: Polynomial(sequence).get_delayed_xor_phase(chosen_delay) for name, sequence in my_hex_polys.items() }

    sequence = "U_forward"
    analyzer = MSequenceAnalyzer(my_hex_polys[sequence], my_shifts[sequence])

    # Valid differential M-sequence fragment for U_forward (0x25) with initial phase b = 10
    base_m_sequence = Polynomial(my_hex_polys[sequence]).generate_reference_sequence()
    target_phase = 14
    requested_length = 25

    # Generate streams dynamically
    perfect_diff_seq = generate_diff_from_m_sequence(
        base_m_sequence, phase=target_phase, num_bits=requested_length, diff_delay=chosen_delay
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


def test_polynomial():
    p = Polynomial(0x25)
    assert(p.reciprocal().poly == 0x29)
    assert (p.degree() == 5)
    print("test_polynomial: PASSED")


def test_sequence():
    phase0 = calculate_reverse_alignment(0x29)
    phase1 = calculate_reverse_alignment(0x25)
    assert phase1 == 0 and phase0 == 0
    phase0 = calculate_reverse_alignment(0x3D)
    phase1 = calculate_reverse_alignment(0x2F)
    assert phase1 == phase0
    phase0 = calculate_reverse_alignment(0x3B)
    phase1 = calculate_reverse_alignment(0x37)
    assert phase1 == phase0


def test_inversion_matrix_gf2():
    """
    Comprehensive test suite for GF(2) Matrix Inverse and Pseudo-Inverse functions.
    Validates algebraic consistency and identity convergence using strict assertions.
    """
    print("Initializing GF(2) Inversion Validation Protocol...")

    # =========================================================================
    # PART 1: TESTING SQUARE INVERSION (3x3 Matrix)
    # =========================================================================
    # Define a known non-singular (invertible) 3x3 binary matrix over GF(2)
    # Definitive 3x3 square matrix definition via flattened row vector
    # A verified, strictly non-singular 3x3 square matrix over GF(2) (Determinant = 1)
    # Row 0: [1, 0, 0], Row 1: [1, 1, 0], Row 2: [1, 1, 1]
    A_flat = np.array([1, 0, 0, 1, 1, 0, 1, 1, 1], dtype=np.uint8)
    A_square = A_flat.reshape(3, 3)

    n_square = A_square.shape[0]
    I_square = np.eye(n_square, dtype=np.uint8)

    # 1.2 Test high-performance LUP decomposition inversion
    A_inv_lup = gf2_matrix_inverse_via_lup(A_square)
    assert A_inv_lup is not None, "LUP Inverse failed: Returned None for non-singular matrix."

    # Mathematical proof: (A @ A^-1_lup) mod 2 == I
    identity_lup = (A_square @ A_inv_lup) % 2
    assert np.array_equal(identity_lup, I_square), "LUP Inverse algebraic violation: (A @ A^-1_lup) % 2 != I"

    # =========================================================================
    # PART 2: TESTING RECTANGULAR PSEUDO-INVERSION (3x4 Matrix)
    # =========================================================================
    # Define a rectangular 3x4 binary matrix with linearly independent columns/rows
    # mathematically tracking your real-world unreduced parity-check configurations.
    # Definitive 3x4 rectangular matrix definition via flattened row vector
    # A verified full-column-rank 4x3 tall rectangular matrix over GF(2)
    # Gramian matrix (H^T @ H) is guaranteed to be non-singular (Determinant = 1)
    H_flat = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1], dtype=np.uint8)
    H_rect = H_flat.reshape(4, 3)

    # Note: For our Moore-Penrose pseudo-inverse formulation H^+ = (H^T @ H)^-1 @ H^T,
    # the matrix H^T @ H will have a size of 4x4.
    H_pseudo_inv = gf2_matrix_pseudo_inverse(H_rect)
    assert H_pseudo_inv is not None, "Pseudo-Inverse failed: Gramian matrix generated unexpected singularity."

    # Assert structural dimensionality properties
    # If H is (3 x 4), H^+ must have flipped dimensions (4 x 3)
    assert H_pseudo_inv.shape == (3, 4), f"Dimensionality violation: Expected (4,3), got {H_pseudo_inv.shape}"

    # Verify the core Moore-Penrose identity constraint over GF(2):
    # For a valid left pseudo-inverse, (H^+ @ H) @ H^+ (mod 2) must strictly equal H^+ (mod 2).
    lhs_identity = (H_pseudo_inv @ H_rect) % 2
    verify_identity = (lhs_identity @ H_pseudo_inv) % 2
    assert np.array_equal(verify_identity, H_pseudo_inv), "Moore-Penrose property violation: (H^+ @ H @ H^+) % 2 != H^+"

    print("-> Part 2: Rectangular 3x4 Pseudo-Inversion passes strict verification.")
    print("=== ALL GF(2) ALGEBRAIC MATRIX INVERSION TESTS PASSED TRIUMPHANTLY ===")



if __name__ == "__main__":
    print("Running assertion-based test suite...")
    test_polynomial()
    test_qr()
    test_inversion_matrix_gf2()
    test_sequence()
    test_build_parity_check_matrix()
    test_gauss_jordan_gf2_valid()
    test_gauss_jordan_gf2_inconsistent()
    test_find_error_via_syndrome_projection()
    test_m_sequence_decoding_and_matching()
    print("All tests completed successfully!")
