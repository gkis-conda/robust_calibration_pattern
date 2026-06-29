# =====================================================================
# SYSTEM COMPONENT: generate.py (Functional Library)
# DESCRIPTION: Creates physical pattern representations. A pattern is a
#              hexagonal lattice mapped into a rectangular space. It
#              consists of two geometric shapes (circles and equilateral
#              triangles) acting as custom binary tokens that are easily
#              extracted and classified by traditional CV blob finders.
#              This module generates the structural matrix, manages IO
#              serialization, and exports the linear vector grid to SVG.
# =====================================================================

import os
import numpy as np
from matcher import AlgebraicGridDecoder32


def generate_triangular_gray_grid(width_nodes: int,
                                  height_nodes: int,
                                  grid_style: str = "normal") -> np.ndarray:
    """
    External wrapper function to generate the triangular grid matrix pattern.

    Variables Description:
        width_nodes (int)  : Number of grid elements along the horizontal axis.
        height_nodes (int) : Number of grid elements along the vertical axis.
        grid_style (str)   : Mapping configuration ("normal" or "zigzag").

    Returns:
        np.ndarray         : Integer matrix carrying token definitions (0 or 1).
    """
    generator = AlgebraicGridDecoder32(width_nodes, height_nodes)
    if grid_style == "zigzag":
        return generator.build_barycentric_matrix_zigzag()
    return generator.build_barycentric_matrix()


class PhysicalMeshGenerator:
    """
    Generates continuous 2D physical world coordinates on an equilateral
    triangular mesh layout, storing the underlying structural grid matrix state
    and providing sequential iterator access over the grid shapes.

    """

    def __init__(self,
                 grid_matrix: np.ndarray,
                 step_mm: float,
                 r_circ: float,
                 circle_points_per_mm: float = 2.0):
        """
        Initializes the spatial geometry mesh engine parameters.

        Variables Description:
            grid_matrix (np.ndarray)   : 2D array of tokens (0=Circle, 1=Triangle, -1=Void).
            step_mm (float)            : Physical distance between adjacent node centers in mm.
            r_circ (float)             : Radius of the canonical circular token in mm.
            circle_points_per_mm (float): Interpolation density factor for generating SVG polygon circles.
        """
        self.grid_matrix = np.array(grid_matrix)
        self.step_mm = float(step_mm)
        self.r_circ = float(r_circ)
        self.circle_points_per_mm = float(circle_points_per_mm)

        # r_tri: Calculated radius of area-matched equilateral triangle tokens
        self.r_tri = self.r_circ * np.sqrt(np.pi / (3.0 * np.sqrt(3.0) / 4.0))

        # Center calibration offsets: Center the grid coordinate space relative to (0,0) world coordinates
        self.center_x_offset = (1.0 - float(self.grid_matrix.shape[1])) / 2.0
        self.center_y_offset = (1.0 - float(self.grid_matrix.shape[0])) / 2.0

    def __iter__(self):
        """
        Iterator implementation yielding (i, j, shape_type, contour)
        for every element sequentially across the stored grid layout.
        """
        H_nodes, W_nodes = self.grid_matrix.shape
        for i in range(H_nodes):
            for j in range(W_nodes):
                shape_type = self.grid_matrix[i, j]
                contour = self.get_shape_contour(i, j)
                yield i, j, shape_type, contour

    def get_shape_center(self, r: int, c: int) -> list:
        """
        Computes 2D physical world positions mapped directly inside the
        true continuous hexagonal space with row-staggering.

        Variables Description:
            r (int)    : Target row coordinate index inside the storage matrix buffer.
            c (int)    : Target column coordinate index inside the storage matrix buffer.

        Returns:
            list       : [x_phys, y_phys] absolute hexagonal screen positions in mm.
        """
        # RESTORED: Pure hexagonal physical lattice generation.
        # Inside the storage matrix, rows and columns are packed flat.
        # But in physical space, every odd row is shifted horizontally by exactly 0.5 * step_mm,
        # and vertical row spacing is compressed by sqrt(3)/2 to maintain equilateral triangulation.
        x_phys = ((float(c) + 0.5 * float(r % 2)) + self.center_x_offset) * self.step_mm
        y_phys = (float(r) * np.sqrt(3.0) / 2.0 + self.center_y_offset) * self.step_mm
        return [x_phys, y_phys]

    def get_shape_contour(self, r: int, c: int) -> list:
        """
        Calculates and returns the explicit boundary contour point sequence
        for a node specified by internal grid matrix indices (r, c).
        """
        shape_type = self.grid_matrix[r, c]
        if shape_type < 0:
            return []

        x_phys, y_phys = self.get_shape_center(r, c)

        if shape_type == 1:
            # Equilateral triangle geometry calculation (area-matched to circles)
            h_top = self.r_tri
            h_bottom = self.r_tri * 0.5
            w_half = self.r_tri * 0.8660254037844386

            p1 = (x_phys, y_phys - h_top)
            p2 = (x_phys - w_half, y_phys + h_bottom)
            p3 = (x_phys + w_half, y_phys + h_bottom)
            return [p1, p2, p3]

        elif shape_type == 0:
            # Circle interpolation into explicit polygon vertices
            circle_perimeter = 2.0 * np.pi * self.r_circ
            num_circle_pts = max(8, int(round(circle_perimeter * self.circle_points_per_mm)))

            angles = np.linspace(0, 2.0 * np.pi, num_circle_pts, endpoint=False)
            circle_poly = []
            for angle in angles:
                cx = x_phys + self.r_circ * np.cos(angle)
                cy = y_phys + self.r_circ * np.sin(angle)
                circle_poly.append((cx, cy))
            return circle_poly

        return []

    def save_to_svg(self, filename: str) -> None:
        """
        Generates and exports the explicitly generated point boundaries into
        an SVG file by consuming the instance's grid iterator directly.
        """
        import svgwrite
        width_mm = (self.grid_matrix.shape[1] + 1) * self.step_mm
        height_mm = (self.grid_matrix.shape[0] + 1) * self.step_mm

        dwg = svgwrite.Drawing(
            filename,
            size=(f"{width_mm}mm", f"{height_mm}mm"),
            viewBox=f"{-width_mm / 2.0} {-height_mm / 2.0} {width_mm} {height_mm}"
        )

        for i, j, shape_type, contour in self:
            if contour:
                dwg.add(dwg.polygon(points=contour, fill='black'))

        dwg.save()


def save_bit_matrix(filename: str, matrix: np.ndarray) -> None:
    """
    Saves a NumPy integer or boolean matrix to a text ASCII file cleanly.

    Variables Description:
        filename (str)     : Target output file text path.
        matrix (np.ndarray): Target grid matrix payload array.
    """
    with open(filename, 'w', encoding='ascii') as f:
        for row in matrix:
            line = "".join('1' if cell else '0' for cell in row)
            f.write(line + '\n')


def load_bit_matrix(filename: str) -> np.ndarray:
    """
    Loads a NumPy boolean matrix from a text ASCII file safely.

    Variables Description:
        filename (str)     : Target string input source file text path.

    Returns:
        np.ndarray         : Evaluated boolean logical tracking canvas array.
    """
    lines = []
    with open(filename, 'r', encoding='ascii') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                lines.append([char == '1' for char in clean_line])
    return np.array(lines, dtype=bool)


def run_matrix_test() -> None:
    """Tests save and load operations using numpy and asserts."""
    test_filename = "test_matrix_numpy.txt"
    rows, cols = 6, 6

    original_matrix = np.random.choice([True, False], size=(rows, cols))

    save_bit_matrix(test_filename, original_matrix)
    loaded_matrix = load_bit_matrix(test_filename)

    if os.path.exists(test_filename):
        os.remove(test_filename)

    assert isinstance(loaded_matrix, np.ndarray), "Loaded object is not a NumPy array"
    assert loaded_matrix.dtype == bool, "Loaded array data type is not boolean"
    assert original_matrix.shape == loaded_matrix.shape, f"Shape mismatch: {original_matrix.shape} vs {loaded_matrix.shape}"
    assert np.array_equal(original_matrix, loaded_matrix), "Matrix data values do not match perfectly"

    print("Test passed: All NumPy assert checks completed successfully.")



STEP = 1

def print_step(t: str) -> None:
    """Prints a formalized orchestration message checkpoint."""
    global STEP
    print(f"--- STEP {STEP}: {t} ---")
    STEP += 1


if __name__ == "__main__":
    # 1. Run internal regression check first
    run_matrix_test()

    # 2. Configure operational framework sizes
    W_NODES = 31
    H_NODES = 31
    STEP_MM = 12.0
    DOT_RADIUS_MM = 2.0

    matrix_filename = f"pattern{W_NODES}x{H_NODES}.txt"
    svg_filename = f"pattern{W_NODES}x{H_NODES}.svg"

    # 3. Synchronize database matrix caching layers
    if os.path.exists(matrix_filename):
        print_step("Debug mode: Loading Digital Mesh Matrix")
        mesh_blueprint = load_bit_matrix(matrix_filename)
    else:
        print_step("Generate Digital Mesh Matrix")
        mesh_blueprint = generate_triangular_gray_grid(width_nodes=W_NODES, height_nodes=H_NODES)
        save_bit_matrix(matrix_filename, mesh_blueprint)

    # 4. Trigger vector rendering export pass
    print_step("Export to SVG via Pure Linear Vector Maps")
    gen = PhysicalMeshGenerator(mesh_blueprint, STEP_MM, DOT_RADIUS_MM)
    gen.save_to_svg(filename=svg_filename)

    print(f"[SUCCESS]: Generated matrix pattern sheet saved to {svg_filename}")
