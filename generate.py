import numpy as np
import os
from matcher import AlgebraicGridDecoder32
# =====================================================================
# 1. GRID GENERATION AND PRINTING UTILITIES
# =====================================================================
def generate_triangular_gray_grid(width_nodes=32, height_nodes=32, type:str = "normal" ):
    """
    External wrapper function to generate the triangular grid matrix pattern.
    Accepts arbitrary dimensions for width and height.
    """
    generator = AlgebraicGridDecoder32(width_nodes, height_nodes)
    return generator.build_barycentric_matrix_zigzag() if type == "zigzag" else generator.build_barycentric_matrix()


class PhysicalMeshGenerator:
    """
    Generates continuous 2D physical world coordinates on an equilateral 
    triangular mesh layout, storing the underlying structural grid matrix state
    and providing sequential iterator access over the grid shapes.
    """
    def __init__(self, grid_matrix, step_mm, r_circ, circle_points_per_mm=2.0):
        self.grid_matrix = np.array(grid_matrix)
        self.step_mm = float(step_mm)
        self.r_circ = float(r_circ)
        self.circle_points_per_mm = float(circle_points_per_mm)
        
        self.r_tri = self.r_circ * np.sqrt(np.pi / (3 * np.sqrt(3) / 4))
        self.center_x_offset = (1.-grid_matrix.shape[1]) / 2.0
        self.center_y_offset = (1.-grid_matrix.shape[0]) / 2.0

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

    def get_shape_center(self, r, c):
        # (Even-R) topology
        x_phys = ((c + 0.5 * (r % 2)) + self.center_x_offset) * self.step_mm
        y_phys = (r * np.sqrt(3) / 2.0 + self.center_y_offset) * self.step_mm
        return [x_phys, y_phys]

    def get_shape_contour(self, r, c):
        """
        Calculates and returns the explicit boundary contour point sequence 
        for a node specified by internal grid matrix indices (i, j).
        
        Returns:
            list of tuples: [(x1, y1), (x2, y2), ...] representing the perimeter vertices.
        """
        shape_type = self.grid_matrix[r, c]
        x_phys, y_phys = self.get_shape_center(r,c)

        if shape_type == 1:
            # Equilateral triangle geometry calculation (area-matched)
            h_top = self.r_tri
            h_bottom = self.r_tri * 0.5
            w_half = self.r_tri * 0.866025
            
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

    def save_to_svg(self, filename):
        """
        Generates and exports the explicitly generated point boundaries into an SVG file 
        by consuming the instance's grid iterator directly.
        """
        import svgwrite
        width_mm = (self.grid_matrix.shape[1] + 1) * self.step_mm
        height_mm = (self.grid_matrix.shape[0] + 1) * self.step_mm
        dwg = svgwrite.Drawing(
            filename, 
            size=(f"{width_mm}mm", f"{height_mm}mm"),
            viewBox=f"{-width_mm/2} {-height_mm/2} {width_mm} {height_mm}"
        )
        
        # Stream shapes directly out of the grid using the built-in iterator
        for i, j, shape_type, contour in self:
            dwg.add(dwg.polygon(points=contour, fill='black'))
            
        dwg.save()


def save_bit_matrix(filename: str, matrix: np.ndarray) -> None:
    """Saves a NumPy boolean matrix to a text ASCII file."""
    with open(filename, 'w', encoding='ascii') as f:
        for row in matrix:
            line = "".join('1' if cell else '0' for cell in row)
            f.write(line + '\n')


def load_bit_matrix(filename: str) -> np.ndarray:
    """Loads a NumPy boolean matrix from a text ASCII file."""
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

    # 1. Generate a random boolean matrix using NumPy
    original_matrix = np.random.choice([True, False], size=(rows, cols))

    # 2. Save and load the matrix
    save_bit_matrix(test_filename, original_matrix)
    loaded_matrix = load_bit_matrix(test_filename)

    # 3. Assertions for structural and data integrity
    assert isinstance(loaded_matrix, np.ndarray), "Loaded object is not a NumPy array"
    assert loaded_matrix.dtype == bool, "Loaded array data type is not boolean"
    assert original_matrix.shape == loaded_matrix.shape, f"Shape mismatch: {original_matrix.shape} vs {loaded_matrix.shape}"
    assert np.array_equal(original_matrix, loaded_matrix), "Matrix data values do not match perfectly"

    print("Test passed: All NumPy assert checks completed successfully.")


# =====================================================================
# 3. PIPELINE INTEGRATION AND SIMULATION EXAMPLE
# =====================================================================
STEP = 1
def print_step(t : str):
    global STEP
    print(f"--- STEP {STEP}: {t} ---")
    STEP += 1

if __name__ == "__main__":
    # Grid configuration parameters
    W_NODES = 31
    H_NODES = 31
    STEP_MM = 12.0
    DOT_RADIUS_MM = 2
    filename = f"pattern{W_NODES}x{H_NODES}.txt"
    if  os.path.exists(filename):
        print_step("Debug mode: Loading Digital Mesh Matrix")
        mesh_blueprint = load_bit_matrix(filename)
    else:
        print_step("Generate Digital Mesh Matrix")
        mesh_blueprint = generate_triangular_gray_grid(width_nodes=W_NODES, height_nodes=H_NODES)
        save_bit_matrix(filename, mesh_blueprint)
    print_step("Export to SVG")
    gen = PhysicalMeshGenerator(mesh_blueprint, STEP_MM, DOT_RADIUS_MM)
    gen.save_to_svg(filename=f"pattern{W_NODES}x{H_NODES}.svg")
