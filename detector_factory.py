import sys
import argparse
import os
import numpy as np

from opencv_baseline_engine import OpenCVCirclesGridMeshGenerator, OpenCVGridDetector
from generate import PhysicalMeshGenerator, generate_triangular_gray_grid
from detector import HexagonalTopologyDetector


def generate_opencv_unary_blueprint(rows, cols):
    """
    OpenCV's cv2.findCirclesGrid demands a pristine array where every node is an identical
    circle tracking element (0). Missing cells or token variances crash the OpenCV graph solver.
    """
    return np.zeros((rows, cols), dtype=np.int32)


def pattern_blueprint(engine_name, rows, cols):
    """
    Virtual Pattern Constructor. Automatically delivers the correct matrix token structure 
    (Binary Multi-Token or Unary Uniform) based on the target execution engine requirements.
    """
    engine_key = str(engine_name).strip().lower()
    
    if engine_key in ["hgp", "hexagonal"]:
        print(f"Blueprint Factory: Constructing a binary hexagonal error-correcting layout matrix ({rows}x{cols})")
        return generate_triangular_gray_grid(rows, cols)
        
    elif engine_key in ["chessboard", "circles", "asymmetric_circles"]:
        print(f"Blueprint Factory: Constructing a unary uniform target grid layout matrix ({rows}x{cols})")
        return generate_opencv_unary_blueprint(rows, cols)

    else:
        raise ValueError(f"Blueprint Factory Error: Unknown engine structure code requested: '{engine_name}'")


def mesh_generator_factory(engine_name, grid_matrix, step_mm, r_circ, density=2.0):
    """
    Virtual 'Constructor' Factory Engine. Resolves named string arguments
    into explicit class instances conforming to the unified generator interface.
    """
    # Force lowercase string verification to capture variations gracefully
    engine_key = str(engine_name).strip().lower()

    if engine_key in ["hgp", "hexagonal"]:
        print("Factory: Constructing Hexagonal Calibration Engine")
        return PhysicalMeshGenerator(
            grid_matrix=grid_matrix,
            step_mm=step_mm,
            r_circ=r_circ,
            circle_points_per_mm = density
        )

    elif engine_key in ["chessboard", "circles", "asymmetric_circles"]:
        print(f"Factory: Constructing standard Baseline Engine")
        return OpenCVCirclesGridMeshGenerator(
            grid_matrix=grid_matrix,
            grid_shape=engine_key,
            step_mm=step_mm,
            r_circ=r_circ,
            circle_points_per_mm=density
        )

    else:
        # Strict validation fallback safety
        raise ValueError(f"Factory Error: Unknown engine token string requested: '{engine_name}'")


def create_detector(engine_name, grid_rows, grid_cols):
    """
    Factory dispatcher generating standard structural node registry interfaces.
    Args:
        engine_type (str): Either 'HEXAGONAL' or 'OPENCV'.
        grid_rows (int): Expected horizontal geometric line partitions.
        grid_cols (int): Expected vertical geometric line partitions.
        pattern_style (str): Variant profile identifier strictly for OpenCV pipelines.
    """
    engine_key = str(engine_name).strip().lower()

    if engine_key in ["hgp", "hexagonal"]:
        print(f" -> [Factory] Allocating Hexagonal Topology Engine ({grid_rows}x{grid_cols})")
        return HexagonalTopologyDetector(grid_rows=grid_rows, grid_cols=grid_cols)

    elif engine_key in ["chessboard", "circles", "asymmetric_circles"]:
        print(f" -> [Factory] Allocating OpenCV Native Engine: {engine_key} ({grid_rows}x{grid_cols})")
        return OpenCVGridDetector(grid_rows=grid_rows, grid_cols=grid_cols, pattern_type=engine_key)

    raise KeyError(f"Unsupported validation engine type token requested: '{engine_key}'")

def parse_arguments(description="Benchmarking Controller"):
    """
    Parses arguments passed behind Blender's native '--' delimiter token.
    """
    # Blender parsing constraint: look only at args trailing the double dash
    if "--" in sys.argv:
        python_args = sys.argv[sys.argv.index("--") + 1:]
    else:
        python_args = sys.argv[1:]

    from lattice_topology import set_debug_output
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-e", "--engine", type=str, default="hexagonal", help="Execution Engine ('hexagonal'/'opencv')")
    parser.add_argument("-p", "--path", type=str, default="./output", help="Working folder path")
    parser.add_argument("-r", "--rows", type=int, default=31, help="Lattice height count")
    parser.add_argument("-c", "--cols", type=int, default=31, help="Lattice width count")
    parser.add_argument("-V", "--verbose", action="store_true", help="Enable debug output to console")
    parser.add_argument("--save-images", action="store_true", help="Export all high-fidelity rendered perspective-warped frames to PNG assets on disk.")
    args = parser.parse_args(python_args)
    set_debug_output(args.verbose)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    engine = args.engine
    
    LATTICE_ROW_COUNT = args.rows
    LATTICE_COL_COUNT = args.cols
    PATTERN_STEP_MM = 40.0
    PRIMITIVE_RADIUS_MM = PATTERN_STEP_MM / 5
    
    BASE_PATH = args.path
    ENGINE_SPECIFIC_DIR = os.path.abspath(f"{BASE_PATH}_{engine.lower()}")
    if not os.path.exists(ENGINE_SPECIFIC_DIR):
        os.makedirs(ENGINE_SPECIFIC_DIR)
        
    print(f"  -> Selected Engine Profile     : {engine.upper()}")
    print(f"  -> Generated Matrix Bounds     : {LATTICE_ROW_COUNT} rows x {LATTICE_COL_COUNT} columns")
    print(f"  -> Concat Target Folder Path   : {ENGINE_SPECIFIC_DIR}")

    try:
        # Invoke Blueprint Factory to dynamically configure layout arrays
        active_blueprint = pattern_blueprint(
            engine_name=engine,
            rows=LATTICE_ROW_COUNT,
            cols=LATTICE_COL_COUNT
        )
        
        print(f"  -> Active Matrix State Layout:\n{active_blueprint}\n")
        
        # 5. Build asset mesh configurations pairs 
        active_generator = mesh_generator_factory(
            engine_name=engine,
            grid_matrix=active_blueprint,
            step_mm=PATTERN_STEP_MM,
            r_circ=PRIMITIVE_RADIUS_MM
        )
        print(f"[SUCCESS] Environment components initialized.")
        active_generator.save_to_svg(os.path.join(ENGINE_SPECIFIC_DIR, "pattern.svg"))
    except ValueError as err:
        print(err)
        sys.exit(1)