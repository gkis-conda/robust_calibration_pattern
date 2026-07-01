import sys
import argparse
import os
import numpy as np

from opencv_baseline_engine import OpenCVCirclesGridMeshGenerator
from generate import PhysicalMeshGenerator, generate_triangular_gray_grid

def generate_opencv_unary_blueprint(rows, cols):
    """
    OpenCV's cv2.findCirclesGrid demands a pristine array where every node is an identical
    circle tracking element (0). Missing cells or token variances crash the OpenCV graph solver.
    """
    return np.zeros((rows, cols), dtype=np.int32)

def pattern_blueprint_factory(engine_name, rows, cols):
    """
    Virtual Pattern Constructor. Automatically delivers the correct matrix token structure 
    (Binary Multi-Token or Unary Uniform) based on the target execution engine requirements.
    """
    engine_key = str(engine_name).strip().lower()
    
    if engine_key in ["hcp"]:
        print(f"Blueprint Factory: Constructing a binary hexagonal error-correcting layout matrix ({rows}x{cols})")
        return generate_triangular_gray_grid(rows, cols)
        
    elif engine_key in ["opencv", "opencv_baseline", "baseline"]:
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
    
    if engine_key in ["hcp"]:
        print("Factory: Constructing Hexagonal Calibration Engine")
        return PhysicalMeshGenerator(
            grid_matrix=grid_matrix, 
            step_mm=step_mm, 
            r_circ=r_circ
        )
        
    elif engine_key in ["opencv", "opencv_baseline", "baseline"]:
        print(f"Factory: Constructing standard Baseline Engine")
        return OpenCVCirclesGridMeshGenerator(
            grid_matrix=grid_matrix, 
            step_mm=step_mm, 
            r_circ=r_circ,
            circle_points_per_mm=density
        )
        
    else:
        # Strict validation fallback safety
        raise ValueError(f"Factory Error: Unknown engine token string requested: '{engine_name}'")


def parse_blender_arguments():
    """
    Parses arguments passed behind Blender's native '--' delimiter token.
    """
    # Blender parsing constraint: look only at args trailing the double dash
    if "--" in sys.argv:
        python_args = sys.argv[sys.argv.index("--") + 1:]
    else:
        python_args = []

    parser = argparse.ArgumentParser(description="TCM Benchmarking Run Parameter Controller")
    parser.add_argument("-e", "--engine", type=str, default="trellis", help="Execution Engine ('trellis'/'opencv')")
    parser.add_argument("-r", "--rows", type=int, default=31, help="Lattice height count")
    parser.add_argument("-c", "--cols", type=int, default=31, help="Lattice width count")
       
    return parser.parse_args(python_args)


if __name__ == "__main__":
    args = parse_blender_arguments()
    engine = args.engine
    
    LATTICE_ROW_COUNT = args.rows
    LATTICE_COL_COUNT = args.cols
    PATTERN_STEP_MM = 20.0
    PRIMITIVE_RADIUS_MM = 6.0
    
    BASE_PATH = "./blender_output"
    ENGINE_SPECIFIC_DIR = os.path.abspath(f"{BASE_PATH}_{engine}")
    if not os.path.exists(ENGINE_SPECIFIC_DIR):
        os.makedirs(ENGINE_SPECIFIC_DIR)
        
    print(f"  -> Selected Engine Profile     : {engine.upper()}")
    print(f"  -> Generated Matrix Bounds     : {LATTICE_ROW_COUNT} rows x {LATTICE_COL_COUNT} columns")
    print(f"  -> Concat Target Folder Path   : {ENGINE_SPECIFIC_DIR}")

    try:
        # Invoke Blueprint Factory to dynamically configure layout arrays
        active_blueprint = pattern_blueprint_factory(
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

    except ValueError as err:
        print(err)
        sys.exit(1)