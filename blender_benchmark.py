import bpy
import mathutils
import math
import os
import numpy as np
import sys
import random
import cv2


# 1. Capture the exact absolute folder path where this script resides
# (Safe for execution via relative paths or system symlinks)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Append to Python's module lookup stack if not already registered
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from detector_factory import *
from camera import *

# ==============================================================================
# [INSERT YOUR PhysicalMeshGenerator CLASS CODE HERE]
# ==============================================================================

# --- MATHEMATICAL TRANSFORMATION MACROS (NAMED CONSTANTS) ---
MM_TO_METER = lambda x: float(x) * 0.001
METER_TO_MM = lambda x: float(x) * 1000.

# --- FIXED HARDWARE AND ENVIRONMENTAL CONSTANTS ---
BLENDER_SENSOR_WIDTH_MM = 36.0                 # Standard full-frame sensor metric
RESOLUTION_PERCENTAGE_FULL = 100               # Scale factor for target frame rendering
Z_PLANE_FIGHTING_OFFSET_METERS = 0.0001        # Slight offset to prevent mesh clipping
CYCLES_RAYTRACING_SAMPLES = 18                 # Computation constraints limit
BLUR_EXPOSURE_SHUTTER_MAX = 1.0                # Keep shutter active across entire frame

# --- JITTER MOTION CHARACTERISTICS ---
TREMOR_DEG = 1./60                              # Sub-pixel rotational hand shake step

# --- CAMERA PARAMETERS DEFINITIONS AND CONSTRAINTS ---
DEFAULT_TX, DEFAULT_TY, Z_DISTANCE = 0.0, 0.0, -1.5
K1_DISTORTION = -0.2
IMG_SHAPE = (1080, 1920)

# --- PATTERN PARAMETERS
PATTERN_STEP_MM = 45.0
PRIMITIVE_RADIUS_MM = 6.0

# --- SHADING AND GRAPHICS CONSTANTS ---

# --- RIGID ENVIRONMENTAL LIGHTING CONSTANTS ---
STUDIO_SOFTBOX_ENERGY_WATTS = 300.0           # Power output for the main AREA light
STUDIO_SOFTBOX_SIZE_METERS = 1.0               # Dimensions of the main AREA light softbox
STUDIO_SOFTBOX_HEIGHT_METERS = 3.0             # Z-axis height position of the softbox
STUDIO_SOFTBOX_RADIUS_METERS = 2.0             # Softbox emitter area size radius

AMBIENT_SUN_ENERGY_WATTS = 0.5                 # Fill light output to soften sharp shadows
AMBIENT_SUN_HEIGHT_METERS = 4.0                # Z-axis height position of the fill light

# --- PHYSICAL BACKGROUND SHEET CONSTANTS ---
WHITE_PAPER_SIZE_METERS = 4.0                  # Total boundary width/height of background sheet
CAMERA_FOV_SAFETY_MARGIN = 1.15                # Strict boundary multiplier padding

# --- MATERIAL ROUGHNESS PROFILES (BRDF SPECIFICATION) ---
COLOR_WHITE_RGB = (1.0, 1.0, 1.0, 1.0)         # Absolute white spectrum reflection
COLOR_BLACK_RGB = (0.0, 0.0, 0.0, 1.0)         # Absolute black token absorption

MATTE_PAPER_ROUGHNESS = 0.6                    # Realistic diffuse micro-texture roughness
MATTE_PAPER_SPECULAR = 0.1                     # Low glancing reflectance gloss index
FILE_OUTPUT_NAME = "TCM_File_Output"
DISTORTION_NODE_NAME = "TCM_Distortion_Node"


# ==============================================================================
# PIPELINE FUNCTIONS WITH EXPLICIT CONSTANTS MAPPING
# ==============================================================================
def calculate_lattice_global_offset(mesh_gen):
    """
    Analytically computes layout bounds using your physical step metric
    to add exactly one full lattice step of white padding around the border.
    """
    H, W = mesh_gen.grid_matrix.shape
    p_tl = mesh_gen.get_shape_center(0, 0)
    p_br = mesh_gen.get_shape_center(H - 1, W - 1)

    # Standard unit conversion to Blender meters
    x_min, x_max = MM_TO_METER(p_tl[0]), MM_TO_METER(p_br[0])
    y_min, y_max = MM_TO_METER(p_tl[1]), MM_TO_METER(p_br[1])

    cx = (x_min + x_max) / 2.
    cy = (y_min + y_max) / 2.

    # Calculate base size of the pattern nodes area
    raw_w = abs(x_max - x_min)
    raw_h = abs(y_max - y_min)

    # PHYSICAL BOUNDARY FIX: Instead of a magic 1.15 multiplier,
    # we add exactly one full physical step_mm to the total width and height.
    step_meters = MM_TO_METER(mesh_gen.step_mm)
    grid_w = raw_w + 2 * step_meters
    grid_h = raw_h + 2 * step_meters

    return cx, cy, grid_w, grid_h


def setup_studio_illumination():
    """
    Instantiates an explicit high-contrast studio softbox lighting array
    to guarantee uniform, mathematically clean background white card exposure.
    """
    # We use a dual key-and-fill lighting setup placed outside the camera frustum,
    # angled at 45 degrees to illuminate the board without shining down into the lens.
    bpy.ops.object.light_add(
        type='AREA',
        radius=1.0,
        location=(-1.5, 0.0, -1.5)  # Positioned 1.5m to the left, 1.5m high
    )
    key_light = bpy.context.active_object
    key_light.name = "Studio_Side_Key_Light"
    key_light.data.energy = STUDIO_SOFTBOX_ENERGY_WATTS
    key_light.data.shape = 'SQUARE'
    key_light.data.size = STUDIO_SOFTBOX_SIZE_METERS
    key_light.rotation_euler = (0.0, math.radians(-135.0), 0.0)

    # Light Source B (Fill Light - Right Side to eliminate harsh shadows)
    bpy.ops.object.light_add(
        type='AREA',
        radius=1.0,
        location=(1.5, 0.0, -1.5)  # Positioned 1.5m to the right, 1.5m high
    )
    fill_light = bpy.context.active_object
    fill_light.name = "Studio_Side_Fill_Light"
    fill_light.data.energy = STUDIO_SOFTBOX_ENERGY_WATTS
    fill_light.data.shape = 'SQUARE'
    fill_light.data.size = STUDIO_SOFTBOX_SIZE_METERS
    fill_light.rotation_euler = (0.0, math.radians(135.0), 0.0)


def white_paper_background(cx, cy, w, h):
    """
    Procedurally constructs a macro white background plane acting as the
    physical canvas backing underneath the active token layout.
    """
    # Create the mesh primitive bounding geometry card
    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=(cx, cy, Z_PLANE_FIGHTING_OFFSET_METERS)
    )

    paper_obj = bpy.context.active_object
    paper_obj.name = "White_Background_Paper_Sheet"
    paper_obj.scale.x = w
    paper_obj.scale.y = h

    # Construct a clean, isolated physical PBR paper material instance
    paper_mat = bpy.data.materials.new(name="White_Paper_Material_Asset")
    paper_mat.use_nodes = True

    # Access the master Principled BSDF node to assign target BRDF properties via constants
    bsdf = paper_mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = COLOR_WHITE_RGB
    bsdf.inputs['Roughness'].default_value = MATTE_PAPER_ROUGHNESS
    bsdf.inputs['Specular'].default_value = MATTE_PAPER_SPECULAR

    paper_obj.data.materials.append(paper_mat)


def initialize_blender_scene(samples=CYCLES_RAYTRACING_SAMPLES, shutter_speed=BLUR_EXPOSURE_SHUTTER_MAX):
    """
    Resets the master workspace environment, configures the Cycles PBR engine,
    and establishes base baseline environment illumination elements.
    """
    # 1. Clear active scene objects
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # 2. Setup rendering engine parameters (Cycles)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = shutter_speed
    scene.cycles.preview_samples = 5

    # 3. Lock down pixel filtering options (Critical for sub-pixel edge detection)
    # A Gaussian filter width of 1.50 px provides the perfect sweet spot:
    # crisp enough for sub-pixel circle blob tracking, smooth enough to eliminate aliasing.
    scene.cycles.pixel_filter_type = 'GAUSSIAN'
    scene.cycles.filter_width = 1.50

    # Disable heavy noise reduction (denoising) artifacts which can smear
    # or warp your black-and-white grid transition borders unevenly
    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = False

    # 4. INDIRECT OBLIQUE STUDIO LIGHTING
    setup_studio_illumination()

    # 5. INITIAL UNBOUND OPTICAL SENSOR CAMERA
    # Spawn camera at the center of the grid layout
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera_obj = bpy.context.active_object
    camera_obj.name = "Benchmark_Optical_Sensor"
    scene.camera = camera_obj
    scene.render.use_compositing = False

    return scene, camera_obj


def create_matte_pbr_material(name, bgr_color=(0.0, 0.0, 0.0, 1.0), roughness=0.4):
    """
    Generates a dedicated Principled BSDF node network mapping custom BRDF parameters.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = bgr_color
    bsdf.inputs['Roughness'].default_value = roughness
    return mat


def build_3d_pattern_mesh(case_name:str, mesh_gen):
    """
    Consumes the custom PhysicalMeshGenerator layout object sequence, builds
    explicit spatial vertex faces structures, and binds PBR properties.
    """    
    blender_mesh = bpy.data.meshes.new(f"Mesh_{case_name}")
    pattern_obj = bpy.data.objects.new(f"Obj_{case_name}", blender_mesh)
    bpy.context.collection.objects.link(pattern_obj)

    # Create the physical white background board centered under grid bounds
    center_x, center_y, grid_w, grid_h = calculate_lattice_global_offset(mesh_gen)
    white_paper_background(center_x, center_y, grid_w, grid_h)

    # Parent the white sheet to the empty/root object alongside pattern_obj if needed,
    # so they rotate together under oblique stress test cases.
    all_vertices = []
    all_faces = []
    vert_index_offset = 0

    for i, j, shape_type, contour in mesh_gen:
        if shape_type < 0 or not contour:
            continue
            
        for pt in contour:
            all_vertices.append((MM_TO_METER(pt[0]), MM_TO_METER(pt[1]), 0.0))
            
        num_pts = len(contour)
        face_indices = list(range(vert_index_offset, vert_index_offset + num_pts))
        all_faces.append(face_indices)
        vert_index_offset += num_pts
        
    blender_mesh.from_pydata(all_vertices, [], all_faces)
    blender_mesh.update()
    
    token_mat = create_matte_pbr_material(f"Mat_Tokens_{case_name}", (0.0, 0.0, 0.0, 1.0), 0.4)
    pattern_obj.data.materials.append(token_mat)
    
    return pattern_obj


def setup_camera_hardware_distortion(scene, camera_obj, k1):
    """
    Applies radial distortion directly into Blender 2.92 camera hardware data block.
    Guarantees that lens warp is evaluated BEFORE motion blur bakes, curving
    your tremor trajectories accurately to match true physical sensors.

    Fully compatible with Blender 2.92.0 Cycles engine.
    """
    # 1. Enforce Cycles rendering engine since standard EEVEE does not
    # support advanced hardware panoramic lens configurations in 2.92
    scene.render.engine = 'CYCLES'

    # 2. Switch the camera architecture to Panoramic to unlock lens controls
    camera_obj.data.type = 'PANO'

    # In Blender 2.92, the true polynomial k1/k2 model is accessed via the
    # 'FISHEYE_LENS_POLYNOMIAL' panorama type.
    camera_obj.data.cycles.panorama_type = 'FISHEYE_LENS_POLYNOMIAL'

    # 3. Inject your k1 distortion parameter straight into the hardware fields.
    # Blender 2.92 maps the radial expansion parameters using explicit sub-field indices:
    # k1 matches 'k1', k2 matches 'k2', etc.
    camera_obj.data.cycles.fisheye_lens_polynomial_k1 = float(k1)
    camera_obj.data.cycles.fisheye_lens_polynomial_k2 = 0.0


def configure_camera(scene,
                     camera_obj,
                     intrinsics: dict,
                     camera_extrinsics: dict,
                     start_frame: int,
                     tremor_frame_delta: int) -> None:
    """
    Transforms explicit positional parameters and orientation metrics from OpenCV format,
    applies physical pixel focal calculations, and executes keyframe animation bindings
    on an isolated segment of the timeline using a multi-frame pseudo-random walk.
    """
    w_px = intrinsics["width_px"]
    h_px = intrinsics["height_px"]
    f_px = intrinsics["f_px"]

    scene.render.resolution_x = w_px * 1.2
    scene.render.resolution_y = h_px * 1.2
    scene.render.resolution_percentage = RESOLUTION_PERCENTAGE_FULL

    camera_obj.data.type = 'PERSP'
    camera_obj.data.sensor_fit = 'HORIZONTAL'
    camera_obj.data.lens_unit = 'MILLIMETERS'
    camera_obj.data.lens = (f_px * BLENDER_SENSOR_WIDTH_MM) / scene.render.resolution_x

    tx = camera_extrinsics.get("tx", 0.0)
    ty = camera_extrinsics.get("ty", 0.0)
    tz = camera_extrinsics.get("tz", 1.0)
    roll = math.radians(camera_extrinsics.get("roll", 0.0))
    pitch = math.radians(camera_extrinsics.get("pitch", 0.0))
    yaw = math.radians(camera_extrinsics.get("yaw", 0.0))

    camera_obj.constraints.clear()

    R_mat = mathutils.Euler((pitch, yaw, roll), 'YXZ').to_matrix().to_4x4()
    T_mat = mathutils.Matrix.Translation((tx, ty, tz))

    cv_to_blender_bridge = mathutils.Matrix((
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ))

    target_matrix = T_mat @ R_mat @ cv_to_blender_bridge
    loc, rot_quat, _ = target_matrix.decompose()
    # -----------------------------------------------------------------
    # FRAME N ENTRY POINT: LOCK THE BASELINE OPENCV POSE
    # -----------------------------------------------------------------
    scene.frame_set(start_frame)

    camera_obj.location = loc
    camera_obj.rotation_mode = 'XYZ'
    camera_obj.rotation_euler = rot_quat.to_euler(camera_obj.rotation_mode)

    camera_obj.keyframe_insert(data_path="location", frame=start_frame)
    camera_obj.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    # Freeze the random state vector seed using your case index to guarantee 100%
    # repeatable, deterministic noise paths across regression test sweeps.
    random.seed(start_frame)

    # Maintain a rolling reference accumulator copy of your primary rotation state
    rolling_rot_x = camera_obj.rotation_euler.x
    rolling_rot_y = camera_obj.rotation_euler.y
    rolling_rot_z = camera_obj.rotation_euler.z

    # Step sequentially through each allocated tremor frame slot step
    for target_frame_idx in range(start_frame + 1, start_frame + tremor_frame_delta + 1):
        scene.frame_set(target_frame_idx)

        # Calculate random gait displacements using a normal Gaussian distribution
        # centered at zero, scaled by your tremor intensity bounds
        delta_pitch = random.uniform(-TREMOR_DEG, TREMOR_DEG)
        delta_yaw = random.uniform(-TREMOR_DEG, TREMOR_DEG)

        # Accumulate the random walk offsets natively
        rolling_rot_x += math.radians(delta_pitch)
        rolling_rot_y += math.radians(delta_yaw)

        # Assign updated fields to primitive tracks and insert keyframe channels
        camera_obj.rotation_euler = (rolling_rot_x, rolling_rot_y, rolling_rot_z)
        camera_obj.keyframe_insert(data_path="rotation_euler", frame=target_frame_idx)

    scene.frame_set(start_frame)
    bpy.context.view_layer.update()


def export_ground_truth_labels(mesh_gen, labels, intrinsics, camera_extrinsics, filepath):
    """
    Computes exact sub-pixel screen space projections for each underlying 3D 
    center coordinates point vector, matching indexing arrays definitions.
    """
    h, w = labels.shape
    camera = ProjectiveCamera((intrinsics["width_px"], intrinsics["height_px"]),
                              intrinsics["f_px"], intrinsics["f_px"],
                              intrinsics["width_px"] / 2, intrinsics["height_px"] / 2,
                              intrinsics["k1"])

    tx = camera_extrinsics.get("tx", 0.0)
    ty = camera_extrinsics.get("ty", 0.0)
    tz = camera_extrinsics.get("tz", 1.0)
    roll = camera_extrinsics.get("roll", 0.0)
    pitch = camera_extrinsics.get("pitch", 0.0)
    yaw = camera_extrinsics.get("yaw", 0.0)

    gt_points = []
    rotation, t = compute_camera_projection_matrix(roll, pitch, yaw, METER_TO_MM(tx), METER_TO_MM(ty), METER_TO_MM(tz))
    for r in range(h):
        for c in range(w):
            shape_type = labels[r, c]
            if shape_type < 0:
                continue

            x_mm, y_mm = mesh_gen.get_shape_center(r, c)
            transformed = camera.project_point(np.asarray([x_mm, y_mm, 0]), rotation, t)
            if transformed is not None:
                if camera.is_visible(transformed):
                    gt_points.append(f"{r},{c},{shape_type},{transformed[0]:.3f},{transformed[1]:.3f}")

    with open(filepath, "w") as f:
        f.write("#Row, Col, Type, x, y\n")
        f.write("\n".join(gt_points))
        f.write("\n")


def cleanup_pattern_instance(pattern_obj):
    """
    Purges evaluated case structures assets from system memory to prevent layout overlay artifacts.
    """
    mesh_data = pattern_obj.data
    bpy.data.objects.remove(pattern_obj, do_unlink=True)
    bpy.data.meshes.remove(mesh_data)


def render(scene, base_output_path, case_name, start_frame: int):
    """
    Forces Python to halt until the Compositor thread completely writes files to disk.
    """
    # 1. Force the timeline context onto the required target frame
    scene.frame_set(start_frame)

    # 2. Force evaluate the dependency graph to bake new geometry/camera transforms
    bpy.context.view_layer.update()
    bpy.context.evaluated_depsgraph_get().update()

    if scene.render.use_compositing:
        scene.use_nodes = True
        file_out_node = scene.node_tree.nodes.get(FILE_OUTPUT_NAME)

        if file_out_node:
            abs_output_path = os.path.abspath(base_output_path)
            file_out_node.base_path = abs_output_path
            file_out_node.file_slots[0].path = f"{case_name}_"
            file_out_node.update()

        print(f" -> [CLI] Active Render: Starting Compositor for {case_name} at Frame {start_frame}")
        bpy.ops.render.render(write_still=False)

    else:
        # Native rendering pipeline automatically blocks the main thread when write_still=True
        native_output_target = os.path.abspath(os.path.join(base_output_path, f"{case_name}_{start_frame:04d}.png"))
        scene.render.filepath = native_output_target

        print(f" -> [CLI] Active Render: Starting Native Engine for {case_name} at Frame {start_frame}")
        bpy.ops.render.render(write_still=True)

    print(f" -> [CLI] Success: Pipeline finished processing {case_name}.\n")


def apply_distortion(image_name, camera_inst, output_path):
    """
    Loads a Blender image buffer directly into a NumPy array, applies
    a custom distortion method, and pushes the modified buffer back
    into Blender for saving.
    """
    blender_image = bpy.data.images.load(output_path, check_existing=True)
    if not blender_image:
        print("Error: Blender image block '{}' not found.".format(image_name))
        return

    # 2. Extract resolution properties from the container
    width, height = blender_image.size

    # Blender stores image pixels as a flat float32 array of RGBA values [0.0 to 1.0]
    # Total array length is always Width * Height * 4
    total_floats = width * height * 4
    pixels_flat = np.empty(total_floats, dtype=np.float32)

    # Direct high-speed C-level block memory copy into the NumPy array
    blender_image.pixels.foreach_get(pixels_flat)

    # Reshape the flat vector to a standard image shape matrix (H, W, 4)
    img_rgba = pixels_flat.reshape((height, width, 4))

    # Flip vertically to align with your standard projection math coordinates
    img_rgba = np.flipud(img_rgba)
    distorted_rgba = distort_image_via_undistort_grid(img_rgba, camera_inst)
    # Flip back vertically to restore Blender's native orientation
    distorted_rgba = np.flipud(distorted_rgba)

    # Flatten the matrix back to a 1D vector before writing to Blender
    pixels_flat_out = distorted_rgba.ravel()
    blender_image = bpy.data.images.new("NewBuffer", camera_inst.img_shape[0],camera_inst.img_shape[1])
    # Direct high-speed C-level block memory write back into Blender's core
    blender_image.pixels.foreach_set(pixels_flat_out)

    # Update the internal image state and force the UI to redraw if necessary
    blender_image.update()
    blender_image.filepath_raw = output_path
    blender_image.save()
    bpy.data.images.remove(blender_image)
    print("-> Successfully processed and saved frame at: {}".format(output_path))


# ==============================================================================
# MAIN TEST BED RUNNER EXECUTION FLOW
# ==============================================================================
if __name__ == "__main__":

    args = parse_arguments()
    
    # 2. Global metric dimensions setup 
    LATTICE_ROW_COUNT = args.rows
    LATTICE_COL_COUNT = args.cols
    
    # 3. Concatenate and build target storage directories paths
    BASE_PATH = "./blender_output"
    ENGINE_SPECIFIC_DIR = os.path.abspath(f"{BASE_PATH}_{args.engine}")
    if not os.path.exists(ENGINE_SPECIFIC_DIR):
        os.makedirs(ENGINE_SPECIFIC_DIR)

    base_blueprint = pattern_blueprint_factory(
         engine_name=args.engine,
         rows=LATTICE_ROW_COUNT,
         cols=LATTICE_COL_COUNT
    )
    intrinsics = {"f_px": 1150.0, "k1": K1_DISTORTION, "width_px": IMG_SHAPE[1], "height_px": IMG_SHAPE[0]}
    # --- TEST CASES DICTIONARY SPECIFICATION ---
    cases = {
        "clean_baseline": {
           "description": "Pristine Baseline Frame (Standard Centered Orientation)",
           "blueprint": np.copy(base_blueprint),
           "camera": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY, "tz": Z_DISTANCE},
           "intrinsics": intrinsics
        },
        "oblique_tilt_high": {
           "description": "Extreme Oblique Viewing Angle Stress Test",
           "blueprint": np.copy(base_blueprint),
           "camera": {"roll": 0.0, "pitch": 45.0, "yaw": 0.0, "tx": DEFAULT_TX, "ty": 1, "tz": Z_DISTANCE},
           "intrinsics": intrinsics
        }
    }
    # Dynamically populate each of the 6 canonical 60-degree roll positions
    for step_idx in range(1, 6):
        target_roll = step_idx * 60.
        cases[f"roll_{int(target_roll)}"] = {
            "description": f"Strict {int(target_roll)}-Degree Roll Skew Around Optical Axis",
            "blueprint": np.copy(base_blueprint),
            "camera": {"roll": target_roll, "pitch": 0.0, "yaw": 0.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY,
                       "tz": Z_DISTANCE},
            "intrinsics": intrinsics
        }

    scene_inst, cam_inst = initialize_blender_scene()

    rolling_frame = 1
    TREMOR_FRAMES_NUM = 2
    # Iterate across individual structured test dictionary cases sequential matrices loops
    for case_name, data in cases.items():
        print(f"\nProcessing Test Case: [{case_name}] - {data['description']}")

        mesh_gen = mesh_generator_factory(
            engine_name=args.engine,
            grid_matrix=data["blueprint"],
            step_mm=PATTERN_STEP_MM,
            r_circ=PRIMITIVE_RADIUS_MM
        )

        pattern_obj = build_3d_pattern_mesh(
            case_name=case_name,
            mesh_gen=mesh_gen
        )

        configure_camera(
            scene=scene_inst,
            camera_obj=cam_inst,
            intrinsics=data["intrinsics"],
            camera_extrinsics=data["camera"],
            start_frame=rolling_frame,
            tremor_frame_delta = TREMOR_FRAMES_NUM
        )

        render(
            scene=scene_inst,
            base_output_path=ENGINE_SPECIFIC_DIR,
            case_name=case_name,
            start_frame=rolling_frame
        )
        img_out_path = os.path.join(ENGINE_SPECIFIC_DIR, f"{case_name}_{rolling_frame:04d}.png")
        camera = ProjectiveCamera((intrinsics["width_px"], intrinsics["height_px"]),
                                  intrinsics["f_px"], intrinsics["f_px"],
                                  intrinsics["width_px"] / 2, intrinsics["height_px"] / 2,
                                  intrinsics["k1"])
        apply_distortion(img_out_path, camera, img_out_path)
        # Clear operational context before launching trailing array cases iterations
        cleanup_pattern_instance(pattern_obj)

        # Process ground truth sub-pixel coordinates text output configurations mappings
        txt_out_path = os.path.join(ENGINE_SPECIFIC_DIR, f"{case_name}_{rolling_frame:04d}_gt.txt")
        export_ground_truth_labels(mesh_gen, data["blueprint"], data["intrinsics"], data["camera"], txt_out_path)
        rolling_frame += 1 + TREMOR_FRAMES_NUM

    print(f"\n[SUCCESS] Completed automated modular cases execution loop inside directory: {ENGINE_SPECIFIC_DIR}")
