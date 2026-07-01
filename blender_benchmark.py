import bpy
import mathutils
import math
import os
import numpy as np
import sys
import random
from datetime import datetime

# 1. Capture the exact absolute folder path where this script resides
# (Safe for execution via relative paths or system symlinks)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Append to Python's module lookup stack if not already registered
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from blender_factory import *
# ==============================================================================
# [INSERT YOUR PhysicalMeshGenerator CLASS CODE HERE]
# ==============================================================================

# --- MATHEMATICAL TRANSFORMATION MACROS (NAMED CONSTANTS) ---
MM_TO_METER = lambda x: float(x) * 0.001
METER_TO_MM = lambda x: float(x) * 1000.

# --- FIXED HARDWARE AND ENVIRONMENTAL CONSTANTS ---
BLENDER_SENSOR_WIDTH_MM = 36.0                 # Standard full-frame sensor metric
RESOLUTION_PERCENTAGE_FULL = 100               # Scale factor for target frame rendering
Z_PLANE_FIGHTING_OFFSET_METERS = 0.001         # Slight offset to prevent mesh clipping
CYCLES_RAYTRACING_SAMPLES = 30                 # Computation constraints limit
BLUR_EXPOSURE_SHUTTER_MAX = 1.0                # Keep shutter active across entire frame

# --- JITTER MOTION CHARACTERISTICS ---
TREMOR_DEG = 0.1                         # Sub-pixel rotational hand shake step

# --- CAMERA PARAMETERS DEFINITIONS AND CONSTRAINTS ---
DEFAULT_TX, DEFAULT_TY, Z_DISTANCE = 0.0, 0.0, -1.5
K1_DISTORTION = -0.1
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


def purge_blender_workspace():
    """
    Safely resets the active workspace environment layers, cleaning out
    residual asset garbage blocks from memory.
    """
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


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


def build_3d_pattern_mesh(case_name:str, engine:str, blueprint, pattern_step_mm=20.0, primitive_radius_mm=6.0):
    """
    Consumes the custom PhysicalMeshGenerator layout object sequence, builds
    explicit spatial vertex faces structures, and binds PBR properties.
    """    
    mesh_gen = mesh_generator_factory(
            engine_name=engine,
            grid_matrix=blueprint,
            step_mm=PATTERN_STEP_MM,
            r_circ=PRIMITIVE_RADIUS_MM
        )

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
    node_centers_3d = {}
    vert_index_offset = 0

    for i, j, shape_type, contour in mesh_gen:
        if shape_type < 0 or not contour:
            continue
            
        cx, cy = mesh_gen.get_shape_center(i, j)
        # CONVERSION FACTOR: Map physical millimeter targets into Blender world meter metrics
        node_centers_3d[(i, j)] = mathutils.Vector((MM_TO_METER(cx), MM_TO_METER(cy), 0.0))
        
        num_pts = len(contour)
        for pt in contour:
            # CONVERSION FACTOR: Transforming bounding vertices stream tracking array indices
            all_vertices.append((MM_TO_METER(pt[0]), MM_TO_METER(pt[1]), 0.0))
            
        face_indices = list(range(vert_index_offset, vert_index_offset + num_pts))
        all_faces.append(face_indices)
        vert_index_offset += num_pts
        
    blender_mesh.from_pydata(all_vertices, [], all_faces)
    blender_mesh.update()
    
    token_mat = create_matte_pbr_material(f"Mat_Tokens_{case_name}", (0.0, 0.0, 0.0, 1.0), 0.4)
    pattern_obj.data.materials.append(token_mat)
    
    return pattern_obj, node_centers_3d, mesh_gen.grid_matrix


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


def setup_camera_lens_distortion(scene, k1):
    """
    Activates Blender 2.92 internal Compositing pipeline with a File Output node
    to guarantee that distorted images are written to disk during Python execution.
    LENS DISTORTION HARDWARE CONFIGURATION LAYER
    """
    dist_node = scene.node_tree.nodes.get("TCM_Distortion_Node")

    if abs(k1) > 1e-8:
        # Distortion layer detected: activate composting pipelines and inject value
        if dist_node:
            scene.render.use_compositing = True
            dist_node.inputs['Distort'].default_value = -float(k1)
    else:
        # No distortion present in metrics: clean reset to zero and deactivate compositor flag
        scene.render.use_compositing = False
        if dist_node:
            dist_node.inputs['Distort'].default_value = 0.0

# Global flag to track the physical end of the render + compositor disk write
_RENDER_BUSY = False

def on_render_complete_callback(scene):
    global _RENDER_BUSY
    # This trigger is executed by Blender when the file is 100% written to disk
    _RENDER_BUSY = False


def initialize_compositor_pipeline(scene):
    """
    Allocates the required file output and distortion nodes exactly once
    at the script barrier checkpoint to prevent runtime memory leaks.
    """
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()  # Clear default template nodes safely once

    # Instantiate required execution nodes
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')

    distortion_node = tree.nodes.new(type='CompositorNodeLensdist')
    distortion_node.name = DISTORTION_NODE_NAME
    if hasattr(distortion_node,"use_project"):
        distortion_node.use_project = False
    elif hasattr(distortion_node,"use_fit"):
        distortion_node.use_fit = False
    if hasattr(distortion_node,"use_jitter"):
        distortion_node.use_jitter = True

    file_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
    file_output_node.name = FILE_OUTPUT_NAME
    file_output_node.label = FILE_OUTPUT_NAME
    # Explicit Image Format Specifications
    file_output_node.format.file_format = 'PNG'
    file_output_node.format.color_mode = 'RGB'
    file_output_node.format.color_depth = '8'
    file_output_node.format.compression = 15

    # Link the node network pipeline array permanently
    links = tree.links
    links.new(render_layers.outputs['Image'], distortion_node.inputs['Image'])
    links.new(distortion_node.outputs['Image'], file_output_node.inputs['Image'])

    # Register the handler once globally during script initialization
    if on_render_complete_callback not in bpy.app.handlers.render_complete:
        bpy.app.handlers.render_complete.append(on_render_complete_callback)

    print(" -> Compositor pipeline infrastructure initialized cleanly.")


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
    k1 = intrinsics.get("k1", 0.0)

    scene.render.resolution_x = w_px
    scene.render.resolution_y = h_px
    scene.render.resolution_percentage = RESOLUTION_PERCENTAGE_FULL

    camera_obj.data.type = 'PERSP'
    camera_obj.data.lens_unit = 'MILLIMETERS'
    camera_obj.data.lens = (f_px * BLENDER_SENSOR_WIDTH_MM) / w_px

    # Dynamic Distortion Processing State Gate Check
    dist_node = scene.node_tree.nodes.get(DISTORTION_NODE_NAME)
    if abs(k1) > 1e-8:
        scene.render.use_compositing = True
        if dist_node:
            dist_node.inputs['Distort'].default_value = -float(k1)
    else:
        scene.render.use_compositing = False
        if dist_node:
            dist_node.inputs['Distort'].default_value = 0.0

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
    print(datetime.now().time(), camera_obj.rotation_euler )

    camera_obj.keyframe_insert(data_path="location", frame=start_frame)
    camera_obj.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    # -----------------------------------------------------------------
    # MULTI-FRAME PSEUDO-RANDOM WALK SEGMENT
    # -----------------------------------------------------------------
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
        delta_pitch = random.gauss(0.0, TREMOR_DEG)
        delta_yaw = random.gauss(0.0, TREMOR_DEG)

        # Accumulate the random walk offsets natively
        rolling_rot_x += math.radians(delta_pitch)
        rolling_rot_y += math.radians(delta_yaw)

        # Assign updated fields to primitive tracks and insert keyframe channels
        camera_obj.rotation_euler = (rolling_rot_x, rolling_rot_y, rolling_rot_z)
        camera_obj.keyframe_insert(data_path="rotation_euler", frame=target_frame_idx)

    # Return timeline context back to the case's base frame so the shutter captures the velocity path
    scene.frame_set(start_frame)
    bpy.context.view_layer.update()


def apply_radial_distortion(x_px, y_px, cx_px, cy_px, f_px, k1):
    """
    Applies the mathematical Brown-Conrady radial distortion model (K1 only)
    to a single 2D pixel coordinate point.
    """
    # 1. Move to normalized camera coordinates space (x, y)
    x_norm = (x_px - cx_px) / f_px
    y_norm = (y_px - cy_px) / f_px

    # Calculate square of radius from principal axis point center
    r2 = (x_norm ** 2) + (y_norm ** 2)

    # 2. Compute distortion scaling factor
    distortion_multiplier = 1.0 + k1 * r2

    # 3. Map back to absolute canvas screen pixels
    x_distorted = (x_norm * distortion_multiplier * f_px) + cx_px
    y_distorted = (y_norm * distortion_multiplier * f_px) + cy_px

    return x_distorted, y_distorted


class Distortion:
    def __init__(self, cx_px, cy_px, f_px, k1):
        self.cx = cx_px
        self.cy = cy_px
        self.f = f_px
        self.k1 = k1

    def __call__(self, x_px, y_px):
        return  apply_radial_distortion(x_px, y_px, self.cx, self.cy, self.f, self.k1)


def export_ground_truth_labels(scene, camera_obj, pattern_obj, node_centers_3d, grid_matrix, distort:Distortion, filepath):
    """
    Computes exact sub-pixel screen space projections for each underlying 3D 
    center coordinates point vector, matching indexing arrays definitions.
    """
    from bpy_extras.object_utils import world_to_camera_view
    
    w_px = scene.render.resolution_x
    h_px = scene.render.resolution_y
    
    gt_lines = ["# Row, Col, Shape_Type, X_pixel, Y_pixel"]
    
    for (r, c), local_center in node_centers_3d.items():
        # Compute exact world matrix position based on transformation stack
        world_coord = pattern_obj.matrix_world @ local_center
        co_2d = world_to_camera_view(scene, camera_obj, world_coord)
        
        # Absolute pixels transformations (Inverting Y bounds to follow CV specs)
        pixel_x = co_2d.x * w_px
        pixel_y = (1.0 - co_2d.y) * h_px  
        
        shape_type = grid_matrix[r, c]
        if distort:
            pixel_x, pixel_y = distort(pixel_x, pixel_y)
        gt_lines.append(f"{r},{c},{shape_type},{pixel_x:.4f},{pixel_y:.4f}")
        
    with open(filepath, "w") as f:
        f.write("\n".join(gt_lines))


def cleanup_pattern_instance(pattern_obj):
    """
    Purges evaluated case structures assets from system memory to prevent layout overlay artifacts.
    """
    mesh_data = pattern_obj.data
    bpy.data.objects.remove(pattern_obj, do_unlink=True)
    bpy.data.meshes.remove(mesh_data)


# =====================================================================
# SYSTEM CONTEXT: render() Loop Execution Update Pass
# =====================================================================

def render(scene, base_output_path, case_name, start_frame: int):
    """
    Standalone rendering pipeline executor. Forces strict blocking file-writes
    by flinging the dependency graph cache parameters clear on every step.
    """
    # 1. Force the timeline pointer directly onto the case's allocated slot
    scene.frame_set(start_frame)

    # 2. THE CRITICAL MATRIX SHIELD: Force Blender to evaluate the graph IMMEDIATELY.
    # This locks down the specific camera rotation in C++ engine memory before rendering.
    bpy.context.view_layer.update()
    bpy.context.evaluated_depsgraph_get()

    if scene.render.use_compositing:
        file_out_node = scene.node_tree.nodes.get(FILE_OUTPUT_NAME)
        if file_out_node:
            file_out_node.base_path = base_output_path
            file_out_node.file_slots[0].path = f"{case_name}_"  # Generates case_name_####.png

        print(f" -> [CLI] Active Render: Starting Compositor for {case_name} at Frame {start_frame}")

        # Trigger blocking render pass
        bpy.ops.render.render(write_still=False)

    else:
        # Standard unwarped direct engine layout path
        native_output_target = os.path.join(base_output_path, f"{case_name}_{start_frame:04d}.png")
        scene.render.filepath = native_output_target

        print(f" -> [CLI] Active Render: Processing Core Route for {case_name} at Frame {start_frame}")
        bpy.ops.render.render(write_still=True)


def render_sync(scene, base_output_path, case_name, start_frame: int):
    """
    ASCII-compatible synchronous rendering executor for Blender CLI execution (-b).
    Forces Python to halt until the Compositor thread completely writes files to disk.
    """
    global _RENDER_BUSY

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

        # Lock the execution thread state before triggering the render operator
        _RENDER_BUSY = True
        bpy.ops.render.render(write_still=False)

        # BLOCKING LOOP: Hold the Python CLI thread until the compositor thread finishes writing
        while _RENDER_BUSY:
            time.sleep(0.1)  # Check every 20 milliseconds to minimize latency

    else:
        # Native rendering pipeline automatically blocks the main thread when write_still=True
        native_output_target = os.path.abspath(os.path.join(base_output_path, f"{case_name}_0001.png"))
        scene.render.filepath = native_output_target

        print(f" -> [CLI] Active Render: Starting Native Engine for {case_name} at Frame {start_frame}")
        bpy.ops.render.render(write_still=True)

    print(f" -> [CLI] Success: Pipeline finished processing {case_name}.\n")


# ==============================================================================
# MAIN TEST BED RUNNER EXECUTION FLOW
# ==============================================================================
if __name__ == "__main__":

    args = parse_blender_arguments()
    
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
        target_roll = float(step_idx * 60)
        cases[f"roll_{int(target_roll)}"] = {
            "description": f"Strict {int(target_roll)}-Degree Roll Skew Around Optical Axis",
            "blueprint": np.copy(base_blueprint),
            "camera": {"roll": target_roll, "pitch": 0.0, "yaw": 0.0, "tx": DEFAULT_TX, "ty": DEFAULT_TY,
                       "tz": Z_DISTANCE},
            "intrinsics": intrinsics
        }
    # Step 1: Uniform global environment initialization phase
    scene_inst, cam_inst = initialize_blender_scene()
    initialize_compositor_pipeline(scene_inst)
    rolling_frame = 1
    TREMOR_FRAMES_NUM = 2
    # Step 2: Iterate across individual structured test dictionary cases sequential matrices loops
    for case_name, data in cases.items():
        print(f"\nProcessing Test Case: [{case_name}] - {data['description']}")
        
        # Step 3: Parse blueprint topology and construct discrete 3D node array structures via lambda macro
        pattern_obj, centers_3d, matrix_state = build_3d_pattern_mesh(
            case_name=case_name,
            engine=args.engine,
            blueprint=data["blueprint"],
            pattern_step_mm=PATTERN_STEP_MM,
            primitive_radius_mm=PRIMITIVE_RADIUS_MM
        )

        # 2. Stage 1: Configure camera transformations on an isolated timeline channel slot
        configure_camera(
            scene=scene_inst,
            camera_obj=cam_inst,
            intrinsics=data["intrinsics"],
            camera_extrinsics=data["camera"],
            start_frame=rolling_frame,
            tremor_frame_delta = TREMOR_FRAMES_NUM
        )

        # 3. Stage 2: Execute rendering pass locked exactly on that frame context position
        render_sync(
            scene=scene_inst,
            base_output_path=ENGINE_SPECIFIC_DIR,
            case_name=case_name,
            start_frame=rolling_frame
        )
        rolling_frame += 1 + TREMOR_FRAMES_NUM
        # Step 6: Process ground truth sub-pixel coordinates text output configurations mappings
        distortion = None
        intrinsics = data["intrinsics"]
        if "k1" in intrinsics:
            distortion = Distortion(intrinsics["width_px"]/2, intrinsics["height_px"]/2, intrinsics["f_px"], intrinsics["k1"])
        txt_out_path = os.path.join(ENGINE_SPECIFIC_DIR, f"{case_name}_gt.txt")
        export_ground_truth_labels(scene_inst, cam_inst, pattern_obj, centers_3d, matrix_state, distortion, txt_out_path)
        
        # Step 7: Clear operational context before launching trailing array cases iterations
        cleanup_pattern_instance(pattern_obj)

    print(f"\n[SUCCESS] Completed automated modular cases execution loop inside directory: {ENGINE_SPECIFIC_DIR}")
