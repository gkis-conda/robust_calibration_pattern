import numpy as np
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple
from datetime import datetime
import json
import os
import cProfile
import io
import pstats

# Type aliases for clarity (Python 3.8 compatible)
NodeCoord = Tuple[int, int]
BlueprintRecord = Tuple[Any, float, float, Any]
BlueprintDict = Dict[NodeCoord, BlueprintRecord]
from camera import compute_camera_projection_matrix, ProjectiveCamera
import camera_io
from detector_factory import mesh_generator_factory, pattern_blueprint
from opencv_baseline_engine import OpenCVMultiFrameCalibrator
from optimization import MultiFrameCalibrator
from run_validation import make_blueprint_dict

def generate_ideal_views(
    base_blueprint: np.ndarray,
    modified_blueprint: Optional[np.ndarray],
    generator: Any,
    camera: Any,
    poses: List[Tuple[np.ndarray, np.ndarray]],
) -> List[BlueprintDict]:
    """Generate noiseless Ground Truth blueprint dictionaries for multiple camera poses.

    Args:
        base_blueprint: Base pattern layout array.
        modified_blueprint: Optional modified layout array.
        generator: Pattern geometry generator instance.
        camera: Camera model instance.
        poses: List of (R, t) tuples representing fixed camera extrinsics.

    Returns:
        List of clean BlueprintDict objects, one per camera pose.
    """
    ideal_views: List[BlueprintDict] = []
    for R, t in poses:
        view_dict = make_blueprint_dict(
            base_blueprint=base_blueprint,
            modified_blueprint=modified_blueprint,
            generator=generator,
            camera=camera,
            R=R,
            t=t,
        )
        ideal_views.append(view_dict)

    return ideal_views


def inject_noise_to_blueprint_dict(
    ideal_dict: BlueprintDict,
    noise_std_px: float,
    rng: Optional[np.random.Generator] = None,
) -> BlueprintDict:
    """Create a noisy copy of an ideal blueprint dictionary without mutating input.

    Args:
        ideal_dict: Clean Ground Truth dictionary mapping (r, c) -> (type, x, y, mod).
        noise_std_px: Standard deviation of isotropic Gaussian pixel noise.
        rng: Optional NumPy random Generator for reproducible Monte Carlo trials.

    Returns:
        New BlueprintDict with Gaussian noise added to x and y coordinates.
    """
    if noise_std_px <= 0.0 or not ideal_dict:
        return dict(ideal_dict)

    if rng is None:
        rng = np.random.default_rng()

    keys = list(ideal_dict.keys())
    num_pts = len(keys)

    # Generate all noise offsets in a single vectorized call
    noise = rng.normal(loc=0.0, scale=noise_std_px, size=(num_pts, 2))

    noisy_dict: BlueprintDict = {}
    for idx, key in enumerate(keys):
        shape_type, x_px, y_px, mod_type = ideal_dict[key]
        noisy_dict[key] = (
            shape_type,
            float(x_px + noise[idx, 0]),
            float(y_px + noise[idx, 1]),
            mod_type,
        )

    return noisy_dict


@dataclass(frozen=True)
class ExperimentCase:
    """Immutable configuration describing a single multi-view camera calibration experiment."""

    case_id: str
    description: str
    blueprint: np.ndarray
    intrinsics: Dict[str, Any]
    num_views: int
    noise_std_px: float
    drop_prob: float
    geometry_seed: int
    noise_seed: int
    sweep_param_name: Optional[str] = None
    sweep_param_value: Optional[float] = None


def default_dropout_model(noise_std_px: float) -> float:
    """Compute node detection dropout probability as a function of pixel noise.

    Models realistic detector degradation: 0% drop at 0 noise, scaling smoothly
    up to ~15% at 1.0 px noise, capped at 0.50 to prevent total grid collapse.
    """
    if noise_std_px <= 0.0:
        return 0.0
    prob = 0.05 * noise_std_px + 0.10 * (noise_std_px * noise_std_px)
    return float(min(prob, 0.50))


def generate_experiment_cases(
    base_blueprint: np.ndarray,
    base_intrinsics: Dict[str, Any],
    model_name: str = "custom_cam",
    noise_std: float = 0.5,
    num_views: int = 3,
    base_seed: int = 2026,
    sweep_param_name: Optional[str] = None,
    sweep_levels: Sequence[float] = (),
    drop_prob_fn: Callable[[float], float] = default_dropout_model,
) -> Generator[ExperimentCase, None, None]:
    """Yield parameterized ExperimentCase instances lazily for a single camera baseline.

    Supports sweeping environment parameters ('noise_std_px', 'num_views', 'drop_prob')
    as well as intrinsic model parameters ('k1', 'fx').
    """
    if sweep_param_name is None or len(sweep_levels) == 0:
        levels_to_run: Sequence[Optional[float]] = (None,)
    else:
        levels_to_run = sweep_levels

    for level_idx, level in enumerate(levels_to_run):
        cur_intrinsics = base_intrinsics.copy()
        cur_noise_std = float(noise_std)
        cur_num_views = int(num_views)
        cur_geom_seed = int(base_seed)
        drop_prob = drop_prob_fn(cur_noise_std)
        sweep_val: Optional[float] = None

        if level is not None:
            if sweep_param_name == "noise_std_px":
                cur_noise_std = float(level)
                drop_prob = drop_prob_fn(cur_noise_std)
                sweep_val = cur_noise_std
            elif sweep_param_name == "num_views":
                cur_num_views = int(level)
                cur_geom_seed = base_seed + level_idx * 1000
                sweep_val = float(cur_num_views)
            elif sweep_param_name == "drop_prob":
                drop_prob = float(level)
                sweep_val = drop_prob
            elif sweep_param_name == "k1":
                cur_intrinsics["k1"] = float(level)
                sweep_val = float(level)
            elif sweep_param_name == "fx":
                cur_intrinsics["fx"] = float(level)
                cur_intrinsics["fy"] = float(level)
                sweep_val = float(level)
            else:
                raise ValueError(f"Unsupported sweep_param_name: '{sweep_param_name}'")

        current_noise_seed = base_seed + level_idx

        if sweep_param_name is not None and sweep_val is not None:
            case_id = f"{level_idx:03d}_{sweep_param_name}_{sweep_val:.4g}"
            desc = f"Step {level_idx} | {sweep_param_name}={sweep_val:.4g}"
        else:
            case_id = f"{level_idx:03d}_baseline"
            desc = f"Step {level_idx} | Baseline (Views={cur_num_views}, Noise={cur_noise_std:.2f}px)"

        yield ExperimentCase(
            case_id=case_id,
            description=desc,
            blueprint=np.copy(base_blueprint),
            intrinsics=cur_intrinsics,
            num_views=cur_num_views,
            noise_std_px=cur_noise_std,
            drop_prob=drop_prob,
            geometry_seed=cur_geom_seed,
            noise_seed=current_noise_seed,
            sweep_param_name=sweep_param_name,
            sweep_param_value=sweep_val
        )


NodeCoord = Tuple[int, int]
BlueprintRecord = Tuple[Any, float, float, Any]
BlueprintDict = Dict[NodeCoord, BlueprintRecord]
ViewGroundTruth = Tuple[BlueprintDict, np.ndarray, np.ndarray]


def generate_ideal_views(
    base_blueprint: np.ndarray,
    modified_blueprint: Optional[np.ndarray],
    generator: Any,
    camera: Any,
    num_views: int,
    geometry_seed: int,
    base_tx: float = 0.0,
    base_ty: float = 0.0,
    base_tz: float = 1000.0,
) -> List[ViewGroundTruth]:
    """Generate noiseless Ground Truth views with tilt-compensated random camera poses.

    Args:
        base_blueprint: Base pattern layout array of shape (H_nodes, W_nodes).
        modified_blueprint: Optional modified layout array.
        generator: Pattern geometry generator implementing get_shape_center(r, c).
        camera: Camera model implementing project_point and is_visible.
        num_views: Number of random tilted views to generate.
        geometry_seed: Deterministic seed controlling pose generation.
        base_tx: Base X translation offset.
        base_ty: Base Y translation offset.
        base_tz: Base Z distance from camera to pattern plane.

    Returns:
        List of tuples (gt_dict, R, t) for each generated view.
    """
    rng = np.random.default_rng(geometry_seed)
    ideal_views: List[ViewGroundTruth] = []

    for _ in range(num_views):
        roll = float(rng.uniform(0.0, 360.0))
        # Randomize sign of pitch and yaw to cover all spatial quadrants
        pitch = float(rng.uniform(5.0, 45.0) * rng.choice([-1.0, 1.0]))
        yaw = float(rng.uniform(3.0, 18.0) * rng.choice([-1.0, 1.0]))

        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        # Compensate translation so tilted pattern stays centered in frame
        comp_tx = (
            base_tx
            - (base_tz * math.tan(yaw_rad))
            + float(rng.uniform(-0.01, 0.01))
        )
        comp_ty = (
            base_ty
            - (base_tz * math.tan(pitch_rad))
            + float(rng.uniform(-0.01, 0.01))
        )
        comp_tz = base_tz

        R, t = compute_camera_projection_matrix(
            roll_deg=roll,
            pitch_deg=pitch,
            yaw_deg=yaw,
            tx=comp_tx,
            ty=comp_ty,
            tz=comp_tz,
        )

        gt_dict = make_blueprint_dict(
            base_blueprint=base_blueprint,
            modified_blueprint=modified_blueprint,
            generator=generator,
            camera=camera,
            R=R,
            t=t,
        )

        ideal_views.append((gt_dict, R, t))

    return ideal_views


def blueprint_dict_to_noisy_frame(
    gt_dict: Dict[Tuple[int, int], Tuple[Any, float, float, Any]],
    grid_shape: Tuple[int, int],
    noise_std_px: float,
    drop_prob: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Gaussian noise and random dropout, returning solver-ready arrays.

    Args:
        gt_dict: Clean Ground Truth mapping (r, c) -> (type, x_px, y_px, mod).
        grid_shape: Tuple (H_nodes, W_nodes) defining topological matrix size.
        noise_std_px: Standard deviation of isotropic Gaussian pixel noise.
        drop_prob: Probability of dropping a visible node [0.0, 1.0).
        rng: Local NumPy random Generator instance.

    Returns:
        Tuple of (topological_matrix, detected_points):
            topological_matrix: int32 array of shape (H_nodes, W_nodes), -1 for missing.
            detected_points: float64 array of shape (K, 2) with noisy pixel coordinates.
    """
    topo_matrix = np.full(grid_shape, -1, dtype=np.int32)
    if not gt_dict:
        return topo_matrix, np.empty((0, 2), dtype=np.float64)

    keys = list(gt_dict.keys())
    num_total = len(keys)

    if drop_prob > 0.0:
        keep_mask = rng.random(num_total) >= drop_prob
    else:
        keep_mask = np.ones(num_total, dtype=bool)

    kept_keys = [k for k, keep in zip(keys, keep_mask) if keep]
    num_kept = len(kept_keys)

    if num_kept == 0:
        return topo_matrix, np.empty((0, 2), dtype=np.float64)

    detected_points = np.empty((num_kept, 2), dtype=np.float64)
    for idx, (r, c) in enumerate(kept_keys):
        _, x_px, y_px, _ = gt_dict[(r, c)]
        topo_matrix[r, c] = idx
        detected_points[idx, 0] = x_px
        detected_points[idx, 1] = y_px

    if noise_std_px > 0.0:
        detected_points += rng.normal(
            loc=0.0, scale=noise_std_px, size=(num_kept, 2)
        )

    return topo_matrix, detected_points


def summarize_trials(
    trials_metrics: List[Dict[str, float]],
    total_attempts: int,
) -> Dict[str, Any]:
    """Compute convergence rate, mean, and standard deviation across Monte Carlo trials.

    Args:
        trials_metrics: List of parameter dictionaries from successful trials.
        total_attempts: Total number of Monte Carlo trials executed.

    Returns:
        Aggregated statistics dictionary with mean and std for every parameter.
    """
    converged_count = len(trials_metrics)
    summary: Dict[str, Any] = {
        "convergence_rate": float(converged_count) / max(total_attempts, 1),
        "converged": converged_count,
        "total": total_attempts,
    }
    if converged_count == 0:
        return summary

    param_names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
    for name in param_names:
        vals = np.array([float(m.get(name, 0.0)) for m in trials_metrics], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.mean(vals))
        summary[f"{name}_std"] = float(np.std(vals))

    return summary

# Factory signatures:
# solver_factory(initial_cam, geometry_generator) -> calibrator instance
# pattern_factory(blueprint) -> pattern geometry generator instance
SolverFactory = Callable[[Any, Any], Any]
PatternGeneratorFactory = Callable[[np.ndarray], Any]

from typing import Any, Dict


def format_calibration_summary(res: Dict[str, Any]) -> str:
    """Format calibration result dictionary with bounded decimal precision for console logs."""
    if res.get("status") != "success":
        return f"{{'status': '{res.get('status')}', 'reason': '{res.get('reason', 'unknown')}'}}"

    fx = float(res.get("fx", 0.0))
    fy = float(res.get("fy", 0.0))
    cx = float(res.get("cx", 0.0))
    cy = float(res.get("cy", 0.0))
    rms = float(res.get("rms", 0.0))

    # Extract distortion coefficients whether flat or nested
    dist = res.get("distortion_coefs", res)
    k1 = float(dist.get("k1", 0.0))
    k2 = float(dist.get("k2", 0.0))
    p1 = float(dist.get("p1", 0.0))
    p2 = float(dist.get("p2", 0.0))

    print (
        f"[OK] fx={fx:.2f}, fy={fy:.2f} | cx={cx:.2f}, cy={cy:.2f} | "
        f"rms={rms:.4f} | k1={k1:.5f}, k2={k2:.5f}, p1={p1:.5f}, p2={p2:.5f}"
    )


def evaluate_single_case_monte_carlo(
    case: ExperimentCase,
    solver_factory: SolverFactory,
    pattern_factory: PatternGeneratorFactory,
    num_trials: int = 30,
) -> Dict[str, Any]:
    """Run Monte Carlo trials for a single solver and pattern generator factory.

    Args:
        case: Immutable ExperimentCase configuration for the current sweep step.
        solver_factory: Callable(initial_cam, generator) returning a calibrator instance.
        pattern_factory: Callable(blueprint) returning a geometry generator instance.
        num_trials: Number of independent Monte Carlo realizations (LLN sample size).

    Returns:
        Dictionary containing case metadata, ground truth, and aggregated solver metrics.
    """
    gt_cam = camera_io.deserialize_camera_from_dict(case.intrinsics)
    generator = pattern_factory(case.blueprint)

    w, h = gt_cam.width, gt_cam.height
    grid_shape = case.blueprint.shape

    successful_trials: List[Dict[str, float]] = []
    resolved_solver_name: str = "UnknownSolver"
    profiler = cProfile.Profile()

# ... your existing trial execution (add_frame + finalize) ...

    for trial_idx in range(num_trials):
        trial_geom_seed = case.geometry_seed + trial_idx
        trial_noise_seed = case.noise_seed + trial_idx * 1000
        rng = np.random.default_rng(trial_noise_seed)

        ideal_views = generate_ideal_views(
            base_blueprint=case.blueprint,
            modified_blueprint=case.blueprint,
            generator=generator,
            camera=gt_cam,
            num_views=case.num_views,
            geometry_seed=trial_geom_seed
        )

        initial_cam = ProjectiveCamera(
            (w, h), fx_px=w / 2.0, fy_px=w / 2.0, cx=w / 2.0, cy=h / 2.0, k1=-0.1
        )
        calibrator = solver_factory(initial_cam, generator)

        if trial_idx == 0:
            resolved_solver_name = str(
                getattr(calibrator, "name", calibrator.__class__.__name__)
            )

        for gt_dict, R_true, t_true in ideal_views:
            topo_mat, det_pts = blueprint_dict_to_noisy_frame(
                gt_dict=gt_dict,
                grid_shape=grid_shape,
                noise_std_px=case.noise_std_px,
                drop_prob=case.drop_prob,
                rng=rng,
            )
            weights = np.ones(len(det_pts), dtype=np.float64)

            calibrator.add_frame(
                topological_matrix=topo_mat,
                detected_points=det_pts,
                point_weights=weights,
                initial_pose=(R_true, t_true),
            )

        profiler.enable()
        cal_res = calibrator.calibrate()
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
        stats.print_stats(15)
        print(stream.getvalue())

        format_calibration_summary(cal_res)
        if cal_res.get("status") == "success":
            successful_trials.append(cal_res)

    return {
        "case_id": case.case_id,
        "solver_name": resolved_solver_name,
        "sweep_param": case.sweep_param_name,
        "sweep_value": case.sweep_param_value,
        "noise_std_px": case.noise_std_px,
        "drop_prob": case.drop_prob,
        "num_views": case.num_views,
        "ground_truth": case.intrinsics,
        "metrics": summarize_trials(successful_trials, num_trials),
    }


def main() -> None:
    import argparse
    """Entry point for running a single-solver Monte Carlo calibration benchmark session."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo Calibration Benchmark Runner"
    )
    parser.add_argument(
        "--solver", type=str, default="algebraic", choices=["algebraic", "opencv"],
        help="Solver backend to evaluate in this session",
    )
    parser.add_argument(
        "--pattern", type=str, default="hexagonal",
        help="Pattern engine type passed to detector_factory (e.g. hgp, circles, chessboard)",
    )
    parser.add_argument(
        "--rows", type=int, default=31,
        help="Number of rows in the target pattern grid",
    )
    parser.add_argument(
        "--cols", type=int, default=31,
        help="Number of columns in the target pattern grid",
    )
    parser.add_argument(
        "--step-mm", type=float, default=25.0,
        help="Physical grid pitch in millimeters",
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of Monte Carlo trials per swept parameter step",
    )
    parser.add_argument(
        "--out-dir", type=str, default="benchmark_sessions",
        help="Directory to store output JSON session statistics",
    )
    parser.add_argument(
        "--seed",  type=int, default=2026,
        help="Base random seed for deterministic reproducibility",
    )
    args = parser.parse_args()

    # 1. Build target blueprint and wrap mesh_generator_factory
    base_blueprint = pattern_blueprint(engine_name=args.pattern, rows=args.rows, cols=args.cols,
    )

    def pattern_factory(bp: np.ndarray) -> Any:
        r_circ_mm = args.step_mm / 5.0
        return mesh_generator_factory(
            engine_name=args.pattern, grid_matrix=bp, step_mm=args.step_mm, r_circ=r_circ_mm,
        )

    # 2. Configure single solver factory for this session
    if args.solver == "opencv":

        def solver_factory(initial_cam: Any, generator: Any) -> Any:
            return OpenCVMultiFrameCalibrator(
                camera_object=initial_cam,
                geometry_generator=generator,
                fix_k3=True,
                fix_principal_point=False,
                use_intrinsic_guess=True,
            )
    else:

        def solver_factory(initial_cam: Any, generator: Any) -> Any:
            return MultiFrameCalibrator( camera_object=initial_cam, N=12, MIN_LEN=15 )

    # 3. Define single ground truth camera baseline
    base_intrinsics: Dict[str, Any] = {
        "width": 1920,
        "height": 1080,
        "cx": 974.5,
        "cy": 531.8,
        "fx": 1452.0,
        "fy": 1450.0,
        "distortion_model": "full",
        "k1": -0.22,
        "k2": 0.005,
        "p1": -0.0020,
        "p2": 0.0030
    }

    # 4. Initialize lazy experiment generator for a single swept parameter
    cases_gen = generate_experiment_cases(
        base_blueprint=base_blueprint,
        base_intrinsics=base_intrinsics,
        noise_std=0.5,
        num_views=3,
        base_seed=args.seed,
        sweep_param_name="noise_std_px",
        sweep_levels=(0.1, 0.5, 1.0),
    )

    # 5. Execute Monte Carlo sweep and collect session records
    session_records: List[Dict[str, Any]] = []
    solver_label = args.solver

    for case in cases_gen:
        print(f"[RUNNING] {case.case_id} ({case.description})")
        record = evaluate_single_case_monte_carlo(
            case=case,
            solver_factory=solver_factory,
            pattern_factory=pattern_factory,
            num_trials=args.trials,
        )
        solver_label = record["solver_name"]
        session_records.append(record)

        m = record["metrics"]
        if m["converged"] > 0:
            print(
                f"  -> {solver_label} | Conv: {m['converged']}/{m['total']} | "
                f"fx: {m['fx_mean']:.2f} (std={m['fx_std']:.2f}) | "
                f"k1: {m['k1_mean']:.5f} (std={m['k1_std']:.5f})"
            )
        else:
            print(f"  -> {solver_label} | FAILED TO CONVERGE (0/{m['total']})")

    # 6. Export session results to JSON for external comparison/visualization
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"session_{solver_label}_{args.pattern}_{timestamp}.json"
    out_path = os.path.join(args.out_dir, out_filename)

    session_payload = {
        "solver_name": solver_label,
        "pattern_engine": args.pattern,
        "grid_shape": [args.rows, args.cols],
        "step_mm": args.step_mm,
        "timestamp": timestamp,
        "num_trials": args.trials,
        "base_seed": args.seed,
        "records": session_records,
    }

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(session_payload, fp, indent=2)

    print(f"\n[DONE] Session statistics saved to: {out_path}")


if __name__ == "__main__":
    main()
