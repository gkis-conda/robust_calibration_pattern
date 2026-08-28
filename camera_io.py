import json
import numpy as np

from camera import ProjectiveCamera


def _as_tuple_shape(v):
    """Normalize various img_shape representations to (W, H) or None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    if isinstance(v, dict):
        if "img_shape" in v:
            return _as_tuple_shape(v["img_shape"])
        if "width" in v and "height" in v:
            return (int(v["width"]), int(v["height"]))
        if "W_img" in v and "H_img" in v:
            return (int(v["W_img"]), int(v["H_img"]))
    return None


def serialize_camera_to_dict(cam, include_K=False):
    """
    Convert ProjectiveCamera -> plain dict with primitive types.
    include_K: include K matrix as list-of-lists if True.
    """
    d = {
        "img_shape": cam.img_shape,
        "fx": cam.fx_px,
        "fy": cam.fy_px,
        "cx": cam.cx,
        "cy": cam.cy,
        "k1": cam.k1,
        "mode": cam.mode,
        "distortion_model": "" if cam.distortion_model is None else cam.distortion_model.name
    }

    if include_K:
        try:
            K = np.asarray(getattr(cam, "K", None), dtype=float)
            if K is not None and K.shape == (3, 3):
                d["K"] = K.tolist()
        except Exception:
            pass

    try:
        d["max_stable_radius"] = float(getattr(cam, "max_stable_radius", np.nan))
    except Exception:
        d["max_stable_radius"] = None

    return d


def deserialize_camera_from_dict(d):
    """
    Create ProjectiveCamera from a dict.

    Rules:
      - 'fx' MUST be present (otherwise ValueError).
      - if no distortion coefficients (k1, dist, dist_coeffs) => no distortion (k1=0.0).
      - other missing fields: fy defaults to fx, cx/cy default to image center,
        img_shape defaults to (1080, 1920).
    """
    if not isinstance(d, dict):
        raise TypeError("Input must be a dict")

    # img_shape detection
    img_shape = None
    if "img_shape" in d:
        img_shape = _as_tuple_shape(d["img_shape"])
    else:
        img_shape = _as_tuple_shape({"W_img": d.get("W_img"), "H_img": d.get("H_img")})
        if img_shape is None:
            img_shape = _as_tuple_shape({"width": d.get("width"), "height": d.get("height")})

    if img_shape is None:
        raise TypeError("Raser shape missed")

    W_img, H_img = int(img_shape[0]), int(img_shape[1])

    # Read K if present (preferred)
    fx = None
    fy = None
    cx = None
    cy = None
    if "K" in d and d["K"] is not None:
        try:
            K = np.array(d["K"], dtype=float)
            if K.shape == (3, 3):
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])
        except Exception:
            pass

    if "fx" in d:
        fx = float(d["fx"])
    if "fy" in d:
        fy = float(d["fy"])
    if "cx" in d:
        cx = float(d["cx"])
    if "cy" in d:
        cy = float(d["cy"])

    # fx is mandatory
    if fx is None:
        raise ValueError("Missing required intrinsic parameter 'fx' (or valid 'K' matrix).")

    # defaults
    if fy is None:
        fy = fx
    if cx is None:
        cx = float(W_img) / 2.0
    if cy is None:
        cy = float(H_img) / 2.0

    # Distortion handling: prefer explicit k1, else look for lists 'dist' or 'dist_coeffs'.
    k1 = 0.0
    distortion_model = None

    if "k1" in d:
        try:
            k1 = float(d["k1"])
            if abs(k1) <= 1e-12:
                k1 = 0.0
            else:
                distortion_model = d.get("distortion_model", None)
        except Exception:
            k1 = 0.0
            distortion_model = None

    mode = d.get("mode", "perspective")

    return ProjectiveCamera((W_img, H_img),
                           float(fx), float(fy),
                           float(cx), float(cy),
                           float(k1),
                           mode,
                           distortion_model=distortion_model)


def save_dict_as_json(path, d):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def load_dict_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_camera_json(path, cam, include_K=False):
    d = serialize_camera_to_dict(cam, include_K=include_K)
    save_dict_as_json(path, d)


def load_camera_json(path):
    d = load_dict_from_json(path)
    return deserialize_camera_from_dict(d)


def find_camera_config(path, load=False):
    import re
    from pathlib import Path

    """
    Find a *_camera.json config in the same folder as `path` and pick the best match by token overlap.

    Rules:
      - Only files matching "*_camera.json" are considered.
      - If `path` is a file, its stem is tokenized and used to score candidates by token intersection.
        The best candidate with score > 0 is returned. If no candidate has score>0 but there's exactly
        one candidate in the folder, that one is returned. Otherwise (tie or all zeros) -> None.
      - If `path` is a directory: if exactly one *_camera.json exists -> return it, else -> None.
      - If load=True -> return load_camera_json(found_path) (ProjectiveCamera object), else return path string.
    """
    p = Path(path)
    if p.is_dir():
        folder = p
        stem = ""
    else:
        folder = p.parent
        stem = p.stem.lower()

    if not folder.exists():
        return None

    # Only files with explicit suffix "_camera.json"
    candidates = sorted(folder.glob("*_camera.json"))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        return None

    # If directory input: only accept unique candidate
    if p.is_dir():
        if len(candidates) == 1:
            chosen = str(candidates[0])
            return load_camera_json(chosen) if load else chosen
        return None

    # If only one candidate file in folder, return it immediately
    if len(candidates) == 1:
        chosen = str(candidates[0])
        return load_camera_json(chosen) if load else chosen

    # Tokenize helper: split on non-alnum, lower-case
    def tokens(s):
        return [t for t in re.split(r'[^0-9a-zA-Z]+', s.lower()) if t]

    stem_tokens = set(tokens(stem))

    # Score each candidate by intersection size of tokens
    best_score = -1
    best_paths = []
    for c in candidates:
        name = c.stem.lower()  # filename without extension
        cand_tokens = set(tokens(name))
        score = len(stem_tokens & cand_tokens) if stem_tokens else 0
        if score > best_score:
            best_score = score
            best_paths = [str(c)]
        elif score == best_score:
            best_paths.append(str(c))

    # If best_score <= 0: ambiguous unless only one candidate (handled earlier)
    if best_score <= 0:
        return None

    # If unique best, return it; if tie, ambiguous -> None
    if len(best_paths) == 1:
        chosen = best_paths[0]
        return load_camera_json(chosen) if load else chosen

    return None


def save_camera_comparison_md(file_path: str, solved_cam, ground_truth: dict, title: str = "CAMERA CALIBRATION EVALUATION"):
    """
    Compares a solved camera instance against Ground Truth metrics and saves
    a publication-ready Markdown table directly to a specified text/markdown file.
    """
    from pathlib import Path
    from os.path import isdir
    img_shape = solved_cam.img_shape
    gt_img_shape = ground_truth.img_shape

    if np.any(gt_img_shape != img_shape):
        raise RuntimeError("Incompatible camera for comparison")
    path = Path(file_path)
    if isdir(file_path):
        summary_file_name = file_path + "-calibration_summary.md"
    else:
        summary_file_name = path.with_name(path.stem + "-calibration_summary.md")

    gt_cx = ground_truth.cx
    gt_cy = ground_truth.cy
    gt_k1 = ground_truth.k1
    gt_fx = ground_truth.fx_px
    gt_fy = ground_truth.fy_px

    sol_cx = solved_cam.cx
    sol_cy = solved_cam.cy
    sol_k1 = solved_cam.k1
    sol_fx = solved_cam.fx_px
    sol_fy = solved_cam.fy_px

    delta_fx = abs(sol_fx - gt_fx)
    delta_fy = abs(sol_fy - gt_fy)
    delta_cx = abs(sol_cx - gt_cx)
    delta_cy = abs(sol_cy - gt_cy)
    delta_k1 = abs(sol_k1 - gt_k1)

    def get_percent(delta, value_base):
        if abs(delta) > abs(value_base):
            return float('inf')
        return (delta / abs(value_base)) * 100.0

    err_fx = get_percent(delta_fx, gt_fx)
    err_fy = get_percent(delta_fy, gt_fy)
    err_cx = get_percent(delta_cx, img_shape[0])
    err_cy = get_percent(delta_cy, img_shape[1])
    err_k1 = get_percent(delta_k1, 0.5 if abs(gt_k1) < 1e-9 else gt_k1)

    def format_metric(value, fmt_spec: str = ".2f"):
        if value is None or np.isnan(value) or np.isinf(value) or abs(value) > 1e10:
            return "--"
        return f"{value:{fmt_spec}}"

    with open(summary_file_name, "w", encoding="utf-8") as f:
        f.write(f"\n### {title}\n")
        f.write("| Parameter | Ground Truth (GT) | Solved Metrics | Absolute Delta | Percentage Error |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Focal Length $f_x$** (px) | {format_metric(gt_fx)} | {format_metric(sol_fx)} |"
                f" {format_metric(delta_fx)} | **{format_metric(err_fx)}%** |\n")
        f.write(f"| **Focal Length $f_y$** (px) | {format_metric(gt_fy)} | {format_metric(sol_fy)} |"
                f" {format_metric(delta_fy)} | **{format_metric(err_fy)}%** |\n")
        f.write(f"| **Principal Point $c_x$** (px) | {format_metric(gt_cx)} | {format_metric(sol_cx)} |"
                f" {format_metric(delta_cx)} | **{format_metric(err_cx)}%** |\n")
        f.write(f"| **Principal Point $c_y$** (px) | {format_metric(gt_cy)} | {format_metric(sol_cy)} |"
                f" {format_metric(delta_cy)} | **{format_metric(err_cy)}%** |\n")
        f.write(f"| **Distortion Coefficient $\\kappa_1$** | {format_metric(gt_k1, '.4f')} |"
                f" {format_metric(sol_k1, '.4f')} | {format_metric(delta_k1, '.4f')} | **{format_metric(err_k1)}%** |\n")
        f.write("\n"
            "**Note on Percentage Error Normalization:**" 
            "To ensure physical relevance and avoid numerical singularities the percentage errors for estimated"
            "parameters are normalized using independent metrics aligned with the sensor's physical domain:\n"
            "1. **Focal Length ($f_x, f_y$):** Normalized by their respective ground truth values ($|f_{\\text{GT}}|$).\n"
            "2. **Principal Point Offsets ($c_x, c_y$):** Normalized by the image raster "
            "width ($W_{\\text{img}}$) and height ($H_{\\text{img}}$).\n"
            "3. **Radial Distortion Coefficient ($\\kappa_1$):** Normalized by the ground truth value"
            " ($|\\kappa_{1,\\text{GT}}|$); under ideal flat-lens conditions where $\kappa_{1,\\text{GT}} \\approx 0$,"
            " the error is normalized by the maximum search boundary ($\kappa_{1,\\text{max}} = 0.5$).\n"
        )