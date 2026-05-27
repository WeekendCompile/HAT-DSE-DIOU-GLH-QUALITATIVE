# -*- coding: utf-8 -*-
"""
inference_visualization.py
--------------------------
End-to-end inference + qualitative visualization pipeline for the
HAT-DSE-DIOU-GLH temporal action detector.

Renders a single output video with three vertically-stacked panels:

    +------------------------------------------------+
    |               original video frame             |
    +------------------------------------------------+
    | GT  : RGB temporal bar (one row per class)     |
    +------------------------------------------------+
    | Pred: RGB temporal bar (one row per class)     |
    +------------------------------------------------+

Bars are drawn on a per-frame timeline with a vertical "playhead" that
follows the current frame so the prediction can be read frame-by-frame.

A consistent colour is assigned to every action class (HSV wheel) so the
GT and Pred panels can be compared at a glance, mirroring the layout used
in the reference figures.

Inference timing is measured end-to-end (model forward + suppression
network + NMS) and the average FPS / total time are reported on screen
and to stdout.

Example
-------
    python inference_visualization.py \
        --config configs/thumos.yaml \
        --checkpoint checkpoint/01_ckp_best.pth.tar \
        --video_path data/I3D/P07-R01-PastaSalad.mp4 \
        --output_dir generated_videos \
        --save_video --show_fps

The --config flag is accepted for CLI parity with the spec but is
optional; when omitted the script falls back to checkpoint/<exp>_opts.json
(which is what main.py writes during training) and finally to the
defaults defined in opts_egtea.py.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

# Local imports
from models import MYNET, SuppressNet


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="HAT-DSE-DIOU-GLH inference + qualitative visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML/JSON config file. If omitted, falls back to "
                        "checkpoint/<exp>_opts.json then opts_egtea defaults.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the main detector checkpoint (.pth.tar).")
    p.add_argument("--suppress_checkpoint", type=str, default=None,
                   help="Path to the SuppressNet checkpoint. "
                        "Defaults to <checkpoint_dir>/ckp_best_suppress.pth.tar.")
    p.add_argument("--video_path", type=str, required=True,
                   help="Path to the input test video (.mp4).")
    p.add_argument("--feature_path", type=str, default=None,
                   help="Path to the pre-extracted I3D feature (.npz with 'rgb'+'flow'). "
                        "Defaults to the .npz sitting next to the video.")
    p.add_argument("--annotation_path", type=str, default=None,
                   help="Path to ground-truth annotation JSON. "
                        "Defaults to data/egtea_annotations_split1.json.")
    p.add_argument("--output_dir", type=str, default="generated_videos",
                   help="Directory where the annotated video is saved.")
    p.add_argument("--exp", type=str, default="01")
    p.add_argument("--split", type=str, default="1")
    p.add_argument("--save_video", action="store_true",
                   help="Save the annotated visualization video to --output_dir.")
    p.add_argument("--show_fps", action="store_true",
                   help="Print and overlay end-to-end inference FPS.")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--fps_out", type=float, default=None,
                   help="Output video FPS. Defaults to the input video FPS.")
    p.add_argument("--score_threshold", type=float, default=None,
                   help="Override classification threshold (default from opts).")
    p.add_argument("--scroll_window", type=float, default=20.0,
                   help="Length (seconds) of the scrolling timeline window. "
                        "When the playhead crosses the end of a window, the bar "
                        "resets to the next 'page' while the video keeps playing.")
    p.add_argument("--min_segment_duration", type=float, default=0.0,
                   help="Drop GT/pred segments shorter than this (seconds). "
                        "Set to 1.0 for cleaner EGTEA demos.")
    return p.parse_args()


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------
def load_opts(args):
    """Build the opt dict the model was trained with.

    Priority: --config  >  <checkpoint_dir>/<exp>_opts.json  >  opts_egtea defaults.
    """
    opt = None

    if args.config and os.path.exists(args.config):
        ext = os.path.splitext(args.config)[1].lower()
        if ext in (".yaml", ".yml"):
            import yaml  # lazy import
            with open(args.config, "r") as f:
                opt = yaml.safe_load(f)
        else:
            with open(args.config, "r") as f:
                opt = json.load(f)

    if opt is None:
        # Try the opts.json saved next to the checkpoint
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        opts_json = os.path.join(ckpt_dir, "{}_opts.json".format(args.exp))
        if os.path.exists(opts_json):
            with open(opts_json, "r") as f:
                opt = json.load(f)

    if opt is None:
        # Fall back to library defaults
        import opts_egtea
        sys_argv_bak = sys.argv
        sys.argv = [sys.argv[0]]  # neutralise argparse inside opts_egtea
        try:
            opt = vars(opts_egtea.parse_opt())
        finally:
            sys.argv = sys_argv_bak

    # Normalise anchors → list[int]
    if isinstance(opt.get("anchors"), str):
        opt["anchors"] = [int(x) for x in opt["anchors"].split(",")]

    # CLI overrides
    opt["exp"] = args.exp
    opt["split"] = args.split
    if args.score_threshold is not None:
        opt["threshold"] = args.score_threshold

    return opt


# ----------------------------------------------------------------------
# Model loading helper
# ----------------------------------------------------------------------
def _strip_module(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_models(opt, args, device):
    print("[Init] Building MYNET ...")
    model = MYNET(opt).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = _strip_module(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Init] Missing keys (showing first 5):", missing[:5])
    if unexpected:
        print("[Init] Unexpected keys (showing first 5):", unexpected[:5])
    model.eval()
    print("[Init] Loaded detector checkpoint: {}".format(args.checkpoint))

    sup_ckpt = args.suppress_checkpoint
    if sup_ckpt is None:
        sup_ckpt = os.path.join(os.path.dirname(args.checkpoint),
                                "ckp_best_suppress.pth.tar")
    sup_model = None
    if os.path.exists(sup_ckpt):
        sup_model = SuppressNet(opt).to(device)
        sc = torch.load(sup_ckpt, map_location=device)
        sup_model.load_state_dict(_strip_module(sc["state_dict"] if "state_dict" in sc else sc))
        sup_model.eval()
        print("[Init] Loaded SuppressNet checkpoint: {}".format(sup_ckpt))
    else:
        print("[Init] WARNING: SuppressNet checkpoint not found ({}). "
              "Falling back to NMS-only post-processing.".format(sup_ckpt))

    return model, sup_model


# ----------------------------------------------------------------------
# Annotation + feature loading
# ----------------------------------------------------------------------
def derive_video_name(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]


def load_annotations(args, video_name):
    """Return (annotations_list, duration, label_name_sorted)."""
    anno_path = args.annotation_path
    if anno_path is None:
        anno_path = os.path.join(os.path.dirname(__file__),
                                 "data", "egtea_annotations_split{}.json".format(args.split))
    if not os.path.exists(anno_path):
        print("[GT ] Annotation file not found: {}".format(anno_path))
        return [], None, []

    with open(anno_path, "r") as f:
        db = json.load(f)["database"]

    # Build the full sorted label list (must match what the dataset/model uses)
    label_set = set()
    for info in db.values():
        for a in info["annotations"]:
            label_set.add(a["label"])
    label_name = sorted(label_set)

    if video_name not in db:
        print("[GT ] Video '{}' not in annotation JSON — visualizing prediction only."
              .format(video_name))
        return [], None, label_name

    info = db[video_name]
    return info["annotations"], float(info["duration"]), label_name


def load_features(args, video_path):
    feat_path = args.feature_path
    if feat_path is None:
        # Same folder as video, same basename, .npz
        feat_path = os.path.splitext(video_path)[0] + ".npz"
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            "Pre-extracted I3D feature not found at '{}'. "
            "Provide --feature_path or place a matching .npz next to the video."
            .format(feat_path))

    npz = np.load(feat_path)
    if "rgb" in npz and "flow" in npz:
        feat = np.concatenate([npz["rgb"], npz["flow"]], axis=1).astype(np.float32)
    elif "feats" in npz:
        feat = npz["feats"].astype(np.float32)
    else:
        raise RuntimeError(
            "Unsupported .npz layout in '{}'. Expected ('rgb','flow') or 'feats'. "
            "Keys present: {}".format(feat_path, list(npz.keys())))
    print("[Feat] Loaded features shape={} from {}".format(feat.shape, feat_path))
    return feat, feat_path


# ----------------------------------------------------------------------
# Inference — mirrors test_online() in main.py but for a single video
# ----------------------------------------------------------------------
def run_inference(model, sup_model, opt, feature, duration_sec):
    """Run the online detector + SuppressNet on a single video's features.

    Returns
    -------
    proposals : list[dict]   merged proposal list (segments in seconds)
    timing    : dict         {'inference_sec', 'snippets', 'feat_fps'}
    """
    device = next(model.parameters()).device
    num_class = opt["num_of_class"]
    unit_size = opt["segment_size"]
    anchors = opt["anchors"]
    feat_dim = opt["feat_dim"]
    threshold = opt["threshold"]
    soft_nms_thr = opt["soft_nms"]
    sup_thr = opt["sup_threshold"]

    T_feat = feature.shape[0]
    if duration_sec is None or duration_sec <= 0:
        # Fall back to "1 feature step = 1 second" if no GT duration is available
        frame_to_time = 1.0
    else:
        # main.py uses: frame_to_time = 100.0*video_time / duration / 100.0
        # which simplifies to video_time / duration.
        frame_to_time = duration_sec / float(T_feat)

    input_queue = torch.zeros((unit_size, feat_dim), device=device)
    sup_queue = torch.zeros((unit_size, num_class - 1), device=device)
    feature_tensor = torch.from_numpy(feature).to(device)

    proposal_dict = []

    start = time.time()

    for idx in range(T_feat):
        # Slide the snippet window: shift left, append current frame
        input_queue[:-1, :] = input_queue[1:, :].clone()
        input_queue[-1:, :] = feature_tensor[idx:idx + 1]

        minput = input_queue.unsqueeze(0)
        with torch.no_grad():
            act_cls, act_reg = model(minput)
            act_cls = torch.softmax(act_cls, dim=-1)

        cls_anc = act_cls.squeeze(0).detach().cpu().numpy()
        reg_anc = act_reg.squeeze(0).detach().cpu().numpy()

        # Per-anchor proposals
        proposal_anc_dict = []
        for anc_idx in range(len(anchors)):
            cls = np.argwhere(cls_anc[anc_idx][:-1] > threshold).reshape(-1)
            if len(cls) == 0:
                continue

            ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
            length = anchors[anc_idx] * np.exp(reg_anc[anc_idx][1])
            st = ed - length

            for cidx in range(len(cls)):
                label = int(cls[cidx])
                proposal_anc_dict.append({
                    "segment": [float(st * frame_to_time), float(ed * frame_to_time)],
                    "score":   float(cls_anc[anc_idx][label]),
                    "label_idx": label,
                    "gentime": float(idx * frame_to_time),
                })

        proposal_anc_dict = _local_nms_by_idx(proposal_anc_dict, soft_nms_thr)

        # SuppressNet pass (optional but matches train-time pipeline)
        if sup_model is not None:
            sup_queue[:-1, :] = sup_queue[1:, :].clone()
            sup_queue[-1, :] = 0
            for proposal in proposal_anc_dict:
                sup_queue[-1, proposal["label_idx"]] = proposal["score"]

            with torch.no_grad():
                suppress_conf = sup_model(sup_queue.unsqueeze(0))
                suppress_conf = suppress_conf.squeeze(0).detach().cpu().numpy()

            for cls in range(num_class - 1):
                if suppress_conf[cls] > sup_thr:
                    for proposal in proposal_anc_dict:
                        if proposal["label_idx"] == cls:
                            if _check_overlap_idx(proposal_dict, proposal, soft_nms_thr) is None:
                                proposal_dict.append(proposal)
        else:
            # NMS-only path
            for proposal in proposal_anc_dict:
                if _check_overlap_idx(proposal_dict, proposal, soft_nms_thr) is None:
                    proposal_dict.append(proposal)

    end = time.time()
    inference_sec = end - start

    timing = {
        "inference_sec": inference_sec,
        "snippets": T_feat,
        "feat_fps": T_feat / max(inference_sec, 1e-9),
    }
    return proposal_dict, timing


def _local_nms_by_idx(proposals, overlap_thr):
    """NMS variant that compares by integer label_idx (avoids string label lookups)."""
    if not proposals:
        return []
    sorted_p = sorted(proposals, key=lambda p: p["score"], reverse=True)
    picked = []
    while sorted_p:
        head = sorted_p.pop(0)
        picked.append(head)
        kept = []
        for p in sorted_p:
            if p["label_idx"] != head["label_idx"]:
                kept.append(p)
                continue
            sst = min(head["segment"][0], p["segment"][0])
            led = max(head["segment"][1], p["segment"][1])
            lst = max(head["segment"][0], p["segment"][0])
            sed = min(head["segment"][1], p["segment"][1])
            iou = (sed - lst) / max(led - sst, 1e-6)
            if iou <= overlap_thr:
                kept.append(p)
        sorted_p = kept
    return picked


def _check_overlap_idx(proposal_list, new_proposal, overlap_thr):
    for p in proposal_list:
        if p["label_idx"] != new_proposal["label_idx"]:
            continue
        sst = min(p["segment"][0], new_proposal["segment"][0])
        led = max(p["segment"][1], new_proposal["segment"][1])
        lst = max(p["segment"][0], new_proposal["segment"][0])
        sed = min(p["segment"][1], new_proposal["segment"][1])
        iou = (sed - lst) / max(led - sst, 1e-6)
        if iou > overlap_thr:
            return p
    return None


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def class_color_palette(label_names):
    """Assign a deterministic, visually distinct BGR colour per class."""
    n = max(len(label_names), 1)
    palette = {}
    for i, name in enumerate(label_names):
        hue = int(180.0 * i / n)  # OpenCV hue is 0..179
        hsv = np.uint8([[[hue, 220, 245]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette[name] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return palette


BAR_BG_COLOR = (192, 192, 192)   # BGR equivalent of #C0C0C0
BAR_BG_PADDING = 5


def _normalize_segments(raw, kind):
    """Normalize GT/pred dicts to a flat list of {label, start, end, duration, score}.

    raw : list of either
        {"segment": [st, ed], "label": str, "score": float?}   (GT/pred from this pipeline)
        {"start": float, "end": float, "label": str, ...}      (already-flat dicts)
    """
    out = []
    for s in raw:
        if "segment" in s:
            st, ed = float(s["segment"][0]), float(s["segment"][1])
        else:
            st, ed = float(s["start"]), float(s["end"])
        if ed <= st:
            continue
        out.append({
            "label":    s["label"],
            "start":    st,
            "end":      ed,
            "duration": ed - st,
            "score":    float(s["score"]) if "score" in s else None,
            "kind":     kind,
        })
    return out


def draw_scrolling_bar(panel_w, segments, current_t, window_start, window_size,
                       palette, title, label_color, page_idx, num_pages,
                       bar_height=46):
    """Draw a single horizontal scrolling bar for one page (window) of the
    timeline.

    - Only segments overlapping [window_start, window_start + window_size]
      are drawn, clipped to that window.
    - Each segment uses its class colour from `palette`; the class label is
      rendered inside the segment when there's room.
    - "Progressive reveal": each segment is drawn only up to
      `min(seg.end, current_t)` so the bar fills in as the video plays.
    - A black playhead marker tracks `current_t` within the current page.
    """
    pad_left = 150
    pad_right = 20
    pad_top = 36
    pad_bottom = 16
    inner_w = panel_w - pad_left - pad_right

    H = pad_top + bar_height + 2 * BAR_BG_PADDING + pad_bottom
    canvas = np.full((H, panel_w, 3), 245, dtype=np.uint8)

    window_end = window_start + window_size

    # ---- Header text: title + window/page indicator -------------------
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_DUPLEX,
                0.62, label_color, 2, cv2.LINE_AA)

    page_str = "Page {}/{}   [{:.1f}s - {:.1f}s]".format(
        page_idx + 1, max(num_pages, 1), window_start, window_end)
    (tw, _), _ = cv2.getTextSize(page_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(canvas, page_str, (panel_w - pad_right - tw, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    # ---- Bar background (grey strip, like the reference design) ------
    bar_y = pad_top
    bg_x0 = pad_left - BAR_BG_PADDING
    bg_x1 = pad_left + inner_w + BAR_BG_PADDING
    bg_y0 = bar_y - BAR_BG_PADDING
    bg_y1 = bar_y + bar_height + BAR_BG_PADDING
    cv2.rectangle(canvas, (bg_x0, bg_y0), (bg_x1, bg_y1), BAR_BG_COLOR, -1)

    # ---- Draw segments inside the current page ------------------------
    # The visible "drawn" extent is clipped at current_t for progressive reveal.
    draw_limit = min(current_t, window_end)
    for seg in segments:
        if seg["end"] < window_start or seg["start"] > window_end:
            continue
        st = max(seg["start"], window_start)
        ed = min(seg["end"], draw_limit)
        if ed <= st:
            continue
        x0 = pad_left + int(inner_w * (st - window_start) / window_size)
        x1 = pad_left + int(inner_w * (ed - window_start) / window_size)
        if x1 <= x0:
            x1 = x0 + 2
        colour = palette.get(seg["label"], (120, 120, 120))
        cv2.rectangle(canvas, (x0, bar_y), (x1, bar_y + bar_height),
                      colour, -1)
        cv2.rectangle(canvas, (x0, bar_y), (x1, bar_y + bar_height),
                      (30, 30, 30), 1)

        # Render the class label centred inside the segment, but only if
        # there is enough horizontal room.
        label = seg["label"]
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        (lw, lh), _ = cv2.getTextSize(label, font, font_scale, 1)
        if (x1 - x0) > lw + 8:
            tx = x0 + ((x1 - x0) - lw) // 2
            ty = bar_y + (bar_height + lh) // 2
            # White outline + dark text for readability over any colour
            cv2.putText(canvas, label, (tx, ty), font, font_scale,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty), font, font_scale,
                        (20, 20, 20), 1, cv2.LINE_AA)

    # ---- Playhead (only meaningful when current_t is inside this page) ---
    if window_start <= current_t <= window_end:
        px = pad_left + int(inner_w * (current_t - window_start) / window_size)
        cv2.line(canvas, (px, bar_y - 6),
                 (px, bar_y + bar_height + 6), (0, 0, 0), 2)

    return canvas


def render_video(args, video_path, gt_segments, pred_segments, label_names,
                 duration_sec, timing):
    """Render the annotated video with a scrolling-window bar layout.

    Layout per frame:
        +-----------------------------------+
        |   original video frame            |
        +-----------------------------------+
        |   Ground Truth   (page k/N)       |   <- single horizontal bar
        +-----------------------------------+
        |   Prediction     (page k/N)       |   <- single horizontal bar
        +-----------------------------------+
        |  t=...   inference=...   fps=...  |
        +-----------------------------------+

    The bars only show segments inside the *current* time window
    [page_k * scroll_window, (page_k + 1) * scroll_window].  When the
    playhead crosses the page boundary, the bar flips to the next page
    while the video keeps playing above.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {}".format(video_path))

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if duration_sec is None or duration_sec <= 0:
        duration_sec = n_frames / max(src_fps, 1.0)

    # ---- Normalize + drop too-short segments --------------------------
    gt_flat = _normalize_segments(gt_segments, "gt")
    pr_flat = _normalize_segments(pred_segments, "pred")
    if args.min_segment_duration > 0:
        gt_flat = [s for s in gt_flat if s["duration"] >= args.min_segment_duration]
        pr_flat = [s for s in pr_flat if s["duration"] >= args.min_segment_duration]

    # ---- Canvas geometry ---------------------------------------------
    panel_w = max(src_w, 960)
    scale = panel_w / float(src_w)
    top_h = int(src_h * scale)

    palette = class_color_palette(label_names)

    window_size = max(args.scroll_window, 1.0)
    num_pages = max(int(np.ceil(duration_sec / window_size)), 1)

    # GT text: blue-ish, Pred text: orange-ish (matches reference convention)
    GT_LABEL_COLOR = (180, 119, 31)
    PR_LABEL_COLOR = (14, 127, 255)

    # Probe heights with a dummy render so we can size the output writer
    dummy_gt = draw_scrolling_bar(panel_w, [], 0.0, 0.0, window_size, palette,
                                  "Ground Truth", GT_LABEL_COLOR, 0, num_pages)
    dummy_pr = draw_scrolling_bar(panel_w, [], 0.0, 0.0, window_size, palette,
                                  "Prediction", PR_LABEL_COLOR, 0, num_pages)
    gt_h = dummy_gt.shape[0]
    pr_h = dummy_pr.shape[0]

    footer_h = 32
    out_h = top_h + gt_h + pr_h + footer_h
    out_w = panel_w

    out_path = None
    writer = None
    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
        out_name = "{}_vis.mp4".format(derive_video_name(video_path))
        out_path = os.path.join(args.output_dir, out_name)
        out_fps = args.fps_out if args.fps_out else src_fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError("Could not open VideoWriter for {}".format(out_path))

    avg_fps = timing["snippets"] / max(timing["inference_sec"], 1e-9)

    print("[Viz ] Writing visualization: {}x{} @ {:.2f} fps   window={:.1f}s, pages={}   -> {}"
          .format(out_w, out_h, args.fps_out if args.fps_out else src_fps,
                  window_size, num_pages, out_path))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, (panel_w, top_h),
                               interpolation=cv2.INTER_LINEAR)

        current_t = frame_idx / max(src_fps, 1.0)
        page_idx = min(int(current_t // window_size), num_pages - 1)
        window_start = page_idx * window_size

        gt_panel = draw_scrolling_bar(
            panel_w, gt_flat, current_t, window_start, window_size,
            palette, "Ground Truth", GT_LABEL_COLOR, page_idx, num_pages)
        pr_panel = draw_scrolling_bar(
            panel_w, pr_flat, current_t, window_start, window_size,
            palette, "Prediction", PR_LABEL_COLOR, page_idx, num_pages)

        footer = np.full((footer_h, panel_w, 3), 30, dtype=np.uint8)
        msg = ("t={:7.2f}s / {:.2f}s   page {}/{}   "
               "inference_total={:.2f}s   avg_fps={:.2f}").format(
            current_t, duration_sec, page_idx + 1, num_pages,
            timing["inference_sec"], avg_fps)
        cv2.putText(footer, msg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)

        composite = np.vstack([frame, gt_panel, pr_panel, footer])

        if writer is not None:
            writer.write(composite)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    return out_path, frame_idx


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    opt = load_opts(args)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("[Init] CUDA requested but unavailable — falling back to CPU.")

    video_name = derive_video_name(args.video_path)
    print("[Init] Video name: {}".format(video_name))

    gt_anno, duration_sec, label_name = load_annotations(args, video_name)
    if not label_name:
        # Try to recover label_name from the default annotation file even if
        # the video isn't listed there.
        if args.annotation_path is None:
            args.annotation_path = os.path.join(os.path.dirname(__file__),
                                                "data",
                                                "egtea_annotations_split{}.json".format(args.split))
        if os.path.exists(args.annotation_path):
            with open(args.annotation_path, "r") as f:
                db = json.load(f)["database"]
            labels = set()
            for info in db.values():
                for a in info["annotations"]:
                    labels.add(a["label"])
            label_name = sorted(labels)

    if not label_name:
        raise RuntimeError("Could not recover the class list — pass --annotation_path.")

    print("[Init] {} classes: {}".format(len(label_name), label_name))

    feature, _ = load_features(args, args.video_path)

    if duration_sec is None:
        cap = cv2.VideoCapture(args.video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        duration_sec = n / max(f, 1.0)
        print("[Init] Duration from video header: {:.2f}s".format(duration_sec))
    else:
        print("[Init] Duration from annotation JSON: {:.2f}s".format(duration_sec))

    model, sup_model = load_models(opt, args, device)

    print("[Run ] Running detector over {} I3D snippets ...".format(feature.shape[0]))
    proposals, timing = run_inference(model, sup_model, opt, feature, duration_sec)

    # Attach string labels for visualization
    for p in proposals:
        p["label"] = label_name[p["label_idx"]]

    avg_fps = timing["snippets"] / max(timing["inference_sec"], 1e-9)
    print("[Time] inference_sec = {:.3f}s   snippets = {}   feat_fps = {:.2f}"
          .format(timing["inference_sec"], timing["snippets"], avg_fps))
    if args.show_fps:
        print("[FPS ] End-to-end inference avg FPS (per I3D snippet): {:.2f}"
              .format(avg_fps))

    print("[Run ] {} predicted segments after suppression.".format(len(proposals)))
    if proposals:
        for p in sorted(proposals, key=lambda x: x["segment"][0])[:10]:
            print("       [{:6.2f}s -> {:6.2f}s]  {:<14s}  score={:.3f}".format(
                p["segment"][0], p["segment"][1], p["label"], p["score"]))

    # Optional: dump predictions/timing JSON next to the video
    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir,
                                "{}_predictions.json".format(video_name))
        with open(log_path, "w") as f:
            json.dump({
                "video": video_name,
                "duration_sec": duration_sec,
                "inference_sec": timing["inference_sec"],
                "avg_fps_snippet": avg_fps,
                "num_predictions": len(proposals),
                "predictions": [
                    {"segment": p["segment"], "label": p["label"], "score": p["score"]}
                    for p in proposals
                ],
                "ground_truth": gt_anno,
            }, f, indent=2)
        print("[Log ] Prediction log saved: {}".format(log_path))

    out_path, n_frames = render_video(
        args, args.video_path, gt_anno, proposals, label_name,
        duration_sec, timing,
    )

    if out_path:
        print("[Done] Annotated video saved: {}".format(out_path))
    else:
        print("[Done] Visualization rendered ({} frames). "
              "Pass --save_video to write it to disk.".format(n_frames))


if __name__ == "__main__":
    main()
