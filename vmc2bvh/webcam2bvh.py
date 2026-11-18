#!/usr/bin/env python3
"""
webcam2bvh - record body motion from a webcam (Google MediaPipe Pose) and
export a BVH for Blender. No VMC app, no login, no avatar rendering.

    python webcam2bvh.py               # 10 s countdown, record, Ctrl+C (or q) to stop
    python webcam2bvh.py -c 5 -t 3     # 5 s countdown, cut last 3 s
    python webcam2bvh.py --video in.mp4 -o out.bvh   # process a video file instead

Coordinates
-----------
MediaPipe world landmarks: meters, origin at the hip centre, x = image right,
y = image down, z = away from the camera (right-handed). The performer faces
the camera, so converting to BVH (right-handed, Y-up, character facing +Z) is a
180-degree turn about X:  bvh = (x, -y, -z). The performer's left side lands on
+X, matching the VMC recorder, and Blender's BVH importer shows the character
facing the front view.

Bone rotations are solved per segment from landmark directions (plus a second
landmark to fix the twist), then made parent-relative and written with the
same BVH writer as vmc2bvh.py.
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import tiling
from vmc2bvh import build_bvh

# MediaPipe Pose landmark indices
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

# Skeleton (Unity humanoid names so Blender retargeting presets recognise them).
BONES = ["Hips", "Spine", "Chest", "Neck", "Head",
         "LeftUpperArm", "LeftLowerArm", "LeftHand",
         "RightUpperArm", "RightLowerArm", "RightHand",
         "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
         "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes"]
PARENT = {"Hips": None, "Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck",
          "LeftUpperArm": "Chest", "LeftLowerArm": "LeftUpperArm", "LeftHand": "LeftLowerArm",
          "RightUpperArm": "Chest", "RightLowerArm": "RightUpperArm", "RightHand": "RightLowerArm",
          "LeftUpperLeg": "Hips", "LeftLowerLeg": "LeftUpperLeg", "LeftFoot": "LeftLowerLeg",
          "LeftToes": "LeftFoot",
          "RightUpperLeg": "Hips", "RightLowerLeg": "RightUpperLeg", "RightFoot": "RightLowerLeg",
          "RightToes": "RightFoot"}

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])
HEAD_FWD_REST = np.array([0.0, -0.3, 1.0]) / math.hypot(0.3, 1.0)  # ear-centre -> nose

# (primary from, primary to, primary rest dir, secondary vector fn, secondary rest dir,
#  landmarks that must be visible)
def _limb_specs():
    specs = {}
    for side, s in (("Left", 1.0), ("Right", -1.0)):
        L = side == "Left"
        sh, el, wr = (L_SH, L_EL, L_WR) if L else (R_SH, R_EL, R_WR)
        ix, pk = (L_INDEX, L_PINKY) if L else (R_INDEX, R_PINKY)
        hip, kn, an = (L_HIP, L_KNEE, L_ANK) if L else (R_HIP, R_KNEE, R_ANK)
        he, ft = (L_HEEL, L_FOOT) if L else (R_HEEL, R_FOOT)
        # Arms, T-pose rest: arm along +-X, palm down, elbow bends forward (+Z).
        specs[f"{side}UpperArm"] = (lambda P, a=sh, b=el: P[b] - P[a], s * X,
                                    lambda P, a=el, b=wr: P[b] - P[a], Z, (sh, el, wr))
        specs[f"{side}LowerArm"] = (lambda P, a=el, b=wr: P[b] - P[a], s * X,
                                    lambda P, i=ix, k=pk: P[i] - P[k], Z, (el, wr, ix, pk))
        specs[f"{side}Hand"] = (lambda P, w=wr, i=ix, k=pk: 0.5 * (P[i] + P[k]) - P[w], s * X,
                                lambda P, i=ix, k=pk: P[i] - P[k], Z, (wr, ix, pk))
        # Legs, rest: straight down, foot pointing forward (+Z).
        specs[f"{side}UpperLeg"] = (lambda P, a=hip, b=kn: P[b] - P[a], -Y,
                                    lambda P, h=he, f=ft: P[f] - P[h], Z, (hip, kn, he, ft))
        specs[f"{side}LowerLeg"] = (lambda P, a=kn, b=an: P[b] - P[a], -Y,
                                    lambda P, h=he, f=ft: P[f] - P[h], Z, (kn, an, he, ft))
        specs[f"{side}Foot"] = (lambda P, a=an, b=ft: P[b] - P[a], Z,
                                lambda P, a=an, k=kn: P[k] - P[a], Y, (kn, an, ft))
    return specs


LIMBS = _limb_specs()


def mp_to_bvh(p):
    """MediaPipe (x right, y down, z away from camera) -> BVH (x, y up, z toward camera)."""
    p = np.asarray(p, dtype=float)
    return p * np.array([1.0, -1.0, -1.0])


def _basis(p, s):
    e1 = p / np.linalg.norm(p)
    s = s - np.dot(s, e1) * e1
    n = np.linalg.norm(s)
    if n < 1e-9:
        return None
    e2 = s / n
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def align(p, p_rest, s, s_rest):
    """Rotation taking p_rest -> p exactly, and s_rest as close as possible to s."""
    b_obs, b_rest = _basis(p, s), _basis(p_rest, s_rest)
    if b_obs is None:
        return None
    return Rotation.from_matrix(b_obs @ b_rest.T)


def _ortho_ratio(p, s):
    pn = p / np.linalg.norm(p)
    return np.linalg.norm(s - np.dot(s, pn) * pn) / max(np.linalg.norm(s), 1e-9)


def _slerp_half(a: Rotation, b: Rotation) -> Rotation:
    return a * Rotation.from_rotvec(0.5 * (a.inv() * b).as_rotvec())


class PoseSolver:
    """33 MediaPipe world landmarks (BVH space) -> local bone rotations."""

    def __init__(self, vis_threshold=0.5, max_speed_deg=1200.0):
        self.vis_threshold = vis_threshold
        # Joints can't really spin faster than this; larger one-frame jumps are
        # tracking glitches (e.g. a forearm twist flip) and get rate-limited.
        self.max_speed = math.radians(max_speed_deg)
        self.prev_t = None
        self.prev_world: dict[str, Rotation] = {}
        self.prev_local: dict[str, Rotation] = {}

    def _visible(self, vis, idx):
        return vis is None or all(vis[i] >= self.vis_threshold for i in idx)

    def solve(self, P, vis=None, t=None):
        P = np.asarray(P, dtype=float)
        dt = None if (t is None or self.prev_t is None) else max(t - self.prev_t, 1e-3)
        self.prev_t = t
        hip_mid = 0.5 * (P[L_HIP] + P[R_HIP])
        sh_mid = 0.5 * (P[L_SH] + P[R_SH])
        ear_mid = 0.5 * (P[L_EAR] + P[R_EAR])
        up = sh_mid - hip_mid

        world: dict[str, Rotation | None] = {}
        world["Hips"] = align(P[L_HIP] - P[R_HIP], X, up, Y)
        world["Chest"] = align(P[L_SH] - P[R_SH], X, up, Y)
        world["Head"] = align(P[L_EAR] - P[R_EAR], X, P[NOSE] - ear_mid, HEAD_FWD_REST)
        if world["Hips"] is not None and world["Chest"] is not None:
            world["Spine"] = _slerp_half(world["Hips"], world["Chest"])
        if world["Chest"] is not None and world["Head"] is not None:
            world["Neck"] = _slerp_half(world["Chest"], world["Head"])
        torso_vis = {"Hips": (L_HIP, R_HIP, L_SH, R_SH), "Spine": (L_HIP, R_HIP, L_SH, R_SH),
                     "Chest": (L_HIP, R_HIP, L_SH, R_SH),
                     "Neck": (L_SH, R_SH, L_EAR, R_EAR, NOSE),
                     "Head": (L_EAR, R_EAR, NOSE)}
        need = dict(torso_vis)

        for name, (pf, pr, sf, sr, idx) in LIMBS.items():
            need[name] = idx
            p, s = pf(P), sf(P)
            if np.linalg.norm(p) < 1e-6:
                world[name] = None
                continue
            # Twist is ill-defined when the helper vector is (nearly) parallel to
            # the bone (e.g. a straight arm): keep the previous frame's twist.
            if _ortho_ratio(p, s) < 0.15 and name in self.prev_world:
                s = self.prev_world[name].apply(sr)
            world[name] = align(p, pr, s, sr)

        # World -> local, in hierarchy order; hold bones whose landmarks are
        # not visible (e.g. legs out of frame) at their previous local pose.
        eff_world: dict[str, Rotation] = {}
        local: dict[str, Rotation] = {}
        for name in BONES:
            par = PARENT[name]
            par_w = eff_world[par] if par else Rotation.identity()
            if name == "LeftToes" or name == "RightToes":
                loc = Rotation.identity()
            elif world.get(name) is not None and self._visible(vis, need[name]):
                loc = par_w.inv() * world[name]
            else:
                loc = self.prev_local.get(name, Rotation.identity())
            if dt is not None and name in self.prev_local:
                step = (self.prev_local[name].inv() * loc).as_rotvec()
                ang, lim = np.linalg.norm(step), self.max_speed * dt
                if ang > lim:
                    loc = self.prev_local[name] * Rotation.from_rotvec(step * (lim / ang))
            local[name] = loc
            eff_world[name] = par_w * loc
        self.prev_local = local
        self.prev_world = eff_world
        return local


def measure_lengths(P):
    """Segment lengths (meters) from one frame of world landmarks."""
    d = lambda a, b: float(np.linalg.norm(P[a] - P[b]))
    mid = lambda a, b: 0.5 * (P[a] + P[b])
    return {
        "hip_half": 0.5 * d(L_HIP, R_HIP),
        "sh_half": 0.5 * d(L_SH, R_SH),
        "torso": float(np.linalg.norm(mid(L_SH, R_SH) - mid(L_HIP, R_HIP))),
        "neck": float(np.linalg.norm(mid(L_EAR, R_EAR) - mid(L_SH, R_SH))),
        "upper_arm": 0.5 * (d(L_SH, L_EL) + d(R_SH, R_EL)),
        "forearm": 0.5 * (d(L_EL, L_WR) + d(R_EL, R_WR)),
        "thigh": 0.5 * (d(L_HIP, L_KNEE) + d(R_HIP, R_KNEE)),
        "shin": 0.5 * (d(L_KNEE, L_ANK) + d(R_KNEE, R_ANK)),
        "foot": 0.5 * (d(L_ANK, L_FOOT) + d(R_ANK, R_FOOT)),
    }


def rest_offsets(L):
    """Joint offsets in the BVH rest T-pose (facing +Z, performer's left = +X)."""
    T = L["torso"]
    off = {"Hips": (0, 0, 0), "Spine": (0, 0.1 * T, 0), "Chest": (0, 0.4 * T, 0),
           "Neck": (0, 0.5 * T, 0), "Head": (0, L["neck"], 0)}
    for side, s in (("Left", 1.0), ("Right", -1.0)):
        off[f"{side}UpperArm"] = (s * L["sh_half"], 0.5 * T, 0)
        off[f"{side}LowerArm"] = (s * L["upper_arm"], 0, 0)
        off[f"{side}Hand"] = (s * L["forearm"], 0, 0)
        off[f"{side}UpperLeg"] = (s * L["hip_half"], 0, 0)
        off[f"{side}LowerLeg"] = (0, -L["thigh"], 0)
        off[f"{side}Foot"] = (0, -L["shin"], 0)
        off[f"{side}Toes"] = (0, 0, L["foot"])
    return {k: np.array(v, dtype=float) for k, v in off.items()}


class OneEuro:
    """One Euro filter (Casiez et al.) for jitter-free yet responsive signals."""

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-4)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


def estimate_root(P_world, img_xy, img_w, img_h, fov_deg):
    """Hip-centre position (BVH space, meters) from one frame.
    y: hip height above the lowest foot point. x/z: pinhole camera estimate
    using the torso's real length (world landmarks) vs its size in pixels."""
    feet = P_world[[L_ANK, R_ANK, L_HEEL, R_HEEL, L_FOOT, R_FOOT], 1]
    hip_mid = 0.5 * (P_world[L_HIP] + P_world[R_HIP])
    y = hip_mid[1] - feet.min()
    f = 0.5 * img_w / math.tan(math.radians(fov_deg) / 2)
    px = img_xy * np.array([img_w, img_h])
    hip_px = 0.5 * (px[L_HIP] + px[R_HIP])
    # Pixels-per-meter from the least foreshortened body segment (a segment
    # pointing at the camera looks short; one across the image plane doesn't).
    sh_px, sh_w = 0.5 * (px[L_SH] + px[R_SH]), 0.5 * (P_world[L_SH] + P_world[R_SH])
    segs = [(sh_px, hip_px, sh_w, hip_mid),
            (px[L_SH], px[R_SH], P_world[L_SH], P_world[R_SH])]
    for a, b in ((L_SH, L_EL), (R_SH, R_EL), (L_HIP, L_KNEE), (R_HIP, R_KNEE),
                 (L_KNEE, L_ANK), (R_KNEE, R_ANK)):
        segs.append((px[a], px[b], P_world[a], P_world[b]))
    best = 0.0
    for pa, pb, wa, wb in segs:
        m = np.linalg.norm(wa - wb)
        if m > 0.15:
            best = max(best, np.linalg.norm(pa - pb) / m)
    if best <= 0:
        return None
    depth = float(np.clip(f / best, 0.5, 10.0))
    x = (hip_px[0] - 0.5 * img_w) * depth / f
    return np.array([x, y, -depth])


def fov_for(fov_deg, rotate, w, h):
    """Horizontal FOV of the (possibly rotated) image. --fov is the camera's
    horizontal FOV in its normal landscape orientation; after a 90-degree turn
    the image's horizontal axis is the camera's original vertical axis."""
    if rotate not in ("cw", "ccw"):
        return fov_deg
    # rotated image is w wide, h tall; the original landscape frame was h x w
    t = math.tan(math.radians(fov_deg) / 2) * (w / h)
    return math.degrees(2 * math.atan(t))


def resample(samples, fps):
    """samples: list of (t, {bone: Rotation}, root_pos). Returns fixed-rate frames."""
    ts = np.array([s[0] for s in samples])
    if len(ts) == 1 or ts[-1] - ts[0] < 1e-6:
        out_t = ts[:1]
    else:
        out_t = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / fps)
    rots = {}
    for b in BONES:
        q = np.array([s[1][b].as_quat() for s in samples])
        if len(ts) == 1:
            rots[b] = Rotation.from_quat(q)
        else:
            rots[b] = Slerp(ts, Rotation.from_quat(q))(out_t)
    root = np.array([s[2] for s in samples])
    root_i = np.column_stack([np.interp(out_t, ts, root[:, k]) for k in range(3)])
    return out_t, rots, root_i


def to_bvh_text(samples, lengths_list, fps, scale, in_place):
    L = {k: float(np.median([l[k] for l in lengths_list])) for k in lengths_list[0]}
    off = rest_offsets(L)
    out_t, rots, root = resample(samples, fps)
    root = root - np.array([root[0, 0], 0.0, root[0, 2]])  # start at the origin
    if in_place:
        root[:, 0] = 0.0
        root[:, 2] = 0.0
    frames = []
    for i in range(len(out_t)):
        bones = {b: (tuple(off[b]), tuple(rots[b][i].as_quat())) for b in BONES}
        bones["Hips"] = (tuple(root[i]), bones["Hips"][1])
        frames.append((bones, None))
    return build_bvh(frames, fps, scale, use_root=False, unity_input=False), len(frames)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Webcam motion capture to BVH (MediaPipe Pose).")
    ap.add_argument("-o", "--output", default=None,
                    help="output .bvh (default: webcam_YYYYmmdd_HHMMSS.bvh)")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--video", default=None, help="process a video file instead of the webcam")
    ap.add_argument("-f", "--fps", type=float, default=24.0, help="BVH frame rate (default 24)")
    ap.add_argument("-c", "--countdown", type=float, default=10.0,
                    help="seconds before recording starts (default 10, 0 = none)")
    ap.add_argument("-t", "--trim-end", type=float, default=5.0,
                    help="seconds cut from the end of the take (default 5, 0 = none)")
    ap.add_argument("-s", "--scale", type=float, default=1.0,
                    help="unit scale (1.0 = meters, 100 = centimeters)")
    ap.add_argument("--model", type=int, choices=(0, 1, 2), default=1,
                    help="MediaPipe model: 0 = fastest, 1 = balanced (default), 2 = most accurate")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="jitter smoothing: higher = smoother but laggier (default 1.0, 0 = off)")
    ap.add_argument("--in-place", action="store_true",
                    help="no horizontal root motion (only hip height)")
    ap.add_argument("--rotate", choices=("cw", "ccw", "180"), default=None,
                    help="turn the camera image upright when the Mac/camera is on its side "
                         "(cw = rotate image clockwise, ccw = counter-clockwise)")
    ap.add_argument("--fov", type=float, default=60.0,
                    help="camera horizontal field of view in degrees, for root motion (default 60)")
    ap.add_argument("--width", type=int, default=640, help="capture width (default 640)")
    ap.add_argument("--height", type=int, default=480, help="capture height (default 480)")
    ap.add_argument("--no-preview", action="store_true", help="don't show the camera window")
    ap.add_argument("--preview", action="store_true",
                    help="preview only: show the camera and tracking to set up your shot; "
                         "nothing is recorded or saved")
    ap.add_argument("--tile", choices=("split", "halves", "off"), default="split",
                    help="window layout: split = macOS Split View, camera left, then click "
                         "Terminal for the right (default); halves = plain windows, camera "
                         "left half, Terminal right half; off = leave windows alone")
    args = ap.parse_args(argv)
    if args.fps <= 0:
        ap.error("--fps must be > 0")
    if args.preview and args.no_preview:
        ap.error("--preview needs the camera window; drop --no-preview")

    os.environ.setdefault("GLOG_minloglevel", "2")      # quiet MediaPipe/TFLite logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import cv2
    import mediapipe as mp
    ROTATIONS = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
                 "180": cv2.ROTATE_180}

    is_video = args.video is not None
    output = args.output or time.strftime("webcam_%Y%m%d_%H%M%S.bvh")
    cap = cv2.VideoCapture(args.video if is_video else args.camera)
    if not is_video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened() and is_video:
        print(f"Could not read video {args.video!r} (try converting it to .mp4).",
              file=sys.stderr)
        return 1
    if not cap.isOpened():
        print("Could not open the camera. In System Preferences > Security & Privacy > "
              "Privacy > Camera, allow Terminal, then run again.", file=sys.stderr)
        return 1
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=args.model,
                                  smooth_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    solver = PoseSolver()
    smooth = args.smooth
    f_world = OneEuro(min_cutoff=1.5 / smooth, beta=0.5) if smooth > 0 else None
    f_img = OneEuro(min_cutoff=1.5 / smooth, beta=0.5) if smooth > 0 else None
    f_root = OneEuro(min_cutoff=0.5 / smooth, beta=0.2) if smooth > 0 else None
    draw = mp.solutions.drawing_utils
    conns = mp.solutions.pose.POSE_CONNECTIONS

    samples, lengths = [], []
    t_start = time.perf_counter()
    countdown = 0.0 if (is_video or args.preview) else max(0.0, args.countdown)
    rec_start = t_start + countdown
    last_announce = None
    n_proc, fps_t, proc_fps = 0, time.perf_counter(), 0.0
    last_root = np.zeros(3)
    frame_idx = 0
    seen_person = False
    show = not args.no_preview
    tile = args.tile if show else "off"
    area = None              # halves mode: usable screen area
    view_size = win_pos = None
    split_win = None         # split mode: our NSWindow once it is tiled

    def setup_halves():
        nonlocal area
        area = tiling.screen_area()
        if area is not None and not tiling.tile_terminal(area):
            print("(Could not move the Terminal window - arrange it by hand, "
                  "or use --tile off.)", file=sys.stderr)

    if tile == "halves":
        setup_halves()
    if show:
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO if tile == "split" else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow("webcam2bvh", flags)

    if args.preview:
        print("PREVIEW - nothing is recorded. Set up your shot, then press q in the camera "
              "window (or Ctrl+C here) to quit.", file=sys.stderr)
    elif is_video:
        print(f"Processing {args.video} -> {output}", file=sys.stderr)
    else:
        print(f"Camera {args.camera} open -> {output}. Ctrl+C here (or q in the preview) "
              f"to stop.", file=sys.stderr)

    while not stop["flag"]:
        ok, frame = cap.read()
        if not ok:
            if is_video:
                break
            continue
        now = frame_idx / video_fps if is_video else time.perf_counter()
        frame_idx += 1
        if args.rotate:
            frame = cv2.rotate(frame, ROTATIONS[args.rotate])
        h, w = frame.shape[:2]
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        recording = not args.preview and (is_video or now >= rec_start)

        if not is_video and not args.preview:
            remain = rec_start - now
            if remain > 0:
                sec = math.ceil(remain)
                if sec != last_announce:
                    print(f"Recording starts in {sec}s...", file=sys.stderr)
                    last_announce = sec
            elif last_announce != "rec":
                print("Recording! Ctrl+C (or q in the preview) to stop and export.",
                      file=sys.stderr)
                last_announce = "rec"

        if res.pose_world_landmarks is not None:
            if not seen_person:
                seen_person = True
                print("Person detected.", file=sys.stderr)
            wl = res.pose_world_landmarks.landmark
            il = res.pose_landmarks.landmark
            Pw = np.array([[p.x, p.y, p.z] for p in wl])
            Pi = np.array([[p.x, p.y] for p in il])
            vis = np.array([p.visibility for p in wl])
            if f_world is not None:
                Pw = f_world(Pw, now)
                Pi = f_img(Pi, now)
            Pb = mp_to_bvh(Pw)
            local = solver.solve(Pb, vis, now)
            root = estimate_root(Pb, Pi, w, h, fov_for(args.fov, args.rotate, w, h))
            if root is not None:
                last_root = f_root(root, now) if f_root is not None else root
            if recording:
                samples.append((now, local, last_root.copy()))
                lengths.append(measure_lengths(Pb))

        n_proc += 1
        if time.perf_counter() - fps_t >= 1.0:
            proc_fps = n_proc / (time.perf_counter() - fps_t)
            n_proc, fps_t = 0, time.perf_counter()

        if show:
            view = frame.copy()
            if res.pose_landmarks is not None:
                draw.draw_landmarks(view, res.pose_landmarks, conns)
            view = cv2.flip(view, 1)  # mirror for the performer
            if split_win is not None:
                view = tiling.letterbox(view, tiling.content_size(split_win))
            elif area is not None:
                if view_size is None:
                    view_size, win_pos = tiling.fit_left_half(w, h, area)
                view = cv2.resize(view, view_size, interpolation=cv2.INTER_LINEAR)
            if args.preview:
                label, col = "PREVIEW (not recording) - q to quit", (255, 200, 0)
            elif is_video:
                label, col = "PROCESSING", (0, 200, 255)
            elif now < rec_start:
                label, col = f"GET IN POSITION {math.ceil(rec_start - now)}", (0, 200, 255)
            else:
                label, col = "REC", (0, 0, 255)
            cv2.putText(view, f"{label}  {proc_fps:.0f} fps", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            if res.pose_landmarks is None:
                cv2.putText(view, "NO PERSON", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
            cv2.imshow("webcam2bvh", view)
            if tile == "split" and split_win is None:
                cv2.waitKey(1)  # let the window appear first
                win = tiling.find_window("webcam2bvh")
                if win and tiling.split_left(win):
                    split_win = win
                    print("Split View: click the Terminal window on the right half of the "
                          "screen.", file=sys.stderr)
                else:
                    print("(Split View not available - using left/right halves instead.)",
                          file=sys.stderr)
                    tile = "halves"
                    setup_halves()
            if win_pos is not None:
                cv2.resizeWindow("webcam2bvh", *view_size)
                cv2.moveWindow("webcam2bvh", *win_pos)
                win_pos = None  # only once, so you can still drag it away
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    t_stop = frame_idx / video_fps if is_video else time.perf_counter()
    cap.release()
    pose.close()
    if show:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    if args.preview:
        print("\nPreview closed. Nothing was recorded.", file=sys.stderr)
        return 0

    n_raw = len(samples)
    if not is_video and args.trim_end > 0 and samples:
        cut_after = t_stop - args.trim_end
        keep = [i for i, s in enumerate(samples) if s[0] <= cut_after]
        samples = [samples[i] for i in keep]
        lengths = [lengths[i] for i in keep]
        print(f"\nCut last {args.trim_end:g}s before stopping "
              f"({n_raw - len(samples)} tracked frames).", file=sys.stderr)

    if not samples:
        if n_raw:
            print(f"Take was shorter than --trim-end ({args.trim_end:g}s) - nothing written. "
                  f"Use -t 0 to keep everything.", file=sys.stderr)
        else:
            print("No person was tracked while recording - nothing written.", file=sys.stderr)
        return 1

    bvh, n = to_bvh_text(samples, lengths, args.fps, args.scale, args.in_place)
    with open(output, "w") as fh:
        fh.write(bvh)
    dur = samples[-1][0] - samples[0][0]
    print(f"Stopped. {n} BVH frames ({dur:.2f}s at {args.fps:g} FPS), tracking ran at "
          f"~{len(samples) / max(dur, 1e-6):.0f} fps.\nWrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
