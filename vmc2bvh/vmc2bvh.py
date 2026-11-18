#!/usr/bin/env python3
"""
vmc2bvh - record VMC (Virtual Motion Capture) OSC bone data and export BVH.

Listens for /VMC/Ext/Bone/Pos and /VMC/Ext/Root/Pos on UDP (default 39539),
samples the latest pose at a fixed FPS, and writes a .bvh file on Ctrl+C.

Coordinate conversion
---------------------
VMC sends Unity data: left-handed, Y-up, +Z forward, meters, quaternions as
(x, y, z, w), and bone transforms that are *local* (relative to the parent
bone in the Unity humanoid hierarchy).

BVH is right-handed, Y-up. We convert by mirroring the X axis (S = diag(-1,1,1)):

    position   p' = S p              -> (-x,  y,  z)
    rotation   R' = S R S            -> quaternion (x, -y, -z, w)

This is a change of basis, not a mirror of the performer: the character's left
hand stays its left hand, and the character keeps facing +Z, which Blender's
BVH importer (forward = -Z, up = Y) maps to Blender's -Y "front" direction.

Because Unity composes child = parent * T(localPos) * R(localRot), exactly the
same as BVH composes joints (OFFSET then channel rotation), local VMC
transforms map one-to-one onto BVH OFFSETs and joint rotations.
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Unity HumanBodyBones hierarchy (child -> parent). Order defines BVH order.
# ---------------------------------------------------------------------------
HUMANOID_PARENTS: "OrderedDict[str, str | None]" = OrderedDict()


def _add(name, parent):
    HUMANOID_PARENTS[name] = parent


_add("Hips", None)
_add("Spine", "Hips")
_add("Chest", "Spine")
_add("UpperChest", "Chest")
_add("Neck", "UpperChest")
_add("Head", "Neck")
_add("LeftEye", "Head")
_add("RightEye", "Head")
_add("Jaw", "Head")
for side in ("Left", "Right"):
    _add(f"{side}Shoulder", "UpperChest")
    _add(f"{side}UpperArm", f"{side}Shoulder")
    _add(f"{side}LowerArm", f"{side}UpperArm")
    _add(f"{side}Hand", f"{side}LowerArm")
    for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
        _add(f"{side}{finger}Proximal", f"{side}Hand")
        _add(f"{side}{finger}Intermediate", f"{side}{finger}Proximal")
        _add(f"{side}{finger}Distal", f"{side}{finger}Intermediate")
for side in ("Left", "Right"):
    _add(f"{side}UpperLeg", "Hips")
    _add(f"{side}LowerLeg", f"{side}UpperLeg")
    _add(f"{side}Foot", f"{side}LowerLeg")
    _add(f"{side}Toes", f"{side}Foot")

ROT_ORDER = "ZXY"  # BVH channel order: Zrotation Xrotation Yrotation


# ---------------------------------------------------------------------------
# Coordinate conversion Unity (LH, Y-up) -> BVH (RH, Y-up)
# ---------------------------------------------------------------------------
def unity_pos_to_bvh(p):
    p = np.asarray(p, dtype=float)
    return p * np.array([-1.0, 1.0, 1.0])


def unity_quat_to_bvh(q_xyzw):
    q = np.asarray(q_xyzw, dtype=float)
    return q * np.array([1.0, -1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class VMCRecorder:
    def __init__(self, fps: float, use_root: bool, verbose: bool = False):
        self.fps = fps
        self.use_root = use_root
        self.verbose = verbose
        self._lock = threading.Lock()
        self._bones: dict[str, tuple] = {}   # latest local (pos, quat) per bone
        self._root = None                    # latest (pos, quat) of root object
        self._unknown: set[str] = set()
        self.frames: list[tuple[dict, tuple | None]] = []
        self.msg_count = 0
        self.frame_msg_counts: list[int] = []
        self._stop = threading.Event()

    # -- OSC handlers -------------------------------------------------------
    def on_bone(self, address, *args):
        if len(args) < 8:
            return
        name = str(args[0])
        if name not in HUMANOID_PARENTS:
            if name not in self._unknown:
                self._unknown.add(name)
                if self.verbose:
                    print(f"[warn] ignoring non-humanoid bone '{name}'", file=sys.stderr)
            return
        pos = tuple(float(v) for v in args[1:4])
        quat = tuple(float(v) for v in args[4:8])
        with self._lock:
            self._bones[name] = (pos, quat)
            self.msg_count += 1

    def on_root(self, address, *args):
        # /VMC/Ext/Root/Pos name px py pz qx qy qz qw [sx sy sz ox oy oz]
        if len(args) < 8:
            return
        pos = tuple(float(v) for v in args[1:4])
        quat = tuple(float(v) for v in args[4:8])
        with self._lock:
            self._root = (pos, quat)
            self.msg_count += 1

    # -- sampling -----------------------------------------------------------
    def sample_loop(self):
        period = 1.0 / self.fps
        next_t = time.perf_counter()
        waiting_msg_shown = False
        while not self._stop.is_set():
            with self._lock:
                have_hips = "Hips" in self._bones
                if have_hips:
                    self.frames.append((dict(self._bones), self._root))
                    self.frame_msg_counts.append(self.msg_count)
            if not have_hips and not waiting_msg_shown:
                print("Waiting for VMC data (Hips bone)...", file=sys.stderr)
                waiting_msg_shown = True
            elif have_hips and len(self.frames) == 1:
                print("Receiving data - recording. Press Ctrl+C to stop and export.",
                      file=sys.stderr)
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                # fell behind (e.g. system sleep); resync instead of bursting
                next_t = time.perf_counter()

    def stop(self):
        self._stop.set()

    def trimmed_frames(self, end_cut_frames: int = 0):
        """Drop the last `end_cut_frames` frames (time spent walking back to
        press Ctrl+C), then drop trailing frames sampled after the last OSC
        message arrived (sender stopped early), keeping one frame of that pose.
        Returns (frames, n_cut, n_idle)."""
        n = max(0, len(self.frames) - max(0, end_cut_frames))
        n_cut = len(self.frames) - n
        m = n
        while m > 1 and self.frame_msg_counts[m - 2] == self.frame_msg_counts[n - 1]:
            m -= 1
        return self.frames[:m], n_cut, n - m


# ---------------------------------------------------------------------------
# BVH construction
# ---------------------------------------------------------------------------
def _effective_parent(name, present):
    p = HUMANOID_PARENTS[name]
    while p is not None and p not in present:
        p = HUMANOID_PARENTS[p]
    return p


def _euler_alternative(e):
    """Second Tait-Bryan solution describing the same rotation (degrees)."""
    a, b, c = e
    return np.array([a + 180.0, 180.0 - b, c + 180.0])


def _wrap_near(angle, ref):
    return angle - 360.0 * np.round((angle - ref) / 360.0)


def continuous_eulers(quats_xyzw: np.ndarray) -> np.ndarray:
    """Quaternions (N,4) -> BVH Euler angles (N,3) in degrees, ROT_ORDER intrinsic,
    choosing per frame the equivalent solution closest to the previous frame so
    curves have no 360-degree jumps or gimbal-branch flips."""
    eul = Rotation.from_quat(quats_xyzw).as_euler(ROT_ORDER, degrees=True)
    out = np.empty_like(eul)
    out[0] = eul[0]
    for i in range(1, len(eul)):
        prev = out[i - 1]
        best = None
        for cand in (eul[i], _euler_alternative(eul[i])):
            cand = _wrap_near(cand, prev)
            d = np.sum(np.abs(cand - prev))
            if best is None or d < best[0]:
                best = (d, cand)
        out[i] = best[1]
    return out


def build_bvh(frames, fps: float, scale: float, use_root: bool,
              unity_input: bool = True) -> str:
    """frames: list of ({bone: (local_pos, local_quat_xyzw)}, root_or_None).
    unity_input=True: data is in Unity space (VMC) and gets converted.
    unity_input=False: data is already in BVH space (right-handed, Y-up)."""
    conv_p = unity_pos_to_bvh if unity_input else (lambda v: np.asarray(v, dtype=float))
    conv_q = unity_quat_to_bvh if unity_input else (lambda v: np.asarray(v, dtype=float))
    if not frames:
        raise ValueError("no frames recorded")

    present = set()
    for bones, _ in frames:
        present.update(bones.keys())
    joints = [n for n in HUMANOID_PARENTS if n in present]
    parent = {n: _effective_parent(n, present) for n in joints}
    children = {n: [c for c in joints if parent[c] == n] for n in joints}
    n_frames = len(frames)

    # Fill per-bone tracks; bones that appear late get back-filled with their
    # first value, gaps are held at the last value.
    pos_track = {n: np.zeros((n_frames, 3)) for n in joints}
    quat_track = {n: np.zeros((n_frames, 4)) for n in joints}
    for n in joints:
        first = next(b[n] for b, _ in frames if n in b)
        last = first
        for i, (bones, _) in enumerate(frames):
            last = bones.get(n, last)
            pos_track[n][i] = conv_p(last[0]) * scale
            quat_track[n][i] = conv_q(last[1])

    # Rest offsets: median of local positions (VMC sends them every frame;
    # they are constant for a rigid skeleton, median rejects any glitches).
    offsets = {n: np.median(pos_track[n], axis=0) for n in joints}

    # Root motion: Hips world = RootObject * Hips_local  (all in BVH space).
    hips_pos = pos_track["Hips"].copy()
    hips_rot = Rotation.from_quat(quat_track["Hips"])
    if use_root and any(r is not None for _, r in frames):
        first_root = next(r for _, r in frames if r is not None)
        last_root = first_root
        rp = np.zeros((n_frames, 3))
        rq = np.zeros((n_frames, 4))
        for i, (_, r) in enumerate(frames):
            last_root = r if r is not None else last_root
            rp[i] = conv_p(last_root[0]) * scale
            rq[i] = conv_q(last_root[1])
        root_rot = Rotation.from_quat(rq)
        hips_pos = rp + root_rot.apply(hips_pos)
        hips_rot = root_rot * hips_rot
    quat_track["Hips"] = hips_rot.as_quat()

    eulers = {n: continuous_eulers(quat_track[n]) for n in joints}
    axis_names = {"X": "Xrotation", "Y": "Yrotation", "Z": "Zrotation"}
    rot_channels = " ".join(axis_names[a] for a in ROT_ORDER)

    def fmt(v):
        return " ".join(f"{x:.6f}" for x in v)

    lines = ["HIERARCHY"]

    def write_joint(n, depth):
        ind = "\t" * depth
        if parent[n] is None:
            lines.append(f"ROOT {n}")
            # Root position channels carry absolute position; OFFSET is zero.
            lines.append(f"{ind}{{")
            lines.append(f"{ind}\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition {rot_channels}")
        else:
            lines.append(f"{ind}JOINT {n}")
            lines.append(f"{ind}{{")
            lines.append(f"{ind}\tOFFSET {fmt(offsets[n])}")
            lines.append(f"{ind}\tCHANNELS 3 {rot_channels}")
        if children[n]:
            for c in children[n]:
                write_joint(c, depth + 1)
        else:
            # End site: continue along the bone direction (half its length).
            off = offsets[n] if parent[n] is not None else np.zeros(3)
            length = np.linalg.norm(off)
            end = off * 0.5 if length > 1e-6 else np.array([0.0, 0.05 * scale, 0.0])
            if n == "Head":
                end = np.array([0.0, max(length, 0.1 * scale), 0.0])
            lines.append(f"{ind}\tEnd Site")
            lines.append(f"{ind}\t{{")
            lines.append(f"{ind}\t\tOFFSET {fmt(end)}")
            lines.append(f"{ind}\t}}")
        lines.append(f"{ind}}}")

    if "Hips" not in joints:
        raise ValueError("Hips bone never received")
    write_joint("Hips", 0)

    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {1.0 / fps:.8f}")
    for i in range(n_frames):
        vals = []
        for n in joints:
            if parent[n] is None:
                vals.extend(hips_pos[i])
            vals.extend(eulers[n][i])
        lines.append(" ".join(f"{v:.6f}" for v in vals))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Record VMC OSC motion and export BVH on Ctrl+C.")
    ap.add_argument("-o", "--output", default=None,
                    help="output .bvh path (default: vmc_YYYYmmdd_HHMMSS.bvh)")
    ap.add_argument("--host", default="0.0.0.0", help="listen address (default 0.0.0.0)")
    ap.add_argument("-p", "--port", type=int, default=39539, help="UDP port (default 39539)")
    ap.add_argument("-f", "--fps", type=float, default=24.0, help="sampling FPS (default 24)")
    ap.add_argument("-s", "--scale", type=float, default=1.0,
                    help="unit scale applied to positions (1.0 = meters, 100 = centimeters)")
    ap.add_argument("--no-root", action="store_true",
                    help="ignore /VMC/Ext/Root/Pos (keep Hips motion relative to avatar root)")
    ap.add_argument("-d", "--duration", type=float, default=None,
                    help="stop automatically after N seconds of recording")
    ap.add_argument("-c", "--countdown", type=float, default=10.0,
                    help="seconds to wait before recording starts, to get into position "
                         "(default 10, 0 = start immediately)")
    ap.add_argument("-t", "--trim-end", type=float, default=5.0,
                    help="seconds cut from the end of the take, to drop walking back to "
                         "press Ctrl+C (default 5, 0 = keep everything)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.fps <= 0:
        ap.error("--fps must be > 0")

    output = args.output or time.strftime("vmc_%Y%m%d_%H%M%S.bvh")
    rec = VMCRecorder(args.fps, use_root=not args.no_root, verbose=args.verbose)

    disp = Dispatcher()
    disp.map("/VMC/Ext/Bone/Pos", rec.on_bone)
    disp.map("/VMC/Ext/Root/Pos", rec.on_root)
    server = BlockingOSCUDPServer((args.host, args.port), disp)
    server.timeout = 0.2

    srv_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                                  daemon=True)
    smp_thread = threading.Thread(target=rec.sample_loop, daemon=True)

    stop_evt = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_evt.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())

    print(f"Listening for VMC on udp://{args.host}:{args.port} at {args.fps:g} FPS -> {output}",
          file=sys.stderr)
    srv_thread.start()

    try:
        remaining = max(0.0, args.countdown)
        while remaining > 0 and not stop_evt.is_set():
            print(f"Recording starts in {math.ceil(remaining):d}s...", file=sys.stderr)
            step = remaining - (math.ceil(remaining) - 1)
            stop_evt.wait(step)
            remaining -= step
        if not stop_evt.is_set():
            smp_thread.start()
        while not stop_evt.is_set():
            stop_evt.wait(0.25)
            if args.duration and len(rec.frames) >= args.duration * args.fps:
                break
    finally:
        rec.stop()
        if smp_thread.is_alive():
            smp_thread.join()
        server.shutdown()
        server.server_close()

    frames, n_cut, n_idle = rec.trimmed_frames(int(round(args.trim_end * args.fps)))
    if n_cut:
        print(f"\nCut last {n_cut} frames ({n_cut / args.fps:.2f}s) before Ctrl+C.",
              file=sys.stderr)
    if n_idle:
        print(f"Trimmed {n_idle} idle trailing frames (no new data).", file=sys.stderr)
    n = len(frames)
    print(f"\nStopped. {n} frames ({n / args.fps:.2f}s), {rec.msg_count} OSC messages.",
          file=sys.stderr)
    if n == 0:
        if n_cut:
            print(f"Take was shorter than --trim-end ({args.trim_end:g}s) - nothing written. "
                  f"Use -t 0 to keep everything.", file=sys.stderr)
        else:
            print("No data received - nothing written.", file=sys.stderr)
        return 1
    bvh = build_bvh(frames, args.fps, args.scale, use_root=not args.no_root)
    with open(output, "w") as fh:
        fh.write(bvh)
    print(f"Wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
