"""Synthetic MediaPipe landmarks from known poses -> webcam2bvh solver -> BVH,
plus expected limb directions (Blender space) for blender_check_webcam.py."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial.transform import Rotation as R
import webcam2bvh as W

rng = np.random.default_rng(7)
LEN = dict(hip_half=0.10, sh_half=0.18, torso=0.50, neck=0.20, upper_arm=0.28,
           forearm=0.25, thigh=0.42, shin=0.40, foot=0.15)
OFF = W.rest_offsets(LEN)
SEGS = {  # child joint landmark pairs whose direction we compare
    "LeftUpperArm": (W.L_SH, W.L_EL), "LeftLowerArm": (W.L_EL, W.L_WR),
    "RightUpperArm": (W.R_SH, W.R_EL), "RightLowerArm": (W.R_EL, W.R_WR),
    "LeftUpperLeg": (W.L_HIP, W.L_KNEE), "LeftLowerLeg": (W.L_KNEE, W.L_ANK),
    "LeftFoot": (W.L_ANK, W.L_FOOT),
    "RightUpperLeg": (W.R_HIP, W.R_KNEE), "RightLowerLeg": (W.R_KNEE, W.R_ANK),
    "RightFoot": (W.R_ANK, W.R_FOOT),
}
CHILD = {"LeftUpperArm": "LeftLowerArm", "LeftLowerArm": "LeftHand",
         "RightUpperArm": "RightLowerArm", "RightLowerArm": "RightHand",
         "LeftUpperLeg": "LeftLowerLeg", "LeftLowerLeg": "LeftFoot", "LeftFoot": "LeftToes",
         "RightUpperLeg": "RightLowerLeg", "RightLowerLeg": "RightFoot",
         "RightFoot": "RightToes"}
E = lambda seq, a: R.from_euler(seq, a, degrees=True)

def pose_landmarks(t):
    """World rotations/positions of a smoothly animated, anatomically plausible pose
    (BVH space) -> 33 landmarks in BVH space."""
    wr, pos = {}, {}
    wr["Hips"] = E("yxz", [60 * np.sin(0.4 * t), 10 * np.sin(t), 8 * np.cos(0.7 * t)])
    wr["Chest"] = wr["Hips"] * E("xyz", [15 * np.sin(0.8 * t), 25 * np.sin(0.5 * t), 0])
    wr["Spine"] = W._slerp_half(wr["Hips"], wr["Chest"])
    wr["Head"] = wr["Chest"] * E("xyz", [20 * np.sin(1.1 * t), 40 * np.sin(0.6 * t), 10 * np.sin(t)])
    wr["Neck"] = W._slerp_half(wr["Chest"], wr["Head"])
    for side, s in (("Left", 1), ("Right", -1)):
        ph = 0 if s > 0 else 1.3
        wr[f"{side}UpperArm"] = wr["Chest"] * E("zyx", [s * 60 * np.sin(0.9 * t + ph),
                                                         -s * 40 * np.sin(0.7 * t + ph),
                                                         50 * np.sin(1.2 * t + ph)])
        bend = 70 + 50 * np.sin(1.3 * t + ph)              # 20..120 deg, forward
        wr[f"{side}LowerArm"] = wr[f"{side}UpperArm"] * E("y", -s * bend) * \
            E("x", 60 * np.sin(0.8 * t + ph))              # + forearm twist
        wr[f"{side}Hand"] = wr[f"{side}LowerArm"] * E("z", 35 * np.sin(1.7 * t + ph))
        wr[f"{side}UpperLeg"] = wr["Hips"] * E("xzy", [-40 * np.sin(0.9 * t + ph) - 10,
                                                        s * 15 * np.sin(0.5 * t),
                                                        s * 20 * np.sin(0.6 * t + ph)])
        knee = 45 + 40 * np.sin(1.1 * t + ph)               # 5..85 deg
        wr[f"{side}LowerLeg"] = wr[f"{side}UpperLeg"] * E("x", knee)
        wr[f"{side}Foot"] = wr[f"{side}LowerLeg"] * E("x", 10 * np.sin(1.4 * t))
        wr[f"{side}Toes"] = wr[f"{side}Foot"]
    pos["Hips"] = np.array([0.3 * np.sin(0.3 * t), 0.9 + 0.05 * np.sin(2 * t), 0.2 * t])
    for b in W.BONES[1:]:
        p = W.PARENT[b]
        pos[b] = pos[p] + wr[p].apply(OFF[b])
    P = np.zeros((33, 3))
    for side, s in (("Left", 1), ("Right", -1)):
        L = side == "Left"
        P[W.L_SH if L else W.R_SH] = pos[f"{side}UpperArm"]
        P[W.L_EL if L else W.R_EL] = pos[f"{side}LowerArm"]
        P[W.L_WR if L else W.R_WR] = pos[f"{side}Hand"]
        h = wr[f"{side}Hand"]
        P[W.L_INDEX if L else W.R_INDEX] = pos[f"{side}Hand"] + h.apply([s * .08, 0, .025])
        P[W.L_PINKY if L else W.R_PINKY] = pos[f"{side}Hand"] + h.apply([s * .08, 0, -.025])
        P[W.L_HIP if L else W.R_HIP] = pos[f"{side}UpperLeg"]
        P[W.L_KNEE if L else W.R_KNEE] = pos[f"{side}LowerLeg"]
        P[W.L_ANK if L else W.R_ANK] = pos[f"{side}Foot"]
        P[W.L_FOOT if L else W.R_FOOT] = pos[f"{side}Toes"]
        P[W.L_HEEL if L else W.R_HEEL] = pos[f"{side}Foot"] + wr[f"{side}Foot"].apply([0, -.06, -.05])
        P[W.L_EAR if L else W.R_EAR] = pos["Head"] + wr["Head"].apply([s * .075, 0, 0])
    P[W.NOSE] = pos["Head"] + wr["Head"].apply([0, -.03, .10])
    return P, pos["Hips"]

if __name__ == "__main__":
    out_bvh, out_json = sys.argv[1], sys.argv[2]
    fps, n = 24, 240
    solver, samples, lengths, expected = W.PoseSolver(), [], [], []
    for i in range(n):
        t = i / fps
        P, hip = pose_landmarks(t)
        # round-trip through MediaPipe's coordinate convention, hip-centred like MP world landmarks
        P_mp = W.mp_to_bvh(P - hip)
        Pb = W.mp_to_bvh(P_mp)
        samples.append((t, solver.solve(Pb, None, t), hip))
        lengths.append(W.measure_lengths(Pb))
        bl = lambda v: [v[0], -v[2], v[1]]                 # BVH -> Blender (fwd -Z, up Y)
        expected.append({
            "dirs": {b: bl(P[j] - P[i0]) for b, (i0, j) in SEGS.items()},
            "hip_line": bl(P[W.L_HIP] - P[W.R_HIP]), "sh_line": bl(P[W.L_SH] - P[W.R_SH]),
            "ear_line": bl(P[W.L_EAR] - P[W.R_EAR]),
        })
    bvh, nf = W.to_bvh_text(samples, lengths, fps, 1.0, in_place=False)
    open(out_bvh, "w").write(bvh)
    json.dump({"child": CHILD, "frames": expected}, open(out_json, "w"))
    print("wrote", out_bvh, nf, "frames")

    # Root estimate check: project a person at known (x, depth) with a pinhole camera.
    w, h, fov = 640, 480, 60.0
    f = 0.5 * w / np.tan(np.radians(fov) / 2)
    worst = 0
    for x_true, d_true in [(0.0, 2.0), (0.5, 2.5), (-0.4, 1.5), (0.2, 3.5)]:
        P, hip = pose_landmarks(0.0)
        P = P - hip
        P[:, 1] -= P[[W.L_ANK, W.R_ANK, W.L_HEEL, W.R_HEEL, W.L_FOOT, W.R_FOOT], 1].min() - 0.9  # irrelevant for x/z
        cam = W.mp_to_bvh(P - 0.5 * (P[W.L_HIP] + P[W.R_HIP]))     # MP axes, hip-centred
        cam = cam + np.array([x_true, 0, d_true])                   # camera space (MP axes)
        img = np.column_stack([(f * cam[:, 0] / cam[:, 2] + w / 2) / w,
                               (f * cam[:, 1] / cam[:, 2] + h / 2) / h])
        est = W.estimate_root(W.mp_to_bvh(cam - cam[[W.L_HIP, W.R_HIP]].mean(0)), img, w, h, fov)
        worst = max(worst, abs(est[0] - x_true), abs(-est[2] - d_true))
    print(f"root estimate max error (x/depth): {worst * 100:.1f} cm")
