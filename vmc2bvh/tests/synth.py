"""Synthetic VMC skeleton + animation, in raw Unity coordinates (LH, Y-up)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Local rest positions (relative to parent, Unity space, meters).
# Unity character faces +Z; its left side is -X.
REST = {
    "Hips": (0.0, 0.95, 0.0),
    "Spine": (0.0, 0.10, 0.0),
    "Chest": (0.0, 0.15, 0.01),
    "UpperChest": (0.0, 0.12, 0.0),
    "Neck": (0.0, 0.15, -0.01),
    "Head": (0.0, 0.10, 0.02),
    "LeftShoulder": (-0.03, 0.10, 0.0),
    "LeftUpperArm": (-0.10, 0.0, 0.0),
    "LeftLowerArm": (-0.26, 0.0, 0.0),
    "LeftHand": (-0.24, 0.0, 0.0),
    "LeftIndexProximal": (-0.08, 0.0, 0.02),
    "RightShoulder": (0.03, 0.10, 0.0),
    "RightUpperArm": (0.10, 0.0, 0.0),
    "RightLowerArm": (0.26, 0.0, 0.0),
    "RightHand": (0.24, 0.0, 0.0),
    "LeftUpperLeg": (-0.09, -0.05, 0.0),
    "LeftLowerLeg": (0.0, -0.42, 0.0),
    "LeftFoot": (0.0, -0.40, 0.0),
    "LeftToes": (0.0, -0.06, 0.12),
    "RightUpperLeg": (0.09, -0.05, 0.0),
    "RightLowerLeg": (0.0, -0.42, 0.0),
    "RightFoot": (0.0, -0.40, 0.0),
    "RightToes": (0.0, -0.06, 0.12),
}
PARENT = {
    "Hips": None, "Spine": "Hips", "Chest": "Spine", "UpperChest": "Chest",
    "Neck": "UpperChest", "Head": "Neck",
    "LeftShoulder": "UpperChest", "LeftUpperArm": "LeftShoulder",
    "LeftLowerArm": "LeftUpperArm", "LeftHand": "LeftLowerArm",
    "LeftIndexProximal": "LeftHand",
    "RightShoulder": "UpperChest", "RightUpperArm": "RightShoulder",
    "RightLowerArm": "RightUpperArm", "RightHand": "RightLowerArm",
    "LeftUpperLeg": "Hips", "LeftLowerLeg": "LeftUpperLeg", "LeftFoot": "LeftLowerLeg",
    "LeftToes": "LeftFoot",
    "RightUpperLeg": "Hips", "RightLowerLeg": "RightUpperLeg", "RightFoot": "RightLowerLeg",
    "RightToes": "RightFoot",
}
NAMES = list(REST)


def pose(t: float):
    """Return ({bone: (pos, quat_xyzw)}, root(pos, quat)) at time t, Unity space."""
    bones = {}
    for k, n in enumerate(NAMES):
        # Smooth, all-axis rotations with distinct phases; large enough to
        # exercise Euler wrapping (Hips spins > 360 degrees over time).
        a = np.array([
            40 * np.sin(1.3 * t + k),
            35 * np.sin(0.9 * t + 2 * k),
            30 * np.sin(1.7 * t + 0.5 * k),
        ])
        if n == "Hips":
            a = np.array([15 * np.sin(t), 90 * t, 10 * np.cos(2 * t)])
        q = Rotation.from_euler("xyz", a, degrees=True).as_quat()
        bones[n] = (tuple(REST[n]), tuple(q))
    root_pos = (0.5 * np.sin(0.5 * t), 0.05 * np.sin(3 * t), 0.8 * t)
    root_q = Rotation.from_euler("y", 20 * np.sin(0.7 * t), degrees=True).as_quat()
    return bones, (root_pos, tuple(root_q))


def world_positions(bones, root):
    """FK in Unity space: world = Root * ... * T(local) * R(local)."""
    rpos, rq = root
    world = {}
    def get(n):
        if n in world:
            return world[n]
        p, q = bones[n]
        if PARENT[n] is None:
            P, Q = np.array(rpos), Rotation.from_quat(rq)
        else:
            P, Q = get(PARENT[n])
        res = (P + Q.apply(p), Q * Rotation.from_quat(q))
        world[n] = res
        return res
    for n in bones:
        get(n)
    return {n: v[0] for n, v in world.items()}
