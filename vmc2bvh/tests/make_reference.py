"""Build a BVH directly from synthetic frames + dump expected world positions
(converted to Blender's Z-up space) for verification inside Blender."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vmc2bvh import build_bvh
from synth import pose, world_positions

FPS, N = 60, 600   # 10 s (Hips spins 900 deg -> exercises unwrapping)
out_bvh, out_json = sys.argv[1], sys.argv[2]
frames, expected = [], []
for i in range(N):
    bones, root = pose(i / FPS)
    frames.append((bones, root))
    w = world_positions(bones, root)
    # Unity (x,y,z) -> BVH (-x,y,z) -> Blender importer (fwd -Z, up Y): (bx, -bz, by)
    expected.append({n: [-p[0], -p[2], p[1]] for n, p in w.items()})
open(out_bvh, "w").write(build_bvh(frames, FPS, 1.0, use_root=True))
json.dump(expected, open(out_json, "w"))
print("wrote", out_bvh, N, "frames")
