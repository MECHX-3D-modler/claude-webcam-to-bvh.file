"""Inside Blender: import webcam BVH and compare limb directions to landmarks."""
import bpy, json, sys
from mathutils import Vector
a = sys.argv[sys.argv.index("--") + 1:]
exp = json.load(open(a[1]))
bpy.ops.wm.read_factory_settings(use_empty=True)
assert bpy.ops.import_anim.bvh(filepath=a[0], axis_forward='-Z', axis_up='Y',
                               update_scene_fps=True, update_scene_duration=True) == {'FINISHED'}
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
sc = bpy.context.scene
print(f"IMPORT OK: bones={len(arm.data.bones)} fps={sc.render.fps} "
      f"frames={len(exp['frames'])} action_range={tuple(arm.animation_data.action.frame_range)}")
rest = {b.name: b.head_local for b in arm.data.bones}
print("rest: LeftUpperArm.x>0", rest["LeftUpperArm"].x > 0, " Head above Hips",
      rest["Head"].z > rest["Hips"].z, " Toes forward(-Y)", (rest["LeftToes"] - rest["LeftFoot"]).y < 0)
H = lambda n: arm.matrix_world @ arm.pose.bones[n].head
worst, worst_at = 0.0, None
lines = 0.0
for i, fr in enumerate(exp["frames"]):
    sc.frame_set(i + 1)
    for b, d in fr["dirs"].items():
        got = H(exp["child"][b]) - H(b)
        ang = got.angle(Vector(d))
        if ang > worst:
            worst, worst_at = ang, (i, b)
    for key, l, r in (("hip_line", "LeftUpperLeg", "RightUpperLeg"),
                      ("sh_line", "LeftUpperArm", "RightUpperArm")):
        lines = max(lines, (H(l) - H(r)).angle(Vector(fr[key])))
import math
print(f"MAX LIMB DIRECTION ERROR: {math.degrees(worst):.4f} deg at {worst_at}")
print(f"MAX HIP/SHOULDER LINE ERROR: {math.degrees(lines):.4f} deg")
sys.exit(0 if math.degrees(worst) < 0.5 and math.degrees(lines) < 0.5 else 1)
