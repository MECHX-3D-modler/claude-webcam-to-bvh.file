"""Headless end-to-end: Mixamo character -> Create Control Rig -> BVH to Mixamo
-> Apply Animation to Control Rig -> compare limb directions with the BVH.
usage: Blender -b --python blender_check_mixamo.py -- char.fbx anim.bvh [ik|fk]"""
import bpy, sys, os, math, importlib.util
a = sys.argv[sys.argv.index("--") + 1:]
fbx, bvh_path, mode = a[0], a[1], (a[2] if len(a) > 2 else "ik")
spec = importlib.util.spec_from_file_location("bvh_to_mixamo",
        os.path.join(os.path.dirname(__file__), "..", "blender", "bvh_to_mixamo.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.preferences.addon_enable(module="mixamo_rig")
B.register()
C = bpy.context
bpy.ops.import_scene.fbx(filepath=fbx)
char = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
char.animation_data_clear()                       # make sure motion comes from our BVH
C.view_layer.objects.active = char; char.select_set(True)
r = bpy.ops.mr.make_rig(bake_anim=False, ik_arms=(mode == "ik"), ik_legs=(mode == "ik"))
print("make_rig:", r, "control rig:", "mr_control_rig" in char.data.keys())

src, mapped, n = B.build_source(C, bvh_path, char)
print(f"source: {src.name} mapped={mapped} frames={n} mix_source={C.scene.mix_source_armature.name}")
C.view_layer.objects.active = char
r = bpy.ops.mr.import_anim_to_rig()
print("apply animation:", r, "char action:", char.animation_data.action.name if char.animation_data and char.animation_data.action else None)

# reference: the BVH itself
before = set(bpy.data.objects)
bpy.ops.import_anim.bvh(filepath=bvh_path, axis_forward="-Z", axis_up="Y", update_scene_fps=False)
ref = [o for o in bpy.data.objects if o not in before][0]
P = "mixamorig:"
SEG = {  # mixamo bone chain (head->head)  vs  bvh joints
    ("LeftArm", "LeftForeArm"): ("LeftUpperArm", "LeftLowerArm"),
    ("LeftForeArm", "LeftHand"): ("LeftLowerArm", "LeftHand"),
    ("RightArm", "RightForeArm"): ("RightUpperArm", "RightLowerArm"),
    ("RightForeArm", "RightHand"): ("RightLowerArm", "RightHand"),
    ("LeftUpLeg", "LeftLeg"): ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLeg", "LeftFoot"): ("LeftLowerLeg", "LeftFoot"),
    ("RightUpLeg", "RightLeg"): ("RightUpperLeg", "RightLowerLeg"),
    ("RightLeg", "RightFoot"): ("RightLowerLeg", "RightFoot"),
    ("LeftFoot", "LeftToeBase"): ("LeftFoot", "LeftToes"),
    ("Neck", "Head"): ("Neck", "Head"),
    ("LeftUpLeg", "RightUpLeg"): ("LeftUpperLeg", "RightUpperLeg"),
    ("LeftArm", "RightArm"): ("LeftUpperArm", "RightUpperArm"),
}
H = lambda o, n: o.matrix_world @ o.pose.bones[n].head
worst = {}
moved = 0.0
f0, f1 = C.scene.frame_start, C.scene.frame_end
C.scene.frame_set(f0); h0 = H(char, P + "LeftHand").copy()
for f in range(f0, f1 + 1, max(1, (f1 - f0) // 60)):
    C.scene.frame_set(f)
    moved = max(moved, (H(char, P + "LeftHand") - h0).length)
    for (ma, mb), (ba, bb) in SEG.items():
        if ba not in ref.pose.bones or bb not in ref.pose.bones:
            continue
        ang = math.degrees((H(char, P + mb) - H(char, P + ma)).angle(H(ref, bb) - H(ref, ba)))
        worst[ma + ">" + mb] = max(worst.get(ma + ">" + mb, 0), ang)
print(f"character LeftHand moved up to {moved:.2f} m (proves it animates)")
for k, v in worst.items():
    print(f"  {k:24s} max direction error {v:6.2f} deg")
print("WORST:", f"{max(worst.values()):.2f} deg")
