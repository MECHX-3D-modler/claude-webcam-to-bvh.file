"""Run inside Blender: import BVH, compare joint world positions to expected."""
import bpy, json, sys
argv = sys.argv[sys.argv.index("--") + 1:]
bvh_path = argv[0]
expected = json.load(open(argv[1])) if len(argv) > 1 else None

bpy.ops.wm.read_factory_settings(use_empty=True)
res = bpy.ops.import_anim.bvh(filepath=bvh_path, axis_forward='-Z', axis_up='Y',
                              global_scale=1.0, frame_start=1, use_fps_scale=False,
                              update_scene_fps=True, update_scene_duration=True,
                              rotate_mode='NATIVE')
assert res == {'FINISHED'}, res
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
act = arm.animation_data.action
sc = bpy.context.scene
print(f"IMPORT OK: armature={arm.name} bones={len(arm.data.bones)} "
      f"frames={sc.frame_start}-{sc.frame_end} fps={sc.render.fps} fcurves={len(act.fcurves)}")
print("hierarchy:", [(b.name, b.parent.name if b.parent else None) for b in arm.data.bones][:6], "...")

if expected:
    worst = (0, None, None)
    step = max(1, len(expected) // 120)
    for i in range(0, len(expected), step):
        sc.frame_set(i + 1)
        for name, exp in expected[i].items():
            pb = arm.pose.bones[name]
            got = arm.matrix_world @ pb.head
            err = max(abs(got[k] - exp[k]) for k in range(3))
            if err > worst[0]:
                worst = (err, i, name)
    print(f"MAX POSITION ERROR: {worst[0]*1000:.4f} mm (frame {worst[1]}, bone {worst[2]})")
    # Semantic checks: character faces -Y (Blender front), left hand on +X, head up.
    sc.frame_set(1)
    print("Frame1 LeftHand.x>0:", (arm.matrix_world @ arm.pose.bones['LeftHand'].head).x > 0,
          " Head.z>Hips.z:", arm.pose.bones['Head'].head.z > arm.pose.bones['Hips'].head.z,
          " Toes forward (-Y):", (arm.pose.bones['LeftToes'].head - arm.pose.bones['LeftFoot'].head).y < 0)
    sys.exit(0 if worst[0] < 1e-3 else 1)
