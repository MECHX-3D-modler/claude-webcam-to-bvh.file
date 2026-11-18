bl_info = {
    "name": "BVH to Mixamo Source",
    "author": "vmc2bvh",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "3D View > Sidebar (N) > Mixamo > BVH to Mixamo",
    "description": "Turn a vmc2bvh/webcam2bvh BVH into a Mixamo-named source armature "
                   "for the Mixamo Rig add-on's 'Apply Animation to Control Rig'",
    "category": "Animation",
}

"""
Why this is needed
------------------
The Mixamo Rig add-on retargets by *bone name* (mixamorig:Hips, mixamorig:LeftArm,
...) and copies each bone's *world orientation*. A BVH has different names
(Hips, LeftUpperArm, ...) and different bone axes/rolls than a Mixamo rig, so
'Apply Animation' either finds no bones (nothing moves) or twists limbs.

What this does
--------------
1. Imports the BVH.
2. Builds a new armature with exactly the target's Mixamo deform bones
   (same names, rest pose, bone rolls, object transform).
3. Bakes the BVH motion onto it as rotation-from-rest: for every mapped bone
       world_new = bvh_delta_from_rest * align * target_rest
   where `align` swings the target's rest bone onto the BVH's T-pose direction
   (so A-pose characters work too). Bone rolls never matter.
4. Scales hip motion by leg length (performer -> character).
5. Sets it as the Mixamo add-on's Source armature.
Then press 'Apply Animation to Control Rig' in the Mixamo panel as usual.
"""

import math

import bpy
from mathutils import Matrix, Quaternion, Vector

# BVH (Unity humanoid names) -> Mixamo names (without prefix)
UNITY_TO_MIXAMO = {
    "Hips": "Hips", "Spine": "Spine", "Chest": "Spine1", "UpperChest": "Spine2",
    "Neck": "Neck", "Head": "Head",
}
for _s in ("Left", "Right"):
    UNITY_TO_MIXAMO.update({
        f"{_s}Shoulder": f"{_s}Shoulder", f"{_s}UpperArm": f"{_s}Arm",
        f"{_s}LowerArm": f"{_s}ForeArm", f"{_s}Hand": f"{_s}Hand",
        f"{_s}UpperLeg": f"{_s}UpLeg", f"{_s}LowerLeg": f"{_s}Leg",
        f"{_s}Foot": f"{_s}Foot", f"{_s}Toes": f"{_s}ToeBase",
    })
    for _uf, _mf in (("Thumb", "Thumb"), ("Index", "Index"), ("Middle", "Middle"),
                     ("Ring", "Ring"), ("Little", "Pinky")):
        for _uj, _mj in (("Proximal", "1"), ("Intermediate", "2"), ("Distal", "3")):
            UNITY_TO_MIXAMO[f"{_s}{_uf}{_uj}"] = f"{_s}Hand{_mf}{_mj}"

# Direction-defining child for each bone (Mixamo names) used for A/T-pose alignment
DIR_CHILD = {"Spine": "Spine1", "Spine1": "Spine2", "Neck": "Head"}
for _s in ("Left", "Right"):
    DIR_CHILD.update({f"{_s}Arm": f"{_s}ForeArm", f"{_s}ForeArm": f"{_s}Hand",
                      f"{_s}UpLeg": f"{_s}Leg", f"{_s}Leg": f"{_s}Foot",
                      f"{_s}Foot": f"{_s}ToeBase", f"{_s}Shoulder": f"{_s}Arm"})
    for _f in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        DIR_CHILD[f"{_s}Hand{_f}1"] = f"{_s}Hand{_f}2"
        DIR_CHILD[f"{_s}Hand{_f}2"] = f"{_s}Hand{_f}3"


def mixamo_prefix(arm_obj):
    for b in arm_obj.data.bones:
        if b.name.endswith("Hips") and (":" in b.name or b.name == "Hips"):
            return b.name[:-4]
    for b in arm_obj.data.bones:
        if b.name.endswith("Hips"):
            return b.name[:-4]
    return None


def _rot(m):
    return m.to_3x3().normalized()


def _swing(a, b):
    """Shortest rotation taking direction a onto direction b (3x3)."""
    a, b = a.normalized(), b.normalized()
    return a.rotation_difference(b).to_matrix()


def _leg_length(heads, up, low, foot):
    if not all(k in heads for k in (up, low, foot)):
        return None
    return (heads[low] - heads[up]).length + (heads[foot] - heads[low]).length


def thin_action(action, step, hold=False, f0=None, f1=None):
    """Keep only keys on frames f0, f0+step, f0+2*step, ... (plus the last frame).
    hold=True makes the pose jump between keys (stepped / 'animated on threes').
    Returns (keys_before, keys_after)."""
    if action is None or step < 1:
        return 0, 0
    if f0 is None or f1 is None:
        r = action.frame_range
        f0, f1 = int(round(r[0])), int(round(r[1]))
    before = after = 0
    for fc in list(action.fcurves):
        n = len(fc.keyframe_points)
        if n == 0:
            continue
        co = [0.0] * (2 * n)
        fc.keyframe_points.foreach_get("co", co)
        kept = []
        for i in range(n):
            fr, v = co[2 * i], co[2 * i + 1]
            k = int(round(fr)) - f0
            if (k % step == 0) or int(round(fr)) >= f1 or i == n - 1:
                kept += [fr, v]
        before += n
        after += len(kept) // 2
        dp, idx = fc.data_path, fc.array_index
        grp = fc.group.name if fc.group else None
        action.fcurves.remove(fc)
        new = action.fcurves.new(dp, index=idx, action_group=grp) if grp \
            else action.fcurves.new(dp, index=idx)
        new.keyframe_points.add(len(kept) // 2)
        new.keyframe_points.foreach_set("co", kept)
        if hold:
            for kp in new.keyframe_points:
                kp.interpolation = "CONSTANT"
        new.update()
    return before, after


def build_source(context, bvh_path, target, fps_from_bvh=True, step=1, hold=False):
    scn = context.scene
    prefix = mixamo_prefix(target)
    if prefix is None:
        raise RuntimeError("Target armature has no Mixamo 'Hips' bone (e.g. mixamorig:Hips).")

    # --- 1. import BVH --------------------------------------------------------
    if context.object and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=bvh_path, axis_forward="-Z", axis_up="Y",
                            global_scale=1.0, frame_start=1, use_fps_scale=False,
                            update_scene_fps=fps_from_bvh, update_scene_duration=True,
                            rotate_mode="NATIVE")
    bvh = [o for o in bpy.data.objects if o not in before and o.type == "ARMATURE"][0]
    act = bvh.animation_data.action
    f0, f1 = int(round(act.frame_range[0])), int(round(act.frame_range[1]))

    # map: mixamo short name -> bvh bone name
    src_of = {}
    for b in bvh.data.bones:
        m = UNITY_TO_MIXAMO.get(b.name)
        if m:
            src_of[m] = b.name
    if "Hips" not in src_of:
        raise RuntimeError("BVH has no Hips bone.")

    # --- 2. new armature = target's Mixamo deform bones only -----------------
    tbones = [b for b in target.data.bones if b.name.startswith(prefix)
              and not b.name.startswith("Ctrl_")]
    tnames = {b.name for b in tbones}
    arm_data = bpy.data.armatures.new(target.name + "_BVH_Source")
    src = bpy.data.objects.new(target.name + "_BVH_Source", arm_data)
    context.collection.objects.link(src)
    src.matrix_world = target.matrix_world.copy()
    for o in context.selected_objects:
        o.select_set(False)
    context.view_layer.objects.active = src
    src.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for b in tbones:
        eb = arm_data.edit_bones.new(b.name)
        eb.head = (0, 0, 0)
        eb.tail = (0, max(b.length, 1e-3), 0)
        eb.matrix = b.matrix_local.copy()
    for b in tbones:
        if b.parent and b.parent.name in tnames:
            arm_data.edit_bones[b.name].parent = arm_data.edit_bones[b.parent.name]
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- 3. rest data ---------------------------------------------------------
    Wt, Wb = src.matrix_world, bvh.matrix_world
    Rt = _rot(Wt)
    short = lambda n: n[len(prefix):]
    t_rest_w = {short(b.name): _rot(Wt @ b.matrix_local) for b in src.data.bones}
    t_head_w = {short(b.name): Wt @ b.head_local for b in src.data.bones}
    b_rest_w = {m: _rot(Wb @ bvh.data.bones[n].matrix_local) for m, n in src_of.items()}
    b_head_w = {m: Wb @ bvh.data.bones[n].head_local for m, n in src_of.items()}

    align = {}
    for m in src_of:
        c = DIR_CHILD.get(m)
        if c and c in src_of and c in t_head_w and m in t_head_w:
            tdir, bdir = t_head_w[c] - t_head_w[m], b_head_w[c] - b_head_w[m]
            if tdir.length > 1e-6 and bdir.length > 1e-6:
                align[m] = _swing(tdir, bdir)
                continue
        align[m] = Matrix.Identity(3)

    # hip motion scale: character leg length / performer leg length
    ratio = 1.0
    for s in ("Left", "Right"):
        lt = _leg_length(t_head_w, f"{s}UpLeg", f"{s}Leg", f"{s}Foot")
        lb = _leg_length(b_head_w, f"{s}UpLeg", f"{s}Leg", f"{s}Foot")
        if lt and lb:
            ratio = lt / lb
            break

    # --- 4. bake --------------------------------------------------------------
    order = [b for b in src.data.bones]          # parents come before children
    for pb in src.pose.bones:
        pb.rotation_mode = "QUATERNION"
    keys = {b.name: [] for b in order}
    hips_key = []
    hips_name = prefix + "Hips"
    t_hips_rest = t_head_w["Hips"]
    bvh_hips_rest = None
    prev_q = {}
    for f in range(f0, f1 + 1):
        scn.frame_set(f)
        bpose = {m: Wb @ bvh.pose.bones[n].matrix for m, n in src_of.items()}
        if bvh_hips_rest is None:
            bvh_hips_rest = bpose["Hips"].to_translation()
        arm_pose = {}
        for b in order:
            m = short(b.name)
            par = b.parent if (b.parent and b.parent.name in tnames) else None
            rel = (par.matrix_local.inverted() @ b.matrix_local) if par else b.matrix_local
            parent_pose = arm_pose[par.name] if par else Matrix.Identity(4)
            base = parent_pose @ rel                        # pose with identity basis
            if m in src_of:
                delta = _rot(bpose[m]) @ b_rest_w[m].inverted()
                want_w = delta @ align[m] @ t_rest_w[m]
                want_rot = Rt.inverted() @ want_w           # armature space
                basis_rot = _rot(base).inverted() @ want_rot
            else:
                basis_rot = Matrix.Identity(3)              # follow parent rigidly
            basis = basis_rot.to_4x4()
            if b.name == hips_name:
                bh = bpose["Hips"].to_translation()
                want_head_w = Vector((t_hips_rest.x + ratio * (bh.x - bvh_hips_rest.x),
                                      t_hips_rest.y + ratio * (bh.y - bvh_hips_rest.y),
                                      ratio * bh.z))
                want_head_a = Wt.inverted() @ want_head_w
                loc = _rot(base).inverted() @ (want_head_a - base.to_translation())
                basis = Matrix.Translation(loc) @ basis
                hips_key.append(loc.copy())
            arm_pose[b.name] = base @ basis
            q = basis_rot.to_quaternion()
            if b.name in prev_q and prev_q[b.name].dot(q) < 0:
                q.negate()
            prev_q[b.name] = q
            keys[b.name].append(q)

    action = bpy.data.actions.new(src.name + "_Action")
    src.animation_data_create()
    src.animation_data.action = action
    frames = list(range(f0, f1 + 1))
    for b in order:
        dp = f'pose.bones["{b.name}"].rotation_quaternion'
        for i in range(4):
            fc = action.fcurves.new(dp, index=i, action_group=b.name)
            fc.keyframe_points.add(len(frames))
            co = []
            for fr, q in zip(frames, keys[b.name]):
                co += [fr, q[i]]
            fc.keyframe_points.foreach_set("co", co)
            fc.update()
    dp = f'pose.bones["{hips_name}"].location'
    for i in range(3):
        fc = action.fcurves.new(dp, index=i, action_group=hips_name)
        fc.keyframe_points.add(len(frames))
        co = []
        for fr, v in zip(frames, hips_key):
            co += [fr, v[i]]
        fc.keyframe_points.foreach_set("co", co)
        fc.update()
    if step > 1:
        thin_action(action, step, hold, f0, f1)

    # --- 5. tidy up -----------------------------------------------------------
    bpy.data.objects.remove(bvh, do_unlink=True)
    if act.users == 0:
        bpy.data.actions.remove(act)
    src.show_in_front = True
    if hasattr(scn, "mix_source_armature"):
        scn.mix_source_armature = src
    scn.frame_start, scn.frame_end = f0, f1
    scn.frame_set(f0)
    # leave the target active so the Mixamo panel's Apply button is ready
    for o in context.selected_objects:
        o.select_set(False)
    context.view_layer.objects.active = target
    target.select_set(True)
    mapped = len([m for m in src_of if prefix + m in tnames])
    return src, mapped, f1 - f0 + 1


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _is_armature(self, obj):
    return obj.type == "ARMATURE"


class BVHMIX_OT_build(bpy.types.Operator):
    """Import the BVH and create a Mixamo-named source armature for 'Apply Animation'"""
    bl_idname = "bvhmix.build_source"
    bl_label = "Make Mixamo Source from BVH"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene
        path = bpy.path.abspath(s.bvhmix_path)
        if not path or not path.lower().endswith(".bvh"):
            self.report({"ERROR"}, "Choose a .bvh file first")
            return {"CANCELLED"}
        if s.bvhmix_target is None:
            self.report({"ERROR"}, "Choose your character's Mixamo armature")
            return {"CANCELLED"}
        try:
            src, mapped, n = build_source(context, path, s.bvhmix_target,
                                          step=s.bvhmix_step, hold=s.bvhmix_hold)
        except Exception as e:  # show the reason in the UI
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created {src.name}: {mapped} bones, {n} frames. "
                              f"Now press 'Apply Animation to Control Rig'.")
        return {"FINISHED"}


class BVHMIX_OT_thin(bpy.types.Operator):
    """Keep a key only every N frames on the character's animation
    (use after 'Apply Animation to Control Rig', which keys every frame)"""
    bl_idname = "bvhmix.thin_keys"
    bl_label = "Thin Keys on Character"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene
        obj = s.bvhmix_target or context.object
        ad = obj.animation_data if obj else None
        if not ad or not ad.action:
            self.report({"ERROR"}, "Character has no animation yet - "
                                   "press 'Apply Animation to Control Rig' first")
            return {"CANCELLED"}
        if s.bvhmix_step < 2:
            self.report({"ERROR"}, "Set 'Key every' to 2 or more")
            return {"CANCELLED"}
        b, a = thin_action(ad.action, s.bvhmix_step, s.bvhmix_hold,
                           s.frame_start, s.frame_end)
        self.report({"INFO"}, f"{obj.name}: {b} keys -> {a} keys "
                              f"(one every {s.bvhmix_step} frames)")
        return {"FINISHED"}


class BVHMIX_PT_panel(bpy.types.Panel):
    bl_label = "BVH to Mixamo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mixamo"

    def draw(self, context):
        s = context.scene
        col = self.layout.column(align=True)
        col.prop(s, "bvhmix_path", text="BVH")
        col.prop(s, "bvhmix_target", text="Character")
        col.prop(s, "bvhmix_step")
        col.prop(s, "bvhmix_hold")
        col.separator()
        col.operator(BVHMIX_OT_build.bl_idname, icon="ARMATURE_DATA")
        if hasattr(s, "mix_source_armature") and s.mix_source_armature:
            col.label(text=f"Source: {s.mix_source_armature.name}", icon="CHECKMARK")
        col.separator()
        col.label(text="After Apply Animation:")
        col.operator(BVHMIX_OT_thin.bl_idname, icon="KEYFRAME")


def register():
    bpy.types.Scene.bvhmix_path = bpy.props.StringProperty(
        name="BVH file", subtype="FILE_PATH", default="")
    bpy.types.Scene.bvhmix_target = bpy.props.PointerProperty(
        name="Character armature", type=bpy.types.Object, poll=_is_armature)
    bpy.types.Scene.bvhmix_step = bpy.props.IntProperty(
        name="Key every", description="Keep a keyframe only every N frames (1 = every frame)",
        default=3, min=1, max=60)
    bpy.types.Scene.bvhmix_hold = bpy.props.BoolProperty(
        name="Hold poses (stepped)", default=False,
        description="Pose jumps from key to key instead of moving smoothly between them")
    bpy.utils.register_class(BVHMIX_OT_build)
    bpy.utils.register_class(BVHMIX_OT_thin)
    bpy.utils.register_class(BVHMIX_PT_panel)


def unregister():
    bpy.utils.unregister_class(BVHMIX_PT_panel)
    bpy.utils.unregister_class(BVHMIX_OT_thin)
    bpy.utils.unregister_class(BVHMIX_OT_build)
    del bpy.types.Scene.bvhmix_hold
    del bpy.types.Scene.bvhmix_step
    del bpy.types.Scene.bvhmix_target
    del bpy.types.Scene.bvhmix_path


if __name__ == "__main__":
    register()
