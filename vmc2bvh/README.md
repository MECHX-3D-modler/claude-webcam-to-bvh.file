# vmc2bvh

Record VMC OSC motion (`/VMC/Ext/Bone/Pos`, `/VMC/Ext/Root/Pos`) and export BVH on Ctrl+C.

    source .venv/bin/activate
    python vmc2bvh.py                      # udp 39539, 24 FPS, meters, writes vmc_<timestamp>.bvh
    python vmc2bvh.py -o take1.bvh -f 30   # custom output / FPS
    python vmc2bvh.py -c 5                 # 5 s countdown before recording (default 10, 0 = none)
    python vmc2bvh.py -t 3                 # cut last 3 s before Ctrl+C (default 5, 0 = none)
    python vmc2bvh.py -s 100               # centimeters instead of meters
    python vmc2bvh.py --no-root            # ignore avatar root motion

Blender 3.6: File > Import > Motion Capture (.bvh), defaults (Forward -Z, Up Y, Scale 1.0).

## Tests
    .venv/bin/python tests/make_reference.py ref.bvh ref.json
    "/Applications/Blender 3.6.app/Contents/MacOS/Blender" -b --factory-startup \
        --python tests/blender_check.py -- ref.bvh ref.json
    # live: run vmc2bvh.py, then: .venv/bin/python tests/send_vmc.py 39539 4

# webcam2bvh (no VMC app, no login)

Tracks your body straight from the webcam with Google MediaPipe and writes BVH.

    source .venv/bin/activate
    python webcam2bvh.py --preview       # just look: camera + tracking, nothing recorded (q to quit)
    python webcam2bvh.py                 # 10 s countdown, record, Ctrl+C (or q in preview) -> BVH
    python webcam2bvh.py -c 5 -t 3       # 5 s countdown, cut last 3 s (defaults 10 / 5)
    python webcam2bvh.py --model 0       # faster, less accurate (2 = most accurate)
    python webcam2bvh.py --in-place      # no walking around, only hip height
    python webcam2bvh.py --no-preview    # no camera window (saves CPU)
    python webcam2bvh.py --tile halves   # plain left/right windows instead of Split View
    python webcam2bvh.py --tile off      # don't arrange the windows
    python webcam2bvh.py --rotate cw     # Mac/camera turned on its side (try ccw if upside down)
    python webcam2bvh.py --video clip.mp4 -o clip.bvh   # process a video file

Windows (preview and recording): the camera window goes into macOS Split View on
the left, like the green button's "Tile Window to Left of Screen". Click the
Terminal window on the right to fill that side. Afterwards Terminal may stay full
screen: press Ctrl+Cmd+F (or the green button) to bring it back.

Tips: stand 2-3 m back so your whole body (feet included) is in frame, good
lighting, plain background. Body/arms/legs/head only - no fingers or face.
First run: macOS asks to let Terminal use the camera - allow it.

Tests: `.venv/bin/python tests/synth_webcam.py wc.bvh wc.json` then
`Blender -b --python tests/blender_check_webcam.py -- wc.bvh wc.json`

# Using the BVH on a Mixamo character (Mixamo Rig add-on)

The Mixamo add-on only copies bones named like `mixamorig:LeftArm`, so a raw BVH
does nothing. Use the included add-on `blender/bvh_to_mixamo.py`
(installed into Blender 3.6; enable "BVH to Mixamo Source" in Preferences > Add-ons).

1. Import your Mixamo character. Select its armature.
2. Mixamo tab > Create Control Rig - **untick IK Hands and IK Legs** (FK copies the
   motion exactly; the add-on's IK mode aims elbows/knees approximately, 10-20 deg off).
3. Mixamo tab > BVH to Mixamo: pick the .bvh and your character's armature,
   click "Make Mixamo Source from BVH" (it also fills in the Mixamo Source field).
4. Select the character armature > Apply Animation to Control Rig.
5. Optional, fewer keys: set "Key every" (default 3) in the BVH to Mixamo panel, then
   click "Thin Keys on Character" (Apply Animation keys every frame; this keeps every 3rd).
   Tick "Hold poses (stepped)" for a choppy animated-on-threes look instead of smooth motion.

Test: `Blender -b --factory-startup --python tests/blender_check_mixamo.py -- character.fbx take.bvh fk`
