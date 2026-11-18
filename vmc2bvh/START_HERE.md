# START HERE — how to run everything

Copy each line into Terminal and press Return.
(Lines starting with `#` are just notes — you don't type those.)

---

## 1. Open the project (do this every time you open a new Terminal)

    cd ~/vmc2bvh
    source .venv/bin/activate

You'll see `(.venv)` at the start of the line. That means it's ready.

---

## 2. Record yourself with the webcam (no extra apps needed)

**Check the camera first** (nothing is recorded):

    python webcam2bvh.py --preview

Press `q` in the camera window to close it.

**Record a take:**

    python webcam2bvh.py

- You get a 10-second countdown — walk back 2-3 m so your feet are in frame.
- Do your moves.
- Press `Ctrl+C` in Terminal (or `q` in the camera window) to stop.
- The last 5 seconds are cut off automatically (the walk back to the keyboard).
- A file named like `webcam_20261118_101500.bvh` appears in this folder.

**Useful variations:**

    python webcam2bvh.py -o jump.bvh        # pick the file name yourself
    python webcam2bvh.py -c 5 -t 3          # 5 s countdown, cut only the last 3 s
    python webcam2bvh.py --model 0          # faster (if the Mac is laggy), less accurate
    python webcam2bvh.py --no-preview       # no camera window, saves CPU
    python webcam2bvh.py --in-place         # stay on the spot (no walking around)
    python webcam2bvh.py --rotate cw        # camera/Mac on its side (try ccw if upside down)
    python webcam2bvh.py --video clip.mp4 -o clip.bvh   # use a video file instead of the webcam

**Window tips:** the camera window snaps to the left half of the screen.
Click the Terminal window to put it on the right. If Terminal stays full screen
afterwards, press `Ctrl+Cmd+F`.

---

## 3. (Optional) Record from a VMC app instead of the webcam

Start your VMC app sending to port 39539, then:

    python vmc2bvh.py

Press `Ctrl+C` to stop. Writes `vmc_<date>_<time>.bvh`.

---

## 4. Put the motion on your Mixamo character in Blender 3.6

1. Open Blender 3.6 and import your Mixamo character (File > Import > FBX).
2. Click the character's armature.
3. **Mixamo** tab (N panel) > **Create Control Rig** —
   **untick IK Hands and IK Legs** (FK is the accurate one).
4. **Mixamo** tab > **BVH to Mixamo**: choose your `.bvh` file and your
   character's armature, then click **Make Mixamo Source from BVH**.
5. Click the character's armature > **Apply Animation to Control Rig**.
6. Optional: **Thin Keys on Character** (keeps every 3rd key), or tick
   **Hold poses (stepped)** for a choppy "on threes" look.

If the BVH to Mixamo panel is missing: Edit > Preferences > Add-ons,
search "BVH to Mixamo Source", tick the box.

---

## When you're done

    deactivate

(or just close the Terminal window).

More detail on every option is in `README.md`.
