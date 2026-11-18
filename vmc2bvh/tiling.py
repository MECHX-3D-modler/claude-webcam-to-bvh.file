"""
Window arrangement for webcam2bvh on macOS.

split  - real macOS Split View: the camera window does the green button's
         "Tile Window to Left of Screen" by itself; you click Terminal on the
         right half. No extra permissions (it only touches its own window).
halves - plain windows: camera on the left half, Terminal moved to the right half.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import subprocess
import sys

import numpy as np

TITLE_BAR = 28  # height of a macOS window title bar, in points


# ---------------------------------------------------------------------------
# halves mode
def screen_area():
    """Usable area of the main screen (left, top, width, height), top-left origin,
    without the menu bar and Dock. None if it can't be found (e.g. not a Mac)."""
    if sys.platform != "darwin":
        return None
    js = ('ObjC.import("AppKit"); var s = $.NSScreen.screens.objectAtIndex(0);'
          'var f = s.frame, v = s.visibleFrame;'
          '[v.origin.x, f.size.height - v.origin.y - v.size.height,'
          ' v.size.width, v.size.height].join(" ")')
    try:
        out = subprocess.run(["osascript", "-l", "JavaScript", "-e", js],
                             capture_output=True, text=True, timeout=5).stdout.split()
        area = tuple(int(float(v)) for v in out)
        return area if len(area) == 4 and area[2] > 0 and area[3] > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def tile_terminal(area):
    """Move the Terminal (or iTerm2) window this script runs in to the right half."""
    left, top, w, h = area
    box = f"{{{left + w // 2}, {top}, {left + w}, {top + h}}}"
    prog = os.environ.get("TERM_PROGRAM", "")
    if prog == "Apple_Terminal":
        try:
            tty = os.ttyname(sys.stdin.fileno())
        except OSError:
            return False
        script = f'''tell application "Terminal"
  repeat with w in windows
    repeat with t in tabs of w
      if tty of t is "{tty}" then
        set bounds of w to {box}
        return "ok"
      end if
    end repeat
  end repeat
end tell'''
    elif prog == "iTerm.app":
        script = f'tell application "iTerm2" to set bounds of current window to {box}'
    else:
        return False
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True,
                           timeout=5)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def fit_left_half(img_w, img_h, area):
    """Preview size and window position that fill the left half of the screen."""
    left, top, w, h = area
    max_w, max_h = w // 2, h - TITLE_BAR
    k = min(max_w / img_w, max_h / img_h)
    view_w, view_h = int(img_w * k), int(img_h * k)
    # cv2.moveWindow on macOS measures from the top-left of the usable area.
    return (view_w, view_h), ((max_w - view_w) // 2, 0)


# ---------------------------------------------------------------------------
# split mode: talk to our own NSWindow through the Objective-C runtime
class _CGRect(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double),
                ("w", ctypes.c_double), ("h", ctypes.c_double)]


_objc = None


def _runtime():
    global _objc
    if _objc is None:
        _objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
        _objc.objc_getClass.restype = ctypes.c_void_p
        _objc.objc_getClass.argtypes = [ctypes.c_char_p]
        _objc.sel_registerName.restype = ctypes.c_void_p
        _objc.sel_registerName.argtypes = [ctypes.c_char_p]
    return _objc


def _msg(obj, sel, *args, restype=ctypes.c_void_p, argtypes=(), stret=False):
    objc = _runtime()
    # On Intel Macs, methods returning big structs (NSRect) use objc_msgSend_stret.
    fn = "objc_msgSend_stret" if stret and platform.machine() == "x86_64" else "objc_msgSend"
    f = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)((fn, objc))
    return f(obj, objc.sel_registerName(sel.encode()), *args)


def _responds(obj, sel):
    return bool(_msg(obj, "respondsToSelector:", _runtime().sel_registerName(sel.encode()),
                     restype=ctypes.c_bool, argtypes=(ctypes.c_void_p,)))


def find_window(title):
    """Our own NSWindow with this title (the OpenCV window), or None."""
    if sys.platform != "darwin":
        return None
    try:
        app = _msg(_runtime().objc_getClass(b"NSApplication"), "sharedApplication")
        wins = _msg(app, "windows")
        for i in range(_msg(wins, "count", restype=ctypes.c_ulong)):
            w = _msg(wins, "objectAtIndex:", i, argtypes=(ctypes.c_ulong,))
            t = _msg(_msg(w, "title"), "UTF8String", restype=ctypes.c_char_p)
            if t is not None and t.decode("utf-8", "replace") == title:
                return w
    except (OSError, AttributeError):
        pass
    return None


def split_left(win):
    """Put our window on the left of a new Split View (like the green button's
    "Tile Window to Left of Screen"). Returns True if macOS accepted it."""
    if not win or not _responds(win, "_tileLeft:"):
        return False
    FULLSCREEN_PRIMARY, ALLOWS_TILING = 1 << 7, 1 << 11
    cur = _msg(win, "collectionBehavior", restype=ctypes.c_ulong)
    _msg(win, "setCollectionBehavior:", cur | FULLSCREEN_PRIMARY | ALLOWS_TILING,
         restype=None, argtypes=(ctypes.c_ulong,))
    if _responds(win, "_canEnterTileMode") and not _msg(win, "_canEnterTileMode",
                                                        restype=ctypes.c_bool):
        return False
    _msg(win, "_tileLeft:", None, restype=None, argtypes=(ctypes.c_void_p,))
    return True


def content_size(win):
    """(width, height) of the window's drawing area, in points."""
    view = _msg(win, "contentView")
    r = _msg(view, "frame", restype=_CGRect, stret=True)
    return int(r.w), int(r.h)


def letterbox(img, size):
    """Scale img to fit inside size (w, h) without stretching; black bars fill the rest."""
    import cv2
    tw, th = size
    h, w = img.shape[:2]
    if tw <= 0 or th <= 0 or (tw, th) == (w, h):
        return img
    k = min(tw / w, th / h)
    nw, nh = max(1, int(w * k)), max(1, int(h * k))
    out = np.zeros((th, tw, 3), np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out
