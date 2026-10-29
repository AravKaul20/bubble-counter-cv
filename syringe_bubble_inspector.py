#!/usr/bin/env python3
"""
syringe_bubble_inspector.py

Measures residual air in a filled syringe from a camera image and checks it against
the EGR 101 project spec:
  * no single air pocket larger than 25 uL
  * total air less than 250 uL per 10 mL of liquid (i.e. 25 uL per mL)

Output is a VOLUME in microliters plus a PASS / MARGINAL / FAIL verdict, not just a
bubble count, so it can serve as an objective bench test for the de-airing device.

Imaging assumptions (the method is built around these):
  1. Backlit: a diffuse light panel BEHIND the syringe, camera in front. Bubbles then
     appear as dark rings on a bright field (light refracts away at the air surface).
  2. Fixed geometry: syringe held in a jig at a fixed distance, so one calibration
     covers every test.
  3. You tell the program which side of the image the air end (the tip) is on.
     Free air rises there and is measured as "headspace".
  4. Fixed exposure and focus (auto-exposure makes the threshold drift between frames).

Workflow
  1) One-time setup (opens windows):
        python syringe_bubble_inspector.py setup --camera 0 --air-end top
     - drag a box just INSIDE the barrel walls (the liquid to inspect), ENTER
     - optionally box the printed numbers so they are ignored, ESC when done
     - click two graduation marks, then type their spacing ("5 ml" or "20 mm")
       and the barrel inner diameter in mm
     This writes bubble_config.json. Every value can also be given on the command
     line instead (see --help), which skips the corresponding window.

  2) Inspect:
        live:    python syringe_bubble_inspector.py run --camera 0
        images:  python syringe_bubble_inspector.py run --image a.png b.png --out results
        video:   python syringe_bubble_inspector.py run --video test.mp4
     Live keys: s = save snapshot + CSV row   d = debug masks   +/- = sensitivity
                p = pause   w = write sensitivity to config   q / Esc = quit

Requires: opencv-python, numpy
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields

import cv2
import numpy as np

SPEC_MAX_POCKET_UL = 25.0     # largest allowed single air pocket
SPEC_TOTAL_UL_PER_ML = 25.0   # 250 uL per 10 mL

# Rotation that puts the air end of the barrel at the top (row 0) of the working image.
TO_WORK = {
    "top": None,
    "bottom": cv2.ROTATE_180,
    "left": cv2.ROTATE_90_CLOCKWISE,
    "right": cv2.ROTATE_90_COUNTERCLOCKWISE,
}


# --------------------------------------------------------------------------- config

@dataclass
class Config:
    roi: list                     # [x, y, w, h] camera pixels, just inside the barrel walls
    mm_per_px: float              # image scale at the syringe (from graduation marks)
    barrel_id_mm: float           # barrel inner diameter
    air_end: str = "top"          # side of the image where the tip / air end is
    exclusions: list = field(default_factory=list)  # [[x, y, w, h], ...] ignored regions
    fill_ml: float = 10.0         # liquid volume in the syringe (sets total-air limit)
    lateral_mag: float = 1.33     # sideways magnification of the liquid-filled barrel (~ n)
    dark_thresh: float = 0.15     # fraction darker than local background to count as "dark"
    polarity: str = "dark"        # "dark" for backlight; "bright" / "both" for other lighting
    min_diam_mm: float = 0.5      # smallest bubble reported (0.5 mm sphere = 0.065 uL)
    meniscus_frac: float = 0.7    # dark band at least this wide (of ROI) = liquid surface
    margin: float = 0.2           # MARGINAL if within this fraction below a limit

    @property
    def ul_per_axial_px(self):
        """Air volume represented by one pixel of headspace height (1 mm^3 = 1 uL)."""
        return math.pi * (self.barrel_id_mm / 2) ** 2 * self.mm_per_px

    @property
    def total_limit_ul(self):
        return SPEC_TOTAL_UL_PER_ML * self.fill_ml

    def px(self, mm):
        return mm / self.mm_per_px

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        known = {fl.name for fl in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def odd(n):
    return max(1, int(round(n)) | 1)


# ------------------------------------------------------------------ geometry helper

class View:
    """The barrel ROI, cropped and rotated so the air end is at the top (row 0)."""

    def __init__(self, frame_shape, cfg):
        H0, W0 = frame_shape[:2]
        x, y, w, h = [int(round(v)) for v in cfg.roi]
        x, y = max(0, x), max(0, y)
        self.x0, self.y0 = x, y
        self.w, self.h = min(w, W0 - x), min(h, H0 - y)
        self.air_end = cfg.air_end
        self.rot = TO_WORK[cfg.air_end]

    def work(self, full_img):
        crop = full_img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        return crop if self.rot is None else cv2.rotate(crop, self.rot)

    def to_frame(self, pts):
        """Map (col, row) points in the working image back to camera pixels."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        u, v = pts[:, 0], pts[:, 1]
        w, h = self.w, self.h
        if self.air_end == "top":
            x, y = u, v
        elif self.air_end == "bottom":
            x, y = w - 1 - u, h - 1 - v
        elif self.air_end == "left":
            x, y = v, h - 1 - u
        else:  # right
            x, y = w - 1 - v, u
        return np.stack([x + self.x0, y + self.y0], axis=1)


# --------------------------------------------------------------------- measurement

@dataclass
class Bubble:
    d_axial_mm: float
    d_lateral_mm: float
    volume_ul: float
    regular: bool            # round and solid; False = odd shape, still counted as air
    contour: np.ndarray      # outline in camera pixels
    center: tuple


@dataclass
class Result:
    bubbles: list
    headspace_ul: float
    headspace_poly: object   # outline of the headspace in camera pixels, or None
    debug: dict

    @property
    def pockets(self):
        v = [b.volume_ul for b in self.bubbles]
        if self.headspace_ul > 0:
            v.append(self.headspace_ul)
        return v

    @property
    def max_pocket_ul(self):
        return max(self.pockets, default=0.0)

    @property
    def total_ul(self):
        return float(sum(self.pockets))


def verdict(max_pocket_ul, total_ul, cfg):
    limit_total = cfg.total_limit_ul
    if max_pocket_ul > SPEC_MAX_POCKET_UL or total_ul > limit_total:
        return "FAIL"
    k = 1.0 - cfg.margin
    if max_pocket_ul > k * SPEC_MAX_POCKET_UL or total_ul > k * limit_total:
        return "MARGINAL"
    return "PASS"


def running_median(v, win):
    win = odd(min(win, 2 * len(v) - 1))
    pad = win // 2
    vp = np.pad(v, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(vp, win), axis=1)


def background_contrast(g, cfg):
    """
    Separable background model: bg(x, y) = column_profile(x) * row_gain(y).
    The barrel's sideways shading, vertical glare streaks and wall shadows run the
    full length of the barrel, so they live in the column profile and cancel out.
    Slow brightness changes along the barrel live in the row gain. Bubbles, ticks and
    the meniscus are local, so medians ignore them and they show up as contrast.
    No kernel sizes to tune, and it works for bubbles of any size.
    """
    col = np.maximum(np.median(g, axis=0), 1.0)
    row = np.median(g / col[None, :], axis=1)
    row = running_median(row, cfg.px(4.0))   # longer than any meniscus band or tick
    bg = np.outer(row, col)
    return (bg - g) / np.maximum(bg, 1.0)


def find_headspace(dark, cfg):
    """
    The liquid surface shows up as a dark band spanning (nearly) the whole barrel width,
    which no tick mark or bubble does. Air is everything between the air end and that
    surface. Volume is integrated assuming the surface is axisymmetric, so a curved
    meniscus is handled correctly:  V = sum over columns of h(x) * pi * |x| * dx.
    """
    H, W = dark.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    cands = [i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] >= cfg.meniscus_frac * W]
    if not cands:
        return None
    i = min(cands, key=lambda j: stats[j, cv2.CC_STAT_TOP])   # closest to the air end
    comp = lab == i

    level = np.full(W, np.nan)
    bottom = np.full(W, np.nan)
    for xc in range(W):
        ys = np.flatnonzero(comp[:, xc])
        if ys.size == 0:
            continue
        gaps = np.flatnonzero(np.diff(ys) > 1)
        y1 = ys[gaps[0]] if gaps.size else ys[-1]       # first dark run from the air end
        level[xc] = (ys[0] + y1) / 2.0
        bottom[xc] = y1
    good = ~np.isnan(level)
    if good.sum() < 0.5 * W:
        return None
    xs = np.arange(W)
    level = np.interp(xs, xs[good], level[good])
    bottom = np.interp(xs, xs[good], bottom[good])

    R = cfg.barrel_id_mm / 2.0
    r_mm = np.abs((xs + 0.5 - W / 2.0) / (W / 2.0)) * R
    dx_mm = R / (W / 2.0)
    h_mm = (level + 0.5) * cfg.mm_per_px
    volume = float(np.sum(h_mm * math.pi * r_mm * dx_mm))

    air = (np.arange(H)[:, None] <= bottom[None, :] + 2) | comp
    return volume, level, air


def fill_holes(mask, max_hole_area):
    """
    Fill the bright centre of each dark bubble ring. The ROI's left/right edges are
    treated as closed (they are the barrel walls), so a bubble stuck to the wall and
    cut by the ROI still gets filled. Regions connected to the top/bottom are liquid.
    """
    H, W = mask.shape
    padded = np.zeros((H, W + 2), np.uint8)
    padded[:, 1:-1] = mask
    padded[:, 0] = padded[:, -1] = 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(padded), connectivity=4)
    top = stats[:, cv2.CC_STAT_TOP]
    bot = top + stats[:, cv2.CC_STAT_HEIGHT]
    holes = [i for i in range(1, n)
             if top[i] > 0 and bot[i] < H and stats[i, cv2.CC_STAT_AREA] <= max_hole_area]
    out = padded.copy()
    if holes:
        out[np.isin(lab, holes)] = 255
    return out[:, 1:-1]


def split_touching(blob, lateral_mag, min_r):
    """
    Split a blob made of touching bubbles. The blob is first squeezed sideways by the
    barrel magnification so each bubble is round again; each bubble is then one peak
    of the distance transform, and pixels go to the nearest peak (weighted by radius).
    """
    h, w = blob.shape
    w2 = max(1, int(round(w / lateral_mag)))
    b2 = cv2.resize(blob, (w2, h), interpolation=cv2.INTER_NEAREST)
    dt = cv2.distanceTransform(b2, cv2.DIST_L2, 5)
    peaks = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= max(2.0, min_r))
    ys, xs = np.nonzero(peaks)
    kept = []
    for j in np.argsort(-dt[ys, xs]):
        y, x, r = ys[j], xs[j], dt[ys[j], xs[j]]
        if all(math.hypot(y - ky, x - kx) > kr for ky, kx, kr in kept):
            kept.append((y, x, r))
    if len(kept) <= 1:
        return [blob]
    yy, xx = np.nonzero(b2)
    cost = np.stack([np.hypot(yy - ky, xx - kx) - kr for ky, kx, kr in kept])
    owner = np.argmin(cost, axis=0)
    pieces = []
    for j in range(len(kept)):
        m2 = np.zeros_like(b2)
        m2[yy[owner == j], xx[owner == j]] = 255
        m = cv2.resize(m2, (w, h), interpolation=cv2.INTER_NEAREST) & blob
        if cv2.countNonZero(m):
            pieces.append(m)
    return pieces


def solidity(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, None
    c = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    return (cv2.contourArea(c) / hull_area if hull_area > 0 else 0.0), c


def measure_piece(piece, ox, oy, cfg, view):
    """
    Size from second moments (robust to ragged edges). Axial size is true size; lateral
    size is divided by the barrel magnification. Volume of a spheroid whose axis lies
    along the barrel: V = pi/6 * d_axial * d_lateral^2  (mm^3 = uL).
    """
    m = cv2.moments(piece, binaryImage=True)
    if m["m00"] < 3:
        return None
    d_ax_mm = 4.0 * math.sqrt(m["mu02"] / m["m00"]) * cfg.mm_per_px
    d_lat_mm = 4.0 * math.sqrt(m["mu20"] / m["m00"]) * cfg.mm_per_px / cfg.lateral_mag
    vol = math.pi / 6.0 * d_ax_mm * d_lat_mm ** 2
    sol, c = solidity(piece)
    aspect = d_lat_mm / max(d_ax_mm, 1e-6)
    regular = sol >= 0.85 and 0.6 <= aspect <= 1.6
    contour = view.to_frame(c[:, 0, :] + [ox, oy]).astype(np.int32)
    cx, cy = m["m10"] / m["m00"] + ox, m["m01"] / m["m00"] + oy
    center = tuple(int(v) for v in view.to_frame([[cx, cy]])[0])
    return Bubble(d_ax_mm, d_lat_mm, vol, regular, contour, center)


def measure(frame, cfg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    view = View(gray.shape, cfg)
    g = cv2.GaussianBlur(view.work(gray).astype(np.float32), (3, 3), 0)
    H, W = g.shape

    excl_full = np.zeros(gray.shape, np.uint8)
    for x, y, w, h in cfg.exclusions:
        excl_full[int(y):int(y + h), int(x):int(x + w)] = 255
    excl = view.work(excl_full)

    # 1. Contrast against the background model
    c = background_contrast(g, cfg)
    feat = {"dark": c, "bright": -c, "both": np.abs(c)}[cfg.polarity]
    dark = ((feat > cfg.dark_thresh) * 255).astype(np.uint8)
    # 3x3 cross (odd size): even-sized kernels shift the mask by a pixel in OpenCV
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

    # 2. Headspace: air between the air end and the liquid surface
    head = find_headspace(dark, cfg)
    headspace_ul, head_poly = 0.0, None
    work_dark = dark.copy()
    if head is not None:
        headspace_ul, level, air = head
        work_dark[air] = 0
        poly = [(0, 0), (W - 1, 0)] + [(x, level[x]) for x in range(W - 1, -1, -1)]
        head_poly = view.to_frame(poly).astype(np.int32)

    # 3. Bubbles: fill ring centres, then an opening with a disk the size of the
    #    smallest reportable bubble deletes everything thinner than that (graduation
    #    ticks, number strokes, noise) while leaving filled bubbles untouched.
    filled = fill_holes(work_dark, max_hole_area=(0.8 * W) ** 2)
    k = odd(cfg.px(cfg.min_diam_mm))
    solid = cv2.morphologyEx(filled, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    solid[excl > 0] = 0

    bubbles = []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        if y == 0 and y + h >= H:      # runs the full length: wall/edge artifact
            continue
        blob = ((lab[y:y + h, x:x + w] == i) * 255).astype(np.uint8)
        sol, _ = solidity(blob)
        pieces = split_touching(blob, cfg.lateral_mag, k / 2.0) if sol < 0.9 else [blob]
        for p in pieces:
            b = measure_piece(p, x, y, cfg, view)
            if b is not None:
                bubbles.append(b)
    bubbles.sort(key=lambda b: -b.volume_ul)

    debug = {"gray": view.work(gray), "dark": dark, "solid": solid}
    return Result(bubbles, headspace_ul, head_poly, debug)


# ------------------------------------------------------------------------- drawing

VERDICT_COLOR = {"PASS": (60, 200, 60), "MARGINAL": (0, 190, 255), "FAIL": (40, 40, 230)}


def annotate(frame, cfg, res, shown_max=None, shown_total=None, note=""):
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    x, y, w, h = [int(v) for v in cfg.roi]
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), 1)
    for ex, ey, ew, eh in cfg.exclusions:
        ex, ey, ew, eh = int(ex), int(ey), int(ew), int(eh)
        cv2.rectangle(out, (ex, ey), (ex + ew, ey + eh), (140, 140, 140), 1)
        cv2.putText(out, "ignored", (ex + 2, ey + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (140, 140, 140), 1, cv2.LINE_AA)

    if res.headspace_poly is not None:
        layer = out.copy()
        cv2.fillPoly(layer, [res.headspace_poly], (40, 40, 230))
        out = cv2.addWeighted(layer, 0.35, out, 0.65, 0)
        cv2.polylines(out, [res.headspace_poly], True, (40, 40, 230), 1, cv2.LINE_AA)

    for b in res.bubbles:
        if b.volume_ul > SPEC_MAX_POCKET_UL:
            col = (40, 40, 230)
        elif b.volume_ul > 0.5 * SPEC_MAX_POCKET_UL:
            col = (0, 190, 255)
        else:
            col = (60, 200, 60)
        if not b.regular:
            col = (255, 0, 255)
        cv2.drawContours(out, [b.contour], -1, col, 2, cv2.LINE_AA)
        if b.volume_ul >= 0.5:
            cv2.putText(out, f"{b.volume_ul:.1f}", (b.center[0] + 6, b.center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    mx = res.max_pocket_ul if shown_max is None else shown_max
    tot = res.total_ul if shown_total is None else shown_total
    v = verdict(mx, tot, cfg)
    lines = [
        (v, VERDICT_COLOR[v], 0.9, 2),
        (f"Largest pocket {mx:6.1f} uL  (limit {SPEC_MAX_POCKET_UL:.0f})", (255, 255, 255), 0.5, 1),
        (f"Total air      {tot:6.1f} uL  (limit {cfg.total_limit_ul:.0f} for {cfg.fill_ml:g} mL)",
         (255, 255, 255), 0.5, 1),
        (f"Bubbles {len(res.bubbles)}   headspace {res.headspace_ul:.1f} uL "
         f"(+/-{cfg.ul_per_axial_px:.1f} per px)", (220, 220, 220), 0.45, 1),
    ]
    irregular = sum(not b.regular for b in res.bubbles)
    if irregular:
        lines.append((f"{irregular} odd-shaped blob(s) counted as air (magenta)", (255, 0, 255), 0.45, 1))
    if note:
        lines.append((note, (200, 200, 200), 0.45, 1))
    # Info goes in a strip beside the image so it never covers the barrel (or the headspace).
    panel = np.zeros((out.shape[0], 400, 3), np.uint8)
    ty = 10
    for text, col, scale, thick in lines:
        ty += int(32 * scale) + 6
        cv2.putText(panel, text, (10, ty), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.85, col, thick,
                    cv2.LINE_AA)
    return np.hstack([out, panel])


def debug_panel(res, height=600):
    tiles = [res.debug["gray"], res.debug["dark"], res.debug["solid"]]
    tiles = [cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in tiles]
    s = height / tiles[0].shape[0]
    tiles = [cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) for t in tiles]
    for t, name in zip(tiles, ["input (air end up)", "dark pixels", "bubbles"]):
        cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return np.hstack(tiles)


def report_text(name, res, cfg):
    v = verdict(res.max_pocket_ul, res.total_ul, cfg)
    lines = [f"{name}: {v}  largest pocket {res.max_pocket_ul:.1f} uL (limit {SPEC_MAX_POCKET_UL:.0f}) | "
             f"total {res.total_ul:.1f} / {cfg.total_limit_ul:.0f} uL | "
             f"{len(res.bubbles)} bubbles | headspace {res.headspace_ul:.1f} uL"]
    for i, b in enumerate(res.bubbles, 1):
        tag = "" if b.regular else "  (odd shape)"
        lines.append(f"   #{i:<3} d_axial {b.d_axial_mm:5.2f} mm  d_lateral {b.d_lateral_mm:5.2f} mm"
                     f"  {b.volume_ul:7.2f} uL{tag}")
    return "\n".join(lines)


# ------------------------------------------------------------------------- logging

CSV_FIELDS = ["timestamp", "source", "verdict", "max_pocket_ul", "total_ul", "headspace_ul",
              "n_bubbles", "frames_averaged", "image"]


def log_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            wr.writeheader()
        wr.writerow(row)


# -------------------------------------------------------------------------- camera

def open_capture(camera=None, video=None, width=None, height=None, exposure=None):
    if video:
        return cv2.VideoCapture(video)
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION)
    elif sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if exposure is not None:
        # Best effort: the meaning of these values differs by OS and camera driver.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if sys.platform.startswith("linux") else 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return cap


def grab_still(args):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Could not read {args.image}")
        return frame
    cap = open_capture(camera=args.camera, width=args.width, height=args.height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    frame = None
    print("Position the syringe, then press SPACE to freeze the frame (q to quit).")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frame = f
        cv2.imshow("setup preview", f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key in (ord("q"), 27):
            cap.release()
            sys.exit(0)
    cap.release()
    cv2.destroyWindow("setup preview")
    if frame is None:
        sys.exit("No frame captured")
    return frame


def click_points(frame, n, title):
    pts, disp = [], frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while len(pts) < n:
        cv2.imshow(title, disp)
        if (cv2.waitKey(20) & 0xFF) == 27:
            break
    cv2.imshow(title, disp)
    cv2.waitKey(400)
    cv2.destroyWindow(title)
    return pts


def parse_spacing_mm(text, barrel_id_mm):
    t = text.strip().lower().replace(" ", "")
    if t.endswith("ml"):
        return float(t[:-2]) * 1000.0 / (math.pi * (barrel_id_mm / 2) ** 2)
    if t.endswith("mm"):
        return float(t[:-2])
    raise ValueError("give the spacing with units, e.g. '5 ml' or '20 mm'")


# -------------------------------------------------------------------------- commands

def cmd_setup(args):
    needs_gui = args.roi is None or args.mm_per_px is None or not args.no_exclusions
    frame = grab_still(args) if needs_gui else None

    if args.roi is not None:
        roi = list(args.roi)
    else:
        r = cv2.selectROI("1/3  Drag a box just INSIDE the barrel walls, then ENTER", frame,
                          showCrosshair=False)
        cv2.destroyAllWindows()
        if r[2] == 0 or r[3] == 0:
            sys.exit("No ROI selected")
        roi = [int(v) for v in r]

    exclusions = []
    if not args.no_exclusions:
        rs = cv2.selectROIs("2/3  Optional: box the printed numbers (ENTER each), ESC when done",
                            frame, showCrosshair=False)
        cv2.destroyAllWindows()
        exclusions = [[int(v) for v in r] for r in rs]

    barrel_id = args.barrel_id_mm
    if barrel_id is None:
        barrel_id = float(input("Barrel inner diameter in mm (measure with calipers): "))

    mm_per_px = args.mm_per_px
    if mm_per_px is None:
        pts = click_points(frame, 2, "3/3  Click two graduation marks far apart (along the barrel)")
        if len(pts) < 2:
            sys.exit("Calibration cancelled")
        dist_px = math.dist(pts[0], pts[1])
        spacing = input("Spacing between those marks, e.g. '5 ml' or '20 mm': ")
        mm_per_px = parse_spacing_mm(spacing, barrel_id) / dist_px

    cfg = Config(roi=roi, mm_per_px=mm_per_px, barrel_id_mm=barrel_id, air_end=args.air_end,
                 exclusions=exclusions, fill_ml=args.fill_ml, lateral_mag=args.lateral_mag)
    cfg.save(args.config)
    print(f"Saved {args.config}")
    print(f"  scale {cfg.mm_per_px:.4f} mm/px | 25 uL bubble = {(6 * 25 / math.pi) ** (1 / 3):.2f} mm "
          f"= {cfg.px((6 * 25 / math.pi) ** (1 / 3)):.0f} px across | headspace resolution "
          f"{cfg.ul_per_axial_px:.1f} uL per px")
    if cfg.ul_per_axial_px > 5:
        print("  WARNING: one pixel of headspace is more than 5 uL. Move the camera closer or use "
              "a higher resolution if headspace near the 25 uL limit matters.")


def apply_overrides(cfg, args):
    for name in ("dark_thresh", "fill_ml", "margin", "lateral_mag", "polarity", "min_diam_mm"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)
    return cfg


def cmd_run(args):
    cfg = apply_overrides(Config.load(args.config), args)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "measurements.csv")

    if args.image:
        worst = "PASS"
        for path in args.image:
            frame = cv2.imread(path)
            if frame is None:
                print(f"{path}: could not read, skipped")
                continue
            res = measure(frame, cfg)
            print(report_text(os.path.basename(path), res, cfg))
            v = verdict(res.max_pocket_ul, res.total_ul, cfg)
            worst = max(worst, v, key=["PASS", "MARGINAL", "FAIL"].index)
            stem = os.path.splitext(os.path.basename(path))[0]
            out_img = os.path.join(args.out, f"{stem}_annotated.png")
            cv2.imwrite(out_img, annotate(frame, cfg, res))
            if args.debug:
                cv2.imwrite(os.path.join(args.out, f"{stem}_debug.png"), debug_panel(res))
            log_row(csv_path, {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "source": path,
                               "verdict": v, "max_pocket_ul": f"{res.max_pocket_ul:.2f}",
                               "total_ul": f"{res.total_ul:.2f}",
                               "headspace_ul": f"{res.headspace_ul:.2f}",
                               "n_bubbles": len(res.bubbles), "frames_averaged": 1,
                               "image": out_img})
        print(f"Annotated images and {csv_path} written to {args.out}/")
        return 0 if worst == "PASS" else 1

    cap = open_capture(camera=args.camera, video=args.video, width=args.width,
                       height=args.height, exposure=args.exposure)
    if not cap.isOpened():
        print("Could not open camera/video")
        return 1

    history = deque(maxlen=args.smooth)
    show_debug, paused, frame = args.debug, False, None
    print("s = save + log   d = debug   +/- = sensitivity   p = pause   w = write config   q = quit")
    try:
        while True:
            if not paused or frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
                res = measure(frame, cfg)
                history.append((res.max_pocket_ul, res.total_ul))
            mx = float(np.median([h[0] for h in history]))
            tot = float(np.median([h[1] for h in history]))
            spread = np.ptp([h[0] for h in history]) if len(history) > 2 else 0.0
            note = (f"median of {len(history)} frames | thr {cfg.dark_thresh:.2f}"
                    + ("  | SETTLING" if spread > 0.2 * SPEC_MAX_POCKET_UL else "")
                    + ("  | PAUSED" if paused else ""))
            view = annotate(frame, cfg, res, mx, tot, note)
            cv2.imshow("Syringe bubble inspector", view)
            if show_debug:
                cv2.imshow("debug", debug_panel(res))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow("debug")
            elif key == ord("p"):
                paused = not paused
            elif key in (ord("+"), ord("=")):
                cfg.dark_thresh = min(0.8, cfg.dark_thresh + 0.01)
                history.clear()
            elif key in (ord("-"), ord("_")):
                cfg.dark_thresh = max(0.02, cfg.dark_thresh - 0.01)
                history.clear()
            elif key == ord("w"):
                cfg.save(args.config)
                print(f"dark_thresh {cfg.dark_thresh:.2f} written to {args.config}")
            elif key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(args.out, f"snap_{stamp}.png")
                cv2.imwrite(img_path, view)
                cv2.imwrite(os.path.join(args.out, f"snap_{stamp}_raw.png"), frame)
                log_row(csv_path, {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "source": "video" if args.video else f"camera {args.camera}",
                                   "verdict": verdict(mx, tot, cfg), "max_pocket_ul": f"{mx:.2f}",
                                   "total_ul": f"{tot:.2f}", "headspace_ul": f"{res.headspace_ul:.2f}",
                                   "n_bubbles": len(res.bubbles), "frames_averaged": len(history),
                                   "image": img_path})
                print(f"saved {img_path}: {verdict(mx, tot, cfg)} max {mx:.1f} uL total {tot:.1f} uL")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main():
    ap = argparse.ArgumentParser(description="Measure residual air in a syringe and check it "
                                             "against the 25 uL / 250 uL-per-10 mL spec.")
    ap.add_argument("--config", default="bubble_config.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="one-time ROI selection and calibration")
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--image", help="use a still image instead of the camera")
    s.add_argument("--width", type=int)
    s.add_argument("--height", type=int)
    s.add_argument("--air-end", choices=list(TO_WORK), default="top",
                   help="side of the image where the syringe tip (air end) is")
    s.add_argument("--fill-ml", type=float, default=10.0)
    s.add_argument("--lateral-mag", type=float, default=1.33,
                   help="~1.33 for saline, ~1.4-1.5 for contrast (its refractive index)")
    s.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    s.add_argument("--mm-per-px", type=float)
    s.add_argument("--barrel-id-mm", type=float)
    s.add_argument("--no-exclusions", action="store_true")

    r = sub.add_parser("run", help="inspect live camera, video, or images")
    src = r.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0)
    src.add_argument("--video")
    src.add_argument("--image", nargs="+")
    r.add_argument("--out", default="bubble_results")
    r.add_argument("--smooth", type=int, default=15, help="frames in the running median")
    r.add_argument("--debug", action="store_true")
    r.add_argument("--width", type=int)
    r.add_argument("--height", type=int)
    r.add_argument("--exposure", type=float, help="lock exposure to this value (driver-specific)")
    r.add_argument("--dark-thresh", type=float)
    r.add_argument("--fill-ml", type=float)
    r.add_argument("--margin", type=float)
    r.add_argument("--lateral-mag", type=float)
    r.add_argument("--min-diam-mm", type=float)
    r.add_argument("--polarity", choices=["dark", "bright", "both"])

    args = ap.parse_args()
    return cmd_setup(args) if args.cmd == "setup" else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
