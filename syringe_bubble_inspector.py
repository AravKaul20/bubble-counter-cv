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
Every photo is checked against these (see "photo quality"). A photo that breaks them
is reported INVALID: its numbers are still shown for diagnosis but carry no verdict.

Workflow
  1) One-time setup (opens windows):
        python syringe_bubble_inspector.py setup --image calibration.jpg --air-end top
     - drag a box just INSIDE the barrel walls (the liquid to inspect), ENTER
     - optionally box the printed numbers so they are ignored, ESC when done
     - type the barrel inner diameter in mm, click two graduation marks, then type
       their spacing ("5 ml" or "20 mm")
     This writes bubble_config.json plus a copy of the calibration photo, which later
     photos are aligned to. Every value can also be given on the command line instead
     (see --help), which skips the corresponding window.

  2) Inspect a syringe with the guided session (recommended):
        python guided_inspection.py --sample S12
     or directly:
        images:  python syringe_bubble_inspector.py run --image a.png b.png --out results
        live:    python syringe_bubble_inspector.py run --camera 0    (preview / aiming)
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
import textwrap
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields, replace

import cv2
import numpy as np

VERSION = "2.0.0"

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
    # Written by setup from the calibration photo; used by the photo-quality checks.
    image_size: list = None       # [w, h] of the calibration photo
    ref_image: str = None         # copy of the calibration photo, relative to the config file
    ref_sharpness: float = None   # edge sharpness of the calibration photo
    calibrated_at: str = None     # when setup was run

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


def read_image(path):
    """cv2.imread that also works with non-ASCII paths on Windows. Applies EXIF rotation."""
    try:
        data = np.fromfile(path, np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise OSError(f"could not encode {path}")
    buf.tofile(path)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


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
    vp = np.pad(v, pad, mode="reflect" if len(v) > pad else "edge")   # edge padding would hide bands at the ends
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
    A band touching the far end of the ROI is the plunger seal, not a liquid surface. A
    row of bubbles can be as wide as the barrel too, but each bubble ring is crossed twice
    by a column (top and bottom of the ring) and is thicker, so the surface must be a thin
    band that most columns cross once.
    """
    H, W = dark.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    cands = sorted((i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] >= cfg.meniscus_frac * W
                    and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] < H),
                   key=lambda j: stats[j, cv2.CC_STAT_TOP])          # closest to the air end first
    max_band = max(3.0, cfg.px(1.5))
    for i in cands:
        x0, y0, w, h = stats[i, :4]
        comp = lab[y0:y0 + h, x0:x0 + w] == i
        level = np.full(W, np.nan)
        bottom = np.full(W, np.nan)
        runs, thick = [], []
        for xc in range(w):
            ys = np.flatnonzero(comp[:, xc])
            if ys.size == 0:
                continue
            gaps = np.flatnonzero(np.diff(ys) > 1)
            y1 = ys[gaps[0]] if gaps.size else ys[-1]       # first dark run from the air end
            level[x0 + xc] = y0 + (ys[0] + y1) / 2.0
            bottom[x0 + xc] = y0 + y1
            runs.append(gaps.size + 1)
            thick.append(y1 - ys[0] + 1)
        good = ~np.isnan(level)
        if (good.sum() >= 0.5 * W and np.mean(np.array(runs) > 1) < 0.3
                and np.median(thick) <= max_band):
            break
    else:
        return None
    comp = lab == i
    # Where the band is cut off by the ROI's air end, its midpoint is too low: place the
    # surface half a typical band thickness above the band's lower edge instead.
    cut = good & (2 * level - bottom <= 0.5)          # band starts at row 0
    full = good & ~cut
    if cut.any() and full.any():
        level[cut] = np.maximum(bottom[cut] - np.median(bottom[full] - level[full]), -0.5)
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
    gray = to_gray(frame)
    view = View(gray.shape, cfg)
    g = cv2.GaussianBlur(view.work(gray).astype(np.float32), (3, 3), 0)
    H, W = g.shape

    excl_full = np.zeros(gray.shape, np.uint8)
    for x, y, w, h in cfg.exclusions:
        # clip: a box moved partly off the photo must not wrap around (negative index)
        excl_full[max(0, int(y)):max(0, int(y + h)), max(0, int(x)):max(0, int(x + w))] = 255
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

    # Exclusion boxes (printed numbers) drop only blobs lying (almost) entirely inside
    # them. A bubble crossing a box edge is measured whole, never cut in half.
    bubbles, excluded_ul, excl_overlap = [], 0.0, 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        if y == 0 and y + h >= H:      # runs the full length: wall/edge artifact
            solid[lab == i] = 0
            continue
        blob = ((lab[y:y + h, x:x + w] == i) * 255).astype(np.uint8)
        inside = cv2.countNonZero(blob & excl[y:y + h, x:x + w]) / max(cv2.countNonZero(blob), 1)
        if inside >= 0.9:
            b = measure_piece(blob, x, y, cfg, view)
            excluded_ul += b.volume_ul if b is not None else 0.0
            solid[y:y + h, x:x + w][blob > 0] = 0
            continue
        excl_overlap += inside > 0
        sol, _ = solidity(blob)
        pieces = split_touching(blob, cfg.lateral_mag, k / 2.0) if sol < 0.9 else [blob]
        for p in pieces:
            b = measure_piece(p, x, y, cfg, view)
            if b is not None:
                bubbles.append(b)
    bubbles.sort(key=lambda b: -b.volume_ul)

    debug = {"gray": view.work(gray), "dark": dark, "solid": solid,
             "air": air if head is not None else np.zeros((H, W), bool), "excl": excl > 0,
             "excluded_ul": excluded_ul, "excl_overlap": excl_overlap}
    return Result(bubbles, headspace_ul, head_poly, debug)


# ------------------------------------------------------------------- photo quality
#
# A photo is only trusted if it looks like the setup the method assumes: backlit, sharp,
# close enough, and in the same place as at calibration. Each check reports OK / WARN /
# FAIL with the measured value and what to do about it. Any FAIL makes the verdict
# INVALID. Under the wrong lighting the result depends more on the threshold than on the
# syringe (a front-lit photo read anywhere from 5 to 156 uL), and a confident wrong
# number is worse than none.

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
SEVERITY = {SKIP: 0, OK: 0, WARN: 1, FAIL: 2}

# (WARN, FAIL) thresholds. See "Photo quality checks" in the README.
QC_UL_PER_PX = (5.0, 10.0)        # headspace uL per pixel of liquid level (above = worse)
QC_MIN_BUBBLE_PX = 3.0            # smallest reportable bubble must span this many pixels
QC_BRIGHTNESS = (100, 60)         # median grey level inside the barrel (below = worse)
QC_CLIPPED = (0.05, 0.25)         # fraction of barrel pixels at 250 or more
QC_BACKGROUND = (0.03, 0.06)      # largest local offset from the background model
QC_BRIGHT_RATIO = (0.5, 1.0)      # area of bright features / area of dark features
QC_SHARPNESS_REL = (0.8, 0.6)     # edge sharpness relative to the calibration photo
QC_SHARPNESS_ABS = (0.06, 0.0)    # no calibration photo: WARN only (depends on resolution)
QC_SHIFT = (0.15, 0.3)            # shift since calibration, as a fraction of barrel width
QC_ALIGN_RESPONSE = 0.5           # wall/end profile correlation below this = shift not found
QC_SCALE = (0.02, 0.05)           # size change since calibration (volume goes as size^3)
QC_ROTATION_DEG = (1.5, 3.0)      # tilt since calibration
QC_THRESH_SPREAD = (0.1, 0.3)     # result change when the threshold is varied, as a
                                  # fraction of the limit (25 uL pocket / total limit)
QC_COVERAGE = 0.25                # bubble mask covering more of the barrel than this


@dataclass
class Check:
    name: str
    status: str               # OK / WARN / FAIL / SKIP
    value: str                # the measured value, formatted
    message: str = ""
    fix: str = ""             # what the operator should do about it


def worst(statuses):
    return max(statuses, key=SEVERITY.get, default=OK)


def grade(value, warn, fail, higher_is_worse=True):
    if higher_is_worse:
        return FAIL if value > fail else WARN if value > warn else OK
    return FAIL if value < fail else WARN if value < warn else OK


def roi_inside(roi, shape):
    x, y, w, h = roi
    return w >= 8 and h >= 8 and x >= 0 and y >= 0 and x + w <= shape[1] and y + h <= shape[0]


def context_box(cfg, shape):
    """The ROI plus the barrel walls and ends around it: features that stay put between
    photos (unlike bubbles), used for alignment and focus."""
    x, y, w, h = cfg.roi
    if cfg.air_end in ("top", "bottom"):
        mx, my = 0.3 * w, 0.06 * h
    else:
        mx, my = 0.06 * w, 0.3 * h
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(shape[1], int(x + w + mx)), min(shape[0], int(y + h + my))
    return slice(y0, y1), slice(x0, x1)


def sharpness(gray, cfg):
    """Steepest edge slope divided by the contrast range: roughly 1 / (edge width in px)."""
    p = cv2.GaussianBlur(gray[context_box(cfg, gray.shape)].astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0) / 8.0
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1) / 8.0
    lo, hi = np.percentile(p, [1, 99])
    return float(np.percentile(np.hypot(gx, gy), 99.5) / max(hi - lo, 1.0))


def _wall_profile(patch, axis=0):
    """Mean intensity profile of a patch (barrel vertical) with slow shading removed.
    axis=0 gives the across-barrel profile, dominated by the two walls; axis=1 the
    along-barrel profile, dominated by the tip shoulder and the plunger seal. Bubbles,
    ticks and the meniscus are small or thin and average out. These large structures are
    the same in every view of the syringe, unlike the printed scale."""
    p = patch.astype(np.float64).mean(axis=axis)
    k = odd(len(p) / 4)
    p = p - cv2.GaussianBlur(p.reshape(1, -1), (k, 1), 0).ravel()
    return (p - p.mean()) / (p.std() + 1e-9)


def _match(a, b, scales, shifts):
    """Best (correlation, scale, shift) such that b(c + (x - c) * scale + shift) ~ a(x)."""
    n = len(a)
    c = (n - 1) / 2.0
    x = np.arange(n, dtype=np.float64)
    best = (-np.inf, 1.0, 0.0)
    for s in scales:
        for t in shifts:
            q = np.interp(c + (x - c) * s + t, x, b, left=np.nan, right=np.nan)
            ok = ~np.isnan(q)
            if ok.sum() < 0.6 * n:
                continue
            r = float(np.dot(a[ok], q[ok]) / ok.sum())
            if r > best[0]:
                best = (r, s, t)
    return best


def _search(a, b, scales, max_shift):
    """Coarse-to-fine version of _match over a scale range and +-max_shift pixels."""
    _, s, t = _match(a, b, scales, np.arange(-max_shift, max_shift + 1, 2.0))
    fine = np.arange(s - 0.01, s + 0.0101, 0.002) if len(scales) > 1 else [s]
    return _match(a, b, fine, np.arange(t - 2, t + 2.01, 0.25))


def register(gray, ref_gray, cfg):
    """
    Where the barrel is in this photo compared with the calibration photo: shift
    (dx, dy) in camera pixels, size ratio, tilt in degrees, and a match score (0..1).
    Sideways shift and size come from the two walls, the shift along the barrel from
    its ends, and the tilt from how far the walls move between the two halves of the
    barrel. Size matters a lot: bubble volume goes as size cubed.
    """
    sl = context_box(cfg, gray.shape)
    rot = TO_WORK[cfg.air_end]
    a, b = ref_gray[sl], gray[sl]
    if rot is not None:
        a, b = cv2.rotate(a, rot), cv2.rotate(b, rot)
    h, w = a.shape
    max_shift = int(0.25 * w)
    r_u, s, tu = _search(_wall_profile(a, 0), _wall_profile(b, 0), np.arange(0.9, 1.1001, 0.01), max_shift)
    r_v, _, tv = _search(_wall_profile(a, 1), _wall_profile(b, 1), [s], max_shift)
    half = h // 2
    _, _, t_top = _match(_wall_profile(a[:half], 0), _wall_profile(b[:half], 0), [s],
                         np.arange(tu - 15, tu + 15.01, 0.25))
    _, _, t_bot = _match(_wall_profile(a[half:], 0), _wall_profile(b[half:], 0), [s],
                         np.arange(tu - 15, tu + 15.01, 0.25))
    # back from the working orientation (air end up) to camera pixels; see View.to_frame
    dx, dy = {"top": (tu, tv), "bottom": (-tu, -tv), "left": (tv, -tu), "right": (-tv, tu)}[cfg.air_end]
    return dx, dy, s, math.degrees(math.atan2(t_bot - t_top, half)), min(r_u, r_v)


def align_to_reference(frame, cfg, ref=None):
    """
    Bring a photo into the calibration's geometry: rescale it to the calibration photo's
    size (same camera at another resolution) and move the ROI to follow a small shift of
    the syringe or camera. Returns (frame, cfg, checks); cfg is a moved copy.
    """
    checks = []
    H, W = frame.shape[:2]
    if cfg.image_size:
        rw, rh = cfg.image_size
        if (W, H) == (rw, rh):
            checks.append(Check("geometry", OK, f"{W}x{H}", "same size as the calibration photo"))
        elif abs(W * rh - H * rw) > 0.01 * W * rh:
            checks.append(Check("geometry", FAIL, f"{W}x{H}",
                                f"photo shape differs from the calibration photo ({rw}x{rh})",
                                "Use the camera, orientation and resolution you calibrated with, "
                                "or run setup again."))
            return frame, cfg, checks
        else:
            smaller = W < rw
            frame = cv2.resize(frame, (rw, rh),
                               interpolation=cv2.INTER_LINEAR if smaller else cv2.INTER_AREA)
            checks.append(Check("geometry", WARN if smaller else OK, f"{W}x{H}",
                                f"rescaled to the calibration size {rw}x{rh}",
                                "Take photos at full resolution." if smaller else ""))
    else:
        checks.append(Check("geometry", SKIP, f"{W}x{H}", "no calibration photo size in the config"))

    if not roi_inside(cfg.roi, frame.shape):
        checks.append(Check("roi", FAIL, str(list(cfg.roi)),
                            "the calibrated barrel box does not fit inside this photo",
                            "Run setup with a photo from this camera."))
        return frame, cfg, checks
    if ref is None or ref.shape[:2] != frame.shape[:2]:
        checks.append(Check("alignment", SKIP, "-", "no matching calibration photo to compare with",
                            "Run setup again to store one."))
        return frame, cfg, checks

    dx, dy, scale, tilt, score = register(to_gray(frame), to_gray(ref), cfg)
    if score < QC_ALIGN_RESPONSE:
        checks.append(Check("alignment", FAIL, f"match {score:.2f}",
                            "the barrel is not where the calibration photo has it",
                            "Re-seat the syringe in the holder and check nothing blocks the view. "
                            "If the camera, zoom or lens changed, run setup again."))
        return frame, cfg, checks
    st = worst([grade(abs(scale - 1), *QC_SCALE), grade(abs(tilt), *QC_ROTATION_DEG)])
    checks.append(Check("scale", st, f"{(scale - 1) * 100:+.1f}% size, {tilt:+.1f} deg",
                        "same distance and tilt as calibration" if st == OK else
                        "the syringe looks a different size or tilted compared with calibration "
                        "(bubble volume goes as size cubed)",
                        "" if st == OK else
                        "Keep the camera distance and zoom fixed (on a phone, stay on the 1x lens "
                        "and turn off automatic macro), seat the syringe straight, or run setup again."))
    width = min(cfg.roi[2], cfg.roi[3])          # barrel width in pixels
    d = math.hypot(dx, dy)
    val = f"{d:.1f} px"
    status = grade(d / width, *QC_SHIFT)
    if status == FAIL:
        checks.append(Check("alignment", FAIL, val,
                            f"syringe or camera moved {d:.0f} px since calibration",
                            "Re-seat the syringe in the holder. If the camera moved, run setup again."))
        return frame, cfg, checks
    ix, iy = int(round(dx)), int(round(dy))
    moved = replace(cfg, roi=[cfg.roi[0] + ix, cfg.roi[1] + iy, cfg.roi[2], cfg.roi[3]],
                    exclusions=[[x + ix, y + iy, w, h] for x, y, w, h in cfg.exclusions])
    if not roi_inside(moved.roi, frame.shape):
        checks.append(Check("alignment", FAIL, val, "the syringe moved partly out of the photo",
                            "Re-seat the syringe in the holder."))
        return frame, cfg, checks
    checks.append(Check("alignment", status, val,
                        "matches calibration" if ix == iy == 0 else
                        f"barrel box moved ({ix:+d}, {iy:+d}) px to follow the syringe",
                        "Seat the syringe the same way each time." if status == WARN else ""))
    return frame, moved, checks


def check_resolution(cfg):
    upx, bpx = cfg.ul_per_axial_px, cfg.px(cfg.min_diam_mm)
    status = grade(upx, *QC_UL_PER_PX)
    msg = f"one pixel of liquid level is {upx:.1f} uL (limit {SPEC_MAX_POCKET_UL:.0f} uL)"
    if bpx < QC_MIN_BUBBLE_PX:
        status = worst([status, WARN])
        msg += f"; a {cfg.min_diam_mm:g} mm bubble is only {bpx:.1f} px across"
    return Check("resolution", status, f"{upx:.1f} uL/px", msg,
                 "" if status == OK else
                 "Move the camera closer (fill the frame with the barrel) and run setup again.")


def check_photo(frame, cfg, res=None):
    """Lighting, exposure and focus. Pass the measurement to leave headspace out of the
    lighting statistics (air looks different from liquid even under good lighting)."""
    gray = to_gray(frame)
    g = View(gray.shape, cfg).work(gray).astype(np.float32)
    H, W = g.shape
    valid = np.ones((H, W), bool)
    if res is not None:
        valid &= ~res.debug["air"] & ~res.debug["excl"]
    checks = [check_resolution(cfg)]

    gv = g[valid] if valid.any() else g.ravel()
    med, clipped = float(np.median(gv)), float(np.mean(gv >= 250))
    s_dark = grade(med, *QC_BRIGHTNESS, higher_is_worse=False)
    s_clip = grade(clipped, *QC_CLIPPED)
    if s_dark != OK:
        msg, fix = f"barrel is dark (median grey level {med:.0f})", "Turn the backlight up or raise the exposure."
    elif s_clip != OK:
        msg, fix = f"{clipped:.0%} of the barrel is blown out", "Lower the exposure or the backlight brightness."
    else:
        msg, fix = "brightness OK", ""
    checks.append(Check("exposure", worst([s_dark, s_clip]),
                        f"grey {med:.0f}, {clipped:.0%} clipped", msg, fix))

    c = background_contrast(cv2.GaussianBlur(g, (3, 3), 0), cfg)
    # Background: on an even backlight the model fits everywhere, so the median offset of
    # every patch is near zero. Objects or room light behind the barrel leave patches off.
    # Detected bubbles are left out, so a syringe full of bubbles is not mistaken for this.
    bg = valid.copy()
    if res is not None:
        k = odd(cfg.px(1.0))
        bg &= cv2.dilate(res.debug["solid"], np.ones((k, k), np.uint8)) == 0
    rows = np.array_split(np.arange(H), max(1, int(round(H / (W / 3.0)))))
    cols = np.array_split(np.arange(W), 3)
    offset = 0.0
    for r in rows:
        for k in cols:
            v = bg[r[0]:r[-1] + 1, k[0]:k[-1] + 1]
            if v.mean() > 0.3:
                offset = max(offset, abs(float(np.median(c[r[0]:r[-1] + 1, k[0]:k[-1] + 1][v]))))
    status = grade(offset, *QC_BACKGROUND)
    checks.append(Check("background", status, f"{offset:.1%}",
                        "even light field behind the barrel" if status == OK else
                        "the background behind the barrel is not an even light field",
                        "" if status == OK else
                        "Put a diffuse light panel directly behind the syringe, filling the whole "
                        "barrel, and move other objects out of the view behind it."))

    # Lighting direction: with a backlight bubbles are dark rings. Bright bubbles mean the
    # light comes from the front, where shading and reflections make the volume unreliable.
    dark_f = float(np.mean(c[valid] > cfg.dark_thresh)) if valid.any() else 0.0
    bright_f = float(np.mean(c[valid] < -cfg.dark_thresh)) if valid.any() else 0.0
    ratio = bright_f / max(dark_f, 1e-4)
    if cfg.polarity != "dark":
        checks.append(Check("lighting", WARN, cfg.polarity,
                            f"polarity '{cfg.polarity}' is not validated; the method assumes a backlight",
                            "Use a backlight and polarity 'dark'."))
    else:
        status = OK if bright_f < 0.002 else grade(ratio, *QC_BRIGHT_RATIO)
        checks.append(Check("lighting", status, f"bright/dark {ratio:.2f}",
                            "bubbles appear dark (backlit)" if status == OK else
                            "bubbles appear bright, not dark: the syringe is lit from the front",
                            "" if status == OK else
                            "Light the syringe from behind with a diffuse panel and switch off or "
                            "block lights on the camera side."))

    s = sharpness(gray, cfg)
    if cfg.ref_sharpness:
        rel = s / cfg.ref_sharpness
        status, val = grade(rel, *QC_SHARPNESS_REL, higher_is_worse=False), f"{rel:.0%} of calibration"
    else:
        status, val = grade(s, *QC_SHARPNESS_ABS, higher_is_worse=False), f"{s:.3f}"
    checks.append(Check("focus", status, val,
                        "sharp" if status == OK else "photo is blurrier than it should be",
                        "" if status == OK else
                        "Refocus on the barrel (lock focus), hold the camera still or use a timer."))
    return checks


def check_result(res, cfg):
    """Sanity checks on the measurement itself."""
    checks = []
    axial_px = cfg.roi[3] if cfg.air_end in ("top", "bottom") else cfg.roi[2]
    barrel_ul = math.pi * (cfg.barrel_id_mm / 2) ** 2 * axial_px * cfg.mm_per_px
    frac = res.headspace_ul / barrel_ul
    if frac > 0.5:
        checks.append(Check("liquid level", FAIL, f"{frac:.0%} air",
                            f"the liquid surface was found {frac:.0%} of the way along the barrel",
                            "Check the syringe is filled and the barrel box ends above the plunger "
                            "(run setup again if not). A dense layer of foam can also look like a "
                            "liquid surface; such a syringe needs de-airing anyway."))
    coverage = float(np.mean(res.debug["solid"] > 0))
    if coverage > QC_COVERAGE:
        checks.append(Check("coverage", WARN, f"{coverage:.0%}",
                            f"{coverage:.0%} of the barrel was read as air",
                            "Check the annotated photo for lighting artefacts. A syringe this full of "
                            "bubbles fails the spec anyway."))
    odd_ul = sum(b.volume_ul for b in res.bubbles if not b.regular)
    if res.total_ul > 1.0 and odd_ul > 0.5 * res.total_ul:
        checks.append(Check("shapes", WARN, f"{odd_ul / res.total_ul:.0%} odd-shaped",
                            "most of the measured air is in odd-shaped blobs (magenta)",
                            "Check the annotated photo: debris, scratches, labels or reflections "
                            "can look like air."))
    if res.debug["excluded_ul"] > 1.0 or res.debug["excl_overlap"]:
        checks.append(Check("scale boxes", WARN,
                            f"{res.debug['excluded_ul']:.1f} uL hidden, {res.debug['excl_overlap']} crossing",
                            "possible air inside the ignored scale boxes was not counted"
                            if res.debug["excluded_ul"] > 1.0 else
                            "a bubble lies partly behind the printed scale (measured whole)",
                            "Check the annotated photo; the turned views see behind the scale."))
    return checks


def check_threshold_stability(frame, cfg, res):
    """
    Re-measure with the threshold at 0.6x and 1.6x. On a good backlit photo the result
    barely moves (bubble edges are sharp and dark). If it swings, the number depends on a
    setting rather than on the syringe: the failure behind the original live tool, where
    one front-lit photo read anywhere from 5 to 156 uL.
    """
    vals = [(res.max_pocket_ul, res.total_ul)]
    for k in (0.6, 1.6):
        r = measure(frame, replace(cfg, dark_thresh=cfg.dark_thresh * k))
        vals.append((r.max_pocket_ul, r.total_ul))
    d_max = float(np.ptp([v[0] for v in vals])) / SPEC_MAX_POCKET_UL
    d_tot = float(np.ptp([v[1] for v in vals])) / cfg.total_limit_ul
    status = grade(max(d_max, d_tot), *QC_THRESH_SPREAD)
    lo_max, hi_max = min(v[0] for v in vals), max(v[0] for v in vals)
    return Check("stability", status, f"largest {lo_max:.1f}-{hi_max:.1f} uL",
                 "result does not depend on the detection threshold" if status == OK else
                 "the result changes a lot with the detection threshold, so it is not reliable",
                 "" if status == OK else
                 "Usually a lighting problem: check the backlight is even and the photo is sharp.")


@dataclass
class Assessment:
    frame: np.ndarray       # the photo as measured (rescaled to the calibration size if needed)
    cfg: Config             # the config used, with the ROI moved to follow any shift
    result: object          # Result, or None if the photo could not be measured at all
    checks: list

    @property
    def status(self):
        return worst(c.status for c in self.checks)

    @property
    def verdict(self):
        if self.result is None or self.status == FAIL:
            return "INVALID"
        return verdict(self.result.max_pocket_ul, self.result.total_ul, self.cfg)

    @property
    def problems(self):
        return [c for c in self.checks if c.status in (WARN, FAIL)]


def assess(frame, cfg, ref=None, use_exclusions=True):
    """Align, measure and quality-check one photo. Set use_exclusions=False for views in
    which the printed scale is not where it was at calibration (syringe turned)."""
    if not use_exclusions:
        cfg = replace(cfg, exclusions=[])
    frame, cfg, checks = align_to_reference(frame, cfg, ref)
    if any(c.name in ("geometry", "roi") and c.status == FAIL for c in checks):
        return Assessment(frame, cfg, None, checks)
    res = measure(frame, cfg)
    checks += check_photo(frame, cfg, res) + check_result(res, cfg)
    checks.append(check_threshold_stability(frame, cfg, res))
    return Assessment(frame, cfg, res, checks)


def load_reference(cfg, config_path):
    """The calibration photo, or None. Warns if the config names one that cannot be read."""
    if not cfg.ref_image:
        return None
    path = os.path.join(os.path.dirname(os.path.abspath(config_path)), cfg.ref_image)
    ref = read_image(path)
    if ref is None:
        print(f"WARNING: calibration photo {path} could not be read, so position, size and focus "
              "cannot be checked. Run setup again.")
    elif cfg.image_size and [ref.shape[1], ref.shape[0]] != list(cfg.image_size):
        print(f"WARNING: calibration photo {path} does not match the size in the config. Run setup again.")
        return None
    return ref


def format_checks(checks, indent="  "):
    lines = []
    for c in checks:
        if c.status == SKIP:
            continue
        lines.append(f"{indent}[{c.status:4s}] {c.name:12s} {c.value:24s} {c.message}")
        if c.fix and c.status != OK:
            lines.append(f"{indent}{'':20s}-> {c.fix}")
    return "\n".join(lines)


# ------------------------------------------------------------------------- drawing

VERDICT_COLOR = {"PASS": (60, 200, 60), "MARGINAL": (0, 190, 255), "FAIL": (40, 40, 230),
                 "INVALID": (170, 170, 170)}
STATUS_COLOR = {OK: (60, 200, 60), WARN: (0, 190, 255), FAIL: (40, 40, 230), SKIP: (150, 150, 150)}


def annotate(frame, cfg, res, shown_max=None, shown_total=None, note="", checks=None):
    """The photo with bubbles outlined, plus a side panel with the verdict and, if given,
    the photo-quality checks. res may be None (photo could not be measured)."""
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    s = max(1.0, out.shape[0] / 900.0)           # scale text and lines with the photo
    lw = max(1, int(round(s)))
    x, y, w, h = [int(v) for v in cfg.roi]
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), lw)
    for ex, ey, ew, eh in cfg.exclusions:
        ex, ey, ew, eh = int(ex), int(ey), int(ew), int(eh)
        cv2.rectangle(out, (ex, ey), (ex + ew, ey + eh), (140, 140, 140), lw)
        cv2.putText(out, "ignored", (ex + 2, ey + int(14 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s,
                    (140, 140, 140), lw, cv2.LINE_AA)

    if res is not None and res.headspace_poly is not None:
        layer = out.copy()
        cv2.fillPoly(layer, [res.headspace_poly], (40, 40, 230))
        out = cv2.addWeighted(layer, 0.35, out, 0.65, 0)
        cv2.polylines(out, [res.headspace_poly], True, (40, 40, 230), lw, cv2.LINE_AA)

    for b in res.bubbles if res is not None else []:
        if b.volume_ul > SPEC_MAX_POCKET_UL:
            col = (40, 40, 230)
        elif b.volume_ul > 0.5 * SPEC_MAX_POCKET_UL:
            col = (0, 190, 255)
        else:
            col = (60, 200, 60)
        if not b.regular:
            col = (255, 0, 255)
        cv2.drawContours(out, [b.contour], -1, col, 2 * lw, cv2.LINE_AA)
        if b.volume_ul >= 0.5:
            cv2.putText(out, f"{b.volume_ul:.1f}", (b.center[0] + int(6 * s), b.center[1] - int(6 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, col, lw, cv2.LINE_AA)

    failed = checks is not None and worst(c.status for c in checks) == FAIL
    if res is None:
        v, lines = "INVALID", [("INVALID", VERDICT_COLOR["INVALID"], 0.9, 2),
                               ("Photo could not be measured", (255, 255, 255), 0.5, 1)]
    else:
        mx = res.max_pocket_ul if shown_max is None else shown_max
        tot = res.total_ul if shown_total is None else shown_total
        v = "INVALID" if failed else verdict(mx, tot, cfg)
        lines = [
            (v, VERDICT_COLOR[v], 0.9, 2),
            (f"Largest pocket {mx:6.1f} uL  (limit {SPEC_MAX_POCKET_UL:.0f})", (255, 255, 255), 0.5, 1),
            (f"Total air      {tot:6.1f} uL  (limit {cfg.total_limit_ul:.0f} for {cfg.fill_ml:g} mL)",
             (255, 255, 255), 0.5, 1),
            (f"Bubbles {len(res.bubbles)}   headspace {res.headspace_ul:.1f} uL "
             f"({cfg.ul_per_axial_px:.1f} uL per px)", (220, 220, 220), 0.45, 1),
        ]
        if failed:
            lines.insert(1, ("Photo failed checks: numbers NOT trusted", (40, 40, 230), 0.5, 1))
        irregular = sum(not b.regular for b in res.bubbles)
        if irregular:
            lines.append((f"{irregular} odd-shaped blob(s) counted as air (magenta)", (255, 0, 255), 0.45, 1))
    if note:
        lines.append((note, (200, 200, 200), 0.45, 1))
    if checks:
        lines.append(("", (0, 0, 0), 0.3, 1))
        lines.append(("Photo checks", (255, 255, 255), 0.5, 1))
        for c in checks:
            if c.status == SKIP:
                continue
            lines.append((f"{c.status:4s} {c.name}: {c.value}", STATUS_COLOR[c.status], 0.42, 1))
            if c.status != OK:
                for part in textwrap.wrap(c.fix or c.message, 56):
                    lines.append(("     " + part, (200, 200, 200), 0.38, 1))
    # Info goes in a strip beside the image so it never covers the barrel (or the headspace).
    panel = np.zeros((out.shape[0], int(430 * s), 3), np.uint8)
    ty = int(10 * s)
    for text, col, scale, thick in lines:
        ty += int((32 * scale + 6) * s)
        cv2.putText(panel, text, (int(10 * s), ty), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.85 * s, col,
                    max(1, int(round(thick * s))), cv2.LINE_AA)
    return np.hstack([out, panel])


def debug_panel(res, height=600):
    tiles = [res.debug["gray"], res.debug["dark"], res.debug["solid"]]
    tiles = [cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in tiles]
    s = height / tiles[0].shape[0]
    tiles = [cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) for t in tiles]
    for t, name in zip(tiles, ["input (air end up)", "dark pixels", "bubbles"]):
        cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return np.hstack(tiles)


def report_text(name, a):
    """Text report for one Assessment."""
    res, cfg = a.result, a.cfg
    if res is None:
        lines = [f"{name}: INVALID  photo could not be measured"]
    else:
        lines = [f"{name}: {a.verdict}  largest pocket {res.max_pocket_ul:.1f} uL "
                 f"(limit {SPEC_MAX_POCKET_UL:.0f}) | total {res.total_ul:.1f} / {cfg.total_limit_ul:.0f} uL | "
                 f"{len(res.bubbles)} bubbles | headspace {res.headspace_ul:.1f} uL"]
        if a.verdict == "INVALID":
            lines.append("   (photo failed the quality checks below; the numbers are not trusted)")
        for i, b in enumerate(res.bubbles, 1):
            tag = "" if b.regular else "  (odd shape)"
            lines.append(f"   #{i:<3} d_axial {b.d_axial_mm:5.2f} mm  d_lateral {b.d_lateral_mm:5.2f} mm"
                         f"  {b.volume_ul:7.2f} uL{tag}")
    if a.problems:
        lines.append(format_checks(a.problems, indent="   "))
    return "\n".join(lines)


def quality_summary(checks):
    return "; ".join(f"{c.name} {c.status}" for c in checks if c.status in (WARN, FAIL)) or "OK"


# ------------------------------------------------------------------------- logging

CSV_FIELDS = ["timestamp", "source", "verdict", "max_pocket_ul", "total_ul", "headspace_ul",
              "n_bubbles", "frames_averaged", "image", "quality"]


def log_row(path, row, fields_=CSV_FIELDS):
    """Append a row. An existing file keeps its own header (older files lack new columns)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), None) or fields_
        missing = [k for k in fields_ if k not in header]
        if missing:
            print(f"Note: {path} has an older header without {', '.join(missing)}; "
                  "those columns are not written. Start a new file to get them.")
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow(row)
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # BOM: Excel reads UTF-8
        wr = csv.DictWriter(f, fieldnames=fields_, extrasaction="ignore")
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
        frame = read_image(args.image)
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

# Typical inner diameters of common plastic syringes (BD-style). Close enough for this
# test; measure with calipers and type the number instead if you have them.
SYRINGE_ID_MM = {1: 4.78, 3: 8.66, 5: 12.06, 10: 14.5, 20: 19.13, 30: 21.7, 60: 26.7}


def parse_barrel_id(text):
    """'10ml' / '10 mL' -> preset inner diameter; a plain number -> mm."""
    t = text.strip().lower().replace(" ", "")
    if t.endswith("ml"):
        size = float(t[:-2])
        if size not in SYRINGE_ID_MM:
            raise ValueError(f"no preset for {size:g} mL; known sizes: "
                             f"{', '.join(f'{k} mL' for k in SYRINGE_ID_MM)}. Type the diameter in mm.")
        return SYRINGE_ID_MM[size]
    return float(t[:-2] if t.endswith("mm") else t)


def cmd_setup(args):
    needs_gui = args.roi is None or args.mm_per_px is None or args.exclusions
    frame = grab_still(args) if needs_gui or args.image else None

    if args.roi is not None:
        roi = list(args.roi)
    else:
        r = cv2.selectROI("1/2  Drag a box just INSIDE the barrel walls, then ENTER", frame,
                          showCrosshair=False)
        cv2.destroyAllWindows()
        if r[2] == 0 or r[3] == 0:
            sys.exit("No ROI selected")
        roi = [int(v) for v in r]

    exclusions = []
    if args.exclusions:
        rs = cv2.selectROIs("Box the printed numbers (ENTER each), ESC when done",
                            frame, showCrosshair=False)
        cv2.destroyAllWindows()
        exclusions = [[int(v) for v in r] for r in rs]

    barrel_id = args.barrel_id_mm
    while barrel_id is None:
        try:
            barrel_id = parse_barrel_id(args.syringe or input(
                "Syringe size (e.g. '10ml'), or barrel inner diameter in mm: "))
        except ValueError as exc:
            print(f"  {exc}")
            args.syringe = None

    mm_per_px = args.mm_per_px
    if mm_per_px is None:
        pts = click_points(frame, 2, "2/2  Click two graduation marks far apart (along the barrel)")
        if len(pts) < 2:
            sys.exit("Calibration cancelled")
        dist_px = math.dist(pts[0], pts[1])
        spacing = input("Spacing between those marks, e.g. '5 ml' or '20 mm': ")
        mm_per_px = parse_spacing_mm(spacing, barrel_id) / dist_px

    cfg = Config(roi=roi, mm_per_px=mm_per_px, barrel_id_mm=barrel_id, air_end=args.air_end,
                 exclusions=exclusions, fill_ml=args.fill_ml, lateral_mag=args.lateral_mag,
                 calibrated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    if frame is not None:
        if not roi_inside(cfg.roi, frame.shape):
            sys.exit(f"ROI {cfg.roi} does not fit inside the {frame.shape[1]}x{frame.shape[0]} photo")
        # Keep the calibration photo: later photos are aligned to it and compared with it.
        ref_name = os.path.splitext(os.path.basename(args.config))[0] + "_ref.png"
        write_image(os.path.join(os.path.dirname(os.path.abspath(args.config)), ref_name), frame)
        cfg.image_size = [frame.shape[1], frame.shape[0]]
        cfg.ref_image = ref_name
        cfg.ref_sharpness = sharpness(to_gray(frame), cfg)
    cfg.save(args.config)
    print(f"Saved {args.config}" + (f" and {cfg.ref_image}" if cfg.ref_image else ""))
    print(f"  scale {cfg.mm_per_px:.4f} mm/px | 25 uL bubble = {(6 * 25 / math.pi) ** (1 / 3):.2f} mm "
          f"= {cfg.px((6 * 25 / math.pi) ** (1 / 3)):.0f} px across | headspace resolution "
          f"{cfg.ul_per_axial_px:.1f} uL per px")
    checks = assess(frame, cfg).checks if frame is not None else [check_resolution(cfg)]
    print("Calibration photo checks:")
    print(format_checks(checks))
    if worst(c.status for c in checks) == FAIL:
        print("  WARNING: the calibration photo fails the checks above. Photos taken like this will be "
              "reported INVALID. Fix the setup and run setup again.")
        return 1
    return 0


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

    ref = load_reference(cfg, args.config)
    if args.image:
        order = ["PASS", "MARGINAL", "FAIL", "INVALID"]
        overall, n_read = "PASS", 0
        for path in args.image:
            frame = read_image(path)
            if frame is None:
                print(f"{path}: could not read, skipped")
                overall = "INVALID"
                continue
            n_read += 1
            a = assess(frame, cfg, ref)
            print(report_text(os.path.basename(path), a))
            overall = max(overall, a.verdict, key=order.index)
            stem = os.path.splitext(os.path.basename(path))[0]
            out_img = os.path.join(args.out, f"{stem}_annotated.png")
            write_image(out_img, annotate(a.frame, a.cfg, a.result, checks=a.checks))
            if args.debug and a.result is not None:
                write_image(os.path.join(args.out, f"{stem}_debug.png"), debug_panel(a.result))
            res = a.result
            log_row(csv_path, {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "source": path,
                               "verdict": a.verdict,
                               "max_pocket_ul": f"{res.max_pocket_ul:.2f}" if res else "",
                               "total_ul": f"{res.total_ul:.2f}" if res else "",
                               "headspace_ul": f"{res.headspace_ul:.2f}" if res else "",
                               "n_bubbles": len(res.bubbles) if res else "", "frames_averaged": 1,
                               "image": out_img, "quality": quality_summary(a.checks)})
        if n_read:
            print(f"Annotated images and {csv_path} written to {args.out}/")
        return {"PASS": 0, "MARGINAL": 1, "FAIL": 1, "INVALID": 2}[overall]

    cap = open_capture(camera=args.camera, video=args.video, width=args.width,
                       height=args.height, exposure=args.exposure)
    if not cap.isOpened():
        print("Could not open camera/video")
        return 1

    history = deque(maxlen=args.smooth)
    show_debug, paused, frame, n_frames, live_checks = args.debug, False, None, 0, None
    live_cfg, res = cfg, None
    print("Live mode is for aiming and setup. For a recorded measurement use guided_inspection.py.")
    print("s = save + log   d = debug   +/- = sensitivity   p = pause   w = write config   q = quit")
    try:
        while True:
            if not paused or frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
                if n_frames % 15 == 0:            # the full photo checks are slower; refresh ~2x/s
                    a = assess(frame, cfg, ref)
                    # Measure every frame in the geometry the checks found (rescaled, ROI moved).
                    live_checks, live_cfg, live_size = a.checks, a.cfg, a.frame.shape[1::-1]
                    measurable = a.result is not None
                n_frames += 1
                res = None
                if measurable:
                    if frame.shape[1::-1] != live_size:
                        frame = cv2.resize(frame, live_size, interpolation=cv2.INTER_AREA)
                    res = measure(frame, live_cfg)
                    history.append((res.max_pocket_ul, res.total_ul))
                else:
                    history.clear()
            mx = float(np.median([h[0] for h in history])) if history else 0.0
            tot = float(np.median([h[1] for h in history])) if history else 0.0
            spread = np.ptp([h[0] for h in history]) if len(history) > 2 else 0.0
            note = (f"median of {len(history)} frames | thr {cfg.dark_thresh:.2f}"
                    + ("  | SETTLING" if spread > 0.2 * SPEC_MAX_POCKET_UL else "")
                    + ("  | PAUSED" if paused else ""))
            view = annotate(frame, live_cfg, res, mx, tot, note, checks=live_checks)
            cv2.imshow("Syringe bubble inspector", view)
            if show_debug and res is not None:
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
                cfg.dark_thresh = live_cfg.dark_thresh = min(0.8, cfg.dark_thresh + 0.01)
                history.clear()
            elif key in (ord("-"), ord("_")):
                cfg.dark_thresh = live_cfg.dark_thresh = max(0.02, cfg.dark_thresh - 0.01)
                history.clear()
            elif key == ord("w"):
                cfg.save(args.config)
                print(f"dark_thresh {cfg.dark_thresh:.2f} written to {args.config}")
            elif key == ord("s") and res is None:
                print("not saved: this frame cannot be measured (see the photo checks)")
            elif key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(args.out, f"snap_{stamp}.png")
                write_image(img_path, view)
                write_image(os.path.join(args.out, f"snap_{stamp}_raw.png"), frame)
                v = "INVALID" if worst(c.status for c in live_checks or []) == FAIL else verdict(mx, tot, cfg)
                log_row(csv_path, {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "source": "video" if args.video else f"camera {args.camera}",
                                   "verdict": v, "max_pocket_ul": f"{mx:.2f}",
                                   "total_ul": f"{tot:.2f}", "headspace_ul": f"{res.headspace_ul:.2f}",
                                   "n_bubbles": len(res.bubbles), "frames_averaged": len(history),
                                   "image": img_path, "quality": quality_summary(live_checks or [])})
                print(f"saved {img_path}: {v} max {mx:.1f} uL total {tot:.1f} uL")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main():
    for stream in (sys.stdout, sys.stderr):   # a Windows console may not encode every name
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
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
    s.add_argument("--syringe", help="syringe size for the barrel diameter preset, e.g. 10ml")
    s.add_argument("--exclusions", action="store_true",
                   help="also box the printed numbers so they are ignored (rarely needed)")
    s.add_argument("--no-exclusions", action="store_true", help=argparse.SUPPRESS)  # old default

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
