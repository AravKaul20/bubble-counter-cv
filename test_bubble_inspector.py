#!/usr/bin/env python3
"""
Synthetic accuracy test for syringe_bubble_inspector.py.

Renders backlit syringe images with KNOWN bubble sizes (graduation ticks, printed
numbers, glare streak, uneven lighting, noise, blur), runs the detector, and checks
the measured volumes and verdicts against ground truth.

    python test_bubble_inspector.py            # run all checks
    python test_bubble_inspector.py --save     # also write the images to synthetic/

Rerun this after changing the detector so you know you have not broken anything.
Passing here does NOT prove it works on real photos; see the README notes.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

from syringe_bubble_inspector import Config, measure, verdict, annotate

MM_PER_PX = 0.04
BARREL_ID = 14.5
MAG = 1.33
FH, FW = 1000, 640
ROI = [140, 60, 362, 880]            # barrel interior
EXCL = [[228, 60, 60, 880]]          # printed number column
UL_PER_ML_MM = 1000 / (math.pi * (BARREL_ID / 2) ** 2)   # axial mm per mL


def d_for_volume(ul):
    return (6 * ul / math.pi) ** (1 / 3)


def sphere_ul(d_mm):
    return math.pi / 6 * d_mm ** 3


def base_image(rng):
    yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
    vign = 1 - 0.18 * (((xx - FW / 2) / FW) ** 2 + ((yy - FH / 2) / FH) ** 2) * 2
    img = 215 * vign
    x0, x1 = ROI[0], ROI[0] + ROI[2]
    xn = np.clip((xx - (x0 + x1) / 2) / ((x1 - x0) / 2), -1.2, 1.2)
    img *= np.where(np.abs(xn) <= 1, 0.95 - 0.22 * xn ** 6, 1.0)          # barrel shading
    for wx in (x0 - 22, x1):                                              # walls
        img[:, wx:wx + 22] *= 0.38
    img[:, 168:173] *= 1.15                                               # glare streak
    tick_dy = 0.2 * UL_PER_ML_MM / MM_PER_PX                              # 0.2 mL spacing
    y, k = ROI[1] + 25, 0
    while y < ROI[1] + ROI[3] - 10:
        x_end = 225 if k % 5 == 0 else 200
        img[int(y) - 1:int(y) + 2, 150:x_end] *= 0.5
        if k % 5 == 0:
            tmp = np.zeros((FH, FW), np.uint8)
            cv2.putText(tmp, str(k // 5), (240, int(y) + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            img[tmp > 0] *= 0.45
        y += tick_dy
        k += 1
    return img


def add_bubble(img, cx, cy, d_mm):
    R = d_mm / 2 / MM_PER_PX
    Rl = R * MAG
    yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
    rho = np.sqrt(((xx - cx) / Rl) ** 2 + ((yy - cy) / R) ** 2)
    outer = np.clip((1 - rho) * R + 0.5, 0, 1)
    inner = np.clip((rho - 0.58) * R + 0.5, 0, 1)
    img *= 1 - 0.72 * outer * inner
    img += 25 * np.exp(-(((xx - cx + 0.2 * Rl) ** 2 + (yy - cy + 0.25 * R) ** 2) / (2 * (0.12 * R) ** 2)))


def add_headspace(img, target_ul, sag_px=10, band_px=6):
    """Concave meniscus: liquid climbs the walls. Returns the true air volume in uL."""
    x0, W = ROI[0], ROI[2]
    R = BARREL_ID / 2
    # mean height weighted by |x| for a parabola is apex - sag/2
    apex = target_ul / (math.pi * R ** 2) / MM_PER_PX + sag_px / 2
    yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
    xn = (xx - (x0 + W / 2)) / (W / 2)
    ym = ROI[1] + apex - sag_px * xn ** 2
    inside = (np.abs(xn) <= 1)
    img[(yy < ym) & inside & (yy >= ROI[1])] *= 1.06
    band = inside & (np.abs(yy - ym) < band_px / 2)
    img[band] *= 0.35
    xs = np.arange(W)
    r_mm = np.abs((xs + 0.5 - W / 2) / (W / 2)) * R
    # +0.5: pixel centres sit at integer rows, so the ROI's top edge is at row -0.5
    h_mm = (apex + 0.5 - sag_px * ((xs + 0.5 - W / 2) / (W / 2)) ** 2) * MM_PER_PX
    return float(np.sum(np.clip(h_mm, 0, None) * math.pi * r_mm * R / (W / 2)))


def finish(img, rng):
    img = img + rng.normal(0, 2.5, img.shape)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    g = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def config(**kw):
    c = dict(roi=ROI, mm_per_px=MM_PER_PX, barrel_id_mm=BARREL_ID, exclusions=EXCL, lateral_mag=MAG)
    c.update(kw)
    return Config(**c)


def scenes(rng):
    out = []

    img = base_image(rng)
    out.append(("clean syringe", finish(img, rng), [], 0.0, "PASS", config()))

    img = base_image(rng)
    d = d_for_volume(30)
    add_bubble(img, 400, 500, d)
    out.append(("one 30 uL bubble", finish(img, rng), [30.0], 0.0, "FAIL", config()))

    img = base_image(rng)
    add_bubble(img, 380, 650, d_for_volume(22))
    out.append(("one 22 uL bubble", finish(img, rng), [22.0], 0.0, "MARGINAL", config()))

    img = base_image(rng)
    smalls = [(330, 150, 0.6), (420, 230, 0.8), (360, 330, 1.0), (450, 420, 1.2),
              (340, 560, 1.4), (430, 720, 1.6), (180, 800, 1.0), (370, 850, 0.7)]
    for x, y, dd in smalls:          # (180, 800) sits across the graduation ticks
        add_bubble(img, x, y, dd)
    out.append(("8 small bubbles, one on the ticks", finish(img, rng),
                [sphere_ul(dd) for *_, dd in smalls], 0.0, "PASS", config()))

    img = base_image(rng)
    dd = 2.0
    rl = dd / 2 / MM_PER_PX * MAG
    add_bubble(img, 380 - rl, 500, dd)
    add_bubble(img, 380 + rl, 500, dd)
    out.append(("two touching 2 mm bubbles", finish(img, rng), [sphere_ul(dd)] * 2, 0.0, "PASS", config()))

    img = base_image(rng)
    dd = 2.5
    add_bubble(img, ROI[0] + ROI[2] - 25, 400, dd)
    out.append(("2.5 mm bubble cut by the wall", finish(img, rng), [sphere_ul(dd)], 0.0, "PASS", config()))

    img = base_image(rng)
    true_head = add_headspace(img, 60)
    add_bubble(img, 400, 600, 1.2)
    out.append(("60 uL headspace + 1 bubble", finish(img, rng), [sphere_ul(1.2)], true_head, "FAIL", config()))

    # same 30 uL scene, syringe lying sideways with the tip on the left
    _, f30, *_ = out[1]
    rot = cv2.rotate(f30, cv2.ROTATE_90_COUNTERCLOCKWISE)
    def rot_box(b):
        x, y, w, h = b
        return [y, FW - x - w, h, w]
    out.append(("30 uL bubble, syringe horizontal", rot, [30.0], 0.0, "FAIL",
                config(roi=rot_box(ROI), exclusions=[rot_box(e) for e in EXCL], air_end="left")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    if args.save:
        os.makedirs("synthetic", exist_ok=True)

    failures = 0
    print(f"{'scene':38s} {'bubbles':>11s} {'largest uL':>17s} {'total uL':>17s} "
          f"{'headspace':>15s} {'verdict':>17s}")
    for i, (name, frame, true_b, true_head, true_v, cfg) in enumerate(scenes(rng)):
        res = measure(frame, cfg)
        v = verdict(res.max_pocket_ul, res.total_ul, cfg)
        t_max = max(true_b + [true_head], default=0.0)
        t_tot = sum(true_b) + true_head
        ok_n = len(res.bubbles) == len(true_b) or "wall" in name
        ok_max = abs(res.max_pocket_ul - t_max) <= max(0.15 * t_max, 0.3)
        ok_head = abs(res.headspace_ul - true_head) <= 2.5 * cfg.ul_per_axial_px
        ok_v = v == true_v
        if "wall" in name:   # half hidden by the wall: only require it is found
            ok_max = len(res.bubbles) >= 1 and res.max_pocket_ul >= 0.3 * t_max
        good = ok_n and ok_max and ok_head and ok_v
        failures += not good
        print(f"{name:38s} {len(res.bubbles):>4d} (true {len(true_b)}) "
              f"{res.max_pocket_ul:7.2f} ({t_max:6.2f}) {res.total_ul:7.2f} ({t_tot:6.2f}) "
              f"{res.headspace_ul:6.1f} ({true_head:5.1f}) {v:>8s} ({true_v:8s}) "
              f"{'ok' if good else 'MISMATCH'}")
        if args.save:
            cv2.imwrite(f"synthetic/{i:02d}_input.png", frame)
            cv2.imwrite(f"synthetic/{i:02d}_annotated.png", annotate(frame, cfg, res))
    print(f"\n{'All checks passed' if not failures else f'{failures} scene(s) mismatched'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
