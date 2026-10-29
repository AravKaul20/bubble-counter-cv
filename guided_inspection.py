#!/usr/bin/env python3
"""
guided_inspection.py

Guided still-photo inspection of one filled syringe. Walks the operator through the
photos the measurement needs, checks every photo before trusting it (lighting, focus,
position, size, resolution), asks for a retake when a photo is unusable, combines the
views and writes a report.

    python guided_inspection.py photo.jpg                # check one photo, done
    python guided_inspection.py front.jpg quarter.jpg    # two views of the same syringe
    python guided_inspection.py                          # asks for the photo (drag it in)
    python guided_inspection.py --camera 0               # captures from a webcam

Calibrate once per camera position first:
    python syringe_bubble_inspector.py setup --image calibration.jpg --air-end top
(the guide offers to start this if there is no calibration yet).

Exit code: 0 PASS, 1 MARGINAL or FAIL, 2 INVALID (a photo failed its checks, a view is
missing, or the session was stopped), 3 error.

Requires: opencv-python, numpy
"""

import argparse
import glob
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field

import cv2
import numpy as np

import syringe_bubble_inspector as sbi
from syringe_bubble_inspector import (FAIL, OK, SKIP, SPEC_MAX_POCKET_UL, STATUS_COLOR, WARN,
                                      VERDICT_COLOR, Check, Config, align_to_reference, annotate,
                                      assess, check_photo, debug_panel, format_checks,
                                      load_reference, log_row, read_image, verdict, worst,
                                      write_image)

# The syringe is turned about its own axis between views. Every view sees the whole
# barrel from another side, so a bubble hidden behind the printed scale or another
# bubble in one view is visible in another.
VIEWS = [
    ("front", "Front", "graduations facing the camera"),
    ("quarter", "Quarter turn", "turn the syringe 90 degrees about its own axis, "
                                "without lifting it out of the holder"),
    ("half", "Half turn", "turn it another 90 degrees (graduations facing away)"),
    ("three-quarter", "Three-quarter turn", "turn it another 90 degrees"),
]

CHECKLIST = [
    "White light BEHIND the syringe (e.g. a tablet showing a white screen), room lights off",
    "Syringe upright, tip up, in the same spot as the calibration photo",
    "Camera in the same place, same zoom; let the syringe rest {settle} s first (don't tap it)",
]

# What the number covers. Shown in every report so nobody reads more into it.
MEASURAND = ("Air inside the barrel box set at calibration (from just below the tip shoulder "
             "to just above the plunger seal), as seen in the photographed views. Not included: "
             "air in the conical tip, nozzle and luer, and air hidden in every view.")
ACCURACY = ("Not yet validated against known air volumes. Known systematic effects: bubble "
            "size depends on depth in the barrel (the liquid acts as a lens), bubbles stuck to "
            "the wall are caps rather than spheres, and headspace is resolved to about half a "
            "pixel of liquid level. Run the validation in the README before quoting values.")

SESSION_FIELDS = ["timestamp", "sample", "operator", "verdict", "max_pocket_ul", "total_ul",
                  "fill_ml", "views_used", "views_planned", "notes", "report"]
EXIT_CODE = {"PASS": 0, "MARGINAL": 1, "FAIL": 1, "INVALID": 2}

# Matching the same bubble across views: turning the syringe about its axis keeps a
# bubble's height along the barrel and its axial size; only its sideways position changes.
MATCH_AXIAL_MM = 0.5          # or 35% of the bubble size, whichever is larger (bubbles drift a little)
MATCH_SIZE = 0.25             # relative difference in axial size


# Checks that describe the rig rather than the photo: reported once at the start and in
# the report, but a warning from them alone does not make the operator confirm each photo.
SETUP_CHECKS = {"resolution"}


class Abort(Exception):
    pass


def photo_status(a):
    return worst(c.status for c in a.checks if c.status == FAIL or c.name not in SETUP_CHECKS)


@dataclass
class ViewRecord:
    key: str
    label: str
    how: str
    assessment: object = None     # the accepted (or last failed) Assessment; None if no photo
    source: str = ""              # file path or camera
    photo: str = ""               # files in the session folder (relative names)
    annotated: str = ""
    debug: str = ""
    sha256: str = ""
    attempts: list = field(default_factory=list)   # every photo tried, with its outcome

    @property
    def usable(self):
        return self.assessment is not None and self.assessment.verdict != "INVALID"


# ------------------------------------------------------------------------- combining

def axial_mm(b, cfg):
    """Distance of a bubble's centre from the air end of the barrel box."""
    x, y, w, h = cfg.roi
    cx, cy = b.center
    px = {"top": cy - y, "bottom": y + h - cy, "left": cx - x, "right": x + w - cx}[cfg.air_end]
    return px * cfg.mm_per_px


def match_pockets(records):
    """
    The air pockets of the syringe from all views. A bubble seen in several views is
    one pocket (its largest reading is kept); a bubble seen in only one view is still
    counted. Matching uses the height along the barrel and the axial size, which do not
    change when the syringe is turned about its axis.
    """
    pockets = []
    for r in records:
        a = r.assessment
        cands = [(axial_mm(b, a.cfg), b.d_axial_mm, b.volume_ul) for b in a.result.bubbles]
        pairs = []
        for i, (z, d, _) in enumerate(cands):
            for j, p in enumerate(pockets):
                big = max(d, p["d_axial_mm"])
                dz, dd = abs(z - p["axial_mm"]), abs(d - p["d_axial_mm"]) / max(big, 1e-6)
                if dz < max(MATCH_AXIAL_MM, 0.35 * big) and dd < MATCH_SIZE:
                    pairs.append((dz / max(big, 1e-6) + dd, i, j))
        used_i, used_j = set(), set()
        for _, i, j in sorted(pairs):
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            pockets[j]["seen"][r.label] = cands[i][2]
            pockets[j]["volume_ul"] = max(pockets[j]["volume_ul"], cands[i][2])
        for i, (z, d, v) in enumerate(cands):
            if i not in used_i:
                pockets.append({"axial_mm": z, "d_axial_mm": d, "volume_ul": v, "seen": {r.label: v}})
    pockets.sort(key=lambda p: -p["volume_ul"])
    return pockets


def combine(records, cfg):
    """Session result from all views; see match_pockets. Headspace: largest reading."""
    usable = [r for r in records if r.usable]
    missing = [r.label for r in records if not r.usable]
    notes = []
    if not usable:
        return {"verdict": "INVALID", "max_pocket_ul": None, "total_ul": None, "headspace_ul": None,
                "views_used": 0, "pockets": [], "notes": ["no usable photo"]}
    pockets = match_pockets(usable)
    heads = [r.assessment.result.headspace_ul for r in usable]
    head = max(heads)
    vols = [p["volume_ul"] for p in pockets]
    mx = max(vols + [head])
    tot = sum(vols) + head
    v = verdict(mx, tot, cfg)
    if missing:
        v = "INVALID"
        notes.append("no usable photo for: " + ", ".join(missing))
    if len(usable) > 1:
        for p in pockets:
            seen = list(p["seen"].values())
            if len(seen) > 1 and max(seen) - min(seen) > max(3.0, 0.3 * max(seen)):
                notes.append(f"the pocket {p['axial_mm']:.1f} mm from the tip end reads "
                             f"{min(seen):.1f} to {max(seen):.1f} uL in different views (largest used)")
            elif len(seen) == 1 and p["volume_ul"] > 3.0:
                notes.append(f"a {p['volume_ul']:.1f} uL pocket {p['axial_mm']:.1f} mm from the tip end "
                             f"was seen only in the {next(iter(p['seen']))} view")
        if max(heads) - min(heads) > 1.5 * cfg.ul_per_axial_px:
            notes.append(f"views disagree on headspace ({min(heads):.1f} to {max(heads):.1f} uL); "
                         "was the syringe moved or tilted between photos?")
    return {"verdict": v, "max_pocket_ul": mx, "total_ul": tot, "headspace_ul": head,
            "views_used": len(usable), "pockets": pockets, "notes": notes}


# --------------------------------------------------------------------------- sources

def ask(prompt, options, default):
    """Ask until the answer starts like one of options. 'q' quits."""
    keys = "/".join(o.upper() if o == default else o for o in options)
    while True:
        try:
            ans = input(f"  {prompt} [{keys}] ").strip().lower()
        except EOFError:
            raise Abort("input closed")
        if not ans:
            return default
        if ans in ("q", "quit") and not any(o.startswith("q") for o in options):
            raise Abort("quit by operator")
        for o in options:
            if o.startswith(ans[0]):
                return o
        print(f"    Please answer one of: {', '.join(options)} (or q to quit)")


def clean_path(s):
    """A path typed or dragged into a terminal: strip quotes, undo backslash-escaped spaces."""
    s = s.strip().strip('"').strip("'").strip()
    if not os.path.exists(s) and "\\ " in s:
        s = s.replace("\\ ", " ")
    return s


class PromptSource:
    """Operator types or drags the photo file for each view into the terminal."""
    interactive = True

    def capture(self, step, title, how):
        while True:
            try:
                s = input("  Photo file for this view (drag it here), 's' to skip, 'q' to quit: ")
            except EOFError:
                raise Abort("input closed")
            s = clean_path(s)
            if s.lower() in ("q", "quit"):
                raise Abort("quit by operator")
            if s.lower() in ("s", "skip"):
                return None, ""
            if not s:
                continue
            if s.lower().endswith((".heic", ".heif")):
                print("    HEIC photos cannot be read. Set the camera to JPEG ('Most compatible') "
                      "or convert the file.")
                continue
            img = read_image(s)
            if img is None:
                print(f"    Could not read '{s}'.")
                continue
            return img, os.path.abspath(s)

    def decide(self, a, annotated_path):
        print(f"  Annotated photo: {annotated_path}")
        status = photo_status(a)
        if status == FAIL:
            print("  This photo cannot be used. Fix the problems above and retake it.")
            return ask("Retake or skip this view (skipping makes the session INVALID)?",
                       ["retake", "skip"], "retake")
        if status == WARN:
            print("  Accepted (the warnings above are noted in the report).")
        return "accept"

    def close(self):
        pass


class ListSource:
    """Photos given on the command line, one per view, no questions."""
    interactive = False

    def __init__(self, paths):
        self.paths = list(paths)

    def capture(self, step, title, how):
        path = self.paths[step - 1]
        img = read_image(path)
        if img is None:
            print(f"  Could not read '{path}'.")
            return None, path
        return img, os.path.abspath(path)

    def decide(self, a, annotated_path):
        return "accept" if a.status != FAIL else "skip"

    def close(self):
        pass


def fit(img, max_h=900, max_w=1600):
    s = min(1.0, max_h / img.shape[0], max_w / img.shape[1])
    return img if s >= 1.0 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def banner(img, lines, color=(255, 255, 255)):
    """Text bar above an image. lines: list of (text, BGR) or plain strings."""
    s = max(1.0, img.shape[1] / 1200.0)
    lh = int(26 * s)
    bar = np.zeros((lh * len(lines) + int(10 * s), img.shape[1], 3), np.uint8)
    for i, line in enumerate(lines):
        text, col = line if isinstance(line, tuple) else (line, color)
        cv2.putText(bar, text, (int(10 * s), lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, col,
                    max(1, int(round(s))), cv2.LINE_AA)
    return np.vstack([bar, img])


class CameraSource:
    """Live preview with the step instructions and running photo checks; SPACE captures."""
    interactive = True
    WIN = "Guided syringe inspection"

    def __init__(self, args, cfg, ref):
        self.cfg, self.ref, self.camera = cfg, ref, args.camera
        self.cap = sbi.open_capture(camera=args.camera, width=args.width, height=args.height,
                                    exposure=args.exposure)
        if not self.cap.isOpened():
            raise Abort(f"could not open camera {args.camera}")

    def capture(self, step, title, how):
        n, checks, cfg = 0, [], self.cfg
        while True:
            ok, frame = self.cap.read()
            if not ok:
                raise Abort("camera stopped delivering frames")
            if n % 10 == 0:          # the checks are slower than the camera; refresh a few times a second
                f2, cfg, checks = align_to_reference(frame, self.cfg, self.ref)
                if not any(c.status == FAIL and c.name in ("geometry", "roi") for c in checks):
                    checks = checks + check_photo(f2, cfg)
            n += 1
            st = worst(c.status for c in checks)
            disp = frame.copy()
            x, y, w, h = [int(v) for v in cfg.roi]
            cv2.rectangle(disp, (x, y), (x + w, y + h), STATUS_COLOR[st], max(2, disp.shape[0] // 400))
            lines = [(f"Step {step}: {title} - {how}", (255, 255, 255))]
            lines += [(f"{c.status} {c.name}: {c.fix or c.message}", STATUS_COLOR[c.status])
                      for c in checks if c.status in (WARN, FAIL)][:4]
            lines.append(("SPACE capture   S skip view   Q quit", (200, 200, 200)))
            cv2.imshow(self.WIN, fit(banner(disp, lines)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                return frame.copy(), f"camera {self.camera}"
            if key == ord("s"):
                return None, ""
            if key in (ord("q"), 27):
                raise Abort("quit by operator")

    def decide(self, a, annotated_path):
        img = read_image(annotated_path) if annotated_path else None
        if photo_status(a) == FAIL:
            keys, hint = {13: "retake", ord("r"): "retake", ord("s"): "skip"}, \
                "Photo cannot be used.  ENTER/R retake   S skip view (session becomes INVALID)   Q quit"
        else:
            keys, hint = {13: "accept", ord("a"): "accept", ord("r"): "retake"}, \
                "ENTER/A accept   R retake   Q quit"
        if img is not None:
            cv2.imshow(self.WIN, fit(banner(img, [(hint, VERDICT_COLOR[a.verdict])])))
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in keys:
                return keys[key]
            if key in (ord("q"), 27):
                raise Abort("quit by operator")

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- reports

def tool_version():
    """Version, plus the git commit and whether the code differs from it."""
    v = sbi.VERSION
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", "*.py"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        if commit:
            v += f" ({commit}{', modified' if dirty else ''})"
    except (OSError, subprocess.SubprocessError):
        pass
    return v


def view_dict(r):
    a = r.assessment
    d = {"view": r.key, "label": r.label, "source": r.source, "photo": r.photo,
         "photo_sha256": r.sha256, "annotated": r.annotated, "debug": r.debug,
         "verdict": a.verdict if a else "MISSING", "attempts": r.attempts}
    if a is not None:
        d["checks"] = [{"name": c.name, "status": c.status, "value": c.value,
                        "message": c.message, "fix": c.fix} for c in a.checks]
        d["roi_used"] = [int(v) for v in a.cfg.roi]
    if a is not None and a.result is not None:
        res = a.result
        d.update(max_pocket_ul=round(res.max_pocket_ul, 2), total_ul=round(res.total_ul, 2),
                 headspace_ul=round(res.headspace_ul, 2), n_bubbles=len(res.bubbles),
                 bubbles=[{"axial_mm": round(axial_mm(b, a.cfg), 2),
                           "d_axial_mm": round(b.d_axial_mm, 3), "d_lateral_mm": round(b.d_lateral_mm, 3),
                           "volume_ul": round(b.volume_ul, 3), "regular": b.regular,
                           "center_px": [int(b.center[0]), int(b.center[1])]} for b in res.bubbles])
    return d


def fmt_ul(v, unit="uL"):
    return "-" if v is None else f"{v:.1f} {unit}"


def write_html(path, rep):
    e = html.escape
    colors = {"PASS": "#1f8a3b", "MARGINAL": "#b26a00", "FAIL": "#c62828", "INVALID": "#616161",
              "MISSING": "#616161", OK: "#1f8a3b", WARN: "#b26a00", FAIL: "#c62828", SKIP: "#757575"}
    valid = rep["verdict"] != "INVALID"

    def badge(text):
        return f'<span class="badge" style="background:{colors.get(text, "#616161")}">{e(text)}</span>'

    def ul(v):
        return fmt_ul(v, "µL")

    def link(f, text=None):
        return f'<a href="{e(f)}">{e(text or f)}</a>' if f else "-"

    summary_note = "" if valid else (
        '<p class="warn">No verdict: the values below are partial readings from the usable '
        'photos only and must not be used.</p>')
    rows = []
    for v in rep["views"]:
        rows.append(f"<tr><td>{e(v['label'])}</td><td>{badge(v['verdict'])}</td>"
                    f"<td>{ul(v.get('max_pocket_ul'))}</td><td>{ul(v.get('total_ul'))}</td>"
                    f"<td>{ul(v.get('headspace_ul'))}</td><td>{v.get('n_bubbles', '-')}</td>"
                    f"<td>{len(v['attempts'])}</td></tr>")
    pockets = "".join(
        f"<tr><td>{i}</td><td>{p['axial_mm']:.1f} mm</td><td>{p['d_axial_mm']:.2f} mm</td>"
        f"<td>{ul(p['volume_ul'])}</td><td>{e(', '.join(f'{k} {x:.1f}' for k, x in p['seen'].items()))}</td></tr>"
        for i, p in enumerate(rep["pockets"][:40], 1))
    if len(rep["pockets"]) > 40:
        pockets += f"<tr><td colspan=5>... {len(rep['pockets']) - 40} more in report.json</td></tr>"
    sections = []
    for v in rep["views"]:
        checks = "".join(
            f"<tr><td>{badge(c['status'])}</td><td>{e(c['name'])}</td><td>{e(c['value'])}</td>"
            f"<td>{e(c['message'])}{'<br><em>' + e(c['fix']) + '</em>' if c['fix'] and c['status'] != OK else ''}</td></tr>"
            for c in v.get("checks", []) if c["status"] != SKIP)
        attempts = "".join(
            f"<li>Photo {t['n']}: {badge(t['verdict'])} {e(t['outcome'])}"
            f"{' - ' + e(', '.join(t['problems'])) if t['problems'] else ''} "
            f"({link(t['photo'], 'photo')}, {link(t['annotated'], 'annotated')})</li>"
            for t in v["attempts"])
        img = (f'<a href="{e(v["annotated"])}"><img src="{e(v["annotated"])}" alt="{e(v["label"])} annotated"></a>'
               if v["annotated"] else "<p>No photo.</p>")
        sections.append(
            f"<section><h3>{e(v['label'])} view &middot; view result {badge(v['verdict'])}</h3>"
            f"<p class='muted'>Original photo: {link(v['photo'])} &middot; source {e(v['source'] or '-')}"
            f"<br>SHA-256 {e(v['photo_sha256'] or '-')}</p>{img}"
            f"<table><tr><th>Status</th><th>Check</th><th>Value</th><th>Detail</th></tr>{checks}</table>"
            f"<p><b>Photos taken for this view</b></p><ul>{attempts or '<li>None</li>'}</ul></section>")
    notes = "".join(f"<li>{e(n)}</li>" for n in rep["notes"]) or "<li>None</li>"
    c, s = rep["calibration"], rep["sample_info"]
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Syringe air report - {e(rep['sample'])}</title>
<style>
 body {{ font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; color: #222; background: #fff;
        max-width: 1100px; margin: 0 auto; padding: 24px 16px; }}
 h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 17px; margin-top: 28px; }} h3 {{ font-size: 15px; }}
 .muted {{ color: #555; }} .warn {{ color: #c62828; font-weight: 600; }}
 .badge {{ color: #fff; border-radius: 4px; padding: 1px 8px; font-weight: 600; font-size: 12px; }}
 .verdict {{ font-size: 28px; font-weight: 700; color: {colors.get(rep['verdict'], '#616161')}; }}
 table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; }}
 th, td {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid #e3e3e3; vertical-align: top; }}
 th {{ background: #f5f5f5; font-weight: 600; }}
 img {{ max-width: 100%; max-height: 640px; border: 1px solid #ddd; }}
 section {{ margin-bottom: 28px; }} code {{ font-size: 12px; }}
 @media print {{ section {{ page-break-inside: avoid; }} }}
</style></head><body>
<h1>Syringe residual-air report</h1>
<table>
<tr><th>Sample</th><td>{e(rep['sample'])}</td><th>Operator</th><td>{e(rep['operator'] or '-')}</td></tr>
<tr><th>Description</th><td colspan=3>{e(s['description'] or '-')}</td></tr>
<tr><th>Fill volume</th><td>{s['fill_ml']:g} mL</td><th>Checklist</th><td>{e(s['checklist'])}</td></tr>
<tr><th>Started</th><td>{e(rep['started'])}</td><th>Finished</th><td>{e(rep['finished'])}</td></tr>
<tr><th>Tool</th><td colspan=3>v{e(rep['tool_version'])}</td></tr>
</table>
<div class="verdict">{e(rep['verdict'])}</div>
{summary_note}
<table>
<tr><th>Quantity</th><th>Measured</th><th>Limit</th></tr>
<tr><td>Largest single air pocket</td><td>{ul(rep['max_pocket_ul'])}</td><td>&le; {rep['spec']['max_pocket_ul']:.0f} µL</td></tr>
<tr><td>Total air</td><td>{ul(rep['total_ul'])}</td><td>&lt; {rep['spec']['total_limit_ul']:.0f} µL ({rep['spec']['fill_ml']:g} mL fill)</td></tr>
<tr><td>of which headspace</td><td>{ul(rep['headspace_ul'])}</td><td></td></tr>
</table>
<p class="muted">{rep['views_used']} of {rep['views_planned']} views usable. A bubble seen in several views is
counted once (its largest reading); a bubble seen in only one view is still counted. MARGINAL means within
{rep['spec']['margin']:.0%} below a limit. INVALID means a photo failed its quality checks or a view is missing,
so no verdict is given.</p>
<p><b>What is measured.</b> {e(MEASURAND)}</p>
<p><b>Accuracy.</b> {e(ACCURACY)}</p>
<h2>Notes</h2><ul>{notes}</ul>
<h2>Views</h2>
<table><tr><th>View</th><th>View result</th><th>Largest pocket</th><th>Total air</th><th>Headspace</th><th>Bubbles</th><th>Photos taken</th></tr>
{''.join(rows)}</table>
<h2>Air pockets</h2>
<table><tr><th>#</th><th>From tip end</th><th>Axial size</th><th>Volume</th><th>Readings by view (µL)</th></tr>
{pockets or '<tr><td colspan=5>None found</td></tr>'}</table>
{''.join(sections)}
<h2>Calibration</h2>
<p>Copy in this folder: {link(rep['calibration_copy'])}, photo {link(rep['calibration_photo_copy'])}.
Calibrated {e(c['calibrated_at'] or 'unknown')}. Scale {c['mm_per_px'] * 1000:.1f} µm per pixel; barrel inner
diameter {c['barrel_id_mm']:g} mm; one pixel of liquid level = {c['ul_per_px']:.1f} µL; sideways magnification of
the liquid {c['lateral_mag']:g}; detection threshold {c['dark_thresh']:.0%} darker than the background.</p>
</body></html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


# ------------------------------------------------------------------------------ flow

def offer_setup(args):
    print(f"No calibration found at {args.config}. Calibration is needed once per camera position.")
    if ask("Calibrate now?", ["yes", "no"], "yes") != "yes":
        return False
    try:
        src = clean_path(input("  Calibration photo file (ENTER to use the camera instead): "))
    except EOFError:
        raise Abort("input closed")
    air_end = ask("Which side of the picture is the syringe tip on?",
                  ["top", "bottom", "left", "right"], "top")
    ns = argparse.Namespace(config=args.config, camera=args.camera if args.camera is not None else 0,
                            image=src or None, width=args.width, height=args.height, air_end=air_end,
                            fill_ml=args.fill_ml or 10.0, lateral_mag=1.33, roi=None,
                            mm_per_px=None, barrel_id_mm=None, syringe=None, exclusions=False,
                            no_exclusions=False)
    sbi.cmd_setup(ns)
    return os.path.exists(args.config)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "sample"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def earlier_hashes(out_root):
    """Photo hashes used in earlier sessions: {sha256: session folder}."""
    seen = {}
    for rp in glob.glob(os.path.join(out_root, "*", "report.json")):
        try:
            with open(rp, encoding="utf-8") as f:
                for v in json.load(f).get("views", []):
                    for t in v.get("attempts", []):
                        if t.get("sha256"):
                            seen[t["sha256"]] = os.path.basename(os.path.dirname(rp))
        except (OSError, ValueError):
            continue
    return seen


def duplicate_checks(digest, frame, records, current, history):
    """FAIL if this photo is one already used for another view of this session (the same
    file, or practically identical pixels); WARN if it was used in an earlier session."""
    checks = []
    for r in records:
        if r is current or r.assessment is None:
            continue
        same = r.sha256 == digest
        if not same and r.assessment.frame.shape == frame.shape:
            same = float(np.mean(cv2.absdiff(r.assessment.frame, frame))) < 0.5
        if same:
            checks.append(Check("duplicate", FAIL, r.label, f"this is the same photo as the {r.label} view",
                                "Turn the syringe as instructed and take a new photo."))
    if digest in history:
        checks.append(Check("duplicate", WARN, history[digest],
                            f"this photo was already used in session {history[digest]}",
                            "Take a new photo of this sample."))
    return checks


def run_view(rec, step, records, source, cfg, ref, out_dir, debug, history):
    print(f"\n[{step}/{len(records)}] {rec.label}: {rec.how}.")
    while True:
        frame, src = source.capture(step, rec.label, rec.how)
        if frame is None:
            print("  View skipped." if not src else "  No usable photo.")
            rec.assessment, rec.source = None, src
            return
        n = len(rec.attempts) + 1
        is_file = os.path.isfile(src)
        digest = file_sha256(src) if is_file else hashlib.sha256(frame.tobytes()).hexdigest()
        # The printed scale is only where the calibration boxed it in the front view.
        a = assess(frame, cfg, ref, use_exclusions=(rec.key == "front"))
        a.checks += duplicate_checks(digest, a.frame, records, rec, history)
        stem = f"{rec.key}_attempt{n}"
        photo = stem + (os.path.splitext(src)[1].lower() if is_file else ".png")
        if is_file:
            shutil.copy2(src, os.path.join(out_dir, photo))        # the original, unchanged
        else:
            write_image(os.path.join(out_dir, photo), frame)
        annotated = stem + "_annotated.png"
        write_image(os.path.join(out_dir, annotated), annotate(a.frame, a.cfg, a.result, checks=a.checks))
        res = a.result
        if res is not None:
            tail = "" if a.verdict != "INVALID" else " (not valid)"
            print(f"  View result {a.verdict}: largest pocket {res.max_pocket_ul:.1f} uL, total "
                  f"{res.total_ul:.1f} uL, {len(res.bubbles)} bubbles, headspace {res.headspace_ul:.1f} uL{tail}")
        else:
            print(f"  View result {a.verdict}: photo could not be measured")
        shown = [c for c in a.checks if c.status not in (OK, SKIP)
                 and (c.status == FAIL or c.name not in SETUP_CHECKS)]
        if shown:
            print(format_checks(shown))
        decision = source.decide(a, os.path.join(out_dir, annotated))
        rec.attempts.append({"n": n, "photo": photo, "annotated": annotated, "sha256": digest,
                             "source": src, "verdict": a.verdict,
                             "outcome": {"accept": "accepted", "retake": "retaken",
                                         "skip": "kept, view skipped"}[decision],
                             "problems": [f"{c.name} {c.status}" for c in a.problems]})
        if decision == "retake":
            print("  Retake: fix the problems listed, then provide a new photo.")
            continue
        # accepted, or skipped after a failed photo (kept so the report shows why)
        rec.assessment, rec.source, rec.sha256 = a, src, digest
        rec.photo, rec.annotated = photo, annotated
        if debug and res is not None:
            rec.debug = f"{rec.key}_debug.png"
            write_image(os.path.join(out_dir, rec.debug), debug_panel(res))
        return


def prompt(text, default=""):
    try:
        ans = input(f"{text}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:
        raise Abort("input closed")
    return ans or default


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Guided still-photo inspection of a filled syringe "
                                             "against the 25 uL / 250 uL-per-10 mL spec.")
    ap.add_argument("photo", nargs="*", help="photo(s) of the syringe: front, then optionally "
                                             "turned 90/180/270 degrees. None: you are asked.")
    ap.add_argument("--config", default="bubble_config.json", help="calibration written by setup")
    ap.add_argument("--sample", help="sample / run ID for the report (default: date and time)")
    ap.add_argument("--operator", default="", help="name or initials for the report")
    ap.add_argument("--description", default="", help="syringe, liquid, device setting, ...")
    ap.add_argument("--camera", type=int, help="capture from this camera instead of photo files")
    ap.add_argument("--photos", nargs="+", help=argparse.SUPPRESS)      # older spelling
    ap.add_argument("--views", type=int, choices=range(1, len(VIEWS) + 1),
                    help="number of views when asking for photos (default 1; 2 adds a quarter turn)")
    ap.add_argument("--out", default="sessions", help="folder for session reports")
    ap.add_argument("--fill-ml", type=float, help="liquid volume in this syringe (default: from config)")
    ap.add_argument("--settle", type=int, default=30, help="seconds the syringe rests before the photos")
    ap.add_argument("--debug", action="store_true", help="also save the detector's intermediate masks")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--exposure", type=float, help="lock camera exposure (driver-specific)")
    args = ap.parse_args(argv)
    args.photos = args.photos or args.photo or None
    if args.photos and args.camera is not None:
        ap.error("give photos or --camera, not both")
    if args.photos:
        if len(args.photos) > len(VIEWS):
            ap.error(f"at most {len(VIEWS)} photos (front, then turned 90, 180, 270 degrees)")
        if args.views and args.views != len(args.photos):
            ap.error(f"--views {args.views} but {len(args.photos)} photo(s) given")
        args.views = len(args.photos)
    args.views = args.views or 1
    return args


def session(args):
    interactive = not args.photos
    views = [ViewRecord(k, label, how) for k, label, how in VIEWS[:args.views]]

    if not os.path.exists(args.config) and not (interactive and offer_setup(args)):
        print(f"No calibration. Run:  python syringe_bubble_inspector.py setup --image <photo> "
              f"--config {args.config}")
        return 2
    cfg = Config.load(args.config)
    ref = load_reference(cfg, args.config)
    if ref is None and not cfg.ref_image:
        print("Note: this calibration has no stored photo, so position, size and focus cannot be "
              "checked. Run setup again to enable those checks.")

    sample = args.sample or time.strftime("%Y-%m-%d_%H%M%S")
    operator, description = args.operator, args.description
    fill = args.fill_ml or cfg.fill_ml
    if fill <= 0:
        raise Abort("the fill volume must be positive")
    cfg.fill_ml = fill

    res_check = sbi.check_resolution(cfg)
    if res_check.status != OK:
        print("\nCalibration warning (applies to every photo):")
        print(format_checks([res_check]))
    if interactive:
        print("\nQuick reminder:")
        for item in CHECKLIST:
            print("  - " + item.format(settle=args.settle))
    checklist = "reminder shown" if interactive else "photos given on the command line"

    source = (ListSource(args.photos) if args.photos else
              CameraSource(args, cfg, ref) if args.camera is not None else PromptSource())
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.out, f"{stamp}_{safe_name(sample)}" if args.sample else stamp)
    out_dir, n = base, 2
    while os.path.exists(out_dir):            # never overwrite an earlier session
        out_dir, n = f"{base}_{n}", n + 1
    history = earlier_hashes(args.out)
    os.makedirs(out_dir)

    aborted = None
    try:
        for i, rec in enumerate(views, 1):
            run_view(rec, i, views, source, cfg, ref, out_dir, args.debug, history)
    except Abort as exc:
        aborted = str(exc)
    except KeyboardInterrupt:
        aborted = "interrupted"
    finally:
        source.close()

    summary = combine(views, cfg)
    if aborted:
        summary["verdict"] = "INVALID"
        summary["notes"].insert(0, f"session stopped ({aborted})")
    calib_copy = "calibration.json"
    shutil.copy2(args.config, os.path.join(out_dir, calib_copy))
    ref_copy = ""
    if ref is not None:
        ref_copy = "calibration_photo.png"
        write_image(os.path.join(out_dir, ref_copy), ref)
    rep = {
        "sample": sample, "operator": operator, "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "tool_version": tool_version(),
        "sample_info": {"description": description, "fill_ml": cfg.fill_ml, "checklist": checklist,
                        "settle_s": args.settle},
        "verdict": summary["verdict"], "max_pocket_ul": summary["max_pocket_ul"],
        "total_ul": summary["total_ul"], "headspace_ul": summary["headspace_ul"],
        "views_used": summary["views_used"], "views_planned": len(views), "notes": summary["notes"],
        "pockets": [{**p, "axial_mm": round(p["axial_mm"], 2), "d_axial_mm": round(p["d_axial_mm"], 3),
                     "volume_ul": round(p["volume_ul"], 3)} for p in summary["pockets"]],
        "measurand": MEASURAND, "accuracy": ACCURACY,
        "spec": {"max_pocket_ul": SPEC_MAX_POCKET_UL, "total_limit_ul": cfg.total_limit_ul,
                 "fill_ml": cfg.fill_ml, "margin": cfg.margin},
        "config_path": os.path.abspath(args.config), "calibration_copy": calib_copy,
        "calibration_photo_copy": ref_copy,
        "calibration": {"mm_per_px": cfg.mm_per_px, "barrel_id_mm": cfg.barrel_id_mm,
                        "lateral_mag": cfg.lateral_mag, "dark_thresh": cfg.dark_thresh,
                        "air_end": cfg.air_end, "ul_per_px": cfg.ul_per_axial_px,
                        "calibrated_at": cfg.calibrated_at},
        "views": [view_dict(r) for r in views],
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    write_html(os.path.join(out_dir, "report.html"), rep)
    valid = rep["verdict"] != "INVALID"
    log_row(os.path.join(args.out, "sessions.csv"), {
        "timestamp": started, "sample": sample, "operator": operator, "verdict": rep["verdict"],
        # an INVALID session has no valid numbers; leave them out so nobody uses them
        "max_pocket_ul": f"{rep['max_pocket_ul']:.2f}" if valid else "",
        "total_ul": f"{rep['total_ul']:.2f}" if valid else "",
        "fill_ml": f"{cfg.fill_ml:g}", "views_used": rep["views_used"], "views_planned": len(views),
        "notes": " | ".join(rep["notes"]), "report": os.path.join(out_dir, "report.html")},
        SESSION_FIELDS)

    print(f"\n=== {sample}: {rep['verdict']} ===")
    if valid:
        print(f"  Largest pocket {fmt_ul(rep['max_pocket_ul'])} (limit {SPEC_MAX_POCKET_UL:.0f} uL) | "
              f"total {fmt_ul(rep['total_ul'])} (limit {cfg.total_limit_ul:.0f} uL) | "
              f"{rep['views_used']}/{len(views)} views")
    else:
        print(f"  No verdict ({rep['views_used']}/{len(views)} views usable). Fix the problems and "
              "inspect this sample again.")
    for n in rep["notes"]:
        print(f"  Note: {n}")
    print(f"  Report: {os.path.join(out_dir, 'report.html')}")
    return EXIT_CODE[rep["verdict"]]


def main(argv=None):
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):   # a Windows console may not encode every name
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        return session(args)
    except (Abort, KeyboardInterrupt) as exc:
        print(f"\nStopped: {exc or 'interrupted'}")
        return 2
    except SystemExit as exc:          # e.g. setup cancelled or failed from inside the guide
        if exc.code not in (None, 0):
            print(exc.code)
        print("Stopped; no report was written.")
        return 2                       # never 0: a stopped session must not look like a PASS
    except Exception:
        traceback.print_exc()
        print("\nUnexpected error; no report was written. Please report this with the traceback.")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
