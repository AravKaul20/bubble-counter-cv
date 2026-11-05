#!/usr/bin/env python3
"""
batch_inspect.py

Mass testing: check a whole folder of syringe photos with one command.

    python batch_inspect.py photos/

How to take the photos (phone, after calibrating once with syringe_bubble_inspector.py setup):
  for each syringe: put it on its spot, let it rest, photo, turn it halfway round about
  its own axis, photo. Then copy all the photos into one folder and run this.

Which photos belong to which syringe:
  * flat folder (photos straight off the phone): photos taken within --gap seconds
    (default 20) of each other are one syringe, using the time the phone stored in each
    photo. So wait a little longer than that between syringes (the syringe should rest
    30 s anyway). Delete a photo you retook, or that syringe is flagged.
    Photos without a stored time (screenshots, edited files) are taken in name order,
    two per syringe.
  * one subfolder per syringe: photos/S12/*.jpg, photos/S13/*.jpg; the folder name is
    the sample ID. Use this if you prefer naming samples yourself.

Each syringe gets one of:
  PASS / MARGINAL / FAIL   the measurement, against 25 uL per pocket and 25 uL per mL total
  RETAKE                   a photo was unusable (lighting, focus, moved, ...) or photos are
                           missing; the reason says what to fix. No numbers are given.
  ERROR                    the program failed on this syringe (please report it)

Output: a results table on screen, batch_<time>_<folder>.csv in the output folder with one
row per syringe, and a full report per syringe (annotated photos, all checks) under it.
Exit code: 0 all PASS, 1 any MARGINAL or FAIL, 2 any RETAKE, 3 any ERROR or setup problem.

Requires: opencv-python, numpy; Pillow to read photo times (otherwise name order is used).
"""

import argparse
import contextlib
import io
import os
import re
import sys
import time
import traceback
from datetime import datetime

import guided_inspection as gi
import syringe_bubble_inspector as sbi
from syringe_bubble_inspector import FAIL, OK, SPEC_MAX_POCKET_UL, Config, log_row

try:
    from PIL import Image
except ImportError:          # optional: without it photos are paired in name order
    Image = None

PHOTO_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
HEIC_EXT = (".heic", ".heif")
BATCH_FIELDS = ["sample", "result", "largest_pocket_ul", "total_air_ul", "reason", "notes",
                "photos", "report"]
RESULT = {"PASS": "PASS", "MARGINAL": "MARGINAL", "FAIL": "FAIL", "INVALID": "RETAKE"}
EXIT = {"PASS": 0, "MARGINAL": 1, "FAIL": 1, "RETAKE": 2, "ERROR": 3}


class LayoutError(Exception):
    pass


# --------------------------------------------------------------------------- grouping

def natural_key(name):
    """IMG_9.jpg before IMG_10.jpg."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def capture_time(path):
    """When the photo was taken (EXIF DateTimeOriginal), or None."""
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            s = im.getexif().get_ifd(0x8769).get(0x9003)
        return datetime.strptime(str(s).strip("\x00 "), "%Y:%m:%d %H:%M:%S") if s else None
    except Exception:
        return None


def photos_in(folder):
    names = sorted(os.listdir(folder), key=natural_key)
    paths = [os.path.join(folder, n) for n in names if os.path.isfile(os.path.join(folder, n))]
    heic = [p for p in paths if p.lower().endswith(HEIC_EXT)]
    if heic:
        raise LayoutError(f"{len(heic)} HEIC photo(s) cannot be read, e.g. {os.path.basename(heic[0])}. "
                          "Set the phone camera to JPEG (iPhone: Settings > Camera > Formats > "
                          "Most Compatible) or convert them.")
    return [p for p in paths if p.lower().endswith(PHOTO_EXT)]


def by_time(paths):
    """Sorted by capture time, then name (several photos can share one second)."""
    return sorted(paths, key=lambda p: (capture_time(p) or datetime.min,
                                        natural_key(os.path.basename(p))))


def group_photos(folder, views, gap, skip=()):
    """[(sample ID, [photo paths], problem or None)], one entry per syringe."""
    skip = {os.path.abspath(s) for s in skip}
    subdirs = [d for d in sorted(os.listdir(folder), key=natural_key)
               if os.path.isdir(os.path.join(folder, d))
               and os.path.abspath(os.path.join(folder, d)) not in skip]
    loose = photos_in(folder)
    per_dir = [(d, photos_in(os.path.join(folder, d))) for d in subdirs]
    per_dir = [(d, p) for d, p in per_dir if p]
    if per_dir and loose:
        raise LayoutError(f"{folder} has photos both loose and in subfolders. Use one layout: "
                          "all photos in one folder, or one subfolder per syringe.")
    if per_dir:
        return [(d, by_time(p), None if len(p) == views else
                 f"{len(p)} photo(s) in this folder, expected {views}") for d, p in per_dir]
    if not loose:
        raise LayoutError(f"no photos ({', '.join(PHOTO_EXT)}) found in {folder}")

    batch = gi.safe_name(os.path.basename(os.path.abspath(folder)))
    times = [capture_time(p) for p in loose]
    if views == 1:
        groups = [[p] for p in by_time(loose)]
    elif all(times):
        groups, last = [], None
        for p in by_time(loose):
            t = capture_time(p)
            if last is None or (t - last).total_seconds() > gap:
                groups.append([])
            groups[-1].append(p)
            last = t
    else:
        if len(loose) % views:
            raise LayoutError(
                f"{len(loose)} photos without a stored capture time, which is not a multiple of {views}, "
                f"so it is not clear which photos belong together. Remove extra photos, or put each "
                f"syringe's photos in its own subfolder.")
        groups = [loose[i:i + views] for i in range(0, len(loose), views)]
    out = []
    for i, g in enumerate(groups, 1):
        problem = None
        if len(g) != views:
            problem = (f"{len(g)} photo(s) taken within {gap:g} s of each other, expected {views}. "
                       f"Delete retaken photos; if syringes were changed faster than {gap:g} s, or one "
                       f"syringe took longer, set --gap")
        out.append((f"{batch}-{i:03d}", g, problem))
    return out


# ---------------------------------------------------------------------------- results

def reason_for(rep):
    """One line: why the syringe did not pass, or what to fix for a retake."""
    v = rep["verdict"]
    if v == "INVALID":
        parts = []
        for view in rep["views"]:
            if view["verdict"] not in ("INVALID", "MISSING"):
                continue
            fails = [c for c in view.get("checks", []) if c["status"] == FAIL]
            if fails:
                c = fails[0]
                parts.append(f"{view['label']} photo: {c['name']} - {c['fix'] or c['message']}")
            else:
                parts.append(f"{view['label']} photo could not be read")
        return "; ".join(parts) or "; ".join(rep["notes"]) or "unusable photos"
    spec = rep["spec"]
    over = []
    k = 1.0 - spec["margin"] if v == "MARGINAL" else 1.0
    if rep["max_pocket_ul"] > k * SPEC_MAX_POCKET_UL:
        over.append(f"largest pocket {rep['max_pocket_ul']:.1f} uL (limit {SPEC_MAX_POCKET_UL:.0f})")
    if rep["total_ul"] > k * spec["total_limit_ul"]:
        over.append(f"total air {rep['total_ul']:.1f} uL (limit {spec['total_limit_ul']:.0f})")
    if v == "MARGINAL":
        return "close to the limit: " + ", ".join(over)
    return ", ".join(over) if v == "FAIL" else ""


def inspect_syringe(sample, photos, args, history):
    """Runs the guided session on these photos without questions. Returns a CSV row."""
    argv = ["--config", args.config, "--out", args.out, "--sample", sample,
            "--operator", args.operator, "--description", args.description,
            "--views", str(len(photos))]
    if args.fill_ml:
        argv += ["--fill-ml", str(args.fill_ml)]
    if args.debug:
        argv.append("--debug")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        rep, out_dir = gi.run_session(gi.parse_args(argv + ["--"] + photos), history)
    with open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8") as f:
        f.write(log.getvalue())
    for view in rep["views"]:                          # later syringes: duplicate-photo check
        for t in view["attempts"]:
            history[t["sha256"]] = os.path.basename(out_dir)
    result = RESULT[rep["verdict"]]
    valid = result != "RETAKE"
    return {"result": result, "reason": reason_for(rep), "notes": " | ".join(rep["notes"]),
            "largest_pocket_ul": f"{rep['max_pocket_ul']:.1f}" if valid else "",
            "total_air_ul": f"{rep['total_ul']:.1f}" if valid else "",
            "report": os.path.join(out_dir, "report.html")}


def short(text, n):
    return text if len(text) <= n else text[:n - 3] + "..."


def run(args):
    if not os.path.isdir(args.folder):
        print(f"{args.folder} is not a folder.")
        return 3
    if not os.path.exists(args.config):
        print(f"No calibration at {args.config}. Calibrate once with a photo from the same phone "
              "position:\n  python syringe_bubble_inspector.py setup --image <photo> --air-end top")
        return 3
    cfg = Config.load(args.config)
    if args.fill_ml:
        cfg.fill_ml = args.fill_ml
    try:
        groups = group_photos(args.folder, args.views, args.gap, skip=[args.out])
    except LayoutError as exc:
        print(exc)
        return 3
    res = sbi.check_resolution(cfg)
    if res.status != OK:
        print("Calibration warning (applies to every syringe):\n" + sbi.format_checks([res]))
    if Image is None and args.views > 1:
        print("Note: Pillow is not installed, so photos are paired in name order "
              "(pip install pillow to use the photo times).")

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out, f"batch_{stamp}_{gi.safe_name(os.path.basename(os.path.abspath(args.folder)))}.csv")
    history = gi.earlier_hashes(args.out)
    print(f"{len(groups)} syringe(s) in {args.folder}, {args.views} photo(s) each. "
          f"Limits: {SPEC_MAX_POCKET_UL:.0f} uL per pocket, {cfg.total_limit_ul:.0f} uL total "
          f"({cfg.fill_ml:g} mL fill).\n")
    print(f"  {'sample':<18} {'result':<9} {'largest':>8} {'total':>8}  photos / reason")
    rows = []
    try:
        for sample, photos, problem in groups:
            names = " + ".join(os.path.basename(p) for p in photos)
            if problem:
                row = {"result": "RETAKE", "reason": problem, "notes": "", "report": "",
                       "largest_pocket_ul": "", "total_air_ul": ""}
            else:
                try:
                    row = inspect_syringe(sample, photos, args, history)
                except Exception as exc:
                    row = {"result": "ERROR", "reason": f"{type(exc).__name__}: {exc}", "notes": "",
                           "report": "", "largest_pocket_ul": "", "total_air_ul": ""}
                    with open(csv_path[:-4] + "_errors.log", "a", encoding="utf-8") as f:
                        f.write(f"--- {sample} ({names})\n{traceback.format_exc()}\n")
            row.update(sample=sample, photos=names)
            log_row(csv_path, row, BATCH_FIELDS)
            rows.append(row)
            mx = f"{row['largest_pocket_ul']} uL" if row["largest_pocket_ul"] else "-"
            tot = f"{row['total_air_ul']} uL" if row["total_air_ul"] else "-"
            print(f"  {short(sample, 18):<18} {row['result']:<9} {mx:>8} {tot:>8}  "
                  f"{short(row['reason'] or names, 90)}", flush=True)
    except KeyboardInterrupt:
        print(f"\nStopped after {len(rows)} of {len(groups)} syringe(s).")

    counts = {k: sum(r["result"] == k for r in rows) for k in EXIT}
    print("\n" + "   ".join(f"{k} {n}" for k, n in counts.items() if n))
    retake = [r for r in rows if r["result"] in ("RETAKE", "ERROR")]
    if retake:
        print("\nPhotograph these syringes again (fix the problem first):")
        for r in retake:
            print(f"  {r['sample']} ({r['photos']}): {r['reason']}")
    print(f"\nResults: {csv_path}\nFull reports: {args.out}")
    if len(rows) < len(groups):
        return 2
    return max((EXIT[r["result"]] for r in rows), default=0)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):   # a Windows console may not encode every name
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Check a folder of syringe photos (mass testing).")
    ap.add_argument("folder", help="folder with the photos (or one subfolder per syringe)")
    ap.add_argument("--views", type=int, default=2, choices=[1, 2, 3, 4],
                    help="photos per syringe: front, back (180), then 90 and 270 degrees (default 2)")
    ap.add_argument("--gap", type=float, default=20,
                    help="photos taken more than this many seconds apart are different syringes "
                         "(default 20)")
    ap.add_argument("--config", default="bubble_config.json", help="calibration written by setup")
    ap.add_argument("--out", default="sessions", help="folder for the results and reports")
    ap.add_argument("--fill-ml", type=float, help="liquid volume per syringe (default: from config)")
    ap.add_argument("--operator", default="", help="name or initials for the reports")
    ap.add_argument("--description", default="", help="syringe, liquid, device setting, ...")
    ap.add_argument("--debug", action="store_true", help="also save the detector's masks")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
