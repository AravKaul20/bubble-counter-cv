#!/usr/bin/env python3
"""
Tests for the photo-quality checks and the guided inspection session.

    python test_guided.py          # or: python -m pytest test_guided.py

The checks must do two things, and both are tested: reject photos that do not match
the imaging setup (front lighting, busy background, blur, bad exposure, moved, zoomed
or tilted syringe, wrong camera), and NOT reject good photos, including a syringe with
many bubbles, which must be measured rather than called a bad photo. Several tests
reproduce bugs found in review; their names say what must not happen again.
"""
import io
import json
import os
import sys
import tempfile
import traceback

import cv2
import numpy as np

import guided_inspection as gi
import test_bubble_inspector as T
from syringe_bubble_inspector import (FAIL, OK, WARN, assess, check_resolution, sharpness,
                                      to_gray, write_image)


def status_of(a, name):
    found = [c.status for c in a.checks if c.name == name]
    return found[0] if found else None


def calibrated(ref):
    """Test config plus what setup stores from the calibration photo."""
    cfg = T.config()
    cfg.image_size = [ref.shape[1], ref.shape[0]]
    cfg.ref_sharpness = sharpness(to_gray(ref), cfg)
    return cfg


def photo(rng, bubbles=(), shift=(0, 0)):
    img = T.base_image(rng)
    for x, y, d in bubbles:
        T.add_bubble(img, x, y, d)
    f = T.finish(img, rng)
    if shift != (0, 0):
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        f = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return f


def warp(f, zoom=1.0, tilt=0.0):
    """The same scene with the camera closer (zoom) or the syringe tilted (degrees)."""
    H, W = f.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), tilt, zoom)
    return cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE)


def frontlit(rng):
    """Front-lit syringe in a room: uneven background behind the barrel, bubbles are
    bright highlights with a dark lower edge (as in the real photo that motivated this)."""
    low = cv2.resize(rng.normal(0, 1, (8, 5)).astype(np.float32), (T.FW, T.FH),
                     interpolation=cv2.INTER_CUBIC)
    img = 150 + 35 * low
    yy, xx = np.mgrid[0:T.FH, 0:T.FW].astype(np.float32)
    x0, x1 = T.ROI[0], T.ROI[0] + T.ROI[2]
    for wx in (x0 - 22, x1):
        img[:, wx:wx + 22] *= 0.7
    for _ in range(120):
        cx, cy = rng.uniform(x0 + 10, x1 - 10), rng.uniform(T.ROI[1] + 300, T.ROI[1] + T.ROI[3] - 10)
        r = rng.uniform(4, 9)
        img += 70 * np.exp(-(((xx - cx) ** 2 + (yy - cy + r / 3) ** 2) / (2 * (r / 2) ** 2)))
        img -= 40 * np.exp(-(((xx - cx) ** 2 + (yy - cy - r * 0.8) ** 2) / (2 * (r / 3) ** 2)))
    return T.finish(img, rng)


# ------------------------------------------------------------------------ the gate

def test_good_photos_pass_the_gate():
    for name, frame, _, _, true_v, cfg in T.scenes(np.random.default_rng(1)):
        a = assess(frame, cfg)
        # the synthetic rig's 6.6 uL/px is coarser than recommended: a resolution WARN is expected
        bad = [(c.name, c.status) for c in a.checks
               if c.status in (WARN, FAIL) and c.name != "resolution"]
        assert not bad, f"{name}: good photo flagged {bad}"
        assert a.verdict == true_v, f"{name}: {a.verdict} != {true_v}"


def test_frontlit_photo_is_invalid():
    a = assess(frontlit(np.random.default_rng(2)), T.config())
    assert a.verdict == "INVALID"
    assert FAIL in (status_of(a, "lighting"), status_of(a, "background"), status_of(a, "stability"))


def test_blur_is_rejected():
    rng = np.random.default_rng(3)
    ref = photo(rng)
    cfg = calibrated(ref)
    f = photo(rng, [(380, 500, 1.2)])
    assert status_of(assess(f, cfg, ref), "focus") == OK
    a = assess(cv2.GaussianBlur(f, (0, 0), 4), cfg, ref)
    assert status_of(a, "focus") == FAIL and a.verdict == "INVALID"


def test_exposure_is_checked():
    f = photo(np.random.default_rng(4), [(380, 500, 1.2)])
    dark = (f * 0.25).astype(np.uint8)
    blown = np.clip(f.astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
    for img in (dark, blown):
        a = assess(img, T.config())
        assert status_of(a, "exposure") == FAIL and a.verdict == "INVALID"


def test_small_shift_is_followed_and_measurement_unchanged():
    rng = np.random.default_rng(5)
    ref = photo(rng)
    cfg = calibrated(ref)
    d30 = T.d_for_volume(30)
    for shift in [(0, 0), (6, -4), (-20, 9)]:
        a = assess(photo(rng, [(400, 500, d30)], shift), cfg, ref)
        assert status_of(a, "alignment") in (OK, WARN), shift
        dx, dy = a.cfg.roi[0] - T.ROI[0], a.cfg.roi[1] - T.ROI[1]
        assert abs(dx - shift[0]) <= 1.5 and abs(dy - shift[1]) <= 1.5, (shift, dx, dy)
        assert abs(a.result.max_pocket_ul - 30) < 4.5 and a.verdict == "FAIL", (shift, a.result.max_pocket_ul)


def test_large_shift_is_rejected():
    rng = np.random.default_rng(6)
    ref = photo(rng)
    a = assess(photo(rng, [(400, 500, 1.0)], (130, 0)), calibrated(ref), ref)
    assert status_of(a, "alignment") == FAIL and a.verdict == "INVALID"


def test_zoom_and_tilt_are_rejected_not_measured():
    # Review finding: a 7% zoom turned a 20 uL bubble into a 25.8 uL FAIL with every check OK.
    rng = np.random.default_rng(12)
    ref = photo(rng)
    cfg = calibrated(ref)
    f = photo(rng, [(400, 500, T.d_for_volume(20))])
    assert status_of(assess(f, cfg, ref), "scale") == OK
    for zoom, tilt in [(1.07, 0), (0.93, 0), (1.0, 8.0)]:
        a = assess(warp(f, zoom, tilt), cfg, ref)
        assert a.verdict == "INVALID", (zoom, tilt, a.verdict, [(c.name, c.value) for c in a.problems])
    for zoom, tilt in [(1.03, 0), (1.0, 2.0)]:           # small changes: warned, still measured
        a = assess(warp(f, zoom, tilt), cfg, ref)
        assert status_of(a, "scale") == WARN and a.verdict != "INVALID", (zoom, tilt, a.verdict)


def test_quarter_view_without_printed_scale_aligns():
    # The turned views have no graduation ticks in front; alignment uses walls and ends.
    rng = np.random.default_rng(13)
    ref = photo(rng)
    cfg = calibrated(ref)
    img = T.base_image(rng)
    sl = (slice(T.ROI[1], T.ROI[1] + T.ROI[3]), slice(145, 300))
    img[sl] = cv2.blur(img[sl], (1, 61))                     # ticks and numbers gone
    T.add_bubble(img, 300, 450, 1.5)
    a = assess(T.finish(img, rng), cfg, ref, use_exclusions=False)
    assert status_of(a, "alignment") == OK and status_of(a, "scale") == OK, a.problems
    assert a.verdict == "PASS"


def test_other_resolution_and_wrong_shape():
    rng = np.random.default_rng(7)
    ref = photo(rng)
    cfg = calibrated(ref)
    f = photo(rng, [(400, 500, T.d_for_volume(30))])
    big = assess(cv2.resize(f, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC), cfg, ref)
    assert status_of(big, "geometry") == OK and big.verdict == "FAIL"
    assert abs(big.result.max_pocket_ul - 30) < 4.5
    small = assess(cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cfg, ref)
    assert status_of(small, "geometry") == WARN
    cropped = assess(f[:, :600], cfg, ref)
    assert status_of(cropped, "geometry") == FAIL and cropped.result is None
    assert cropped.verdict == "INVALID"


def test_bubble_behind_scale_box_is_measured_whole():
    # Review finding: the exclusion box deleted a 30 uL bubble (read 2.1 uL, PASS).
    rng = np.random.default_rng(14)
    for x in (258, 300, 320):                       # box covers x = 228..288
        a = assess(photo(rng, [(x, 500, T.d_for_volume(30))]), T.config())
        assert abs(a.result.max_pocket_ul - 30) < 4.5 and a.verdict == "FAIL", (x, a.result.max_pocket_ul)
        assert status_of(a, "scale boxes") == WARN
    a = assess(photo(rng, [(258, 500, 1.5)]), T.config())     # small bubble wholly inside the box
    assert status_of(a, "scale boxes") == WARN


def test_many_bubbles_are_measured_not_rejected():
    # ~100 uL in ~100 bubbles, overlapping in places: the checks must not call this a bad
    # photo. (Far denser foam does break the background model and is reported INVALID;
    # see Limitations in the README.)
    rng = np.random.default_rng(8)
    bubbles, true_ul = [], 0.0
    while true_ul < 100:
        d = rng.uniform(0.8, 1.6)
        bubbles.append((rng.uniform(T.ROI[0] + 20, T.ROI[0] + T.ROI[2] - 20),
                        rng.uniform(T.ROI[1] + 120, T.ROI[1] + T.ROI[3] - 20), d))
        true_ul += T.sphere_ul(d)
    a = assess(photo(rng, bubbles), T.config())
    assert a.verdict == "PASS", (a.verdict, [(c.name, c.status, c.value) for c in a.problems])
    assert abs(a.result.total_ul - true_ul) < 0.25 * true_ul, (a.result.total_ul, true_ul)


def test_large_bubble_is_not_taken_for_the_liquid_surface():
    # Review finding: a 150 uL bubble was read as 3000 uL of headspace (INVALID).
    rng = np.random.default_rng(15)
    for vol in (80, 150, 300):
        a = assess(photo(rng, [(380, 520, T.d_for_volume(vol))]), T.config())
        assert a.result.headspace_ul == 0.0 and a.verdict == "FAIL", (vol, a.verdict, a.result.headspace_ul)
        assert abs(a.result.max_pocket_ul - vol) < 0.15 * vol, (vol, a.result.max_pocket_ul)


def test_headspace_near_the_limit():
    # Review finding: headspace read 3-5 uL high near the 25 uL limit, and a small
    # headspace close to the barrel box's top edge was missed entirely.
    rng = np.random.default_rng(16)
    cfg = T.config()
    for target, sag in [(20, 10), (25, 10), (30, 10), (25, 4), (30, 4)]:
        img = T.base_image(rng)
        true = T.add_headspace(img, target, sag_px=sag)
        a = assess(T.finish(img, rng), cfg)
        err = abs(a.result.headspace_ul - true)
        assert err < 0.5 * cfg.ul_per_axial_px, (target, sag, true, a.result.headspace_ul)
    img = T.base_image(rng)
    true = T.add_headspace(img, 13, sag_px=4)
    assert assess(T.finish(img, rng), cfg).result.headspace_ul > 0.5 * true


def test_domed_plunger_seal_at_roi_end_adds_no_air():
    rng = np.random.default_rng(9)
    img = T.base_image(rng)
    yy, xx = np.mgrid[0:T.FH, 0:T.FW].astype(np.float32)
    xn = (xx - (T.ROI[0] + T.ROI[2] / 2)) / (T.ROI[2] / 2)
    top = T.ROI[1] + T.ROI[3] - 40 + 30 * xn ** 2                    # seal rises in the middle
    img[(yy >= top) & (np.abs(xn) <= 1.05)] *= 0.25
    a = assess(T.finish(img, rng), T.config())
    assert a.result.headspace_ul == 0.0 and a.result.total_ul < 1.0, a.result.total_ul
    assert a.verdict == "PASS", a.verdict


def test_shifted_scale_box_off_the_photo_does_not_wrap():
    # Review finding: a box moved to a negative x was dropped by Python's negative slicing.
    rng = np.random.default_rng(17)
    cfg = T.config(roi=[10, 60, 300, 880], exclusions=[[-3, 290, 80, 60]])
    img = T.base_image(rng)
    img[300:340, 12:60] *= 0.3                                      # dark label in the box
    a = assess(T.finish(img, rng), cfg)
    assert a.result.total_ul < 1.0, a.result.total_ul


def test_resolution_grades():
    assert check_resolution(T.config(mm_per_px=0.025)).status == OK       # 4.1 uL/px
    assert check_resolution(T.config()).status == WARN                     # 6.6 uL/px
    assert check_resolution(T.config(mm_per_px=0.065)).status == FAIL      # 10.7 uL/px


def test_tiny_roi_is_refused_not_crashed():
    a = assess(photo(np.random.default_rng(18)), T.config(roi=[300, 60, 2, 880], exclusions=[]))
    assert a.verdict == "INVALID" and a.result is None


# -------------------------------------------------------------------- the session

class _B:
    """A stand-in bubble: what match_pockets reads."""
    def __init__(self, axial_px, d_mm, vol):
        self.center = (300, T.ROI[1] + axial_px)
        self.d_axial_mm, self.volume_ul = d_mm, vol


def _rec(key, bubbles=(), head=0.0, verdict_="PASS"):
    class R:
        pass
    res = R()
    res.bubbles, res.headspace_ul = list(bubbles), head
    a = R()
    a.result, a.verdict, a.cfg = res, verdict_, T.config()
    return gi.ViewRecord(key, key, "", assessment=a)


def test_same_bubble_in_two_views_counts_once():
    s = gi.combine([_rec("a", [_B(200, 3.0, 14.0)]), _rec("b", [_B(202, 3.1, 15.0)])], T.config())
    assert len(s["pockets"]) == 1 and s["max_pocket_ul"] == 15.0 and s["total_ul"] == 15.0, s


def test_bubbles_seen_in_one_view_are_added():
    s = gi.combine([_rec("a", [_B(200, 3.0, 14.0)]),
                    _rec("b", [_B(202, 3.0, 14.0), _B(600, 3.4, 20.0)])], T.config())
    assert len(s["pockets"]) == 2 and s["total_ul"] == 34.0, s
    assert any("only in the b view" in n for n in s["notes"]), s["notes"]
    s = gi.combine([_rec("a", [_B(200, 3.0, 14.0)]), _rec("b", [_B(600, 3.8, 27.0)])], T.config())
    assert s["verdict"] == "FAIL"


def test_combine_missing_view_is_invalid():
    s = gi.combine([_rec("a"), gi.ViewRecord("b", "b", "")], T.config())
    assert s["verdict"] == "INVALID" and any("b" in n for n in s["notes"])
    s = gi.combine([_rec("a"), _rec("b", verdict_="INVALID")], T.config())
    assert s["verdict"] == "INVALID"


def test_combine_flags_disagreeing_views():
    s = gi.combine([_rec("a", [_B(200, 3.0, 8.0)]), _rec("b", [_B(201, 3.2, 15.0)])], T.config())
    assert any("reads" in n for n in s["notes"]), s["notes"]
    s = gi.combine([_rec("a", head=10.0), _rec("b", head=30.0)], T.config())
    assert s["headspace_ul"] == 30.0 and any("headspace" in n for n in s["notes"])


def _setup_dir(tmp, rng):
    ref = photo(rng)
    cfg = calibrated(ref)
    cfg.ref_image = "bubble_config_ref.png"
    write_image(os.path.join(tmp, cfg.ref_image), ref)
    cfg.save(os.path.join(tmp, "bubble_config.json"))


def _session(tmp, args, stdin=None):
    """Run a session in tmp; returns (exit code, report, session folder)."""
    cwd, old_in = os.getcwd(), sys.stdin
    os.chdir(tmp)
    if stdin is not None:
        sys.stdin = io.StringIO(stdin)
    try:
        code = gi.main(args)
    finally:
        os.chdir(cwd)
        sys.stdin = old_in
    root = os.path.join(tmp, "sessions")
    runs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    folder = max(runs, key=os.path.getmtime)
    with open(os.path.join(folder, "report.json"), encoding="utf-8") as f:
        return code, json.load(f), folder


def test_guided_session_end_to_end():
    rng = np.random.default_rng(10)
    with tempfile.TemporaryDirectory() as tmp:
        _setup_dir(tmp, rng)
        names = {
            "good_a.png": photo(rng, [(380, 400, 1.0), (420, 700, 1.4)]),
            "good_b.png": photo(rng, [(330, 400, 1.0), (420, 710, 1.4)], (3, -2)),
            "bad_a.png": photo(rng, [(400, 500, T.d_for_volume(30))]),
            "blurry.png": cv2.GaussianBlur(photo(rng, [(380, 400, 1.0)]), (0, 0), 4),
        }
        for n, img in names.items():
            write_image(os.path.join(tmp, n), img)

        code, rep, folder = _session(tmp, ["--sample", "S1", "--photos", "good_a.png", "good_b.png"])
        assert code == 0 and rep["verdict"] == "PASS" and rep["views_used"] == 2, rep["notes"]
        assert len(rep["pockets"]) == 2, rep["pockets"]          # the same 2 bubbles in both views
        for f in ("report.html", "calibration.json", "calibration_photo.png",
                  rep["views"][0]["photo"], rep["views"][0]["annotated"], rep["views"][1]["annotated"]):
            assert os.path.exists(os.path.join(folder, f)), f
        assert len(rep["views"][0]["photo_sha256"]) == 64

        code, rep, _ = _session(tmp, ["--sample", "S2", "--photos", "bad_a.png", "good_b.png"])
        assert code == 1 and rep["verdict"] == "FAIL" and rep["max_pocket_ul"] > 25

        code, rep, _ = _session(tmp, ["--sample", "S3", "--photos", "blurry.png", "good_b.png"])
        assert code == 2 and rep["verdict"] == "INVALID"
        assert any(c["name"] == "focus" and c["status"] == FAIL for c in rep["views"][0]["checks"])

        code, rep, _ = _session(tmp, ["--sample", "S4", "--photos", "missing.png", "good_b.png"])
        assert code == 2 and rep["views"][0]["verdict"] == "MISSING"

        code, rep, _ = _session(tmp, ["--sample", "S5", "--photos", "good_a.png", "good_a.png"])
        assert code == 2 and rep["verdict"] == "INVALID", "the same photo twice must be rejected"
        assert any(c["name"] == "duplicate" for c in rep["views"][1]["checks"])

        code, rep, _ = _session(tmp, ["--sample", "S-李", "--operator", "王",
                                      "--photos", "bad_a.png", "good_b.png"])
        assert code == 1                      # non-ASCII names must not crash the CSV log
        with open(os.path.join(tmp, "sessions", "sessions.csv"), encoding="utf-8-sig") as f:
            rows = f.read().strip().splitlines()
        assert len(rows) == 1 + 6 and "王" in rows[-1]
        invalid_row = [r for r in rows if ",S3," in r][0].split(",")
        assert invalid_row[4] == "" and invalid_row[5] == "", "INVALID rows must carry no numbers"


def test_rejected_photo_is_never_used():
    # Review finding: the operator chose "retake" on a photo, then skipped the view, and
    # the rejected photo still counted: PASS with exit 0. It must be INVALID.
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as tmp:
        _setup_dir(tmp, rng)
        f = photo(rng, [(380, 400, 1.0)])
        write_image(os.path.join(tmp, "blurry.png"), cv2.GaussianBlur(f, (0, 0), 4))
        write_image(os.path.join(tmp, "good_b.png"), photo(rng, [(330, 400, 1.0)]))
        answers = "blurry.png\nr\ns\ngood_b.png\n"      # bad photo, retake, skip, next view
        code, rep, _ = _session(tmp, ["--views", "2"], stdin=answers)
        assert code == 2 and rep["verdict"] == "INVALID", (code, rep["verdict"])
        assert rep["views"][0]["verdict"] == "MISSING"
        assert [t["outcome"] for t in rep["views"][0]["attempts"]] == ["retaken"]


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
