#!/usr/bin/env python3
"""
Tests for batch_inspect.py (mass testing from a folder of photos).

    python test_batch.py          # or: python -m pytest test_batch.py

The risky part of batch mode is deciding which photos belong to which syringe, so most
tests are about that: photos grouped by the time the phone stored, a retaken photo
flagged instead of silently pairing the wrong photos, name-order pairing refused when
the count does not add up, and one subfolder per syringe.
"""
import csv
import glob
import os
import sys
import tempfile
import traceback

import cv2
import numpy as np
from PIL import Image

import batch_inspect as bi
import test_bubble_inspector as T
import test_guided as TG
from syringe_bubble_inspector import write_image


def save_jpg(path, img, when=None):
    """Phone-like JPEG; when = 'HH:MM:SS' stores it as the capture time."""
    rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()
    if when:
        exif[0x8769] = {0x9003: f"2026:11:03 {when}"}
    rgb.save(path, quality=95, exif=exif)


def run(tmp, folder, *extra):
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        code = bi.main([folder, *extra])
    finally:
        os.chdir(cwd)
    files = glob.glob(os.path.join(tmp, "sessions", "batch_*.csv"))
    rows = []
    if files:
        with open(max(files, key=os.path.getmtime), encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    return code, rows


def test_flat_folder_grouped_by_photo_time():
    rng = np.random.default_rng(20)
    with tempfile.TemporaryDirectory() as tmp:
        TG._setup_dir(tmp, rng)
        d = os.path.join(tmp, "run1")
        os.makedirs(d)
        big = T.d_for_volume(30)
        shots = [  # (name, photo, time): names out of order on purpose; times decide
            ("IMG_0010.jpg", TG.photo(rng, [(380, 400, 1.0)]), "10:00:00"),
            ("IMG_0011.jpg", TG.photo(rng, [(340, 400, 1.0)]), "10:00:08"),
            ("IMG_0009.jpg", TG.photo(rng, [(400, 500, big)]), "10:01:00"),
            ("IMG_0012.jpg", TG.photo(rng, [(360, 505, big)]), "10:01:06"),
            ("IMG_0013.jpg", TG.photo(rng), "10:02:00"),       # third syringe: a retake,
            ("IMG_0014.jpg", TG.photo(rng), "10:02:05"),       # three photos together
            ("IMG_0015.jpg", TG.photo(rng), "10:02:12"),
        ]
        for name, img, when in shots:
            save_jpg(os.path.join(d, name), img, when)
        code, rows = run(tmp, "run1")
        assert [r["sample"] for r in rows] == ["run1-001", "run1-002", "run1-003"], rows
        assert rows[0]["photos"] == "IMG_0010.jpg + IMG_0011.jpg"
        assert rows[0]["result"] == "PASS", rows[0]
        assert rows[1]["photos"] == "IMG_0009.jpg + IMG_0012.jpg"
        assert rows[1]["result"] == "FAIL" and float(rows[1]["largest_pocket_ul"]) > 25, rows[1]
        assert "largest pocket" in rows[1]["reason"]
        assert rows[2]["result"] == "RETAKE" and "3 photo(s)" in rows[2]["reason"], rows[2]
        assert code == 2
        assert os.path.exists(os.path.join(tmp, rows[0]["report"])), rows[0]["report"]


def test_bad_photo_gives_retake_with_the_fix():
    rng = np.random.default_rng(21)
    with tempfile.TemporaryDirectory() as tmp:
        TG._setup_dir(tmp, rng)
        d = os.path.join(tmp, "run2")
        os.makedirs(d)
        write_image(os.path.join(d, "a1.png"), TG.photo(rng, [(380, 400, 1.0)]))
        write_image(os.path.join(d, "a2.png"), TG.photo(rng, [(340, 400, 1.0)]))
        write_image(os.path.join(d, "b1.png"), cv2.GaussianBlur(TG.photo(rng), (0, 0), 4))
        write_image(os.path.join(d, "b2.png"), TG.photo(rng))
        code, rows = run(tmp, "run2")          # PNGs have no capture time: name order, pairs
        assert [r["photos"] for r in rows] == ["a1.png + a2.png", "b1.png + b2.png"]
        assert rows[0]["result"] == "PASS"
        assert rows[1]["result"] == "RETAKE" and "focus" in rows[1]["reason"], rows[1]
        assert rows[1]["largest_pocket_ul"] == "", "a RETAKE row must carry no numbers"
        assert code == 2


def test_name_order_refused_when_count_is_odd():
    rng = np.random.default_rng(22)
    with tempfile.TemporaryDirectory() as tmp:
        TG._setup_dir(tmp, rng)
        d = os.path.join(tmp, "run3")
        os.makedirs(d)
        for n in ("1.png", "2.png", "3.png"):
            write_image(os.path.join(d, n), TG.photo(rng))
        code, rows = run(tmp, "run3")
        assert code == 3 and rows == []
        code, rows = run(tmp, "run3", "--views", "1")    # one photo per syringe is fine
        assert code == 0 and len(rows) == 3 and all(r["result"] == "PASS" for r in rows), rows


def test_one_subfolder_per_syringe():
    rng = np.random.default_rng(23)
    with tempfile.TemporaryDirectory() as tmp:
        TG._setup_dir(tmp, rng)
        for sample, n in (("S12", 2), ("S13", 1)):
            os.makedirs(os.path.join(tmp, "run4", sample))
            for i in range(n):
                write_image(os.path.join(tmp, "run4", sample, f"{i}.png"),
                            TG.photo(rng, [(380 - 40 * i, 400, 1.0)]))
        code, rows = run(tmp, "run4")
        assert [(r["sample"], r["result"]) for r in rows] == [("S12", "PASS"), ("S13", "RETAKE")], rows
        write_image(os.path.join(tmp, "run4", "loose.png"), TG.photo(rng))
        code, rows = run(tmp, "run4")
        assert code == 3, "photos both loose and in subfolders must be refused"


def test_heic_is_refused_with_advice():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "IMG_1.HEIC"), "wb").close()
        try:
            bi.group_photos(tmp, 2, 20)
        except bi.LayoutError as exc:
            assert "JPEG" in str(exc)
        else:
            raise AssertionError("HEIC must be refused")


def test_phone_server_flow():
    """What the phone page does: send two photos, ask for the result, show the preview."""
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    import phone_server as ps
    rng = np.random.default_rng(24)
    with tempfile.TemporaryDirectory() as tmp:
        TG._setup_dir(tmp, rng)
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            args = ps.argparse.Namespace(port=0, views=2, config="bubble_config.json", out="sessions",
                                         fill_ml=None, operator="", description="", debug=False)
            app = ps.App(args)
            srv = ThreadingHTTPServer(("127.0.0.1", 0), ps.make_handler(app))
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"

            def call(path, data=None):
                req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET")
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read()
                    return json.loads(body) if r.headers["Content-Type"].startswith("application/json") else body

            def jpg(img):
                return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()

            st = call("/status")
            assert st["calibrated"] and st["views"] == 2 and len(st["steps"]) == 2
            assert call("/photo?sample=A1&view=0", jpg(TG.photo(rng, [(380, 400, 1.0)]))) == {"ok": True}
            assert "1 of 2" in call("/inspect?sample=A1", b"")["error"]
            assert call("/photo?sample=A1&view=1", jpg(TG.photo(rng, [(340, 400, 1.0)]))) == {"ok": True}
            r = call("/inspect?sample=A1", b"")
            assert r["result"] == "PASS" and r["largest_pocket_ul"], r
            assert len(r["previews"]) == 2 and call(r["previews"][0])[:3] == b"\xff\xd8\xff"
            assert "HEIC" in call("/photo?sample=A2&view=0", b"\0\0\0\x18ftypheic" + b"\0" * 100)["error"]
            for bad in ("../bubble_config.json", "../bubble_config_ref.png"):   # outside the output folder
                try:
                    call("/preview?f=" + bad)
                except urllib.error.HTTPError as exc:
                    assert exc.code == 404
                else:
                    raise AssertionError("served a file outside the output folder: " + bad)
        finally:
            os.chdir(cwd)
        with open(os.path.join(tmp, app.csv), encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert [(r["sample"], r["result"]) for r in rows] == [("A1", "PASS")], rows
        srv.shutdown()
        srv.server_close()


def main():
    tests =[(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
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
