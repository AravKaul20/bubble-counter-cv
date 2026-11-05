#!/usr/bin/env python3
"""
phone_server.py

Test syringes from your phone: no cables, no copying files, no terminal while testing.

    python phone_server.py

It prints a web address. Open it in the phone's browser (phone and computer on the same
Wi-Fi). The page walks you through each syringe: take the front photo, turn it halfway round,
take the second photo, and the result (PASS / MARGINAL / FAIL / RETAKE) shows on the
phone a few seconds later, with the annotated photo. Then the next syringe.

Everything is saved on the computer like batch_inspect.py does: photos under
sessions/phone_photos/, a full report per syringe under sessions/, and one CSV per run
(sessions/batch_<time>_phone.csv). A retaken syringe gets a new row; the last one counts.

Calibration: the page also has a "calibration photo" button. It saves the photo on the
computer as calibration.jpg; then run setup on the computer once (the page shows the
command). Calibrate with a photo taken through this page, so the photo size matches.

Anyone on the same network can open the page while it runs; stop it with Ctrl+C.
Requires: opencv-python, numpy. Optional: qrcode (pip install qrcode) prints a QR code.
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

import batch_inspect as bi
import guided_inspection as gi
from syringe_bubble_inspector import SPEC_MAX_POCKET_UL, Config, log_row, read_image

MAX_UPLOAD = 60 * 1024 * 1024


def sniff(data):
    """File type from the first bytes: 'jpg', 'png', 'heic' or None."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[4:8] == b"ftyp":
        return "heic"
    return None


def lan_address():
    """This computer's address on the local network (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class App:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()                  # one measurement at a time
        self.history = gi.earlier_hashes(args.out)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.inbox = os.path.join(args.out, "phone_photos", stamp)
        self.csv = os.path.join(args.out, f"batch_{stamp}_phone.csv")
        self.prefix = time.strftime("%m%d") + "-"

    def status(self):
        cfg = Config.load(self.args.config) if os.path.exists(self.args.config) else None
        return {"calibrated": cfg is not None, "views": self.args.views, "prefix": self.prefix,
                "steps": [{"title": t, "how": h} for _, t, h in gi.VIEWS[:self.args.views]],
                "limits": cfg and f"limits {SPEC_MAX_POCKET_UL:.0f} uL per pocket, "
                                  f"{(self.args.fill_ml or cfg.fill_ml) * 25:.0f} uL total",
                "setup_cmd": self.setup_cmd()}

    def setup_cmd(self):
        img = os.path.relpath(self.calib_path())
        cmd = f"python syringe_bubble_inspector.py setup --image {img if ' ' not in img else repr(img)} --syringe 10ml"
        if os.path.abspath(self.args.config) != os.path.abspath("bubble_config.json"):
            cmd += f' --config "{self.args.config}"'
        return cmd

    def calib_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(self.args.config)), "calibration.jpg")

    def sample_dir(self, sample):
        return os.path.join(self.inbox, gi.safe_name(sample))

    def save_photo(self, sample, view, data):
        kind = sniff(data)
        if kind == "heic":
            return {"error": "HEIC photo. Set the camera to JPEG (iPhone: Settings > Camera > Formats > "
                             "Most Compatible)."}
        if kind is None:
            return {"error": "That file is not a JPEG or PNG photo."}
        d = self.sample_dir(sample)
        os.makedirs(d, exist_ok=True)
        key = gi.VIEWS[view][0]
        for old in os.listdir(d):                     # a retake replaces the earlier photo
            if old.startswith(f"{view}_"):
                os.remove(os.path.join(d, old))
        with open(os.path.join(d, f"{view}_{key}.{kind}"), "wb") as f:
            f.write(data)
        return {"ok": True}

    def save_calibration(self, data):
        if sniff(data) != "jpg":
            return {"error": "Calibration photo must be a JPEG."}
        with open(self.calib_path(), "wb") as f:
            f.write(data)
        return {"ok": True, "cmd": self.status()["setup_cmd"]}

    def inspect(self, sample):
        if not os.path.exists(self.args.config):
            return {"error": "Not calibrated yet. Send a calibration photo and run setup on the computer."}
        d = self.sample_dir(sample)
        names = sorted(os.listdir(d)) if os.path.isdir(d) else []
        photos = [os.path.join(d, n) for n in names if n.split("_")[0].isdigit()
                  and int(n.split("_")[0]) < self.args.views]
        if len(photos) != self.args.views:
            return {"error": f"{len(photos)} of {self.args.views} photos received for {sample}."}
        with self.lock:
            try:
                row = bi.inspect_syringe(sample, photos, self.args, self.history)
            except Exception as exc:
                row = {"result": "ERROR", "reason": f"{type(exc).__name__}: {exc}", "notes": "",
                       "report": "", "largest_pocket_ul": "", "total_air_ul": ""}
            row.update(sample=sample, photos=" + ".join(os.path.basename(p) for p in photos))
            log_row(self.csv, row, bi.BATCH_FIELDS)
        previews = []
        if row["report"]:
            folder = os.path.dirname(row["report"])
            with open(os.path.join(folder, "report.json"), encoding="utf-8") as f:
                for v in json.load(f)["views"]:
                    if v["annotated"]:
                        previews.append("/preview?f=" + os.path.relpath(os.path.join(folder, v["annotated"]),
                                                                        self.args.out).replace("\\", "/"))
        print(f"  {sample:<14} {row['result']:<9} {row['largest_pocket_ul'] or '-':>6} uL  {row['reason']}",
              flush=True)
        out = {k: row[k] for k in ("sample", "result", "largest_pocket_ul", "total_air_ul", "reason", "notes")}
        out["previews"] = previews
        return out

    def preview(self, rel):
        """Annotated photo, shrunk to a phone-friendly JPEG. Only files under the output folder."""
        root = os.path.abspath(self.args.out)
        path = os.path.abspath(os.path.join(root, rel))
        if not path.startswith(root + os.sep) or not path.lower().endswith(".png"):
            return None
        img = read_image(path)
        if img is None:
            return None
        s = min(1.0, 1400 / max(img.shape[:2]))
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):            # keep the terminal for results
            pass

        def send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self.send(200, PAGE, "text/html; charset=utf-8")
            if u.path == "/status":
                return self.send(200, app.status())
            if u.path == "/preview":
                data = app.preview(q.get("f", [""])[0])
                return self.send(200, data, "image/jpeg") if data else self.send(404, {"error": "not found"})
            self.send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                return self.send(413, {"error": "photo too large"})
            data = self.rfile.read(n)
            sample = (q.get("sample", [""])[0] or "").strip()[:60]
            try:
                if u.path == "/photo" and sample:
                    view = int(q.get("view", ["-1"])[0])
                    if not 0 <= view < app.args.views:
                        return self.send(400, {"error": "bad view"})
                    return self.send(200, app.save_photo(sample, view, data))
                if u.path == "/inspect" and sample:
                    return self.send(200, app.inspect(sample))
                if u.path == "/calibration":
                    return self.send(200, app.save_calibration(data))
            except Exception as exc:
                return self.send(500, {"error": f"{type(exc).__name__}: {exc}"})
            self.send(404, {"error": "not found"})
    return Handler


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Syringe check</title>
<style>
 :root { --bg:#f6f6f4; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e2e2dc;
         --pass:#1f8a3b; --marg:#b26a00; --fail:#c62828; --retake:#5b5b8a; --accent:#1d4ed8; }
 @media (prefers-color-scheme: dark) { :root { --bg:#151514; --card:#1f1f1d; --ink:#ededea;
         --muted:#a3a39c; --line:#33332f; --accent:#7aa2ff; } }
 * { box-sizing:border-box; }
 body { margin:0; background:var(--bg); color:var(--ink);
        font:16px/1.4 system-ui,-apple-system,Segoe UI,sans-serif; }
 main { max-width:560px; margin:0 auto; padding:16px; }
 .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; margin-bottom:14px; }
 h1 { font-size:20px; margin:0 0 4px; } .muted { color:var(--muted); font-size:14px; }
 label.big, button.big { display:block; width:100%; text-align:center; border:0; border-radius:12px;
        padding:18px; font-size:19px; font-weight:650; background:var(--accent); color:#fff; cursor:pointer; }
 button.alt { background:transparent; color:var(--accent); border:1px solid var(--line); margin-top:10px;
        font-size:16px; padding:12px; }
 .step { font-size:22px; font-weight:700; margin:6px 0; } .how { margin-bottom:14px; }
 .row { display:flex; gap:8px; align-items:center; } .row input { flex:1; font-size:17px; padding:9px;
        border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--ink); }
 .result { font-size:34px; font-weight:800; } .PASS{color:var(--pass)} .MARGINAL{color:var(--marg)}
 .FAIL{color:var(--fail)} .RETAKE,.ERROR{color:var(--retake)}
 img { width:100%; border-radius:8px; margin-top:10px; border:1px solid var(--line); }
 table { width:100%; border-collapse:collapse; font-size:14px; } td { padding:5px 4px; border-top:1px solid var(--line); }
 .warn { color:var(--fail); font-weight:600; } input[type=file] { display:none; } code { word-break:break-all; }
</style></head><body><main>
<div class="card"><h1>Syringe check</h1><div class="muted" id="limits">Connecting...</div>
 <div id="nocal" class="warn" hidden>Not calibrated yet: send a calibration photo (bottom of this page).</div></div>

<div class="card" id="shoot">
 <div class="row"><span class="muted">Sample</span><input id="sample" autocomplete="off"></div>
 <div class="step" id="step"></div><div class="how muted" id="how"></div>
 <label class="big" for="cam" id="camlabel">Take photo</label>
 <input type="file" id="cam" accept="image/*" capture="environment">
 <div class="muted" id="busy" style="margin-top:10px"></div>
</div>

<div class="card" id="res" hidden>
 <div class="result" id="verdict"></div><div id="numbers"></div><div class="muted" id="reason"></div>
 <div id="imgs"></div>
 <button class="big" id="next" style="margin-top:14px">Next syringe</button>
 <button class="big alt" id="again">Retake this syringe</button>
</div>

<div class="card"><b>This run</b><table id="log"></table></div>

<div class="card"><b>Calibration</b>
 <div class="muted">Once per setup: a photo of a filled syringe on its spot, same as the test photos.</div>
 <label class="big alt" for="calcam">Send calibration photo</label>
 <input type="file" id="calcam" accept="image/*" capture="environment">
 <div id="calmsg" class="muted"></div></div>
</main>
<script>
const $ = id => document.getElementById(id);
let S = null, n = 1, view = 0;
const pad = k => String(k).padStart(3, "0");
async function post(url, body) {
  const r = await fetch(url, {method: "POST", body});
  let j; try { j = await r.json(); } catch (e) { j = {error: "server error " + r.status}; }
  if (j.error) throw new Error(j.error);
  return j;
}
function show() {
  const st = S.steps[view];
  $("step").textContent = `Photo ${view + 1} of ${S.views}: ${st.title}`;
  $("how").textContent = view === 0 ? "Let the syringe rest 30 s first. " + st.how : st.how[0].toUpperCase() + st.how.slice(1) + ".";
  $("camlabel").textContent = "Take photo " + (view + 1);
}
function newSample() { $("sample").value = S.prefix + pad(n); view = 0; $("res").hidden = true; $("shoot").hidden = false; show(); }
async function start() {
  try { S = await (await fetch("/status")).json(); } catch (e) { $("limits").textContent = "Cannot reach the computer."; return; }
  $("limits").textContent = S.calibrated ? S.limits : "";
  $("nocal").hidden = S.calibrated;
  newSample();
}
$("cam").onchange = async e => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f) return;
  const sample = $("sample").value.trim() || (S.prefix + pad(n));
  $("camlabel").style.opacity = .5; $("busy").textContent = "Sending photo...";
  try {
    await post(`/photo?sample=${encodeURIComponent(sample)}&view=${view}`, f);
    if (view + 1 < S.views) { view++; show(); $("busy").textContent = "Got it."; }
    else {
      $("busy").textContent = "Checking... (about 10 s)";
      const r = await post(`/inspect?sample=${encodeURIComponent(sample)}`);
      result(r);
    }
  } catch (err) { $("busy").textContent = ""; alert(err.message); }
  $("camlabel").style.opacity = 1;
};
function result(r) {
  $("busy").textContent = ""; $("shoot").hidden = true; $("res").hidden = false;
  $("verdict").textContent = r.result; $("verdict").className = "result " + r.result;
  $("numbers").textContent = r.largest_pocket_ul ? `Largest pocket ${r.largest_pocket_ul} uL, total air ${r.total_air_ul} uL` : "No numbers: photos not usable.";
  $("reason").textContent = [r.reason, r.notes].filter(Boolean).join(" | ");
  $("imgs").innerHTML = ""; for (const p of r.previews) { const i = new Image(); i.src = p; $("imgs").appendChild(i); }
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${r.sample}</td><td class="${r.result}"><b>${r.result}</b></td><td>${r.largest_pocket_ul ? r.largest_pocket_ul + " uL" : ""}</td>`;
  $("log").prepend(tr);
  window.scrollTo(0, 0);
}
$("next").onclick = () => { n++; newSample(); };
$("again").onclick = () => { view = 0; $("res").hidden = true; $("shoot").hidden = false; show(); };
$("calcam").onchange = async e => {
  const f = e.target.files[0]; e.target.value = ""; if (!f) return;
  $("calmsg").textContent = "Sending...";
  try { const r = await post("/calibration", f);
        $("calmsg").innerHTML = "Saved on the computer. Now run this there, then reload this page:<br><code></code>";
        $("calmsg").querySelector("code").textContent = r.cmd; }
  catch (err) { $("calmsg").textContent = err.message; }
};
start();
</script></body></html>
"""


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Test syringes from a phone browser.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--views", type=int, default=2, choices=[1, 2, 3, 4], help="photos per syringe")
    ap.add_argument("--config", default="bubble_config.json")
    ap.add_argument("--out", default="sessions")
    ap.add_argument("--fill-ml", type=float)
    ap.add_argument("--operator", default="")
    ap.add_argument("--description", default="")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    app = App(args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(app))
    url = f"http://{lan_address()}:{server.server_address[1]}"
    print(f"\nOn the phone (same Wi-Fi), open:   {url}\n")
    try:
        import qrcode
        q = qrcode.QRCode(border=1)
        q.add_data(url)
        q.print_ascii(invert=True)
        print("(or scan this with the phone camera)\n")
    except ImportError:
        pass
    if not os.path.exists(args.config):
        print("Not calibrated yet: send a calibration photo from the page first.")
    print(f"Results go to {app.csv}. Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
