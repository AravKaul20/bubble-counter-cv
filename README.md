# Syringe Bubble Inspector

Measures the residual air in a filled syringe from photos and checks it against the
EGR 101 de-airing spec:

| Requirement | Limit |
|---|---|
| Largest single air pocket | ≤ 25 µL |
| Total air | < 250 µL per 10 mL of liquid (25 µL/mL) |

The output is a **volume in microliters** plus a verdict:

| Verdict | Meaning |
|---|---|
| **PASS** | Both values are under their limits |
| **MARGINAL** | A value is within 20% below a limit (set by `margin`) |
| **FAIL** | A value is over its limit |
| **INVALID** | A photo failed its quality checks or a view is missing, so no verdict is given |

INVALID exists because a photo taken under the wrong conditions does not give a slightly
worse number; it gives an arbitrary one. On a front-lit test photo the old live tool
reported anything from 5 µL (PASS) to 156 µL (FAIL) depending only on its threshold and
polarity settings. The tool now checks every photo, refuses to give a verdict for one
that breaks the imaging setup, and tells the operator what to fix.

## Quick start

**Setup (once):**

1. Put a white light *behind* the syringe (a tablet or monitor showing a white screen
   works) and turn off room lights in front of it.
2. Stand the syringe upright, tip up, and fix the phone or camera close to it, so the
   barrel fills most of the photo.
3. Take one photo, then run:
   ```bash
   python syringe_bubble_inspector.py setup --image calibration.jpg --syringe 10ml
   ```
   Drag a box just inside the barrel walls (ENTER), then click two graduation marks far
   apart and type their spacing, e.g. `5 ml`.

**Each syringe:** take a photo from the same spot and run

```bash
python guided_inspection.py photo.jpg
```

It prints PASS / MARGINAL / FAIL with the air volume and writes a report (path
printed). If the photo is bad, for example lit from the front, moved, zoomed or blurry,
it says INVALID and tells you what to fix. Optional: add a second photo with the
syringe turned 90° (`python guided_inspection.py front.jpg turned.jpg`) to catch
bubbles hidden behind the printed scale.

The rest of this README is reference detail.

## Files

| File | Purpose |
|---|---|
| `guided_inspection.py` | **Start here.** Guided photo session for one syringe, with report |
| `syringe_bubble_inspector.py` | Detector, photo-quality checks, and tools: `setup` (calibration), `run` (batch images, live preview, video) |
| `test_bubble_inspector.py` | Synthetic accuracy test with known bubble sizes |
| `test_guided.py` | Tests for the quality checks and the guided session |
| `requirements.txt` | Python dependencies |

## Install

Python 3.8+.

```bash
pip install -r requirements.txt
```

`setup`, `--camera` and live mode open windows, so they need the full `opencv-python`
package. On a headless machine `opencv-python-headless` is enough for photo-file
sessions, `run --image` and the tests.

## Imaging setup

The method depends on these conditions. The quality checks enforce them, so a photo
that breaks one is reported INVALID with the reason.

1. **Backlit.** A diffuse light panel *behind* the syringe, filling the view behind the
   whole barrel, camera in front. Bubbles then show up as dark rings on a bright field.
   A tablet or monitor showing a plain white screen works as a panel. Switch off or
   block lights on the camera side: they make bubbles look bright and add reflections.
2. **Vertical, tip up.** Free air rises to the tip, where it is measured as headspace.
   The camera may be rotated (tell setup which side of the *picture* the tip is on with
   `--air-end`), but the syringe itself must stand tip up.
3. **Fixed geometry.** Hold the syringe in a holder at a fixed distance from the camera,
   so one calibration covers every test. Small shifts when the syringe is re-seated are
   detected and followed automatically; a change of size (camera distance, zoom, lens)
   or tilt is not, and makes the photo INVALID. Index marks on the holder make the
   quarter turns repeatable.
4. **Close enough.** Fill the frame with the barrel along the photo's long side. One
   pixel of liquid level should be at most 5 µL (setup prints this figure): about
   0.025 mm per pixel for a 10 mL syringe, which a 12 MP phone camera reaches when the
   barrel spans the frame.
5. **Fixed camera settings.** Lock focus and exposure (on most phones, long-press on the
   barrel). On a phone, stay on the 1× lens with automatic macro switching off, and turn
   off HDR and night mode. Expose so the middle of the barrel is bright but not white.
   With a webcam, `--exposure` locks exposure if the driver supports it.

## Workflow

### 1. Calibrate (once per camera position)

Take a photo with the syringe in the holder under the setup above, then:

```bash
python syringe_bubble_inspector.py setup --image calibration.jpg --air-end top
```

1. Drag a box just **inside** the barrel walls, from just below the tip shoulder to
   just above the plunger seal, then ENTER.
2. Click two graduation marks far apart along the barrel, then type their spacing,
   e.g. `5 ml` or `20 mm`.

The barrel inner diameter comes from `--syringe 10ml` (presets for 1, 3, 5, 10, 20, 30
and 60 mL); without it setup asks, and you can also type a caliper measurement in mm.
`--exclusions` adds an optional step to box the printed numbers so they are ignored
(front view only).

This writes `bubble_config.json` and keeps the photo as `bubble_config_ref.png`. Later
photos are compared with it for position, size, tilt and focus. Setup then runs the
quality checks on the calibration photo. If any fails, fix the setup and calibrate
again, because photos taken the same way will be rejected. Use `--camera 0` instead of
`--image` to calibrate from a webcam (SPACE freezes the frame).

Each step can also be given on the command line, which skips its window:

```bash
python syringe_bubble_inspector.py setup --image calibration.jpg --roi 140 60 362 880 \
    --mm-per-px 0.04 --barrel-id-mm 14.5 --air-end top
```

Other setup options: `--fill-ml` (default liquid volume, 10; the guide asks for each
sample) and `--lateral-mag` (about 1.33 for saline, 1.4–1.5 for contrast media; see
[How it works](#how-it-works)).

### 2. Inspect a syringe

```bash
python guided_inspection.py photo.jpg                 # one photo
python guided_inspection.py front.jpg turned.jpg      # optional second view, turned 90°
python guided_inspection.py                           # asks for the photo (drag it in)
```

The guide checks each photo. If a photo fails, it says what is wrong and how to fix
it, and (when it asked for the photo) asks for a retake. Warnings are noted in the
report but never stop you. Every photo taken is kept and listed in the report.
Optional report details: `--sample`, `--operator`, `--description`, `--fill-ml`
(default from setup; sets the total-air limit). Let the syringe rest untouched for
about 30 s before the photo; don't tap it (tapping is itself de-airing).

With two or more views, the syringe is turned about its own axis between photos;
`--views N` sets how many photos the guide asks for.

Photo sources:

| Command | Source |
|---|---|
| `guided_inspection.py` | Asks for each photo file (type the path or drag the file into the terminal). Use this with a phone. |
| `guided_inspection.py --camera 0` | Live webcam preview with the step instructions and running checks. SPACE captures. |
| `guided_inspection.py front.jpg quarter.jpg` | Photos given up front, no questions. |

Phone photos: use JPEG, not HEIC (on iPhone: Settings → Camera → Formats → Most
Compatible), and the same phone, lens, zoom and orientation as the calibration photo.
A photo at another resolution with the same shape is rescaled automatically.

**How views are combined.** Turning the syringe about its axis does not change a
bubble's height along the barrel or its size along the barrel, so bubbles are matched
across views by those two values. A bubble seen in several views is counted once, at
its largest reading. A bubble seen in only one view (hidden in the others) is still
counted. Headspace is the largest reading. The report notes bubbles whose readings
differ a lot between views and large bubbles seen in only one view.

The same photo given for two views, or a photo already used in an earlier session, is
detected (by file hash and pixel comparison).

### Output

Each session gets its own folder under `sessions/` (set with `--out`):

- `report.html`: sample details, verdict, values against limits, what is and is not
  measured, every air pocket with its reading in each view, each view's photo checks
  and annotated photo, every photo taken, and the calibration used. Prints cleanly.
- `report.json`: the same data plus every bubble's size and position, for analysis.
- `<view>_attempt<n>.*`: every photo taken, unchanged, and `<view>_attempt<n>_annotated.png`
  (bubbles outlined and labelled in µL: green small, orange over half the limit, red
  over the limit, magenta odd shape still counted as air; headspace shaded red; side
  panel with the verdict and checks). `--debug` adds the detector's masks.
- `calibration.json`, `calibration_photo.png`: copies of the calibration used, so the
  session stays traceable after setup is run again.
- `sessions/sessions.csv`: one row per session (sample, operator, verdict, values,
  notes, report path). INVALID rows have no values.

Exit code: 0 PASS, 1 MARGINAL or FAIL, 2 INVALID or stopped, 3 error.

### Other tools

```bash
# batch-check photos (same checks, no guidance); exit code 0 PASS, 1 MARGINAL/FAIL, 2 INVALID
python syringe_bubble_inspector.py run --image a.jpg b.jpg --out results

# live preview, for setting up lighting and position (not for recorded measurements)
python syringe_bubble_inspector.py run --camera 0

# recorded video
python syringe_bubble_inspector.py run --video test.mp4
```

Live keys: `s` save snapshot + CSV row, `d` debug masks, `+`/`-` sensitivity
(`dark_thresh`), `w` write sensitivity to config, `p` pause, `q`/Esc quit. Live and
video results are the median of the last 15 frames (`--smooth`), and the overlay shows
**SETTLING** while the reading is still moving.

Useful `run` options: `--debug`, `--dark-thresh`, `--fill-ml`, `--margin`,
`--lateral-mag`, `--min-diam-mm`, `--polarity dark|bright|both` (only `dark`, the
backlight mode, is validated; the others are flagged with a warning).

## Photo quality checks

Each photo gets these checks. FAIL makes the photo INVALID; WARN is reported but the
photo can be used. Thresholds are the `QC_*` constants at the top of the photo-quality
section in `syringe_bubble_inspector.py`.

| Check | What it measures | WARN | FAIL |
|---|---|---|---|
| geometry | Photo size vs. the calibration photo | lower resolution (upscaled) | different shape (other camera or orientation) |
| alignment | Where the barrel is, from the walls (sideways) and the tip shoulder and plunger (along the barrel) | moved > 15% of barrel width (still followed) | moved > 30%, or barrel not found |
| scale | Size and tilt vs. the calibration photo (wall spacing; wall shift between barrel halves) | > 2% size or > 1.5° | > 5% size or > 3° |
| resolution | µL per pixel of liquid level; pixels across the smallest reportable bubble | > 5 µL/px, or < 3 px | > 10 µL/px |
| exposure | Median grey level in the barrel; fraction blown out | < 100, or > 5% clipped | < 60, or > 25% clipped |
| background | Largest local offset from the background model, ignoring bubbles and headspace | > 3% | > 6% |
| lighting | Area of bright features vs. dark features (bubbles are dark when backlit) | ratio > 0.5 | ratio > 1 |
| focus | Edge sharpness of the barrel vs. the calibration photo | < 80% | < 60% |
| stability | Change of the result when the detection threshold is set to 0.6× and 1.6× | > 10% of a limit | > 30% of a limit |
| liquid level | Headspace larger than half the barrel | | yes |
| coverage | Fraction of the barrel read as air | > 25% | |
| shapes | Share of the air volume in odd-shaped blobs | > 50% | |
| scale boxes | Possible air inside the ignored scale boxes, or a bubble crossing one | yes | |
| duplicate | Same photo as another view of this session / an earlier session | earlier session | this session |

The stability check tests the measurement directly: on a good backlit photo the result
barely moves with the threshold (bubble edges are sharp and dark); on the front-lit
photo that motivated this redesign it swung by a factor of ten.

## How it works

1. **Background model.** The barrel ROI is cropped and rotated so the air end is at
   the top. The background is modelled as a column profile × a row gain. Shading,
   glare streaks and wall shadows run the whole length of the barrel, so they live in
   the column profile and cancel out. Pixels much darker than this background
   (`dark_thresh`, default 15%) are kept.
2. **Headspace.** The liquid surface is the first dark band that spans at least 70% of
   the barrel width (`meniscus_frac`), is thin, is crossed once by most columns (a row
   of bubble rings is crossed twice), and does not touch the plunger end. Air volume
   above it is integrated assuming the surface is symmetric around the barrel axis, so
   a curved meniscus is handled.
3. **Bubbles.** The bright centre of each dark ring is filled in. An opening with a
   disk the size of the smallest reportable bubble (`min_diam_mm`, default 0.5 mm)
   removes graduation ticks, number strokes and noise. Touching bubbles are separated
   using peaks of the distance transform. Blobs lying entirely inside a scale box are
   ignored (and reported); a bubble crossing a box edge is measured whole.
4. **Volume.** Each bubble's size comes from its second moments. The liquid-filled
   barrel acts as a cylindrical lens that stretches things sideways, so the sideways
   size is divided by `lateral_mag`. Volume is that of a spheroid:
   V = π/6 · d_axial · d_lateral².
5. **Quality.** The photo is registered to the calibration photo and checked as above.
6. **Verdict.** All pockets (bubbles plus headspace) are checked against the largest
   pocket and total air limits; the guided session first combines the views.

## Validation protocol

The tests use synthetic photos drawn with the same optical model the detector assumes,
so passing them shows the code does what it was designed to do, not that the design
matches real syringes. Before quoting any value in a design report, run these with the
real rig and record the results:

1. **Blank.** A carefully bubble-free filled syringe, 5 sessions. Every reading is a
   false positive; the largest sets the detection floor.
2. **Known volumes.** Inject known air volumes with a gastight microsyringe: 5, 10,
   20, 25, 30 and 50 µL, 3 repeats each, both as a free pocket at the tip and as a
   bubble stuck to the wall. Plot measured vs. injected: the slope and offset are the
   bias, the scatter the precision. This also shows how much air sits in the tip and
   nozzle, which is not measured.
3. **Repeatability.** Re-seat and re-photograph one syringe 10 times; the standard
   deviation is the repeatability.
4. **Reproducibility.** Repeat step 3 with 2–3 operators.
5. **Daily check.** Photograph a fixed reference (for example glass beads of known size
   in a sealed filled syringe) at the start of each test day and record the reading.

Then set the MARGINAL band (`margin`) from the measured uncertainty instead of the
default 20%, and state results as value ± uncertainty.

## Testing

```bash
python test_bubble_inspector.py          # detector accuracy on synthetic photos
python test_bubble_inspector.py --save   # also write the images to synthetic/
python test_guided.py                    # quality checks and guided sessions (or: pytest)
```

`test_bubble_inspector.py` renders backlit syringe images with known bubble sizes,
including graduation ticks, printed numbers, the tip shoulder and plunger seal, a glare
streak, uneven lighting, noise and blur, and checks the measured volumes and verdicts:
a clean syringe, single 30 µL (FAIL) and 22 µL (MARGINAL) bubbles, eight small bubbles
(one across the ticks), two touching bubbles, a bubble cut off by the wall, 60 µL of
headspace with a curved meniscus, and a camera rotated 90°.

`test_guided.py` checks that good photos pass and that front-lit, blurred, badly
exposed, moved, zoomed, tilted and wrong-camera photos are rejected; that bubbles
behind the scale boxes, large bubbles, many bubbles and headspace near the limit are
measured correctly; how views are combined; and complete sessions: reports, exit codes,
duplicate photos, retakes and non-ASCII names.

Run both after any change.

## Limitations

- **Not validated on real photos yet.** See [Validation protocol](#validation-protocol).
  The quality-check thresholds were set from synthetic photos and one real front-lit
  photo; expect to adjust them once you have real backlit photos.
- **What is measured.** Air inside the barrel box only. Air in the conical tip, nozzle
  and luer is not measured, nor is air hidden in every view or behind the plunger seal.
- **Known systematic errors**, to be quantified by the validation:
  - The liquid-filled barrel magnifies sideways by about 1.0 near the front wall, about
    the refractive index (1.33) on the axis and up to about 2 near the back wall. One
    fixed `lateral_mag` therefore over- or under-sizes bubbles by depth.
  - Bubbles stuck to the wall are caps, not spheres, so their volume is overestimated.
  - The scale is taken from graduations on the front surface of the barrel, not its
    axis, which biases sizes by a few percent at short camera distances. A longer
    working distance with zoom reduces this.
  - Headspace is resolved to about half a pixel of liquid level (±3 µL at 6.6 µL/px).
    A headspace thinner than the meniscus band (one or two pixels) is over-read.
- **Very dense foam** (bubbles overlapping across most of the barrel) breaks the
  background model. The photo is then reported INVALID rather than FAIL.
- `lateral_mag` depends on the liquid's refractive index. Set it for the liquid you are
  testing.
- Camera exposure and focus control through OpenCV depend on the OS and driver.
