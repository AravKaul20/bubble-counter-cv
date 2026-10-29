# Syringe Bubble Inspector

Measures the residual air in a filled syringe from a camera image and checks it
against the EGR 101 de-airing spec:

| Requirement | Limit |
|---|---|
| Largest single air pocket | ≤ 25 µL |
| Total air | < 250 µL per 10 mL of liquid (25 µL/mL) |

The output is a **volume in microliters** plus a **PASS / MARGINAL / FAIL** verdict,
not only a bubble count, so it can be used as an objective bench test for the
de-airing device. MARGINAL means a value is within 20% below a limit (set by `margin`).

## Files

| File | Purpose |
|---|---|
| `syringe_bubble_inspector.py` | The detector and command-line tool (setup, live camera, video, images) |
| `test_bubble_inspector.py` | Synthetic accuracy test with known bubble sizes |
| `requirements.txt` | Python dependencies |

## Install

Python 3.8+.

```bash
pip install -r requirements.txt
```

`setup` and live/video mode open windows, so they need the full `opencv-python`
package. On a headless machine (CI, a server) `opencv-python-headless` is enough for
`run --image` and the test.

## Imaging setup

The method depends on these conditions. Results get worse the further you are from them.

1. **Backlit.** A diffuse light panel *behind* the syringe, camera in front. Bubbles
   then show up as dark rings on a bright field.
2. **Fixed geometry.** Hold the syringe in a jig at a fixed distance from the camera,
   so one calibration covers every test.
3. **Known air end.** Tell the program which side of the image the tip (air end) is
   on: `top`, `bottom`, `left` or `right`. Free air collects there and is measured as
   headspace.
4. **Fixed exposure and focus.** Auto-exposure moves the threshold between frames.
   Use `--exposure` if your camera driver supports it, or lock it in the camera app.

## Usage

### 1. One-time setup

```bash
python syringe_bubble_inspector.py setup --camera 0 --air-end top
```

Press SPACE to freeze a frame, then:

1. Drag a box just **inside** the barrel walls (the liquid to inspect), then ENTER.
2. Optionally box the printed graduation numbers so they are ignored (ENTER after
   each box), ESC when done.
3. Type the barrel inner diameter in mm (measure it with calipers).
4. Click two graduation marks far apart along the barrel, then type their spacing,
   e.g. `5 ml` or `20 mm`.

This writes `bubble_config.json`. Use `--image photo.png` to calibrate from a still
image instead of the camera. Each step can also be given on the command line, which
skips its window:

```bash
python syringe_bubble_inspector.py setup --roi 140 60 362 880 --mm-per-px 0.04 \
    --barrel-id-mm 14.5 --air-end top --no-exclusions
```

Other setup options: `--fill-ml` (liquid volume, default 10; sets the total-air
limit) and `--lateral-mag` (about 1.33 for saline, 1.4–1.5 for contrast media; see
[How it works](#how-it-works)).

Setup prints the headspace resolution in µL per pixel. If it warns that this is
above 5 µL/px, move the camera closer or raise the resolution.

### 2. Inspect

```bash
# live camera
python syringe_bubble_inspector.py run --camera 0

# recorded video
python syringe_bubble_inspector.py run --video test.mp4

# still images
python syringe_bubble_inspector.py run --image a.png b.png --out results
```

Live keys:

| Key | Action |
|---|---|
| `s` | Save a snapshot and append a CSV row |
| `d` | Toggle the debug masks |
| `+` / `-` | Adjust sensitivity (`dark_thresh`) |
| `w` | Write the current sensitivity to the config |
| `p` | Pause |
| `q` / Esc | Quit |

Live and video results are the median of the last 15 frames (`--smooth`). The overlay
shows **SETTLING** while the reading is still moving, for example right after the
syringe is tapped.

Useful `run` options: `--debug` (also save or show the intermediate masks),
`--dark-thresh`, `--fill-ml`, `--margin`, `--lateral-mag`, `--min-diam-mm`,
`--polarity dark|bright|both` (for lighting other than a backlight).

### Output

Everything goes to `--out` (default `bubble_results/`):

- `<name>_annotated.png`: the image with each bubble outlined and labelled in µL
  (green = small, orange = over half the limit, red = over the limit, magenta = odd
  shape still counted as air), headspace shaded red, and a summary panel on the right.
- `measurements.csv`: one row per image or snapshot with timestamp, source, verdict,
  largest pocket, total air, headspace, bubble count and frames averaged.

Example text output:

```
01_input.png: FAIL  largest pocket 31.3 uL (limit 25) | total 31.3 / 250 uL | 1 bubbles | headspace 0.0 uL
   #1   d_axial  3.93 mm  d_lateral  3.90 mm    31.34 uL
```

With `--image`, the exit code is 0 only if every image passes, so it can be used in a
script.

## How it works

1. **Background model.** The barrel ROI is cropped and rotated so the air end is at
   the top. The background is modelled as a column profile × a row gain. Shading,
   glare streaks and wall shadows run the whole length of the barrel, so they live in
   the column profile and cancel out. Pixels much darker than this background
   (`dark_thresh`, default 15%) are kept.
2. **Headspace.** The liquid surface is the first dark band that spans at least 70% of
   the barrel width (`meniscus_frac`). Air volume above it is integrated assuming the
   surface is symmetric around the barrel axis, so a curved meniscus is handled.
3. **Bubbles.** The bright centre of each dark ring is filled in. An opening with a
   disk the size of the smallest reportable bubble (`min_diam_mm`, default 0.5 mm)
   removes graduation ticks, number strokes and noise. Touching bubbles are separated
   using peaks of the distance transform.
4. **Volume.** Each bubble's size comes from its second moments. The liquid-filled
   barrel acts as a cylindrical lens that stretches things sideways by about the
   liquid's refractive index, so the sideways size is divided by `lateral_mag`. Volume
   is that of a spheroid: V = π/6 · d_axial · d_lateral².
5. **Verdict.** All pockets (bubbles plus headspace) are checked against the largest
   pocket and total air limits.

## Testing

```bash
python test_bubble_inspector.py          # run all checks
python test_bubble_inspector.py --save   # also write the images to synthetic/
```

The test renders backlit syringe images with known bubble sizes, including
graduation ticks, printed numbers, a glare streak, uneven lighting, noise and blur,
and checks the measured volumes and verdicts. Scenes covered:

- clean syringe
- single 30 µL (FAIL) and 22 µL (MARGINAL) bubbles
- eight small bubbles, one lying across the graduation ticks
- two touching 2 mm bubbles
- a bubble cut off by the barrel wall
- 60 µL of headspace with a curved meniscus plus one bubble
- a syringe lying sideways (`air_end="left"`)

Run it after any change to the detector.

## Limitations

- **The synthetic test passing does not prove the tool works on real photos.**
  Before trusting it, check it against syringes with known injected air volumes (for
  example from a calibrated microsyringe) under your actual lighting and camera.
- Only the side of the barrel facing the camera is seen. Bubbles hidden behind other
  bubbles or the plunger seal are missed, and a bubble pressed against the wall is
  only partly visible (its volume is underestimated).
- Headspace resolution is one pixel of height. Check the µL/px figure from setup.
- `lateral_mag` depends on the liquid's refractive index. Set it for the liquid you
  are testing.
- Camera exposure and focus control through OpenCV depend on the OS and driver.
