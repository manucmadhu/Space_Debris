# 🛰️ Orbital Debris Tracker — User Manual

**Version:** 1.0  
**Project:** Space Debris Detection & Photometry Pipeline  
**Platform:** Windows 10/11 (Python 3.10+)

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Requirements](#2-system-requirements)
3. [Installation & Setup](#3-installation--setup)
4. [Project Structure](#4-project-structure)
5. [Data Preparation](#5-data-preparation)
6. [Using the GUI (Recommended)](#6-using-the-gui-recommended)
7. [Using the Command-Line Interface](#7-using-the-command-line-interface)
8. [Understanding the Pipeline Steps](#8-understanding-the-pipeline-steps)
9. [Reading the Results](#9-reading-the-results)
10. [Output Files](#10-output-files)
11. [Troubleshooting](#11-troubleshooting)
12. [Technical Reference](#12-technical-reference)

---

## 1. Overview

The **Orbital Debris Tracker** is a Python-based scientific pipeline that automatically:

1. **Detects** moving objects (space debris, satellites) in sequences of astronomical FITS images
2. **Measures** the brightness of detected streaks using aperture photometry
3. **Identifies** the object by cross-referencing its position against a live TLE/3LE orbital catalog
4. **Visualises** results in an interactive dark-theme GUI with a zoomable image viewer

The pipeline is specifically designed to work with raw telescope captures in FITS format, including those from the **Frigate** dataset (which strips geographic metadata and uses geocentric fallback for orbital correlation).

### How it works — in plain English

When a satellite or debris passes over your telescope's field of view, it leaves a **streak** across multiple frames. This pipeline:

- **Stacks** a sequence of N frames together by computing the **average absolute difference** between consecutive frames — stars cancel out (they don't move), but moving objects accumulate into bright streaks
- **Finds** those streaks using computer vision (Canny edges + Hough line transform)
- **Computes** how bright the streak is (aperture photometry → instrumental magnitude)
- **Looks up** which known satellite was at that exact sky position at that exact time

---

## 2. System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 | Windows 11 |
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16+ GB |
| GPU | — | NVIDIA (CUDA, optional) |
| Storage | 10 GB | 50+ GB (for large FITS datasets) |
| Plate Solver | Optional | Astrometry.net (WSL or native) |

> **GPU Note:** The pre-processing step can use an NVIDIA GPU via CuPy for Gaussian blur operations. This is **optional** — the pipeline runs fully on CPU if no GPU is available.

---

## 3. Installation & Setup

### Step 1 — Clone or download the project

```
d:\Space Debris\Project\
```

### Step 2 — Install Python dependencies

Open a terminal in the project folder and run:

```powershell
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Array math and image filtering |
| `matplotlib` | GUI image viewer |
| `astropy` | FITS file I/O and WCS coordinate handling |
| `skyfield` | Orbital propagation (TLE → sky position) |
| `sgp4` | SGP4 satellite propagation engine |
| `opencv-python` | Computer vision (edge detection, Hough lines) |
| `tqdm` | Progress bars in the console |
| `pandas` | Data handling |
| `requests` | (Optional) catalog downloads |

### Step 3 — (Optional) Install CuPy for GPU acceleration

If you have an NVIDIA GPU with CUDA:

```powershell
pip install cupy-cuda12x   # Replace 12x with your CUDA version
```

### Step 4 — (Optional) Install Astrometry.net Plate Solver

The plate solver converts pixel coordinates to true sky coordinates (RA/Dec). Without it, the pipeline uses **Geocentric fallback** (still works, but with wider matching tolerance).

**Windows via WSL:**
```bash
# In WSL terminal:
sudo apt-get install astrometry.net
```

The pipeline automatically tries native `solve-field` first, then falls back to WSL.

### Step 5 — Prepare your TLE catalog

Download a current TLE catalog in **3-line format (3LE)**. The default location is:

```
Data/3le.txt
```

You can download the latest catalog from:
- https://celestrak.org/SOCRATES/query.php
- https://www.space-track.org (registration required)

---

## 4. Project Structure

```
d:\Space Debris\Project\
│
├── Data/
│   ├── raw/                   ← Put your raw FITS files here
│   ├── preprocessed/          ← Auto-created: preprocessed PNGs
│   └── 3le.txt                ← TLE/3LE orbital catalog
│
├── Output/                    ← All pipeline outputs go here
│   ├── diff_*.fits            ← Difference images (FITS)
│   ├── diff_*.png             ← Difference images (PNG preview)
│   ├── diff_*_correlated.png  ← Annotated result image ← YOUR RESULT
│   └── pipeline_report.txt    ← Text summary of all detected tracks
│
├── Code/                      ← Pipeline module source files
│   ├── pre_process.py         ← Stage 1: Background subtraction
│   ├── difference.py          ← Stage 2: Frame differencing
│   ├── plate_solver.py        ← Stage 3: WCS coordinate solving
│   ├── correlate.py           ← Stage 4: CV detection + TLE matching
│   ├── annotate.py            ← Standalone annotation utility
│   ├── flux_extract.py        ← Aperture photometry engine
│   └── display.py             ← Image stretch utilities
│
├── GUI/
│   └── gui_app.py             ← Main GUI application
│
├── main.py                    ← Command-line entry point
├── run_gui.py                 ← GUI launcher script
└── requirements.txt
```

---

## 5. Data Preparation

### Raw FITS Files

- Place all raw telescope captures inside `Data/raw/`
- Files must be in `.fits` or `.fit` format
- Files are sorted **alphabetically** — ensure your filenames sort in chronological order (e.g., `Capture_00001.fits`, `Capture_00002.fits`, …)
- Each file should contain a **single 2D image** in the primary HDU (Header Data Unit)
- The FITS header should ideally contain `DATE-OBS` in ISO 8601 format: `YYYY-MM-DDTHH:MM:SS.ffffff`

### FITS Header Requirements for Full Correlation

| Header Key | Required? | Description |
|---|---|---|
| `DATE-OBS` | **Yes** | Observation timestamp (UTC) |
| `SITELAT` or `OBSGEO-B` | Optional | Observatory latitude (degrees) |
| `SITELONG` or `OBSGEO-L` | Optional | Observatory longitude (degrees) |
| `SITEELEV` or `OBSGEO-H` | Optional | Observatory elevation (metres) |

> **Missing location data?** The pipeline automatically detects this and switches to **Geocentric fallback** mode with a wider 5° matching tolerance. This is the behaviour for Frigate dataset files.

---

## 6. Using the GUI (Recommended)

### Launching the GUI

```powershell
# From the project folder:
python run_gui.py
```

The GUI window titled **"ORBITAL DEBRIS TRACKER"** will open.

---

### GUI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  ORBITAL DEBRIS TRACKER  •  Detection & Photometry Dashboard     │
│                                              ● READY  [N TRACKS] │
├─────────────┬───────────────────────────────────────────────────┤
│             │  ◉ IMAGE VIEWER  |  ◈ DETECTED TRACKS  |  ≡ CONSOLE│
│  SIDEBAR    ├───────────────────────────────────────────────────┤
│             │                                                     │
│  ◈ PATHS    │                                                     │
│  ◆ PARAMS   │              MATPLOTLIB IMAGE VIEWER                │
│  ▶ STEPS    │          (zoom, pan, coordinate readout)            │
│             │                                                     │
│ [⚡ RUN ALL] │                                                     │
└─────────────┴───────────────────────────────────────────────────┘
```

---

### Sidebar — PATHS Section

| Field | Default | Description |
|---|---|---|
| **Raw FITS** | `Data/raw` | Folder containing your raw `.fits` telescope captures |
| **Catalog** | `Data/3le.txt` | Path to your TLE/3LE orbital catalog text file |
| **Output** | `Output` | Folder where all results will be saved |

Click the **`…`** button next to each field to browse for files/folders.

---

### Sidebar — PARAMETERS Section

| Control | Description |
|---|---|
| **Start Frame** | Dropdown listing all `.fits` files found in the Raw FITS folder. Select the first frame of the sequence you want to analyse. Selecting a frame also loads it into the viewer. |
| **Frames to Stack** | Slider (2–50). The number of consecutive frames to combine into the difference image. More frames = longer integration time = fainter objects visible, but also more motion blur. **10 is a good starting point.** |
| **Run Pre-processing** | Toggle checkbox. If enabled, the Full Pipeline will first run a background subtraction pass on all raw frames. Usually not needed if your FITS data is already clean. |
| **Run Plate Solver (WCS)** | Toggle checkbox. If enabled, the pipeline will attempt to plate-solve the start frame using Astrometry.net to get true sky coordinates. Disable if you don't have Astrometry.net installed. |

---

### Sidebar — PIPELINE STEPS Section

Each step has a **pulsing status dot** on the left:
- ⚫ **Grey** — Not yet run
- 🔵 **Pulsing cyan** — Currently running
- 🟢 **Solid green** — Completed successfully
- 🔴 **Red** — Failed with an error

You can run each step individually, or use the **Full Pipeline** button to run all steps in sequence.

---

### Step-by-Step Walkthrough

#### Step 1 — Pre-process Frames *(Optional)*

**What it does:** Applies background subtraction and unsharp masking to every raw FITS file in the input folder. Saves sharpened PNG previews to `Output/preprocessed/`.

**When to use it:** Use this if your raw FITS files have strong background gradients or want PNG previews of all frames for inspection.

**When to skip it:** Skip it (the default) if you just want to run the detection pipeline quickly. This step can take many minutes for thousands of frames.

---

#### Step 2 — Compute Difference

**What it does:**
1. Reads N consecutive frames starting from your selected **Start Frame**
2. Computes the **average absolute pixel difference** between each consecutive pair of frames
3. Saves the result as a FITS file (`diff_*.fits`) and a PNG preview (`diff_*.png`) in the Output folder
4. Displays the difference image in the viewer tab

**How to interpret the difference image:** Stars appear as faint noise (they cancel out). A moving satellite or debris object appears as a **bright streak or trail**. The brighter the streak, the more the object moved relative to the exposure length.

> **Tip:** If the difference image looks mostly uniform/grey with no bright streaks, try increasing the frame count (the object may be moving very slowly) or check that you selected the right start frame.

---

#### Step 3 — Plate Solve (WCS)

**What it does:**
1. Calls Astrometry.net's `solve-field` on your raw start frame
2. Determines the precise sky coordinates (RA/Dec) for every pixel in the image
3. Injects these WCS (World Coordinate System) keys into the difference image FITS header

**If this step fails:** The pipeline prints a warning and automatically runs in **Geocentric fallback** mode during correlation. This mode uses Earth's centre as the observer reference, which introduces some parallax error, compensated by using a wider 5° matching tolerance.

**Requirements:** `solve-field` must be installed (natively on Windows, or via WSL). The correct Astrometry.net index files for your image scale must also be installed (the pipeline expects ~26–29 arcseconds/pixel).

---

#### Step 4 — Detect & Correlate *(The Main Step)*

This is the core of the pipeline. Click this after running **Compute Difference**.

**What it does, internally:**

1. **Background subtraction:** Applies a Gaussian spatial filter (σ=15px) to estimate and remove the background gradient
2. **Sharpening:** Applies unsharp masking to enhance edge contrast of streaks
3. **Normalisation:** Crushes the darkest 85% of pixels to pure black, eliminating most sensor noise and faint stars
4. **Edge detection:** Runs Canny edge detection to find sharp edges in the image
5. **Line detection:** Runs the Probabilistic Hough Line Transform to find line-like features (the minimum line length is 40 pixels)
6. **Line grouping:** Merges nearby parallel line segments (within 10° angle tolerance and 50px distance) into single **tracks** using a connected-components graph algorithm
7. **Photometry:** For each track, performs aperture photometry — measures the integrated flux inside a 5-pixel-wide aperture along the streak, subtracts local background, and calculates the instrumental magnitude and SNR
8. **Orbital correlation:** Converts each track's centroid pixel to sky coordinates (RA/Dec) using the WCS, then queries all ~31,000 objects in your TLE catalog using Skyfield's SGP4 propagator to find which known satellite was at that exact position at the time of observation
9. **Annotation:** Draws coloured bounding boxes and labels on the image:
   - 🟢 **Green** = Matched to a known catalog object
   - 🔴 **Red** = Unidentified (UCT — Unknown Correlated Target)
10. **Displays** the annotated image in the viewer and populates the Detected Tracks table

---

### The Image Viewer Tab

After running any pipeline step, the result image appears here.

| Control | Function |
|---|---|
| **Stretch** dropdown | Changes how pixel values are mapped to display brightness. `percentile` (default) auto-scales to the 0.1–99.9th percentile. `log` is good for wide dynamic range images. |
| **Colormap** dropdown | Changes the colour palette. `gray` (default) is standard. `viridis` or `inferno` can highlight faint structures. |
| **Coordinate readout** (top-right) | Shows the pixel X/Y, sky RA/Dec (if WCS is available), and raw pixel value under your mouse cursor. |
| **Matplotlib toolbar** (bottom) | Standard navigation: 🏠 Home (reset zoom), ↩ Back, ↪ Forward, ✛ Pan, 🔍 Zoom, 💾 Save |

**To zoom in on a track:** Use the 🔍 Zoom tool in the bottom toolbar, then draw a rectangle around the area you want to zoom into.

**To pan:** Use the ✛ Pan tool, then click and drag.

**To reset:** Click the 🏠 Home button.

---

### The Detected Tracks Tab

After running **Detect & Correlate** or **Full Pipeline**, this table is populated with one row per detected track.

| Column | Description |
|---|---|
| **ID** | Track number (T1, T2, …) |
| **Object** | Matched satellite name from catalog, or `UCT (Unknown Debris)` |
| **Mag** | Instrumental magnitude (brightness). Lower = brighter. `N/A` means the flux was too low/noisy to measure. |
| **SNR** | Signal-to-noise ratio. Values > 3 are reliable detections. Values 1–3 are marginal. Values < 1 are noise-dominated. |
| **Flux** | Net background-subtracted integrated flux in aperture (arbitrary units) |
| **Peak** | Peak pixel value inside the aperture |
| **Length** | Length of the detected streak in pixels |
| **RA** | Right Ascension of the track centroid (degrees) |
| **Dec** | Declination of the track centroid (degrees) |
| **Sep°** | Angular separation between the detected centroid and the matched catalog object's predicted position (degrees). Smaller = better match. |

**Row colours:**
- 🟢 **Green row** = Successfully matched to a catalog object
- 🔴 **Red row** = Unmatched / unidentified object

**Double-click any row** to zoom the Image Viewer to centre on that track's location.

---

### The Console Tab

Displays real-time output from the pipeline as it runs, including progress, per-track correlation results, and any warnings or errors. Useful for debugging.

Click **Clear Console** in the sidebar to reset it.

---

### Running the Full Pipeline

Click **⚡ RUN FULL PIPELINE** to run all four stages in sequence:

1. Pre-processing (if enabled in Parameters)
2. Frame Differencing
3. Plate Solving (if enabled in Parameters)
4. Detection, Photometry & Correlation

All four step dots will light up progressively as each stage completes. The annotated result image is displayed automatically at the end.

> **Note:** The "Run Full Pipeline" button is **disabled** while a pipeline step is running. It re-enables automatically when the step finishes or fails.

---

## 7. Using the Command-Line Interface

For batch processing or headless server use, the pipeline can be run without the GUI.

### Basic usage

```powershell
python main.py
```

This runs with all defaults (reads from `Data/raw/`, catalog `Data/3le.txt`, outputs to `Output/`, uses the first FITS file in the folder, stacks 10 frames).

### Full syntax

```powershell
python main.py [OPTIONS]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--fits-dir PATH` | `Data/raw` | Directory containing raw FITS captures |
| `--catalog PATH` | `Data/3le.txt` | Path to TLE/3LE catalog file |
| `--output-dir PATH` | `Output` | Directory for output files |
| `--start-frame NAME` | First file found | Filename of the starting FITS frame (e.g. `Capture_00001 02_55_24Z.fits`) |
| `--num-frames N` | `10` | Number of consecutive frames to stack |
| `--preprocess` | Off | Add this flag to run batch preprocessing first |
| `--no-solve` | Off | Add this flag to skip the plate solver |

### Examples

Run with 20 frames, skip the plate solver:
```powershell
python main.py --num-frames 20 --no-solve
```

Run from a specific start frame:
```powershell
python main.py --start-frame "Capture_00500 03_15_00Z.fits" --num-frames 15
```

Run the full pipeline including preprocessing:
```powershell
python main.py --preprocess --num-frames 10
```

---

## 8. Understanding the Pipeline Steps

### Pre-processing (`pre_process.py`)

The preprocessor runs a **background-subtraction + unsharp-masking** pass on every raw FITS file:

1. For each target frame, it loads the `N` adjacent frames (default: 2 on each side = 4 neighbours)
2. Computes the **mean of the neighbours** as an estimate of the background
3. Applies **Gaussian blur** (σ=5) to smooth it
4. Subtracts the background from the target: `residual = target - background`
5. Applies **unsharp masking**: `sharpened = residual + (residual - blur(residual, σ=10)) × 10`
6. Saves as a uint8 PNG with percentile stretch normalisation

Uses an **async prefetch loader** — the next batch of frames loads from disk in a background thread while the current batch is being processed, keeping the CPU busy.

### Differencing (`difference.py`)

Computes a **temporal difference image** from N consecutive frames:

```
diff = average( |frame[i+1] - frame[i]| ) for i in 1..N-1
```

This is a simple but highly effective technique: anything that is **stationary** (stars, background) cancels to near-zero, while anything that **moves** (satellites, aircraft, meteors) produces a non-zero signal.

The output FITS file inherits all header metadata from the first frame (telescope, date, coordinates).

### Plate Solving (`plate_solver.py`)

Uses **Astrometry.net's** `solve-field` command to perform blind astrometric calibration:

1. Detects stars in the raw image
2. Matches the star pattern against index files (star catalogs)
3. Determines the RA/Dec pointing, pixel scale, and orientation of the telescope
4. Writes WCS (CRVAL, CRPIX, CD matrix, SIP distortion) keys into the difference image FITS header

Settings used: 4× downsampling for speed, depth 50, 2nd-order polynomial distortion correction, scale 26–29 arcsec/pixel.

### Detection & Correlation (`correlate.py`)

The detection stage:
1. **Gaussian background subtraction** (σ=15) to remove large-scale gradients
2. **Unsharp masking** (factor 5×) to enhance streak edges
3. **Percentile stretch** — the bottom 85% of pixels are crushed to black
4. **Border masking** — outer 50px ring zeroed out to remove sensor artefacts
5. **Canny edge detection** (thresholds 100/200) on the 5×5 Gaussian-blurred image
6. **Probabilistic Hough Transform** — minimum line length 40px, max gap 50px, vote threshold 150
7. **Graph-based line grouping** — lines within 10° angle and 50px distance are merged into a single track

The correlation stage:
1. Converts the track centroid pixel to RA/Dec using the WCS
2. Queries Skyfield's SGP4 propagator for each of the ~31,648 catalog objects at the exact `DATE-OBS` timestamp
3. Computes angular separation using the haversine-like formula:
   ```
   sep = sqrt( (ΔRA × cos(dec))² + (ΔDec)² )
   ```
4. Returns the closest match
5. A match is accepted if `sep ≤ tolerance`:
   - `tolerance = 0.5°` when geographic location is known (topocentric)
   - `tolerance = 5.0°` in geocentric fallback mode

### Aperture Photometry (`flux_extract.py`)

For each detected streak, the photometer:

1. Defines an **aperture zone** — a ribbon of width ±5px around the streak line segment
2. Defines a **background annulus** — a ribbon from ±7px to ±12px around the streak
3. Computes `net_flux = sum(aperture pixels) - N_aperture × mean(background pixels)`
4. Computes **instrumental magnitude**: `mag = -2.5 × log10(net_flux) + ZP` (zeropoint = 25.0)
5. Computes **SNR**: `SNR = net_flux / sqrt(net_flux + N_aperture × σ²_background)`

---

## 9. Reading the Results

### Magnitude Scale

Astronomical magnitudes are on a **reverse logarithmic scale** — lower numbers mean brighter objects:

| Magnitude | Brightness |
|---|---|
| 10–12 | Very bright (visible to naked eye) |
| 13–15 | Bright satellite |
| 16–18 | Moderate — typical LEO debris |
| 19–21 | Faint — small debris or high altitude |
| N/A | Below detection threshold (negative flux) |

### SNR Interpretation

| SNR | Reliability |
|---|---|
| > 5 | Highly reliable detection |
| 3–5 | Good detection |
| 1–3 | Marginal — may be noise |
| < 1 | Unreliable — likely noise artefact |

### Separation Angle

The separation between the detected position and the predicted catalog position:

| Separation | Meaning |
|---|---|
| < 0.5° | Strong match (topocentric) |
| 0.5° – 2° | Possible match (geocentric) |
| > 5° | Poor match / unidentified object |

---

## 10. Output Files

After running the pipeline, the `Output/` folder will contain:

| File | Description |
|---|---|
| `diff_{frame}_{N}f.fits` | The difference image in FITS format (with WCS if solved) |
| `diff_{frame}_{N}f.png` | PNG preview of the difference image |
| `diff_{frame}_{N}f_correlated.png` | **The main result** — annotated PNG with bounding boxes and labels for all detected tracks |
| `pipeline_report.txt` | Text summary table of all detected tracks with photometry and coordinates |
| `preprocessed/` | (If preprocessing was run) Sharpened PNG versions of every raw FITS frame |

---

## 11. Troubleshooting

### No streaks detected in the difference image

- Try **increasing the number of frames** to stack (more frames = more integration = fainter streaks visible)
- Check that the object is actually moving relative to the stars (geo-stationary satellites may not produce streaks)
- The start frame might not contain any moving objects — try a different frame sequence
- Check the Console tab for any errors during the differencing step

### Too many false detections

- The image may be very noisy. The pipeline will abort if > 5,000 raw line segments are found
- Try reducing the number of frames (fewer frames = less noise accumulation in some cases)
- Check that your raw FITS files don't have hot pixels or strong optical artefacts

### Correlation shows "UCT (Unknown Debris)" for everything

- This most likely means the **plate solver did not run** or **failed** — without WCS, all tracks are labelled as UCT
- Check the Console for a line saying `[!] Falling back to Geocentric (Earth-center) correlation` — this is normal for Frigate dataset files
- Verify your `DATE-OBS` header is in the correct format: `YYYY-MM-DDTHH:MM:SS.ffffff`
- Make sure your TLE catalog (`3le.txt`) is up to date — old catalogs won't contain recently launched objects

### Plate solver fails

- `solve-field` must be installed. Test in WSL: `which solve-field`
- The correct Astrometry.net index files must be installed for your image scale (26–29 arcsec/px corresponds to **index-4107** through **index-4110**)
- Your image must contain a sufficient number of detectable stars (at least 10–20)

### GUI crashes or freezes

- The GUI runs pipeline steps in background threads — do not close the window while a step is running
- If the FITS viewer appears blank, try clicking a different tab and switching back, or select a start frame from the dropdown to re-load the image

### CUDA / CuPy warning on startup

```
UserWarning: CUDA path could not be detected.
```

This is **non-critical** and can be safely ignored. The pipeline will use the CPU for all blur operations instead of the GPU. To suppress it, either install CUDA properly or install the CPU-only build of the dependencies.

### Memory errors with large FITS files

- Large FITS files (100+ MB) are loaded using an `io.BytesIO` buffer to avoid Windows file locking issues
- If you run out of RAM, reduce the number of frames to stack
- Close other applications to free RAM before running the pipeline

---

## 12. Technical Reference

### Supported File Formats

| Input | Format |
|---|---|
| Raw images | `.fits`, `.fit` |
| Orbital catalog | `.txt` (3-line TLE / 3LE format) |

| Output | Format |
|---|---|
| Difference image | `.fits` (float32) |
| Preview images | `.png` (uint8, 8-bit) |
| Annotated result | `.png` (BGR, OpenCV output) |
| Text report | `.txt` |

### Coordinate Systems

- **Pixel coordinates:** Origin at top-left (OpenCV convention). In the GUI viewer, the Matplotlib axes show pixels with origin at bottom-left.
- **Sky coordinates:** ICRS (J2000) Right Ascension in degrees (0–360), Declination in degrees (−90 to +90)
- **WCS:** Standard FITS WCS with optional SIP polynomial distortion (injected by Astrometry.net)

### Correlation Mode Selection

The pipeline automatically selects the correlation mode based on available header data:

```
If SITELAT and SITELONG present in header:
    → Topocentric mode (observer at known ground location)
    → Tolerance = 0.5°
Else:
    → Geocentric mode (observer at Earth's centre)
    → Tolerance = 5.0°
```

### Hough Transform Parameters

| Parameter | Value | Effect |
|---|---|---|
| `threshold` | 150 | Minimum Hough vote count — higher = fewer but more certain lines |
| `minLineLength` | 40 px | Minimum streak length to consider — filters out point sources and hot pixels |
| `maxLineGap` | 50 px | Maximum gap allowed within a single streak (handles interrupted trails) |

### Photometry Aperture Parameters

| Parameter | Value | Description |
|---|---|---|
| `W_aper` | 5 px | Half-width of the streak aperture |
| `W_bg1` | 7 px | Inner edge of background annulus |
| `W_bg2` | 12 px | Outer edge of background annulus |
| `zeropoint` | 25.0 | Instrumental magnitude zeropoint |

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────┐
│         QUICK START — 3 STEPS                    │
│                                                  │
│ 1. python run_gui.py                             │
│ 2. Select your Start Frame in the sidebar        │
│ 3. Click  ⚡ RUN FULL PIPELINE                   │
│                                                  │
│    Result: annotated PNG in Output/ folder       │
│    + populated Detected Tracks table             │
│    + interactive viewer with zoom                │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│         KEYBOARD / MOUSE VIEWER CONTROLS         │
│                                                  │
│ 🔍 Zoom tool → drag rectangle to zoom            │
│ ✛ Pan tool → click & drag to pan                 │
│ 🏠 Home → reset to full view                     │
│ Mouse hover → live coordinates in top-right bar  │
│ Double-click table row → zoom to that track      │
└──────────────────────────────────────────────────┘
```

---

*Manual generated for Orbital Debris Tracker v1.0 — June 2026*
