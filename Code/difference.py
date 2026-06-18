# This is intended to preprocess the FITS file and generate the png files for substraction procedure
import argparse
import sys
import time
import threading
import queue
from pathlib import Path
import multiprocessing
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from pre_process import normalise_u8_fast,save_png_fast,get_sorted_fits



def compute_difference_image(fits_dir: Path, start_filename: str,
                              num_frames: int):
    """
    Compute average absolute-difference image over a sequence of frames.
    Returns (diff_array, header) where header is copied from the first frame
    so that DATE-OBS, TELESCOP, INSTRUME, RA, DEC etc. are preserved.
    """
    files = get_sorted_fits(fits_dir)
    if not files:
        raise FileNotFoundError(f"No FITS files in {fits_dir}")
    names = [f.name for f in files]
    if start_filename not in names:
        raise ValueError(f"'{start_filename}' not found.\nFirst 10: {names[:10]}")

    s, e = names.index(start_filename), 0
    e    = min(s + num_frames, len(files))
    seg  = files[s:e]
    print(f"Using frames {s}-{e-1}  ({len(seg)} frames)")

    # Load first frame and keep its header
    with fits.open(str(seg[0]), memmap=False) as h:
        prev        = h[0].data.astype(np.float32)
        base_header = h[0].header.copy()

    diff_accum = np.zeros_like(prev, dtype=np.float64)
    n_diffs    = 0

    # Also grab the last frame header to record DATE-END
    last_header = None
    for f in tqdm(seg[1:], desc="  Differencing"):
        with fits.open(str(f), memmap=False) as h:
            curr        = h[0].data.astype(np.float32)
            last_header = h[0].header.copy()
        diff_accum += np.abs(curr.astype(np.float64) - prev.astype(np.float64))
        n_diffs    += 1
        prev        = curr

    diff = (diff_accum / max(n_diffs, 1)).astype(np.float32)

    # Build output header: start from first frame, add pipeline metadata
    out_header = base_header.copy()
    out_header["NAXIS1"]   = diff.shape[1]
    out_header["NAXIS2"]   = diff.shape[0]
    out_header["BITPIX"]   = -32   # float32
    out_header["PIPELINE"] = "FRIGATE difference image"
    out_header["NFRAMES"]  = (n_diffs + 1, "Number of frames combined")
    out_header["FSTART"]   = (seg[0].name,  "First frame in sequence")
    out_header["FEND"]     = (seg[-1].name, "Last frame in sequence")

    # Copy DATE-END from last frame if available
    if last_header is not None:
        for key in ["DATE-END", "DATE_END", "DATEEND"]:
            if key in last_header:
                out_header["DATE-END"] = last_header[key]
                break
        # Also grab MJD-END if present
        for key in ["MJD-END", "MJD_END"]:
            if key in last_header:
                out_header["MJD-END"] = last_header[key]
                break

    # Keep only keys that are valid for a 2D float image
    for key in ["BZERO", "BSCALE"]:
        if key in out_header:
            del out_header[key]

    return diff, out_header


def run_difference(fits_dir: Path, start_frame: str, num_frames: int, output_dir: Path) -> tuple:
    """
    Main function to compute the difference image, write both FITS and PNG outputs,
    and return their paths.
    """
    fits_dir = Path(fits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    diff, header = compute_difference_image(fits_dir, start_frame, num_frames)

    # Print key header values so user can verify
    for key in ["DATE-OBS", "DATE-END", "TELESCOP", "INSTRUME",
                "OBJECT", "RA", "DEC", "EXPTIME", "FILTER"]:
        if key in header:
            print(f"  {key:10s} = {header[key]}")

    start_stem = Path(start_frame).stem.replace(" ", "_")
    fits_out = output_dir / f"diff_{start_stem}_{num_frames}f.fits"
    fits.writeto(str(fits_out), diff, header, overwrite=True)
    print(f"FITS  -> {fits_out}")
    
    png_out = output_dir / f"diff_{start_stem}_{num_frames}f.png"
    save_png_fast(diff, png_out)
    print(f"PNG   -> {png_out}")
    
    return diff, header, fits_out, png_out


def cmd_difference(args):
    run_difference(Path(args.fits_dir), args.start_frame, args.num_frames, Path(args.output_dir))

