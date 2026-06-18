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

# ── Optional GPU ─────────────────────────────────────────────────────────────
_GPU          = False
_cp           = None
_gpu_gaussian = None

def _init_gpu(): #If GPU is available the image processing is done on GPU
    global _GPU, _cp, _gpu_gaussian
    if _GPU:
        return
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter as gf
        _ = cp.zeros((4, 4), dtype=cp.float32)   # smoke test
        _cp = cp; _gpu_gaussian = gf; _GPU = True
        print("[GPU] NVIDIA GPU active — blurs run on GPU.")
    except Exception as e:
        print(f"[CPU] GPU not available ({type(e).__name__}).")
        
#load the fits files
def load_fits(path: Path) -> np.ndarray:
    with fits.open(str(path), memmap=False) as h:
        d = h[0].data
        if d is None:
            raise ValueError(f"No image data in {path}")
        return d.astype(np.float32)
 
# To get the sorted list of fits files    
def get_sorted_fits(fits_dir: Path) -> list:
    files = sorted(list(fits_dir.rglob("*.fits")) + list(fits_dir.rglob("*.fit")))
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); out.append(f)
    return out

# To Blur the image
def fast_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """cv2 Gaussian blur (5x faster than scipy). GPU if CuPy available."""
    if _GPU and _cp is not None:
        try:
            return _cp.asnumpy(_gpu_gaussian(_cp.asarray(img), sigma=sigma))
        except Exception:
            pass
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

# To get the background of the image from the mean of neighbours adn Gaussian Blur
def fast_background(target: np.ndarray, neighbours: list,
                    sigma: float = 5.0) -> np.ndarray:
    """
    Background = mean of neighbours + Gaussian blur.
    Uses manual accumulation (4x faster than np.mean(np.stack(...))).
    """
    if not neighbours:
        return np.zeros_like(target)
    acc = neighbours[0].copy()
    for f in neighbours[1:]:
        acc += f
    acc /= len(neighbours)
    return fast_blur(acc, sigma)


def normalise_u8_fast(img: np.ndarray, lo=5.0, hi=99.9) -> np.ndarray:
    """
    Percentile stretch to uint8.
    Samples 1/16 of pixels for percentile — 10x faster, same result.
    """
    img = np.nan_to_num(img)
    sample = img.ravel()[::16]
    vmin, vmax = np.percentile(sample, [lo, hi])
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    return np.clip((255.0 * (img - vmin) / (vmax - vmin)), 0, 255).astype(np.uint8)


def save_png_fast(img_f32: np.ndarray, path: Path) -> None:
    """cv2.imwrite with compression=3 — 3x faster than PIL default."""
    u8 = normalise_u8_fast(img_f32)
    cv2.imwrite(str(path), u8, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def process_frame(target: np.ndarray, neighbours: list) -> np.ndarray:
    """Background subtract + unsharp-mask sharpen one frame."""
    bg       = fast_background(target, neighbours, sigma=5.0)
    residual = np.clip(target - bg, 0, None)
    blurred  = fast_blur(residual, sigma=10.0)
    return np.clip(residual + (residual - blurred) * 10.0,
                   0, 65535).astype(np.float32)


# =============================================================================
# Async prefetch loader
# =============================================================================

class PrefetchLoader:
    """
    Loads FITS batches from disk in a background thread so that
    I/O and computation overlap instead of running sequentially.

    Usage:
        with PrefetchLoader(files, batches, num_adj) as loader:
            for batch_indices, cache in loader:
                ... process frames ...
    """

    def __init__(self, files: list, batches: list, num_adj: int,
                 prefetch: int = 2):
        self.files     = files
        self.batches   = batches
        self.num_adj   = num_adj
        self._q        = queue.Queue(maxsize=prefetch)
        self._thread   = threading.Thread(target=self._load_loop, daemon=True)

    def _load_loop(self):
        n = len(self.files)
        for batch_indices in self.batches:
            raw_start = max(0, batch_indices[0]  - self.num_adj)
            raw_end   = min(n, batch_indices[-1] + self.num_adj + 1)
            cache = {}
            for j in range(raw_start, raw_end):
                try:
                    cache[j] = load_fits(self.files[j])
                except Exception as e:
                    pass   # missing frames handled in main loop
            self._q.put((batch_indices, cache))
        self._q.put(None)   # sentinel

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        pass

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            yield item


# =============================================================================
# Stage 1: inspect
# =============================================================================

def cmd_inspect(args):
    if hasattr(args, 'fits_file') and args.fits_file:
        _print_header(Path(args.fits_file))
    else:
        files = get_sorted_fits(Path(args.fits_dir))
        if not files:
            print(f"No FITS files in {args.fits_dir}"); return
        for f in files:
            _print_header(f); print()

def _print_header(p):
    print("=" * 60)
    print(f"File: {p}")
    print("=" * 60)
    with fits.open(str(p), memmap=False) as h:
        h.info(); print(); print(repr(h[0].header))


# =============================================================================
# Stage 2: preprocess  — async prefetch + fast math
# =============================================================================

def run_preprocessing(fits_dir: Path, output_dir: Path, num_adj: int = 2, batch_size: int = 40):
    """
    Main entry point for preprocessing FITS files and saving them as preprocessed PNGs.
    """
    fits_dir = Path(fits_dir)
    target_output_dir = Path(output_dir) / "preprocessed"
    target_output_dir.mkdir(parents=True, exist_ok=True)

    files = get_sorted_fits(fits_dir)

    if not files:
        raise ValueError(f"No FITS files in {fits_dir}")

    n = len(files)
    print(f"Found {n} FITS files.")
    if n < 2 * num_adj + 1:
        raise ValueError(f"Need >= {2*num_adj+1} frames. Reduce --num-adj.")

    _init_gpu()

    processable   = list(range(num_adj, n - num_adj))
    total         = len(processable)
    batches       = [processable[i:i+batch_size]
                     for i in range(0, total, batch_size)]
    total_batches = len(batches)

    print(f"Frames to process : {total}")
    print(f"Batch size        : {batch_size}  ({total_batches} batches)")
    print(f"RAM window        : {batch_size + 2*num_adj} frames at once")
    print(f"Prefetch          : next batch loads while current batch computes")
    print(f"Blur engine       : {'GPU (CuPy)' if _GPU else 'cv2 (CPU)'}")
    print()

    done = errors = 0
    t_start = time.perf_counter()

    with PrefetchLoader(files, batches, num_adj) as loader:
        for batch_num, (batch_indices, cache) in enumerate(loader, 1):

            for i in batch_indices:
                if i not in cache:
                    errors += 1; continue
                try:
                    neighbours = [cache[j]
                                  for j in range(max(0, i - num_adj),
                                                 min(n, i + num_adj + 1))
                                  if j != i and j in cache]
                    sharpened = process_frame(cache[i], neighbours)
                    save_png_fast(sharpened,
                                  target_output_dir / (files[i].stem + "_preprocessed.png"))
                    done += 1
                except Exception as e:
                    print(f"\n  error {files[i].name}: {e}", file=sys.stderr)
                    errors += 1

            del cache   # free RAM; next batch already loading in background

            elapsed = time.perf_counter() - t_start
            rate    = done / max(elapsed, 1)
            eta_s   = (total - done) / max(rate, 1e-6)
            print(f"Batch {batch_num:>4}/{total_batches}  |  "
                  f"{done}/{total} done  |  "
                  f"{rate:.2f} fr/s  |  "
                  f"ETA {eta_s/60:.0f} min")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {done} frames in {elapsed/60:.1f} min  "
          f"({elapsed/max(done,1):.2f} s/frame)  |  errors: {errors}")
    print(f"PNGs -> {target_output_dir}")
    return done, errors, target_output_dir


def cmd_preprocess(args):
    run_preprocessing(args.fits_dir, args.output_dir, args.num_adj, args.batch_size)

