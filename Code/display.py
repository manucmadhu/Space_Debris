import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

def load_fits_data(fits_path: Path) -> tuple:
    """
    Loads a FITS image and returns its data array and header.
    """
    try:
        with fits.open(str(fits_path), memmap=False) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError("FITS file contains no data in primary HDU.")
            return data.astype(np.float32), hdul[0].header
    except Exception as e:
        raise RuntimeError(f"Error loading {fits_path.name}: {e}")

def apply_stretch(img: np.ndarray, stretch_type: str = "percentile", 
                  lo_val: float = 0.5, hi_val: float = 99.5) -> np.ndarray:
    """
    Applies a stretching function to fit float FITS data into a normalized range [0, 1].
    
    Supported stretches:
      - 'percentile': Scale between low and high percentiles.
      - 'linear': Scale between actual min and max values.
      - 'log': Logarithmic scaling to bring out faint details.
    """
    img_clean = np.nan_to_num(img)
    
    if stretch_type == "percentile":
        vmin, vmax = np.percentile(img_clean, [lo_val, hi_val])
        if vmax <= vmin:
            return np.zeros_like(img_clean)
        return np.clip((img_clean - vmin) / (vmax - vmin), 0.0, 1.0)
        
    elif stretch_type == "linear":
        vmin, vmax = np.min(img_clean), np.max(img_clean)
        if vmax <= vmin:
            return np.zeros_like(img_clean)
        return (img_clean - vmin) / (vmax - vmin)
        
    elif stretch_type == "log":
        # Shift data to be positive
        vmin = np.min(img_clean)
        shifted = img_clean - vmin + 1.0
        log_img = np.log10(shifted)
        lmin, lmax = np.min(log_img), np.max(log_img)
        if lmax <= lmin:
            return np.zeros_like(img_clean)
        return (log_img - lmin) / (lmax - lmin)
        
    else:
        # Fallback to linear
        return apply_stretch(img, "linear")

def plot_fits_figure(data: np.ndarray, header: fits.Header = None, 
                      stretch_type: str = "percentile", cmap: str = "gray", 
                      tracks: list = None, title: str = "FITS Image Viewer") -> plt.Figure:
    """
    Generates a styled Matplotlib Figure of the FITS data with optional WCS grid
    and streak annotations.
    """
    fig = plt.figure(figsize=(10, 8))
    
    wcs = WCS(header, relax=True) if header is not None else None
    has_wcs = wcs is not None and wcs.has_celestial
    
    if has_wcs:
        ax = fig.add_subplot(111, projection=wcs)
        ax.coords[0].set_axislabel('Right Ascension (RA)')
        ax.coords[1].set_axislabel('Declination (Dec)')
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
    # Scale image data
    scaled_data = apply_stretch(data, stretch_type)
    
    im = ax.imshow(scaled_data, cmap=cmap, origin='lower')
    fig.colorbar(im, ax=ax, label='Scaled Intensity')
    ax.set_title(title)
    
    # Draw any debris tracks if provided
    if tracks:
        for idx, track in enumerate(tracks):
            # track is a tuple of ((x1, y1), (x2, y2), label) or similar
            # If it is just endpoints, handle it
            if len(track) >= 2:
                p1, p2 = track[0], track[1]
                label = track[2] if len(track) > 2 else f"T{idx+1}"
                
                # Plot line
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='orange', linestyle='--', linewidth=1.5)
                # Plot endpoints
                ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], color='red', s=30)
                
                # Bounding box
                x_min, x_max = min(p1[0], p2[0]) - 10, max(p1[0], p2[0]) + 10
                y_min, y_max = min(p1[1], p2[1]) - 10, max(p1[1], p2[1]) + 10
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     fill=False, edgecolor='cyan', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(x_min, y_max + 2, label, color='cyan', fontsize=10, fontweight='bold')
                
    fig.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Standalone FITS File Viewer")
    parser.add_argument("--file", type=str, required=True, help="Path to the FITS file to view")
    parser.add_argument("--stretch", type=str, default="percentile", choices=["percentile", "linear", "log"],
                        help="Image scaling algorithm")
    parser.add_argument("--cmap", type=str, default="gray", help="Matplotlib color map")
    
    args = parser.parse_args()
    
    fits_path = Path(args.file)
    if not fits_path.exists():
        print(f"Error: file not found at {fits_path}")
        sys.exit(1)
        
    try:
        print(f"Loading {fits_path.name}...")
        data, header = load_fits_data(fits_path)
        print(f"Image shape: {data.shape}")
        
        fig = plot_fits_figure(data, header, stretch_type=args.stretch, cmap=args.cmap, title=fits_path.name)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
