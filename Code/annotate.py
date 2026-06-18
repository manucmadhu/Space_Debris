import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent))
from flux_extract import extract_streak_flux

try:
    from astropy.io import fits
except ImportError:
    import sys
    sys.exit("astropy not installed. Run: pip install astropy")

# =============================================================================
# Helper Functions
# =============================================================================

def load_fits(path: Path) -> np.ndarray:
    """Loads a FITS file safely, accounting for BZERO/BSCALE scaling."""
    try:
        with fits.open(str(path), memmap=False) as h:
            d = h[0].data
            if d is None:
                raise ValueError(f"No image data in {path}")
            return d.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")

def normalise_u8(img: np.ndarray, lo=5.0, hi=99.9) -> np.ndarray:
    """Converts a float32 array to a displayable 8-bit image."""
    img = np.nan_to_num(img)
    vmin, vmax = np.percentile(img, lo), np.percentile(img, hi)
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    return np.clip((255.0 * (img - vmin) / (vmax - vmin)), 0, 255).astype(np.uint8)

def group_lines(lines, angle_tol=10.0, dist_tol=50.0):
    """Merges broken line segments into a single track using connected components."""
    if not lines: return []
    
    def ang(x1, y1, x2, y2): 
        return np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
    
    def point_to_segment_dist(px, py, x1, y1, x2, y2):
        """Finds the shortest distance from a point to a line segment."""
        l2 = (x2 - x1)**2 + (y2 - y1)**2
        if l2 == 0: 
            return np.hypot(px - x1, py - y1)
        # t is the projection of the point onto the line, clamped to [0, 1] for segment bounds
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        return np.hypot(px - proj_x, py - proj_y)

    def seg_to_seg_dist(l1, l2):
        """Calculates the minimum distance between two line segments."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        return min(
            point_to_segment_dist(x3, y3, x1, y1, x2, y2),
            point_to_segment_dist(x4, y4, x1, y1, x2, y2),
            point_to_segment_dist(x1, y1, x3, y3, x4, y4),
            point_to_segment_dist(x2, y2, x3, y3, x4, y4)
        )

    n = len(lines)
    adj = {i: [] for i in range(n)}
    
    # Build a graph of which lines are near each other and share similar angles
    for i in range(n):
        ai = ang(*lines[i])
        for j in range(i + 1, n):
            aj = ang(*lines[j])
            ang_diff = min(abs(ai - aj), 180 - abs(ai - aj))
            
            if ang_diff <= angle_tol:
                if seg_to_seg_dist(lines[i], lines[j]) <= dist_tol:
                    adj[i].append(j)
                    adj[j].append(i)
                    
    # Group connected lines (Breadth-First Search)
    visited = set()
    groups = []
    for i in range(n):
        if i not in visited:
            queue = [i]
            visited.add(i)
            comp = []
            while queue:
                curr = queue.pop(0)
                comp.append(lines[curr])
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            groups.append(comp)
            
    return groups

# =============================================================================
# Helper function for dynamic adaptive parameter estimation
# =============================================================================
def compute_adaptive_params(h: int, w: int) -> dict:
    """
    Compute detection parameters dynamically from image resolution.
    This interpolates between the verified parameters for 6.4 MP (M100)
    and 61.6 MP (Frigate) to adapt to any sensor size.
    """
    megapixels = (h * w) / 1_000_000.0
    lo_percentile = 85.0
    canny_low = 100
    canny_high = 200
    hough_threshold = max(20, min(200, int(20 + (megapixels - 6.4) * 2.35)))
    hough_min_line_len = max(25, min(50, int(30 + (megapixels - 6.4) * 0.18)))
    hough_max_line_gap = max(15, min(60, int(15 + (megapixels - 6.4) * 0.63)))
    group_angle_tol = 10.0
    group_dist_tol = 50.0
    return {
        'lo_percentile': lo_percentile,
        'canny_low': canny_low,
        'canny_high': canny_high,
        'hough_threshold': hough_threshold,
        'hough_min_line_len': hough_min_line_len,
        'hough_max_line_gap': hough_max_line_gap,
        'group_angle_tol': group_angle_tol,
        'group_dist_tol': group_dist_tol
    }

# =============================================================================
# Main Processing Logic
# =============================================================================
def process_single_fits(fits_path: str, output_png: str):
    print(f"Loading {fits_path}...")
    img_f32 = load_fits(Path(fits_path))

    # 1. Spatial Background Subtraction
    print("Applying spatial background subtraction and sharpening...")
    bg = gaussian_filter(img_f32, sigma=15.0)  
    residual = np.clip(img_f32 - bg, 0, None)
    
    # 2. Unsharp Masking
    blurred = gaussian_filter(residual, sigma=5.0)
    sharpened = np.clip(residual + (residual - blurred) * 5.0, 0, 65535).astype(np.float32)

    # 2.5 Compute Adaptive Parameters
    h, w = img_f32.shape
    params = compute_adaptive_params(h, w)

    # 3. Convert to 8-bit (CRUSH THE BLACKS)
    img_u8 = normalise_u8(sharpened, lo=params['lo_percentile'], hi=99.9)

    # =================================================================
    # 3.5 Border Masking (NEW)
    # Black out the outer edges to destroy sensor artifacts
    # =================================================================
    margin = 50  # Number of pixels to ignore on the edges. Increase if needed.
    img_u8[0:margin, :] = 0      # Black out top edge
    img_u8[-margin:, :] = 0      # Black out bottom edge
    img_u8[:, 0:margin] = 0      # Black out left edge
    img_u8[:, -margin:] = 0      # Black out right edge

    # 4. Find Edges (Stricter Canny)
    print("Detecting edges and lines...")
    edge_blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    edges = cv2.Canny(edge_blur, params['canny_low'], params['canny_high'])

    # 5. Detect Lines (Adaptive Hough Transform)
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                                threshold=params['hough_threshold'], 
                                minLineLength=params['hough_min_line_len'], 
                                maxLineGap=params['hough_max_line_gap'])
    lines = [tuple(l[0]) for l in raw_lines] if raw_lines is not None else []
    print(f"Found {len(lines)} raw line segments.")

    # --- SAFETY NET TO PREVENT CRASHES ---
    if len(lines) > 5000:
        print("\n[!] ERROR: Too many lines detected. The image is too noisy, or it is picking up too many stars.")
        print("Try increasing 'lo=85.0' to 'lo=95.0', or increasing 'minLineLength'.")
        # Save what the edge detector sees so you can debug what is going wrong
        cv2.imwrite("debug_edges.png", edges)
        print("Saved 'debug_edges.png' to help you see what the algorithm thinks are edges.")
        return
    if len(lines) == 0:
        print("\n[!] No streaks found. If there is a streak, try lowering 'minLineLength' to 20.")
        return

    # 6. Group adjacent lines into single tracks
    tracks = group_lines(lines, angle_tol=params['group_angle_tol'], dist_tol=params['group_dist_tol'])
    print(f"Merged into {len(tracks)} distinct tracks.")

    # 7. Draw the bounding boxes and lines
    ann = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    for idx, grp in enumerate(tracks):
        pts = [(x, y) for x1, y1, x2, y2 in grp for x, y in [(x1,y1), (x2,y2)]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = 10
        x = max(min(xs) - pad, 0)
        y = max(min(ys) - pad, 0)
        x2 = min(max(xs) + pad, img_u8.shape[1])
        y2 = min(max(ys) + pad, img_u8.shape[0])
        
        # Calculate start and end pixels (furthest apart points) for photometry
        max_dist = 0
        start_pixel, end_pixel = pts[0], pts[0]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                if d > max_dist:
                    max_dist, start_pixel, end_pixel = d, pts[i], pts[j]
        
        # Extract flux details
        import math
        phot = extract_streak_flux(img_f32, start_pixel, end_pixel)
        mag_val = phot['magnitude']
        mag_str = f"{mag_val:.2f}" if not math.isnan(mag_val) else "N/A"
        
        cv2.rectangle(ann, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(ann, f"T{idx+1} (M:{mag_str})", (x, max(y-5, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for x1, y1, x2, y2 in grp:
            cv2.line(ann, (x1, y1), (x2, y2), (0, 128, 255), 1)

    # 8. Save the final image
    cv2.imwrite(output_png, ann)
    print(f"Success! Annotated image saved to: {output_png}")

# --- RUN IT HERE ---
if __name__ == "__main__":
    # 1. Define your input folder and output folder here
    # (Using r"..." makes it a raw string, preventing Windows folder path errors)
    input_folder = Path(r"frigate-main\frigate-main\output")
    output_folder = Path(r"frigate-main\frigate-main\output\rawAnnotated_Outputs")

    # 2. Create the output folder automatically if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # 3. Find all FITS files in the directory
    fits_files = list(input_folder.glob("*.fits")) + list(input_folder.glob("*.fit"))

    if not fits_files:
        print(f"No FITS files found in {input_folder}")
    else:
        print(f"Found {len(fits_files)} FITS files. Starting batch processing...")

        # 4. Loop through each file and process it
        for fits_path in fits_files:
            # Create a unique output filename (e.g., img-0001.fits -> img-0001_annotated.png)
            output_filename = f"{fits_path.stem}_annotated.png"
            output_png = output_folder / output_filename

            print(f"\n--- Processing {fits_path.name} ---")
            try:
                process_single_fits(str(fits_path), str(output_png))
            except Exception as e:
                # If one file fails, print the error and continue to the next one
                print(f"[!] Failed to process {fits_path.name}: {e}")

        print(f"\nBatch complete! All annotated images saved to: {output_folder}")