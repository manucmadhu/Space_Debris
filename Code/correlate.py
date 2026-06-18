import cv2
import math
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent))
from flux_extract import extract_streak_flux

# Astrometry & Orbital Imports
from astropy.io import fits
from astropy.wcs import WCS
from skyfield.api import load, wgs84
from datetime import datetime, timezone

# =============================================================================
# 1. Image Processing & Computer Vision Helpers
# =============================================================================

def load_fits(path: Path) -> np.ndarray:
    with fits.open(str(path), memmap=False) as h:
        d = h[0].data
        if d is None: raise ValueError(f"No image data in {path}")
        return d.astype(np.float32)

def normalise_u8(img: np.ndarray, lo=85.0, hi=99.9) -> np.ndarray:
    img = np.nan_to_num(img)
    vmin, vmax = np.percentile(img, lo), np.percentile(img, hi)
    if vmax <= vmin: return np.zeros_like(img, dtype=np.uint8)
    return np.clip((255.0 * (img - vmin) / (vmax - vmin)), 0, 255).astype(np.uint8)

def group_lines(lines, angle_tol=10.0, dist_tol=50.0):
    if not lines: return []
    def ang(x1, y1, x2, y2): return np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
    def point_to_segment_dist(px, py, x1, y1, x2, y2):
        l2 = (x2 - x1)**2 + (y2 - y1)**2
        if l2 == 0: return np.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
        return np.hypot(px - (x1 + t * (x2 - x1)), py - (y1 + t * (y2 - y1)))
    def seg_to_seg_dist(l1, l2):
        return min(point_to_segment_dist(l2[0], l2[1], *l1), point_to_segment_dist(l2[2], l2[3], *l1),
                   point_to_segment_dist(l1[0], l1[1], *l2), point_to_segment_dist(l1[2], l1[3], *l2))

    n = len(lines)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        ai = ang(*lines[i])
        for j in range(i + 1, n):
            aj = ang(*lines[j])
            if min(abs(ai - aj), 180 - abs(ai - aj)) <= angle_tol:
                if seg_to_seg_dist(lines[i], lines[j]) <= dist_tol:
                    adj[i].append(j); adj[j].append(i)
                    
    visited, groups = set(), []
    for i in range(n):
        if i not in visited:
            queue, comp = [i], []
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(lines[curr])
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor); queue.append(neighbor)
            groups.append(comp)
    return groups

def get_track_endpoints(group):
    pts = [(x1, y1) for x1, y1, _, _ in group] + [(x2, y2) for _, _, x2, y2 in group]
    max_dist, start_pixel, end_pixel = 0, pts[0], pts[0]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d > max_dist:
                max_dist, start_pixel, end_pixel = d, pts[i], pts[j]
    return start_pixel, end_pixel

# =============================================================================
# 2. Orbital Correlation Engine
# =============================================================================

def correlate_track(wcs_info, obs_time_dt, start_pixel, end_pixel, satellites, observer):
    ra_A, dec_A = wcs_info.pixel_to_world_values(start_pixel[0], start_pixel[1])
    ra_B, dec_B = wcs_info.pixel_to_world_values(end_pixel[0], end_pixel[1])
    
    target_ra, target_dec = (ra_A + ra_B) / 2.0, (dec_A + dec_B) / 2.0
    
    print(f"  -> Extracted Streak Centroid: RA {target_ra:.4f}°, Dec {target_dec:.4f}°")
    
    ts = load.timescale()
    t = ts.from_datetime(obs_time_dt)
    
    closest_sat = None
    min_sep = float('inf')
    
    print(f"  -> Scanning {len(satellites)} catalog objects... (this may take ~10 seconds)")
    
    for sat in satellites:
        # Switch between Topocentric (known location) and Geocentric (unknown location)
        if observer is None:
            position = sat.at(t)
        else:
            position = (sat - observer).at(t)

        ra_skyfield, dec_skyfield, _ = position.radec()

        ra_deg = ra_skyfield.hours * 15.0
        dec_deg = dec_skyfield.degrees

        ra_diff = ra_deg - target_ra
        dec_diff = dec_deg - target_dec
        
        # Correct for RA wrap-around
        if ra_diff > 180: ra_diff -= 360
        if ra_diff < -180: ra_diff += 360
        
        # Pythagorean angular separation
        separation = math.sqrt((ra_diff * math.cos(math.radians(target_dec)))**2 + dec_diff**2)
        
        if separation < min_sep:
            min_sep = separation
            closest_sat = sat
            
    return closest_sat, min_sep

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
    
    # 1. Normalisation percentile (stable contrast for both sensors)
    lo_percentile = 85.0
    
    # 2. Canny thresholds (stricter to prevent noise flooding on large images)
    canny_low = 100
    canny_high = 200
    
    # 3. Hough transform parameters (interpolated dynamically)
    hough_threshold = max(20, min(200, int(20 + (megapixels - 6.4) * 2.35)))
    hough_min_line_len = max(25, min(50, int(30 + (megapixels - 6.4) * 0.18)))
    hough_max_line_gap = max(15, min(60, int(15 + (megapixels - 6.4) * 0.63)))
    
    # 4. Grouping parameters
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
# 3. Main Master Function
# =============================================================================

def process_and_correlate(fits_path: str, catalog_path: str, output_png: str):
    print(f"\n[{fits_path}] Starting automated extraction & correlation...")
    
    # --- 1. Load Image & Extract Header Data ---
    img_f32 = load_fits(Path(fits_path))
    with fits.open(fits_path, memmap=False) as hdul:
        header = hdul[0].header.copy()
        
        # Added relax=True to force extraction of all Astrometry.net WCS keys
        wcs_info = WCS(header, relax=True)
        
        # Check if Astrometry injected the celestial coordinates
        has_wcs = wcs_info is not None and wcs_info.has_celestial
        
        obs_time_dt = None
        observer = None
        tolerance = 0.0
        satellites = []
        
        if not has_wcs:
            print("[!] WARNING: This image is missing celestial WCS coordinates!")
            print("[!] Running in Pixel-space fallback mode. Orbital correlation will be skipped.")
        else:
            # Handle high-precision fractional seconds in DATE-OBS
            date_obs_str = header.get('DATE-OBS', None)
            if not date_obs_str:
                print("[!] WARNING: 'DATE-OBS' missing in header. Skipping orbital correlation.")
                has_wcs = False
            else:
                try:
                    if '.' in date_obs_str:
                        base_time, fraction = date_obs_str.split('.')
                        fraction = fraction[:6] # Truncate to exactly 6 digits maximum
                        clean_date_str = f"{base_time}.{fraction}"
                        obs_time_dt = datetime.strptime(clean_date_str, '%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc)
                    else:
                        obs_time_dt = datetime.strptime(date_obs_str, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)
                except Exception as e:
                    print(f"[!] WARNING: Failed to parse DATE-OBS '{date_obs_str}': {e}. Skipping correlation.")
                    has_wcs = False

            if has_wcs:
                # --- Dynamically Extract Telescope Location ---
                lat = header.get('SITELAT', header.get('OBSGEO-B'))
                lon = header.get('SITELONG', header.get('OBSGEO-L'))
                elev = header.get('SITEELEV', header.get('OBSGEO-H', 0.0))

                # Catch missing Frigate metadata and fall back to Earth-center
                if lat is None or lon is None:
                    print("[!] WARNING: Geographic metadata stripped (Frigate Dataset).")
                    print("[!] Falling back to Geocentric (Earth-center) correlation.")
                    observer = None
                    tolerance = 5.0  # Massive tolerance to account for Parallax error
                else:
                    observer = wgs84.latlon(lat, lon, elevation_m=elev)
                    tolerance = 0.5  # Strict tolerance for known locations

    # --- 2. Load Orbital Catalog ---
    if has_wcs:
        try:
            satellites = load.tle_file(catalog_path)
        except Exception as e:
            print(f"[!] WARNING: Failed to load catalog '{catalog_path}': {e}. Skipping correlation.")
            has_wcs = False

    # --- 2.5 Compute Adaptive Parameters ---
    h, w = img_f32.shape
    params = compute_adaptive_params(h, w)

    # --- 3. Clean & Threshold Image ---
    print("Applying computer vision filters...")
    bg = gaussian_filter(img_f32, sigma=15.0)  
    residual = np.clip(img_f32 - bg, 0, None)
    blurred = gaussian_filter(residual, sigma=5.0)
    sharpened = np.clip(residual + (residual - blurred) * 5.0, 0, 65535).astype(np.float32)
    img_u8 = normalise_u8(sharpened, lo=params['lo_percentile'], hi=99.9)

    margin = 50
    img_u8[0:margin, :] = img_u8[-margin:, :] = img_u8[:, 0:margin] = img_u8[:, -margin:] = 0 

    # --- 4. Detect Lines ---
    edge_blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    edges = cv2.Canny(edge_blur, params['canny_low'], params['canny_high'])
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                                 threshold=params['hough_threshold'], 
                                 minLineLength=params['hough_min_line_len'], 
                                 maxLineGap=params['hough_max_line_gap'])
    lines = [tuple(l[0]) for l in raw_lines] if raw_lines is not None else []
    
    if len(lines) > 5000:
        print("[!] Image too noisy. Aborting to prevent crash.")
        return []
    if len(lines) == 0:
        print("[!] No streaks detected.")
        return []

    tracks = group_lines(lines, angle_tol=params['group_angle_tol'], dist_tol=params['group_dist_tol'])
    print(f"Detected {len(tracks)} distinct debris tracks. Analyzing...")

    # --- 5. Correlate, Photometer & Annotate ---
    ann = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    results = []
    
    for idx, grp in enumerate(tracks):
        start_pixel, end_pixel = get_track_endpoints(grp)
        
        if has_wcs:
            closest_sat, min_sep = correlate_track(wcs_info, obs_time_dt, start_pixel, end_pixel, satellites, observer)
            is_match = min_sep <= tolerance
            label = closest_sat.name if is_match else "UCT (Unknown Debris)"
            color = (0, 255, 0) if is_match else (0, 0, 255)
        else:
            closest_sat = None
            min_sep = float('inf')
            is_match = False
            label = "UCT (No WCS)"
            color = (0, 0, 255)
        
        # Extract flux and magnitude details for the streak
        phot = extract_streak_flux(img_f32, start_pixel, end_pixel, W_aper=5.0, W_bg1=7.0, W_bg2=12.0, zeropoint=25.0)
        mag_val = phot['magnitude']
        mag_str = f"{mag_val:.2f}" if not math.isnan(mag_val) else "N/A"
        
        # Calculate centroid RA/Dec
        centroid_ra, centroid_dec = 0.0, 0.0
        if has_wcs:
            try:
                mid_x = (start_pixel[0] + end_pixel[0]) / 2.0
                mid_y = (start_pixel[1] + end_pixel[1]) / 2.0
                centroid_ra, centroid_dec = wcs_info.pixel_to_world_values(mid_x, mid_y)
            except Exception:
                pass
            
        print(f"  -> Track {idx+1}: {label} (Separation: {min_sep:.4f} deg)")
        print(f"     Photometry: Net Flux = {phot['net_flux']:.1f}, Peak = {phot['peak_value']:.1f}, Mag = {mag_str}, SNR = {phot['snr']:.1f}")

        results.append({
            'track_id': idx + 1,
            'start_pixel': start_pixel,
            'end_pixel': end_pixel,
            'centroid_ra': float(centroid_ra),
            'centroid_dec': float(centroid_dec),
            'label': label,
            'is_match': bool(is_match),
            'separation_deg': float(min_sep),
            'photometry': phot
        })

        pts = [(x, y) for x1, y1, x2, y2 in grp for x, y in [(x1,y1), (x2,y2)]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = 15
        x, y = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
        x2, y2 = min(max(xs) + pad, img_u8.shape[1]), min(max(ys) + pad, img_u8.shape[0])
        
        cv2.rectangle(ann, (x, y), (x2, y2), color, 2)
        # Add magnitude details to the box label
        label_text = f"{label} (M:{mag_str})"
        cv2.putText(ann, label_text, (x, max(y-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        for x1, y1, x2, y2 in grp:
            cv2.line(ann, (x1, y1), (x2, y2), (0, 128, 255), 1)

    # --- 6. Save Final Image ---
    cv2.imwrite(output_png, ann)
    print(f"\n[SUCCESS] Annotated image saved to: {output_png}")
    return results


# =============================================================================
# Execution Block
# =============================================================================

if __name__ == "__main__":
    # FIX: Corrected all paths to point to the `output` folder instead of `output_test`
    FITS_FILE = r"frigate-main/frigate-main/output/difference_image.fits"
    TLE_CATALOG = r"frigate-main/frigate-main/3le.txt"
    OUTPUT_IMAGE = r"frigate-main/frigate-main/output/correlated_result.png"

    process_and_correlate(FITS_FILE, TLE_CATALOG, OUTPUT_IMAGE)