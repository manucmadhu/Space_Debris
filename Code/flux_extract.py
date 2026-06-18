import numpy as np

def extract_streak_flux(img: np.ndarray, start_pixel: tuple, end_pixel: tuple, 
                        W_aper: float = 5.0, W_bg1: float = 7.0, W_bg2: float = 12.0, 
                        zeropoint: float = 25.0) -> dict:
    """
    Performs aperture photometry along a line segment representing a debris streak.
    
    Parameters:
      img: 2D numpy float array of the image (e.g. difference image).
      start_pixel: (x1, y1) tuple.
      end_pixel: (x2, y2) tuple.
      W_aper: Half-width of the streak aperture (pixels).
      W_bg1: Inner boundary of the background estimation region (pixels).
      W_bg2: Outer boundary of the background estimation region (pixels).
      zeropoint: Instrumental magnitude zero-point.
      
    Returns:
      A dictionary containing photometry parameters:
        - 'raw_flux': Integrated pixel values in the aperture.
        - 'net_flux': Background-subtracted integrated flux.
        - 'bg_mean': Average pixel value in the background region.
        - 'bg_std': Standard deviation of background pixels.
        - 'peak_value': Peak pixel value inside the aperture.
        - 'aperture_pixels': Count of pixels inside the aperture.
        - 'magnitude': Instrumental magnitude.
        - 'snr': Signal-to-noise ratio.
        - 'streak_length': Length of the streak in pixels.
    """
    x1, y1 = start_pixel
    x2, y2 = end_pixel
    
    # Calculate streak length
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    
    # Define bounding box around the segment, expanded by the outer background radius
    h, w = img.shape
    x_min = max(0, int(np.floor(min(x1, x2) - W_bg2)))
    x_max = min(w - 1, int(np.ceil(max(x1, x2) + W_bg2)))
    y_min = max(0, int(np.floor(min(y1, y2) - W_bg2)))
    y_max = min(h - 1, int(np.ceil(max(y1, y2) + W_bg2)))
    
    if x_max <= x_min or y_max <= y_min:
        return {
            'raw_flux': 0.0, 'net_flux': 0.0, 'bg_mean': 0.0, 'bg_std': 0.0,
            'peak_value': 0.0, 'aperture_pixels': 0, 'magnitude': float('nan'),
            'snr': 0.0, 'streak_length': L
        }
    
    # Generate coordinates in the bounding box
    yy, xx = np.mgrid[y_min:y_max+1, x_min:x_max+1]
    
    # Compute the projection parameter t to find the distance to the line segment
    l2 = dx**2 + dy**2
    if l2 == 0:
        # The line is just a point
        dist = np.hypot(xx - x1, yy - y1)
    else:
        # Distance calculation to segment: project pixel coords onto line segment
        t = ((xx - x1) * dx + (yy - y1) * dy) / l2
        t = np.clip(t, 0.0, 1.0)
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = np.hypot(xx - proj_x, yy - proj_y)
        
    # Mask of pixels inside the aperture
    aperture_mask = (dist <= W_aper)
    # Mask of background pixels in a ribbon surrounding the aperture
    bg_mask = (dist >= W_bg1) & (dist <= W_bg2)
    
    aperture_vals = img[yy[aperture_mask], xx[aperture_mask]]
    bg_vals = img[yy[bg_mask], xx[bg_mask]]
    
    # Compute background metrics
    if len(bg_vals) > 0:
        bg_mean = np.mean(bg_vals)
        bg_std = np.std(bg_vals)
    else:
        # Fallback to local image percentile if background region has no pixels
        bg_mean = np.median(img[y_min:y_max+1, x_min:x_max+1])
        bg_std = np.std(img[y_min:y_max+1, x_min:x_max+1])
        
    n_aper = np.sum(aperture_mask)
    if n_aper > 0:
        raw_flux = np.sum(aperture_vals)
        net_flux = raw_flux - n_aper * bg_mean
        peak_value = np.max(aperture_vals)
    else:
        raw_flux = 0.0
        net_flux = 0.0
        peak_value = 0.0
        
    # Instrumental Magnitude calculation
    if net_flux > 0:
        magnitude = -2.5 * np.log10(net_flux) + zeropoint
    else:
        magnitude = float('nan') # Too faint/noisy to measure
        
    # SNR calculation (Standard CCD noise equation)
    # SNR = Signal / sqrt(Signal + Area * Background_Variance)
    bg_variance = bg_std**2
    noise = np.sqrt(max(0.0, net_flux) + n_aper * bg_variance)
    snr = net_flux / noise if noise > 0 else 0.0
    
    return {
        'raw_flux': float(raw_flux),
        'net_flux': float(net_flux),
        'bg_mean': float(bg_mean),
        'bg_std': float(bg_std),
        'peak_value': float(peak_value),
        'aperture_pixels': int(n_aper),
        'magnitude': float(magnitude),
        'snr': float(snr),
        'streak_length': float(L)
    }
