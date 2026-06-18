import subprocess
import os
from astropy.io import fits
from astropy.wcs import WCS

def solve_and_apply_wcs(raw_image_path, difference_image_path=None):
    """
    Solves a raw image using offline Astrometry.net and directly copies 
    the resulting WCS header keys to a difference image.
    """
    import cv2
    import numpy as np

    # 1. Ensure the raw image exists
    if not os.path.exists(raw_image_path):
        raise FileNotFoundError(f"Cannot find raw image at {raw_image_path}")

    print(f"Starting plate solve on raw image: {raw_image_path}...")

    # Check for Bayer pattern in FITS header
    is_bayer = False
    bayer_pat = None
    try:
        with fits.open(raw_image_path, memmap=False) as hdul:
            hdr = hdul[0].header
            if 'BAYERPAT' in hdr:
                is_bayer = True
                bayer_pat = hdr['BAYERPAT'].strip()
    except Exception as e:
        print(f"[!] Warning: failed to check FITS header for Bayer pattern: {e}")

    # If it is a Bayer image, debayer it to a temp grayscale FITS file
    solve_image_path = raw_image_path
    base_raw, ext_raw = os.path.splitext(raw_image_path)
    temp_raw_path = f"{base_raw}_debayered_temp{ext_raw}"

    if is_bayer:
        print(f"  -> Bayer pattern detected: {bayer_pat}. Debayering to grayscale for solver...")
        try:
            with fits.open(raw_image_path, memmap=False) as hdul:
                data = hdul[0].data
                header = hdul[0].header.copy()
            
            # Map Bayer code
            pat_upper = bayer_pat.upper() if bayer_pat else 'BGGR'
            if pat_upper == 'BGGR':
                code = cv2.COLOR_BayerBG2GRAY
            elif pat_upper == 'GBRG':
                code = cv2.COLOR_BayerGB2GRAY
            elif pat_upper == 'RGGB':
                code = cv2.COLOR_BayerRG2GRAY
            elif pat_upper == 'GRBG':
                code = cv2.COLOR_BayerGR2GRAY
            else:
                code = cv2.COLOR_BayerBG2GRAY
                
            data_u16 = data.astype(np.uint16)
            gray_data = cv2.cvtColor(data_u16, code)
            
            if 'BAYERPAT' in header:
                del header['BAYERPAT']
                
            fits.writeto(temp_raw_path, gray_data, header, overwrite=True)
            solve_image_path = temp_raw_path
            print(f"  -> Saved clean grayscale temp file at: {temp_raw_path}")
        except Exception as e:
            print(f"[!] Error during debayering: {e}. Solving original raw image as fallback.")

    # 2. Dynamically determine the pixel scale from FITS header or guess based on instrument/resolution
    scale_low = "0.5"
    scale_high = "60.0"
    downsample_val = "2"
    
    try:
        with fits.open(raw_image_path, memmap=False) as hdul:
            hdr = hdul[0].header
            
            # Adaptive downsampling based on dimension
            naxis1 = hdr.get('NAXIS1', 0)
            if naxis1 > 5000:
                downsample_val = "4"
            
            # Try to get focal length and pixel size
            pixsz = None
            focal = None
            
            for key in ['PIXSZ', 'XPIXSZ', 'YPIXSZ']:
                if key in hdr:
                    pixsz = float(hdr[key])
                    break
            
            for key in ['FOCAL', 'FOCALLEN']:
                if key in hdr:
                    val = hdr[key]
                    if isinstance(val, str):
                        # Extract float from string like "28.0"
                        import re
                        m = re.search(r'[\d\.]+', val)
                        if m:
                            focal = float(m.group(0))
                    else:
                        focal = float(val)
                    break
                    
            if pixsz is not None and focal is not None and focal > 0:
                scale_val = (pixsz / 1000.0) / focal * 206265.0
                scale_low = f"{scale_val * 0.95:.3f}"
                scale_high = f"{scale_val * 1.05:.3f}"
                print(f"  -> Calculated pixel scale dynamically: {scale_val:.3f} arcsec/pixel (range: {scale_low} - {scale_high})")
            else:
                # Fallback based on image dimensions
                naxis2 = hdr.get('NAXIS2', 0)
                instrume = str(hdr.get('INSTRUME', '')).lower()
                
                if 'stellina' in instrume or (naxis1 == 3072 and naxis2 == 2080):
                    scale_low = "1.10"
                    scale_high = "1.40"
                    print("  -> Stellina detected by dimensions/instrument. Using scale range 1.10 - 1.40 arcsec/pixel")
                elif 'qhy600' in instrume or (naxis1 == 9600 and naxis2 == 6422):
                    scale_low = "26"
                    scale_high = "29"
                    print("  -> QHY600 detected by dimensions/instrument. Using scale range 26 - 29 arcsec/pixel")
                else:
                    scale_low = "0.5"
                    scale_high = "60.0"
                    print("  -> Unknown camera. Using broad search scale range 0.5 - 60.0 arcsec/pixel")
    except Exception as e:
        print(f"[!] Warning: failed to parse header for pixel scale: {e}. Falling back to broad search.")

    # 3. Define the solve-field command with speed optimizations
    base_args = [
        "--overwrite",
        "--no-plots",
        "--downsample", downsample_val,          
        "--depth", "50",              
        "--tweak-order", "2",         
        "--scale-units", "arcsecperpix",
        "--scale-low", scale_low,          
        "--scale-high", scale_high          
    ]

    # Convert solve_image_path to a WSL-friendly absolute path
    abs_path = os.path.abspath(solve_image_path).replace('\\', '/')
    if len(abs_path) > 1 and abs_path[1] == ':':
        drive = abs_path[0].lower()
        wsl_path = f"/mnt/{drive}{abs_path[2:]}"
    else:
        wsl_path = abs_path

    # Run the command, falling back to WSL if native command is not in Windows PATH
    command = ["solve-field", solve_image_path] + base_args
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="ignore")
    except FileNotFoundError:
        print("[WARNING] 'solve-field' not found in Windows PATH. Trying via WSL...")
        wsl_command = [
            "wsl",
            "solve-field",
            wsl_path,
        ] + base_args + ["--config", "/mnt/d/Space Debris/Project/Code/local_astrometry.cfg"]
        try:
            result = subprocess.run(wsl_command, capture_output=True, text=True, errors="ignore")
        except FileNotFoundError:
            print("[ERROR] 'solve-field' command not found natively or via WSL. Plate solving failed.")
            if is_bayer and os.path.exists(temp_raw_path):
                try: os.remove(temp_raw_path)
                except: pass
            return None

    if result.returncode != 0:
        print("[ERROR] Astrometry.net crashed.")
        print("Error log:\n", result.stderr)
        if is_bayer and os.path.exists(temp_raw_path):
            try: os.remove(temp_raw_path)
            except: pass
        return None

    base_name = os.path.splitext(solve_image_path)[0]
    wcs_file = f"{base_name}.new"

    wcs_info = None

    # 4. Check if the solve was actually successful
    if os.path.exists(wcs_file):
        print(f"[SUCCESS] Successfully solved! True WCS generated at {wcs_file}")
        
        # --- THE FIX: Extract the RAW header directly, bypassing the WCS abstraction ---
        with fits.open(wcs_file, memmap=False) as hdul:
            solved_header = hdul[0].header.copy()
            wcs_info = WCS(solved_header) # Keep this just for the print statement at the end
        
        # 5. Apply the RAW WCS to the difference image
        if difference_image_path:
            if os.path.exists(difference_image_path):
                print(f"Applying true coordinates to difference image: {difference_image_path}...")
                
                with fits.open(difference_image_path, memmap=False) as diff_hdul:
                    diff_data = diff_hdul[0].data.copy()
                    diff_header = diff_hdul[0].header.copy()
                    
                # 'extend' with 'update=True' forces all new WCS and SIP distortion keys 
                # into the target header, replacing existing ones without stripping data.
                diff_header.extend(solved_header, update=True)
                
                fits.writeto(difference_image_path, diff_data, diff_header, overwrite=True)
                print("[SUCCESS] Difference image updated successfully! It is now ready for orbital analysis.")
            else:
                print(f"[WARNING] Warning: WCS solved, but difference image not found at {difference_image_path}")
    else:
        print("\n[ERROR] Astrometry.net finished searching, but COULD NOT SOLVE the raw image.")
        print("Usually this means the image lacks clear stars, or you are missing the correct index files.")
        print("\n--- LAST 15 LINES OF ASTROMETRY LOG ---")
        lines = result.stdout.strip().split('\n')
        print('\n'.join(lines[-15:]))
        print("---------------------------------------\n")

    # Clean up temporary files here
    if is_bayer and os.path.exists(temp_raw_path):
        temp_basename = os.path.splitext(temp_raw_path)[0]
        # Clean up files created by solve-field
        for ext in ['.fits', '.fit', '.axy', '.solved', '.new', '-indx.xyls', '.match', '.rdls', '.wcs', '.corr']:
            path_to_del = temp_basename + ext
            if os.path.exists(path_to_del):
                try:
                    os.remove(path_to_del)
                except Exception as e:
                    pass

    return wcs_info


if __name__ == "__main__":
    RAW_CAPTURE = "frigate/frigate/raw/Capture_01419 03_09_43Z.fits"
    DIFFERENCE_IMAGE = "frigate-main/frigate-main/output/difference_image.fits" 
    
    wcs_data = solve_and_apply_wcs(RAW_CAPTURE, DIFFERENCE_IMAGE)

    if wcs_data:
        print("\n--- Final Output ---")
        print(f"Center Right Ascension (RA):  {wcs_data.wcs.crval[0]:.4f} degrees")
        print(f"Center Declination (Dec):     {wcs_data.wcs.crval[1]:.4f} degrees")