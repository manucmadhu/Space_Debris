import sys
import argparse
import os
import math
from pathlib import Path

# Add 'Code' directory to the path so we can import our modules
CODE_DIR = Path(__file__).parent / "Code"
sys.path.append(str(CODE_DIR))

try:
    from pre_process import run_preprocessing, get_sorted_fits
    from difference import run_difference
    from plate_solver import solve_and_apply_wcs
    from correlate import process_and_correlate
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Please ensure the 'Code' folder contains all necessary python files.")
    sys.exit(1)

def run_pipeline(fits_dir: Path, catalog_path: Path, output_dir: Path, 
                 start_frame: str, num_frames: int, run_preprocess_step: bool, 
                 run_solve_step: bool):
    """
    Orchestrates the entire space debris detection and correlation pipeline.
    """
    fits_dir = Path(fits_dir)
    catalog_path = Path(catalog_path)
    output_dir = Path(output_dir)
    
    print("=" * 70)
    print("          SPACE DEBRIS STREAK TRACKING & PHOTOMETRY PIPELINE")
    print("=" * 70)
    print(f" Raw FITS Folder : {fits_dir}")
    print(f" Orbital Catalog : {catalog_path}")
    print(f" Output Folder   : {output_dir}")
    print(f" Start Frame     : {start_frame}")
    print(f" Frame Count     : {num_frames}")
    print("=" * 70)

    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. OPTIONAL: Run Batch Preprocessing
    if run_preprocess_step:
        print("\n--- STAGE 1: Preprocessing raw files ---")
        try:
            run_preprocessing(fits_dir, output_dir, num_adj=2, batch_size=40)
        except Exception as e:
            print(f"[!] Preprocessing step failed: {e}. Moving on...")

    # 2. RUN DIFFERENCING
    print("\n--- STAGE 2: Computing Frame Differencing ---")
    try:
        diff, header, diff_fits, diff_png = run_difference(
            fits_dir, start_frame, num_frames, output_dir
        )
    except Exception as e:
        print(f"[ERROR] Error during differencing: {e}")
        return False

    # 3. RUN PLATE SOLVER (WCS CALIBRATION)
    has_wcs = False
    if run_solve_step:
        print("\n--- STAGE 3: Plate Solving with Astrometry.net ---")
        raw_solve_file = fits_dir / start_frame
        if not raw_solve_file.exists():
            # Find the actual path from sorted files
            files = get_sorted_fits(fits_dir)
            filenames = [f.name for f in files]
            if start_frame in filenames:
                raw_solve_file = files[filenames.index(start_frame)]
                
        if raw_solve_file.exists():
            try:
                wcs_info = solve_and_apply_wcs(str(raw_solve_file), str(diff_fits))
                if wcs_info:
                    has_wcs = True
                    print("[SUCCESS] WCS Coordinates successfully calculated and written to difference image.")
                else:
                    print("[WARNING] Astrometry.net was unable to solve the image. TLE correlation will run in Geocentric fallback.")
            except Exception as e:
                print(f"[WARNING] Plate solving failed: {e}. Proceeding without active WCS calibration.")
        else:
            print(f"[WARNING] Start frame {start_frame} not found at {raw_solve_file}. Solver skipped.")

    # 4. RUN DETECTION, PHOTOMETRY & ORBITAL CORRELATION
    print("\n--- STAGE 4: Debris Detection, Aperture Photometry & Correlation ---")
    start_stem = Path(start_frame).stem.replace(" ", "_")
    output_png = output_dir / f"diff_{start_stem}_{num_frames}f_correlated.png"
    
    try:
        results = process_and_correlate(str(diff_fits), str(catalog_path), str(output_png))
        
        if results:
            print(f"\nPipeline run completed successfully! Detected {len(results)} tracks.")
            print("-" * 70)
            print(f"{'ID':<4} | {'Label':<25} | {'Flux':<10} | {'Mag':<6} | {'SNR':<5} | {'RA (deg)':<10} | {'Dec (deg)':<10}")
            print("-" * 70)
            for res in results:
                phot = res['photometry']
                mag_str = f"{phot['magnitude']:.2f}" if not math.isnan(phot['magnitude']) else "N/A"
                print(f"T{res['track_id']:<3} | {res['label']:<25} | {phot['net_flux']:<10.1f} | {mag_str:<6} | {phot['snr']:<5.1f} | {res['centroid_ra']:<10.4f} | {res['centroid_dec']:<10.4f}")
            print("-" * 70)
            
            # Save a text summary report
            report_path = output_dir / "pipeline_report.txt"
            with open(report_path, "w") as f:
                f.write("SPACE DEBRIS ANALYSIS REPORT\n")
                f.write("============================\n")
                f.write(f"Source Folder: {fits_dir}\n")
                f.write(f"Start Frame:   {start_frame}\n")
                f.write(f"Frames Used:   {num_frames}\n\n")
                f.write(f"{'Track ID':<10} | {'Object Label':<25} | {'Net Flux':<12} | {'Magnitude':<10} | {'SNR':<8} | {'RA (deg)':<12} | {'Dec (deg)':<12}\n")
                f.write("-" * 105 + "\n")
                for res in results:
                    phot = res['photometry']
                    mag_str = f"{phot['magnitude']:.2f}" if not math.isnan(phot['magnitude']) else "N/A"
                    f.write(f"T{res['track_id']:<9} | {res['label']:<25} | {phot['net_flux']:<12.1f} | {mag_str:<10} | {phot['snr']:<8.1f} | {res['centroid_ra']:<12.4f} | {res['centroid_dec']:<12.4f}\n")
            print(f"Text report saved to: {report_path}")
            return True
        else:
            print("\n[WARNING] No debris tracks were detected or matched in this sequence.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error during detection/correlation stage: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Fetch default configurations
    default_fits_dir = "Data/raw"
    default_catalog = "Data/3le.txt"
    default_output = "Output"
    
    # Try to scan fits folder to extract the first fits filename automatically
    default_start_frame = "Capture_00001 02_55_24Z.fits"
    if os.path.exists(default_fits_dir):
        files = sorted([f for f in os.listdir(default_fits_dir) if f.endswith('.fits') or f.endswith('.fit')])
        if files:
            default_start_frame = files[0]

    parser = argparse.ArgumentParser(description="Modular Space Debris Streak Detection & Photometry Pipeline")
    parser.add_argument("--fits-dir", type=str, default=default_fits_dir, help="Directory containing raw FITS captures")
    parser.add_argument("--catalog", type=str, default=default_catalog, help="Path to the TLE/3LE space tracking catalog txt file")
    parser.add_argument("--output-dir", type=str, default=default_output, help="Directory to save output FITS, PNGs, and logs")
    parser.add_argument("--start-frame", type=str, default=default_start_frame, help="The starting FITS file name in the sequence")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to combine for the difference image")
    parser.add_argument("--preprocess", action="store_true", help="Run batch conversion of FITS to preprocessed PNG files first")
    parser.add_argument("--no-solve", action="store_true", help="Skip running the plate WCS solver (uses cached coordinates or geocentric fallback)")

    args = parser.parse_args()

    # Orchestrate
    success = run_pipeline(
        fits_dir=Path(args.fits_dir),
        catalog_path=Path(args.catalog),
        output_dir=Path(args.output_dir),
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        run_preprocess_step=args.preprocess,
        run_solve_step=not args.no_solve
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()