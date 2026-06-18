import os
import sys
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

# Add directories to system path
ROOT_DIR = Path(__file__).parent.parent
CODE_DIR = ROOT_DIR / "Code"
sys.path.append(str(ROOT_DIR))
sys.path.append(str(CODE_DIR))

# Matplotlib integration
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

# Pipeline imports
try:
    from astropy.io import fits
    from astropy.wcs import WCS
    # pyrefly: ignore [missing-import]
    from pre_process import run_preprocessing, get_sorted_fits
    from difference import run_difference
    from plate_solver import solve_and_apply_wcs
    from correlate import process_and_correlate
    from display import apply_stretch
    from flux_extract import extract_streak_flux
except ImportError as e:
    print(f"Error importing dependencies in GUI: {e}")
    messagebox.showerror("Import Error", f"Missing required dependency: {e}\nPlease check requirements.txt")

# Thread-safe log redirector
class QueueLogger:
    def __init__(self, text_widget, log_queue, original_stream=None):
        self.text_widget = text_widget
        self.log_queue = log_queue
        self.original_stream = original_stream
        self.text_widget.config(state=tk.DISABLED)
        
    def write(self, string):
        self.log_queue.put(string)
        if self.original_stream:
            self.original_stream.write(string)
            self.original_stream.flush()
            
    def flush(self):
        if self.original_stream:
            self.original_stream.flush()

class DebrisTrackerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Space Debris Track & Photometry Dashboard")
        self.geometry("1280x850")
        self.minsize(1024, 768)
        
        # Color Palette - Modern Sleek Dark Theme
        self.bg_dark = "#121216"
        self.bg_card = "#1e1e24"
        self.accent_blue = "#00adb5"
        self.accent_orange = "#ff9f1c"
        self.text_light = "#eeeeee"
        self.text_dim = "#9a9aa2"
        self.border_color = "#33333d"
        
        self.configure(bg=self.bg_dark)
        
        # Styling Setup
        self.setup_styles()
        
        # State Variables
        self.raw_dir_var = tk.StringVar(value=str(ROOT_DIR / "Data" / "raw"))
        self.catalog_var = tk.StringVar(value=str(ROOT_DIR / "Data" / "3le.txt"))
        self.output_dir_var = tk.StringVar(value=str(ROOT_DIR / "Output"))
        self.start_frame_var = tk.StringVar()
        self.num_frames_var = tk.IntVar(value=10)
        self.preprocess_var = tk.BooleanVar(value=False)
        self.solve_var = tk.BooleanVar(value=True)
        self.show_annotated_var = tk.BooleanVar(value=True)
        
        self.current_fits_data = None
        self.current_fits_header = None
        self.current_wcs = None
        self.current_png_path = None
        self.detected_tracks = []
        self.selected_fits_path = None
        
        # Logging Queue
        self.log_queue = queue.Queue()
        
        # GUI Task Queue
        self.gui_queue = queue.Queue()
        
        # Build UI Elements
        self.build_ui()
        
        # Populate FITS dropdown list
        self.scan_raw_directory()
        
        # Start logging queue check loop
        self.after(100, self.process_log_queue)
        
        # Start GUI task queue check loop
        self.after(50, self.process_gui_queue)
        
        # Redirect stdout and stderr
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = QueueLogger(self.log_console, self.log_queue, self.old_stdout)
        sys.stderr = QueueLogger(self.log_console, self.log_queue, self.old_stderr)
        
        # Print welcome banner
        print("=" * 60)
        print("  Space Debris Detection & Photometry Pipeline GUI Active")
        print("=" * 60)
        print("Use the left panel to configure paths and trigger pipeline steps.")
        print("Double-click rows in the results list to view target details.")
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Custom styles for dark theme
        style.configure("TLabel", background=self.bg_card, foreground=self.text_light, font=("Segoe UI", 10))
        style.configure("TButton", background=self.accent_blue, foreground="#ffffff", font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TButton", background=[("active", "#00dcd4"), ("pressed", "#008a90")])
        
        style.configure("Card.TFrame", background=self.bg_card, borderwidth=1, relief="flat")
        style.configure("Title.TLabel", background=self.bg_card, foreground=self.accent_blue, font=("Segoe UI", 12, "bold"))
        
        style.configure("TCombobox", fieldbackground=self.bg_dark, background=self.bg_dark, foreground=self.text_light, bordercolor=self.border_color)
        style.configure("TSpinbox", fieldbackground=self.bg_dark, background=self.bg_dark, foreground=self.text_light)
        
        # Notebook (Tabs) Styling
        style.configure("TNotebook", background=self.bg_dark, borderwidth=0)
        style.configure("TNotebook.Tab", background=self.bg_card, foreground=self.text_dim, padding=[12, 4], font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", self.bg_dark)], foreground=[("selected", self.accent_blue)])
        
        # Treeview (Table) Styling
        style.configure("Treeview", background=self.bg_card, fieldbackground=self.bg_card, foreground=self.text_light, 
                        rowheight=25, borderwidth=0, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=self.bg_dark, foreground=self.text_light, font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", self.accent_blue)], foreground=[("selected", "#ffffff")])
        
    def build_ui(self):
        # Outer container with padding
        main_container = tk.Frame(self, bg=self.bg_dark)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- LEFT PANEL: Settings & Pipeline (Width: 340) ---
        left_panel = ttk.Frame(main_container, style="Card.TFrame", padding=4)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        left_panel.config(width=340)

        # Scrollable Canvas setup for the left control panel
        canvas = tk.Canvas(left_panel, bg=self.bg_card, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="Card.TFrame")

        # Configure scrollregion on resize
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=310)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind mouse wheel to scroll canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Bind mousewheel when cursor enters the left panel, and unbind when it leaves
        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            
        left_panel.bind("<Enter>", _bind_mousewheel)
        left_panel.bind("<Leave>", _unbind_mousewheel)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Title (child of scrollable_frame)
        title_lbl = ttk.Label(scrollable_frame, text="PIPELINE CONTROL PANEL", style="Title.TLabel")
        title_lbl.pack(anchor=tk.W, pady=(10, 15))
        
        # Section 1: Paths configuration (child of scrollable_frame)
        path_group = ttk.LabelFrame(scrollable_frame, text="Directories", padding=8, style="Card.TFrame")
        path_group.pack(fill=tk.X, pady=(0, 12))
        
        # Raw FITS folder
        ttk.Label(path_group, text="Raw FITS Folder:").pack(anchor=tk.W)
        raw_f = tk.Frame(path_group, bg=self.bg_card)
        raw_f.pack(fill=tk.X, pady=(2, 6))
        ttk.Entry(raw_f, textvariable=self.raw_dir_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(raw_f, text="...", width=3, command=self.browse_raw_dir).pack(side=tk.RIGHT, padx=(4, 0))
        
        # TLE/3LE File
        ttk.Label(path_group, text="TLE Catalog File:").pack(anchor=tk.W)
        tle_f = tk.Frame(path_group, bg=self.bg_card)
        tle_f.pack(fill=tk.X, pady=(2, 6))
        ttk.Entry(tle_f, textvariable=self.catalog_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(tle_f, text="...", width=3, command=self.browse_catalog).pack(side=tk.RIGHT, padx=(4, 0))
        
        # Output folder
        ttk.Label(path_group, text="Output Directory:").pack(anchor=tk.W)
        out_f = tk.Frame(path_group, bg=self.bg_card)
        out_f.pack(fill=tk.X, pady=(2, 2))
        ttk.Entry(out_f, textvariable=self.output_dir_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_f, text="...", width=3, command=self.browse_output_dir).pack(side=tk.RIGHT, padx=(4, 0))
        
        # Section 2: Pipeline Parameters (child of scrollable_frame)
        param_group = ttk.LabelFrame(scrollable_frame, text="Parameters", padding=8, style="Card.TFrame")
        param_group.pack(fill=tk.X, pady=(0, 12))
        
        # Start Frame Dropdown
        ttk.Label(param_group, text="Start Frame (Differencing):").pack(anchor=tk.W)
        self.frame_combo = ttk.Combobox(param_group, textvariable=self.start_frame_var, state="readonly", style="TCombobox")
        self.frame_combo.pack(fill=tk.X, pady=(2, 6))
        self.frame_combo.bind("<<ComboboxSelected>>", self.on_start_frame_selected)
        
        # Num Frames Spinbox
        ttk.Label(param_group, text="Number of Frames:").pack(anchor=tk.W)
        self.num_spin = ttk.Spinbox(param_group, from_=3, to=200, textvariable=self.num_frames_var, width=10, style="TSpinbox")
        self.num_spin.pack(anchor=tk.W, pady=(2, 6))
        
        # Switches (Checkbuttons)
        pre_chk = tk.Checkbutton(param_group, text="Run pre-processing (FITS to PNG)", variable=self.preprocess_var, 
                                 bg=self.bg_card, fg=self.text_light, activebackground=self.bg_card, activeforeground=self.text_light, selectcolor=self.bg_dark)
        pre_chk.pack(anchor=tk.W, pady=2)
        
        solve_chk = tk.Checkbutton(param_group, text="Run Plate Solver (WCS Calibration)", variable=self.solve_var,
                                   bg=self.bg_card, fg=self.text_light, activebackground=self.bg_card, activeforeground=self.text_light, selectcolor=self.bg_dark)
        solve_chk.pack(anchor=tk.W, pady=2)
        
        # Section 3: Run pipeline buttons (child of scrollable_frame)
        btn_group = ttk.LabelFrame(scrollable_frame, text="Actions", padding=8, style="Card.TFrame")
        btn_group.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(btn_group, text="Run Pre-Process Only", command=self.trigger_preprocess, width=22).pack(fill=tk.X, pady=4)
        ttk.Button(btn_group, text="Run Differencing Only", command=self.trigger_difference, width=22).pack(fill=tk.X, pady=4)
        ttk.Button(btn_group, text="Run Plate Solver Only", command=self.trigger_solve, width=22).pack(fill=tk.X, pady=4)
        ttk.Button(btn_group, text="Detect & Correlate", command=self.trigger_correlation, width=22).pack(fill=tk.X, pady=4)
        
        # Separator
        sep = tk.Frame(btn_group, height=2, bg=self.border_color)
        sep.pack(fill=tk.X, pady=6)
        
        # Run Full Pipeline Button
        self.run_all_btn = ttk.Button(btn_group, text="RUN FULL PIPELINE", command=self.trigger_full_pipeline, width=22)
        self.run_all_btn.pack(fill=tk.X, pady=(2, 4))
        
        # Clear Console button
        ttk.Button(btn_group, text="Clear Console Logs", command=self.clear_logs, width=22).pack(fill=tk.X, pady=4)
        
        # --- RIGHT PANEL: Visualizer & Results ---
        right_panel = tk.Frame(main_container, bg=self.bg_dark)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Upper visual notebook
        self.notebook = ttk.Notebook(right_panel, style="TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: FITS Viewer
        self.fits_tab = tk.Frame(self.notebook, bg=self.bg_dark)
        self.notebook.add(self.fits_tab, text=" 🌌 FITS IMAGE VIEWER ")
        self.build_fits_viewer_tab()
        
        # Tab 2: Detected Streaks & Table
        self.results_tab = tk.Frame(self.notebook, bg=self.bg_dark)
        self.notebook.add(self.results_tab, text=" 📊 DETECTED STREAKS & PHOTOMETRY ")
        self.build_results_tab()
        
        # Bottom Console window
        console_frame = tk.Frame(right_panel, bg=self.bg_card, height=180, highlightthickness=1, highlightbackground=self.border_color)
        console_frame.pack(fill=tk.X, pady=(10, 0))
        console_frame.pack_propagate(False)
        
        console_lbl = tk.Label(console_frame, text=" SYSTEM LOGGER CONSOLE OUTPUT ", bg=self.bg_dark, fg=self.accent_blue, font=("Segoe UI", 9, "bold"), anchor="w", padx=10)
        console_lbl.pack(fill=tk.X)
        
        log_scroll = ttk.Scrollbar(console_frame, orient="vertical")
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_console = tk.Text(console_frame, bg=self.bg_dark, fg="#00ff66", font=("Consolas", 9), wrap=tk.WORD, 
                                   yscrollcommand=log_scroll.set, padx=8, pady=4)
        self.log_console.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_console.yview)
        
    def build_fits_viewer_tab(self):
        # Toolbar and Settings area
        viewer_tools = tk.Frame(self.fits_tab, bg=self.bg_card, height=40)
        viewer_tools.pack(fill=tk.X)
        viewer_tools.pack_propagate(False)
        
        # Stretching selector
        tk.Label(viewer_tools, text="Stretch:", bg=self.bg_card, fg=self.text_light, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(10, 4))
        self.stretch_combo = ttk.Combobox(viewer_tools, values=["percentile", "linear", "log"], width=12, state="readonly", style="TCombobox")
        self.stretch_combo.pack(side=tk.LEFT, padx=4)
        self.stretch_combo.set("percentile")
        self.stretch_combo.bind("<<ComboboxSelected>>", lambda e: self.update_fits_display())
        
        # Colormap selector
        tk.Label(viewer_tools, text="Colormap:", bg=self.bg_card, fg=self.text_light, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(15, 4))
        self.cmap_combo = ttk.Combobox(viewer_tools, values=["gray", "viridis", "inferno", "plasma", "bone", "hot"], width=12, state="readonly", style="TCombobox")
        self.cmap_combo.pack(side=tk.LEFT, padx=4)
        self.cmap_combo.set("gray")
        self.cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.update_fits_display())
        
        # Show Annotated checkbox
        self.show_annotated_chk = tk.Checkbutton(
            viewer_tools, text="Show Annotated", variable=self.show_annotated_var,
            command=self.update_fits_display, bg=self.bg_card, fg=self.text_light,
            selectcolor=self.bg_dark, activebackground=self.bg_card, activeforeground=self.text_light
        )
        self.show_annotated_chk.pack(side=tk.LEFT, padx=(20, 4))
        
        # Hover coordinate label
        self.coords_lbl = tk.Label(viewer_tools, text="Coordinate: X: --, Y: -- | RA: --, Dec: --", bg=self.bg_card, fg=self.accent_orange, font=("Segoe UI", 9, "bold"))
        self.coords_lbl.pack(side=tk.RIGHT, padx=15)
        
        # Canvas space
        self.canvas_frame = tk.Frame(self.fits_tab, bg=self.bg_dark)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize Matplotlib Figure
        self.fits_fig, self.fits_ax = plt.subplots(figsize=(6, 5), facecolor=self.bg_dark)
        self.fits_ax.set_facecolor(self.bg_dark)
        self.fits_ax.tick_params(colors=self.text_dim)
        for spine in self.fits_ax.spines.values():
            spine.set_color(self.border_color)
            
        self.fits_canvas = FigureCanvasTkAgg(self.fits_fig, master=self.canvas_frame)
        self.fits_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Embed Matplotlib Navigation toolbar (hidden standard, customized or native layout)
        self.toolbar_frame = tk.Frame(self.fits_tab, bg="#e0e0e0")
        self.toolbar_frame.pack(fill=tk.X)
        self.fits_toolbar = NavigationToolbar2Tk(self.fits_canvas, self.toolbar_frame)
        self.fits_toolbar.config(background="#e0e0e0")
        for child in self.fits_toolbar.winfo_children():
            try:
                child.configure(background="#e0e0e0")
            except Exception:
                pass
            try:
                child.configure(foreground="#121216")
            except Exception:
                pass
        self.fits_toolbar.update()
        
        # Bind Mouse Motion Event for coordinates tracking
        self.fits_fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        
    def build_results_tab(self):
        # Treeview (table) container
        tree_frame = tk.Frame(self.results_tab, bg=self.bg_card)
        tree_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        lbl = tk.Label(tree_frame, text=" DETECTED ORBITAL STREAKS AND PHOTOMETRY TABLE (Double click row to find streak in viewer) ", 
                       bg=self.bg_dark, fg=self.accent_blue, font=("Segoe UI", 9, "bold"), anchor="w", padx=10, pady=4)
        lbl.pack(fill=tk.X)
        
        scroll_y = ttk.Scrollbar(tree_frame, orient="vertical")
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        cols = ("id", "label", "mag", "snr", "net_flux", "peak", "length", "ra", "dec", "start_px", "end_px", "sep")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)
        
        # Headings and widths
        headings = {
            "id": ("ID", 40),
            "label": ("Debris / Satellite ID", 180),
            "mag": ("Mag (Inst)", 80),
            "snr": ("SNR", 60),
            "net_flux": ("Net Flux", 90),
            "peak": ("Peak Pix", 80),
            "length": ("Len (px)", 70),
            "ra": ("RA (deg)", 90),
            "dec": ("Dec (deg)", 90),
            "start_px": ("Start (X,Y)", 100),
            "end_px": ("End (X,Y)", 100),
            "sep": ("Catalog Sep (°)", 110)
        }
        
        for col, (title, width) in headings.items():
            self.tree.heading(col, text=title, anchor=tk.CENTER)
            self.tree.column(col, width=width, anchor=tk.CENTER)
            
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_tree_row_double_click)
        
    def scan_raw_directory(self):
        raw_path = Path(self.raw_dir_var.get())
        if not raw_path.exists():
            return
            
        try:
            files = get_sorted_fits(raw_path)
            filenames = [f.name for f in files]
            
            if filenames:
                self.frame_combo.config(values=filenames)
                if self.start_frame_var.get() not in filenames:
                    self.start_frame_var.set(filenames[0])
            else:
                self.frame_combo.config(values=[])
                self.start_frame_var.set("")
        except Exception as e:
            print(f"Error scanning directory: {e}")
            
    def browse_raw_dir(self):
        path = filedialog.askdirectory(title="Select Raw FITS Folder", initialdir=self.raw_dir_var.get())
        if path:
            self.raw_dir_var.set(path)
            self.scan_raw_directory()
            
    def browse_catalog(self):
        path = filedialog.askopenfilename(title="Select TLE/3LE File", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")], 
                                          initialdir=os.path.dirname(self.catalog_var.get()))
        if path:
            self.catalog_var.set(path)
            
    def browse_output_dir(self):
        path = filedialog.askdirectory(title="Select Output Directory", initialdir=self.output_dir_var.get())
        if path:
            self.output_dir_var.set(path)
            
    def on_start_frame_selected(self, event):
        # Load the selected raw FITS file automatically when selected
        start_f = self.start_frame_var.get()
        raw_path = Path(self.raw_dir_var.get()) / start_f
        if raw_path.exists():
            self.load_fits_image(raw_path)
            
    def load_fits_image(self, fits_path):
        try:
            fits_path = Path(fits_path)
            print(f"\nLoading FITS image in viewer: {fits_path.name}")
            
            # Read file bytes to avoid file locking on Windows
            import io
            with open(str(fits_path), 'rb') as f:
                fits_bytes = f.read()
                
            # Open FITS data from memory buffer
            with fits.open(io.BytesIO(fits_bytes), memmap=False) as hdul:
                self.current_fits_data = np.array(hdul[0].data, dtype=np.float32)
                self.current_fits_header = hdul[0].header.copy()
                self.current_wcs = WCS(self.current_fits_header, relax=True)
                
            self.selected_fits_path = fits_path
            
            # Check for correlated PNG matching this FITS image
            png_candidate = fits_path.parent / f"{fits_path.stem}_correlated.png"
            if not png_candidate.exists():
                output_dir = Path(self.output_dir_var.get())
                clean_stem = fits_path.stem.replace(" ", "_")
                num_f = self.num_frames_var.get()
                png_candidate = output_dir / f"diff_{clean_stem}_{num_f}f_correlated.png"
                
            if png_candidate.exists():
                self.current_png_path = png_candidate
                print(f"[DEBUG] Found corresponding annotated PNG: {self.current_png_path.name}")
            else:
                self.current_png_path = None
                
            self.update_fits_display()
            
            # Switch view to FITS viewer
            self.notebook.select(self.fits_tab)
        except Exception as e:
            print(f"Error loading FITS file: {e}")
            messagebox.showerror("FITS Error", f"Failed to load FITS file: {e}")
            
    def update_fits_display(self, overlay_tracks=True):
        print(f"[DEBUG] update_fits_display: overlay_tracks={overlay_tracks}, detected_tracks count={len(self.detected_tracks) if self.detected_tracks else 0}")
        if self.current_fits_data is None:
            return
            
        stretch = self.stretch_combo.get()
        cmap = self.cmap_combo.get()
        
        # Preserve existing zoom/pan limits if available to prevent snapping back on settings change
        old_xlim = None
        old_ylim = None
        if hasattr(self, 'fits_ax') and self.fits_ax is not None:
            try:
                old_xlim = self.fits_ax.get_xlim()
                old_ylim = self.fits_ax.get_ylim()
            except Exception:
                pass
                
        self.fits_fig.clf()
        
        # Check if we should render the beautifully annotated PNG instead of the raw difference FITS file
        show_annotated = self.show_annotated_var.get() and hasattr(self, 'current_png_path') and self.current_png_path is not None and self.current_png_path.exists()
        
        if show_annotated:
            import matplotlib.image as mpimg
            try:
                img = mpimg.imread(str(self.current_png_path))
                H, W = self.current_fits_data.shape
                
                self.fits_ax = self.fits_fig.add_subplot(111)
                self.fits_ax.set_xlabel('X (pixels)', color=self.text_dim)
                self.fits_ax.set_ylabel('Y (pixels)', color=self.text_dim)
                self.fits_ax.tick_params(colors=self.text_dim)
                for spine in self.fits_ax.spines.values():
                    spine.set_color(self.border_color)
                    
                self.fits_ax.set_facecolor(self.bg_dark)
                self.fits_fig.patch.set_facecolor(self.bg_dark)
                
                # Draw the PNG image with origin='upper' matching standard image coordinate system,
                # and map to the exact same extent [0, W, 0, H] so WCS coordinates remain perfectly aligned!
                self.fits_ax.imshow(img, origin='upper', extent=[0, W, 0, H])
                
                # Re-apply preserved zoom limits
                if old_xlim is not None and old_ylim is not None:
                    try:
                        if old_xlim[0] >= -100 and old_xlim[1] <= W + 100:
                            self.fits_ax.set_xlim(old_xlim)
                        if old_ylim[0] >= -100 and old_ylim[1] <= H + 100:
                            self.fits_ax.set_ylim(old_ylim)
                    except Exception:
                        pass
                
                title_str = f"{self.selected_fits_path.name} (Annotated)"
                self.fits_ax.set_title(title_str, color=self.text_light, fontsize=11, fontweight="bold")
                
                self.fits_canvas.draw()
                return
            except Exception as e:
                print(f"[ERROR] Failed to load annotated PNG in viewer: {e}. Falling back to FITS view.")
        
        has_wcs = self.current_wcs is not None and self.current_wcs.has_celestial
        
        # Always utilize a standard pixel-space subplot to display overlays correctly
        self.fits_ax = self.fits_fig.add_subplot(111)
        self.fits_ax.set_xlabel('X (pixels)', color=self.text_dim)
        self.fits_ax.set_ylabel('Y (pixels)', color=self.text_dim)
        self.fits_ax.tick_params(colors=self.text_dim)
        for spine in self.fits_ax.spines.values():
            spine.set_color(self.border_color)
            
        self.fits_ax.set_facecolor(self.bg_dark)
        self.fits_fig.patch.set_facecolor(self.bg_dark)
        
        # Get dimensions and determine downsampling factor for smooth interaction
        H, W = self.current_fits_data.shape
        factor = max(1, min(W, H) // 1200)
        
        # Downsample data for rendering (speeds up Matplotlib rendering by 16x+)
        downsampled = self.current_fits_data[::factor, ::factor]
        
        # Apply stretch to downsampled data
        scaled = apply_stretch(downsampled, stretch_type=stretch)
        
        # Draw image using 'extent' to map downsampled pixels back to original coordinate system
        im = self.fits_ax.imshow(scaled, cmap=cmap, origin='lower', extent=[0, W, 0, H])
        
        # Re-apply preserved zoom limits
        if old_xlim is not None and old_ylim is not None:
            try:
                # Sanity check to ensure limits are within image bounds
                if old_xlim[0] >= -100 and old_xlim[1] <= W + 100:
                    self.fits_ax.set_xlim(old_xlim)
                if old_ylim[0] >= -100 and old_ylim[1] <= H + 100:
                    self.fits_ax.set_ylim(old_ylim)
            except Exception:
                pass
        
        # Draw annotations if available and requested
        if overlay_tracks and self.detected_tracks:
            # Set appropriate pixel space coordinate transform
            pixel_transform = self.fits_ax.transData
            
            for idx, track in enumerate(self.detected_tracks):
                p1 = track['start_pixel']
                p2 = track['end_pixel']
                label = track['label']
                color = "green" if track['is_match'] else "red"
                mag_val = track['photometry']['magnitude']
                mag_str = f"M:{mag_val:.2f}" if not math.isnan(mag_val) else "M:N/A"
                
                # Plot segment line
                self.fits_ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='orange', linestyle='--', linewidth=1.5, transform=pixel_transform)
                # Scatter endpoints
                self.fits_ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], color=color, s=25, transform=pixel_transform)
                
                # Bounding box
                x_min, x_max = min(p1[0], p2[0]) - 15, max(p1[0], p2[0]) + 15
                y_min, y_max = min(p1[1], p2[1]) - 15, max(p1[1], p2[1]) + 15
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     fill=False, edgecolor=color, linewidth=1.5, transform=pixel_transform)
                self.fits_ax.add_patch(rect)
                
                lbl_text = f"T{track['track_id']}: {label} ({mag_str})"
                self.fits_ax.text(x_min, y_max + 4, lbl_text, color=color, fontsize=9, fontweight='bold', transform=pixel_transform)
                
        title_str = self.selected_fits_path.name if self.selected_fits_path else "FITS Image"
        self.fits_ax.set_title(title_str, color=self.text_light, fontsize=11, fontweight="bold")
        
        self.fits_canvas.draw()

    def show_annotated_png(self, png_path):
        """Display the annotated PNG (with track overlays drawn by OpenCV) directly in the viewer."""
        png_path = Path(png_path)
        if not png_path.exists():
            print(f"[ERROR] Annotated PNG not found: {png_path}")
            return
        
        import matplotlib.image as mpimg
        try:
            img = mpimg.imread(str(png_path))
            # img shape: (H, W, 3) for RGB PNG
            img_h, img_w = img.shape[0], img.shape[1]
            
            self.fits_fig.clf()
            self.fits_ax = self.fits_fig.add_subplot(111)
            self.fits_ax.set_facecolor(self.bg_dark)
            self.fits_fig.patch.set_facecolor(self.bg_dark)
            for spine in self.fits_ax.spines.values():
                spine.set_color(self.border_color)
            self.fits_ax.tick_params(colors=self.text_dim)
            self.fits_ax.set_xlabel('X (pixels)', color=self.text_dim)
            self.fits_ax.set_ylabel('Y (pixels)', color=self.text_dim)
            
            # Draw the annotated PNG in pixel coordinates
            # origin='upper' matches OpenCV's top-left origin
            self.fits_ax.imshow(img, origin='upper', extent=[0, img_w, 0, img_h])
            self.fits_ax.set_title(
                f"{png_path.name} — ANNOTATED TRACKS",
                color=self.accent_orange, fontsize=11, fontweight='bold'
            )
            
            self.fits_canvas.draw()
            
            # Switch to viewer tab
            self.notebook.select(self.fits_tab)
            print(f"[INFO] Annotated image displayed in viewer: {png_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to render annotated PNG: {e}")
        
    def on_mouse_move(self, event):
        if event.inaxes is None or self.current_fits_data is None:
            return
            
        x, y = event.xdata, event.ydata
        
        ra_str, dec_str = "--", "--"
        if self.current_wcs and self.current_wcs.has_celestial:
            try:
                ra, dec = self.current_wcs.pixel_to_world_values(x, y)
                ra_str = f"{ra:.5f}°"
                dec_str = f"{dec:.5f}°"
            except Exception:
                pass
                
        # Get pixel value
        try:
            px_val = self.current_fits_data[int(round(y)), int(round(x))]
            px_str = f"{px_val:.1f}"
        except Exception:
            px_str = "--"
            
        self.coords_lbl.config(
            text=f"X: {x:.1f}, Y: {y:.1f} [Val: {px_str}] | RA: {ra_str}, Dec: {dec_str}"
        )
        
    def on_tree_row_double_click(self, event):
        item = self.tree.selection()
        if not item:
            return
            
        values = self.tree.item(item[0], "values")
        track_id = int(values[0])
        
        # Find this track inside our detected tracks
        target_track = None
        for track in self.detected_tracks:
            if track['track_id'] == track_id:
                target_track = track
                break
                
        if target_track:
            p1 = target_track['start_pixel']
            p2 = target_track['end_pixel']
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            
            # Switch back to FITS viewer
            self.notebook.select(self.fits_tab)
            
            # Pan the Matplotlib axes to center around mid_x, mid_y
            x_lim = self.fits_ax.get_xlim()
            y_lim = self.fits_ax.get_ylim()
            
            # Maintain current zoom scale but shift center
            half_w = (x_lim[1] - x_lim[0]) / 2.0
            half_h = (y_lim[1] - y_lim[0]) / 2.0
            
            # If default/full viewport is set, zoom in slightly
            if half_w > 100:
                half_w = 80.0
                half_h = 80.0
                
            self.fits_ax.set_xlim(mid_x - half_w, mid_x + half_w)
            self.fits_ax.set_ylim(mid_y - half_h, mid_y + half_h)
            
    def _release_viewer_locks(self):
        """Releases any active file handles in the viewer and forces garbage collection."""
        self.fits_fig.clf()
        self.current_fits_data = None
        self.current_fits_header = None
        self.current_wcs = None
        import gc
        gc.collect()

    def clear_logs(self):
        self.log_console.config(state=tk.NORMAL)
        self.log_console.delete("1.0", tk.END)
        self.log_console.config(state=tk.DISABLED)
        
    def process_log_queue(self):
        try:
            while True:
                string = self.log_queue.get_nowait()
                self.log_console.config(state=tk.NORMAL)
                self.log_console.insert(tk.END, string)
                self.log_console.see(tk.END)
                self.log_console.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(100, self.process_log_queue)

    def process_gui_queue(self):
        try:
            while True:
                callable_task = self.gui_queue.get_nowait()
                try:
                    callable_task()
                except Exception as e:
                    print(f"Error executing GUI task: {e}")
        except queue.Empty:
            pass
        self.after(50, self.process_gui_queue)

    def safe_gui_call(self, func, *args, **kwargs):
        """Queues a GUI-modifying function to run safely on the main thread."""
        self.gui_queue.put(lambda: func(*args, **kwargs))

    def enable_run_button(self):
        self.run_all_btn.config(state=tk.NORMAL)
        
    # --- PIPELINE TRIGGER THREADS ---
    
    def trigger_preprocess(self):
        self.run_in_thread(self._preprocess_worker)
        
    def trigger_difference(self):
        self._release_viewer_locks()
        self.run_in_thread(self._difference_worker)
        
    def trigger_solve(self):
        self._release_viewer_locks()
        self.run_in_thread(self._solve_worker)
        
    def trigger_correlation(self):
        self._release_viewer_locks()
        self.run_in_thread(self._correlation_worker)
        
    def trigger_full_pipeline(self):
        self._release_viewer_locks()
        self.run_in_thread(self._full_pipeline_worker)
        
    def run_in_thread(self, target):
        self.run_all_btn.config(state=tk.DISABLED)
        t = threading.Thread(target=target, daemon=True)
        t.start()
        
    def _preprocess_worker(self):
        try:
            fits_dir = Path(self.raw_dir_var.get())
            output_dir = Path(self.output_dir_var.get())
            print("\n>>> STARTING BATCH PREPROCESSING...")
            run_preprocessing(fits_dir, output_dir, num_adj=2, batch_size=40)
            print(">>> PREPROCESSING BATCH COMPLETED!")
            self.safe_gui_call(messagebox.showinfo, "Success", "Preprocessing completed successfully!")
        except Exception as e:
            print(f"\n[ERROR] Error during preprocessing: {e}")
            self.safe_gui_call(messagebox.showerror, "Pipeline Error", f"Preprocessing failed: {e}")
        finally:
            self.safe_gui_call(self.enable_run_button)
            
    def _difference_worker(self):
        try:
            fits_dir = Path(self.raw_dir_var.get())
            start_frame = self.start_frame_var.get()
            num_frames = self.num_frames_var.get()
            output_dir = Path(self.output_dir_var.get())
            
            if not start_frame:
                print("[ERROR] Start frame selection missing!")
                return
                
            print("\n>>> COMPUTING DIFFERENCE IMAGE...")
            diff, header, fits_out, png_out = run_difference(fits_dir, start_frame, num_frames, output_dir)
            print(">>> DIFFERENCE IMAGE CREATED SUCCESSFULLY!")
            
            # Load difference image into FITS viewer safely on main thread
            self.safe_gui_call(self.load_fits_image, fits_out)
        except Exception as e:
            print(f"\n[ERROR] Error during differencing: {e}")
            self.safe_gui_call(messagebox.showerror, "Pipeline Error", f"Differencing failed: {e}")
        finally:
            self.safe_gui_call(self.enable_run_button)
            
    def _solve_worker(self):
        try:
            fits_dir = Path(self.raw_dir_var.get())
            start_frame = self.start_frame_var.get()
            output_dir = Path(self.output_dir_var.get())
            
            raw_path = fits_dir / start_frame
            num_frames = self.num_frames_var.get()
            start_stem = Path(start_frame).stem.replace(" ", "_")
            diff_path = output_dir / f"diff_{start_stem}_{num_frames}f.fits"
            
            if not raw_path.exists():
                print(f"[ERROR] Cannot find raw image for solving at {raw_path}")
                return
            if not diff_path.exists():
                print(f"[ERROR] Cannot find difference image at {diff_path}. Run differencing first.")
                return
                
            print("\n>>> RUNNING PLATE SOLVER...")
            wcs_data = solve_and_apply_wcs(str(raw_path), str(diff_path))
            if wcs_data:
                print(">>> SUCCESS! True coordinates written to difference image.")
                # Reload FITS data in viewer safely on main thread
                self.safe_gui_call(self.load_fits_image, diff_path)
            else:
                print(">>> WARNING: Solver could not calibrate image coordinates.")
                self.safe_gui_call(messagebox.showwarning, "WCS Solver Failed", "Plate solver was unable to resolve fields. Running in Geocentric fallback.")
        except Exception as e:
            print(f"\n[ERROR] Error during WCS solving: {e}")
            self.safe_gui_call(messagebox.showerror, "Pipeline Error", f"Solver failed: {e}")
        finally:
            self.safe_gui_call(self.enable_run_button)
            
    def _correlation_worker(self):
        try:
            output_dir = Path(self.output_dir_var.get())
            catalog_path = Path(self.catalog_var.get())
            
            start_frame = self.start_frame_var.get()
            num_frames = self.num_frames_var.get()
            start_stem = Path(start_frame).stem.replace(" ", "_")
            
            diff_fits = output_dir / f"diff_{start_stem}_{num_frames}f.fits"
            output_png = output_dir / f"diff_{start_stem}_{num_frames}f_correlated.png"
            
            if not diff_fits.exists():
                print(f"[ERROR] Cannot find difference image at {diff_fits}. Compute differencing first!")
                return
            if not catalog_path.exists():
                print(f"[ERROR] Cannot find catalog file missing at {catalog_path}!")
                return
                
            print("\n>>> RUNNING TRACK DETECTION, PHOTOMETRY & CORRELATION...")
            results = process_and_correlate(str(diff_fits), str(catalog_path), str(output_png))
            
            if results:
                self.detected_tracks = results
                print(f"\n>>> PIPELINE CORRELATION COMPLETE: Detected {len(results)} distinct tracks.")
                self.safe_gui_call(self.populate_results_table, results)
                # Show the annotated PNG directly — not the raw difference FITS
                self.safe_gui_call(self.show_annotated_png, output_png)
            else:
                self.detected_tracks = []
                self.safe_gui_call(self.populate_results_table, [])
                print("\n>>> PROCESS COMPLETE: No debris tracks detected.")
        except Exception as e:
            print(f"\n[ERROR] Error during detection/correlation: {e}")
            self.safe_gui_call(messagebox.showerror, "Pipeline Error", f"Correlation failed: {e}")
        finally:
            self.safe_gui_call(self.enable_run_button)
            
    def _full_pipeline_worker(self):
        try:
            fits_dir = Path(self.raw_dir_var.get())
            catalog = Path(self.catalog_var.get())
            output = Path(self.output_dir_var.get())
            start_f = self.start_frame_var.get()
            num_f = self.num_frames_var.get()
            
            run_pre = self.preprocess_var.get()
            run_sol = self.solve_var.get()
            
            if not start_f:
                print("[ERROR] Start frame is not selected!")
                return
                
            print("\n" + "="*50)
            print(">>> STARTING PIPELINE INTEGRATION RUN")
            print("="*50)
            
            # Step 1: Pre-process
            if run_pre:
                print("\n[Step 1/4] Pre-processing FITS frames...")
                run_preprocessing(fits_dir, output, num_adj=2, batch_size=40)
            else:
                print("\n[Step 1/4] Pre-processing skipped.")
                
            # Step 2: Differencing
            print("\n[Step 2/4] Running image differencing...")
            diff, header, diff_fits, diff_png = run_difference(fits_dir, start_f, num_f, output)
            
            # Step 3: Solve WCS
            if run_sol:
                print("\n[Step 3/4] Calibrating WCS coordinates (Plate Solving)...")
                raw_path = fits_dir / start_f
                solve_and_apply_wcs(str(raw_path), str(diff_fits))
            else:
                print("\n[Step 3/4] WCS Calibration skipped.")
                
            # Step 4: Debris tracking, Photometry & Correlation
            start_stem = Path(start_f).stem.replace(" ", "_")
            output_png = output / f"diff_{start_stem}_{num_f}f_correlated.png"
            results = process_and_correlate(str(diff_fits), str(catalog), str(output_png))
            
            if results:
                self.detected_tracks = results
                self.safe_gui_call(self.populate_results_table, results)
                # Show the annotated PNG directly — not the raw difference FITS
                self.safe_gui_call(self.show_annotated_png, output_png)
                print(f"\n[SUCCESS] FULL PIPELINE SUCCESSFULLY COMPLETED! Detected {len(results)} streaks.")
                self.safe_gui_call(messagebox.showinfo, "Pipeline Complete", f"Success! Detected {len(results)} debris tracks.\nReport saved to: Output/pipeline_report.txt")
            else:
                self.detected_tracks = []
                self.safe_gui_call(self.populate_results_table, [])
                print("\n[SUCCESS] FULL PIPELINE SUCCESSFULLY COMPLETED! No debris tracks detected.")
                self.safe_gui_call(messagebox.showinfo, "Pipeline Complete", "Completed successfully. No debris tracks detected.")
                
        except Exception as e:
            print(f"\n[ERROR] Pipeline integration crashed: {e}")
            self.safe_gui_call(messagebox.showerror, "Pipeline Failure", f"Pipeline crashed: {e}")
        finally:
            self.safe_gui_call(self.enable_run_button)
            
    def populate_results_table(self, results):
        # Clear table
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if not results:
            return
            
        for res in results:
            phot = res['photometry']
            mag_val = phot['magnitude']
            mag_str = f"{mag_val:.2f}" if not math.isnan(mag_val) else "N/A"
            
            p1 = res['start_pixel']
            p2 = res['end_pixel']
            
            self.tree.insert("", tk.END, values=(
                res['track_id'],
                res['label'],
                mag_str,
                f"{phot['snr']:.1f}",
                f"{phot['net_flux']:.1f}",
                f"{phot['peak_value']:.1f}",
                f"{phot['streak_length']:.1f}",
                f"{res['centroid_ra']:.4f}",
                f"{res['centroid_dec']:.4f}",
                f"({p1[0]:.1f},{p1[1]:.1f})",
                f"({p2[0]:.1f},{p2[1]:.1f})",
                f"{res['separation_deg']:.4f}"
            ))

    def destroy(self):
        # Restore old stdout/stderr
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        super().destroy()

if __name__ == "__main__":
    app = DebrisTrackerGUI()
    app.mainloop()
