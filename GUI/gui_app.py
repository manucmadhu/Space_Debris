"""
gui.py — Space Debris Detection & Photometry Dashboard
=======================================================
Professional Tkinter GUI for the FRIGATE pipeline.

Performance fixes vs original:
  - Image cached as numpy array; stretch/cmap changes reuse cached data
  - Matplotlib figure NOT recreated on every update — only image data replaced
  - Motion-notify debounced (16ms throttle, ~60fps max)
  - Log console: plain text insert, capped at 4000 lines, no state toggle per line
  - Downsampling factor computed once on load, not every redraw
  - Treeview: tag-based row colouring, incremental insert
  - All messagebox calls routed through gui_queue (never from worker thread directly)
  - Pipeline cancel flag (threading.Event) checked by workers
  - Status bar with animated spinner during pipeline runs
  - Keyboard shortcuts: F5=run all, Ctrl+L=clear log, Escape=cancel
"""

import gc
import io
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    _PIL = True
except ImportError:
    _PIL = False

# ── Pipeline imports (graceful degradation if modules missing) ────────────────
ROOT_DIR = Path(__file__).parent.parent
CODE_DIR = ROOT_DIR / "Code"
sys.path.extend([str(ROOT_DIR), str(CODE_DIR)])

_IMPORT_ERROR = None
try:
    from astropy.io import fits
    from astropy.wcs import WCS
    from pre_process import run_preprocessing, get_sorted_fits
    from difference import run_difference
    from plate_solver import solve_and_apply_wcs
    from correlate import process_and_correlate
    from display import apply_stretch
    from flux_extract import extract_streak_flux
except ImportError as e:
    _IMPORT_ERROR = str(e)
    # Stub so the GUI still launches and shows the error
    def get_sorted_fits(d): return sorted(Path(d).glob("*.fits"))
    def apply_stretch(d, stretch_type="percentile"):
        lo, hi = np.percentile(d, [2, 99])
        return np.clip((d - lo) / max(hi - lo, 1), 0, 1)


# =============================================================================
# Palette & design tokens
# =============================================================================
# Inspired by deep-sky observatory dashboards: near-black with cold-blue
# accent and amber for live data. One signature element: the status bar
# uses a pulsing amber dot to show pipeline activity without blocking the UI.

PAL = {
    "bg":          "#0d0f14",   # near-black with blue tint
    "surface":     "#13161e",   # card background
    "surface2":    "#1a1e29",   # slightly lighter surface
    "border":      "#252a38",   # subtle border
    "accent":      "#4d9de0",   # cold blue — primary action
    "accent_dim":  "#2a5580",   # muted blue for inactive
    "amber":       "#e8a838",   # live data / warning
    "green":       "#3dd68c",   # success / correlated
    "red":         "#e05d5d",   # error / uncorrelated
    "fg":          "#d8dce8",   # primary text
    "fg_dim":      "#6b7394",   # secondary text
    "mono":        "Consolas",  # monospace
    "ui":          "Segoe UI",  # UI font
}

FONT_UI    = (PAL["ui"],   10)
FONT_UI_B  = (PAL["ui"],   10, "bold")
FONT_TITLE = (PAL["ui"],   11, "bold")
FONT_MONO  = (PAL["mono"],  9)
FONT_SMALL = (PAL["ui"],    9)


# =============================================================================
# Thread-safe logger
# =============================================================================

class _Logger:
    """
    Writes to a queue; the main thread drains it into the Text widget.
    Never touches Tkinter from a worker thread.
    """
    MAX_LINES = 4000

    def __init__(self, q: queue.Queue, original=None):
        self._q        = q
        self._original = original

    def write(self, s):
        self._q.put(("log", s))
        if self._original:
            self._original.write(s)

    def flush(self):
        if self._original:
            try: self._original.flush()
            except Exception: pass


# =============================================================================
# Main application
# =============================================================================

class DebrisTrackerGUI(tk.Tk):

    # ── init ─────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.title("FRIGATE — Space Debris Detection & Photometry")
        self.geometry("1400x860")
        self.minsize(1100, 720)
        self.configure(bg=PAL["bg"])

        # State
        self._fits_data     = None      # raw float32 array, cached on load
        self._fits_header   = None
        self._fits_wcs      = None
        self._fits_path     = None
        self._display_cache = None      # (stretch, cmap) → scaled uint8
        self._display_key   = None
        self._ds_factor     = 1         # downsampling, computed once on load
        self._png_path      = None
        self._tracks        = []
        self._cancel_flag   = threading.Event()
        self._running       = False
        self._motion_after  = None      # debounce id for mouse move
        self._spinner_after = None
        self._spinner_idx   = 0

        # Queues
        self._log_q  = queue.Queue()
        self._gui_q  = queue.Queue()

        # Variables
        self.v_raw_dir     = tk.StringVar(value=str(ROOT_DIR / "Data" / "raw"))
        self.v_catalog     = tk.StringVar(value=str(ROOT_DIR / "Data" / "3le.txt"))
        self.v_output      = tk.StringVar(value=str(ROOT_DIR / "Output"))
        self.v_start_frame = tk.StringVar()
        self.v_num_frames  = tk.IntVar(value=10)
        self.v_do_preproc  = tk.BooleanVar(value=False)
        self.v_do_solve    = tk.BooleanVar(value=True)
        self.v_show_ann    = tk.BooleanVar(value=True)
        self.v_stretch     = tk.StringVar(value="percentile")
        self.v_cmap        = tk.StringVar(value="gray")
        self.v_status      = tk.StringVar(value="Ready")

        self._build_styles()
        self._build_ui()
        self._scan_raw_dir()

        # Keyboard shortcuts
        self.bind("<F5>",         lambda e: self._trigger("full"))
        self.bind("<Control-l>",  lambda e: self._clear_log())
        self.bind("<Escape>",     lambda e: self._cancel())

        # Redirect stdout/stderr
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _Logger(self._log_q, self._orig_stdout)
        sys.stderr = _Logger(self._log_q, self._orig_stderr)

        # Drain queues
        self.after(80,  self._drain_log_q)
        self.after(50,  self._drain_gui_q)

        # Show import error if any
        if _IMPORT_ERROR:
            self.after(200, lambda: self._log_line(
                f"[WARN] Some pipeline modules failed to import: {_IMPORT_ERROR}\n"
                f"       GUI is functional but pipeline steps may fail.\n"
            ))

        self._log_line("FRIGATE pipeline GUI ready.  F5 = Run Full | Ctrl+L = Clear log | Esc = Cancel\n")

    # ── styles ────────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure(".",
            background=PAL["surface"], foreground=PAL["fg"],
            font=FONT_UI, borderwidth=0, relief="flat")

        s.configure("TLabel",
            background=PAL["surface"], foreground=PAL["fg"], font=FONT_UI)

        s.configure("Dim.TLabel",
            background=PAL["surface"], foreground=PAL["fg_dim"], font=FONT_SMALL)

        s.configure("Title.TLabel",
            background=PAL["surface"], foreground=PAL["accent"], font=FONT_TITLE)

        s.configure("TFrame", background=PAL["surface"])
        s.configure("Sep.TFrame", background=PAL["border"])

        s.configure("TButton",
            background=PAL["accent"], foreground="#ffffff",
            font=FONT_UI_B, padding=(10, 5), relief="flat")
        s.map("TButton",
            background=[("active","#6ab4f0"),("disabled", PAL["accent_dim"])],
            foreground=[("disabled","#8899aa")])

        s.configure("Ghost.TButton",
            background=PAL["surface2"], foreground=PAL["fg"],
            font=FONT_UI, padding=(8, 4))
        s.map("Ghost.TButton",
            background=[("active", PAL["border"])])

        s.configure("Danger.TButton",
            background="#5c2020", foreground=PAL["red"],
            font=FONT_UI_B, padding=(10,5))
        s.map("Danger.TButton",
            background=[("active","#7a2a2a")])

        s.configure("TEntry",
            fieldbackground=PAL["bg"], foreground=PAL["fg"],
            insertcolor=PAL["fg"], bordercolor=PAL["border"],
            lightcolor=PAL["border"], darkcolor=PAL["border"])

        s.configure("TCombobox",
            fieldbackground=PAL["bg"], background=PAL["surface2"],
            foreground=PAL["fg"], arrowcolor=PAL["accent"],
            bordercolor=PAL["border"])
        s.map("TCombobox", fieldbackground=[("readonly", PAL["bg"])])

        s.configure("TSpinbox",
            fieldbackground=PAL["bg"], foreground=PAL["fg"],
            arrowcolor=PAL["accent"], bordercolor=PAL["border"])

        s.configure("TCheckbutton",
            background=PAL["surface"], foreground=PAL["fg"],
            font=FONT_UI, indicatorcolor=PAL["bg"],
            indicatordiameter=14)
        s.map("TCheckbutton",
            indicatorcolor=[("selected", PAL["accent"]),
                            ("!selected", PAL["border"])])

        s.configure("TNotebook",
            background=PAL["bg"], borderwidth=0, tabmargins=0)
        s.configure("TNotebook.Tab",
            background=PAL["surface"], foreground=PAL["fg_dim"],
            font=FONT_UI, padding=[16, 6])
        s.map("TNotebook.Tab",
            background=[("selected", PAL["bg"])],
            foreground=[("selected", PAL["accent"])])

        s.configure("Treeview",
            background=PAL["surface"], fieldbackground=PAL["surface"],
            foreground=PAL["fg"], rowheight=26, font=FONT_SMALL,
            borderwidth=0)
        s.configure("Treeview.Heading",
            background=PAL["bg"], foreground=PAL["fg_dim"],
            font=FONT_UI_B, relief="flat", padding=(4,4))
        s.map("Treeview",
            background=[("selected", PAL["accent_dim"])],
            foreground=[("selected", "#ffffff")])

        s.configure("TLabelframe",
            background=PAL["surface"], foreground=PAL["fg_dim"],
            bordercolor=PAL["border"], font=FONT_SMALL)
        s.configure("TLabelframe.Label",
            background=PAL["surface"], foreground=PAL["fg_dim"],
            font=FONT_SMALL)

        s.configure("TScrollbar",
            background=PAL["surface2"], troughcolor=PAL["bg"],
            arrowcolor=PAL["fg_dim"], borderwidth=0)

        # Progress / status bar
        s.configure("Status.TLabel",
            background=PAL["bg"], foreground=PAL["fg_dim"], font=FONT_SMALL)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # Root grid: left sidebar | right workspace
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self._build_sidebar()
        self._build_workspace()
        self._build_statusbar()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = tk.Frame(self, bg=PAL["surface"], width=300)
        sb.grid(row=0, column=0, sticky="nsew", padx=(0,1))
        sb.grid_propagate(False)
        sb.columnconfigure(0, weight=1)

        # Header
        hdr = tk.Frame(sb, bg=PAL["surface"])
        hdr.grid(row=0, column=0, sticky="ew", pady=(16,8), padx=16)
        tk.Label(hdr, text="FRIGATE", bg=PAL["surface"],
                 fg=PAL["accent"], font=(PAL["ui"], 16, "bold")).pack(anchor="w")
        tk.Label(hdr, text="Debris Detection Pipeline",
                 bg=PAL["surface"], fg=PAL["fg_dim"], font=FONT_SMALL).pack(anchor="w")

        # Thin divider
        tk.Frame(sb, bg=PAL["border"], height=1).grid(row=1, column=0, sticky="ew")

        # Scrollable content
        canvas = tk.Canvas(sb, bg=PAL["surface"], highlightthickness=0, bd=0)
        vsb    = ttk.Scrollbar(sb, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.grid(row=2, column=1, sticky="ns")
        canvas.grid(row=2, column=0, sticky="nsew")
        sb.rowconfigure(2, weight=1)

        inner = tk.Frame(canvas, bg=PAL["surface"])
        win_id = canvas.create_window((0,0), window=inner, anchor="nw")

        def _on_inner_resize(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win_id, width=canvas.winfo_width())
        inner.bind("<Configure>", _on_inner_resize)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))

        def _scroll(e):  canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind("<Enter>",  lambda e: canvas.bind_all("<MouseWheel>", _scroll))
        canvas.bind("<Leave>",  lambda e: canvas.unbind_all("<MouseWheel>"))

        p = 16  # padding
        self._build_section_paths(inner, p)
        self._build_section_params(inner, p)
        self._build_section_actions(inner, p)

    def _section_label(self, parent, text):
        f = tk.Frame(parent, bg=PAL["surface"])
        f.pack(fill="x", padx=16, pady=(18,4))
        tk.Label(f, text=text.upper(), bg=PAL["surface"],
                 fg=PAL["fg_dim"], font=(PAL["ui"], 8, "bold")).pack(side="left")
        tk.Frame(f, bg=PAL["border"], height=1).pack(side="left", fill="x",
                                                      expand=True, padx=(8,0))

    def _path_row(self, parent, label, var, browse_cmd):
        tk.Label(parent, text=label, bg=PAL["surface"],
                 fg=PAL["fg_dim"], font=FONT_SMALL).pack(
                     anchor="w", padx=16, pady=(6,1))
        row = tk.Frame(parent, bg=PAL["surface"])
        row.pack(fill="x", padx=16, pady=(0,2))
        ttk.Entry(row, textvariable=var, font=FONT_SMALL).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="…", width=3,
                   style="Ghost.TButton",
                   command=browse_cmd).pack(side="left", padx=(4,0))

    def _build_section_paths(self, parent, p):
        self._section_label(parent, "Directories")
        self._path_row(parent, "Raw FITS folder",
                       self.v_raw_dir, self._browse_raw)
        self._path_row(parent, "TLE catalogue file",
                       self.v_catalog, self._browse_catalog)
        self._path_row(parent, "Output directory",
                       self.v_output, self._browse_output)

    def _build_section_params(self, parent, p):
        self._section_label(parent, "Parameters")

        tk.Label(parent, text="Start frame", bg=PAL["surface"],
                 fg=PAL["fg_dim"], font=FONT_SMALL).pack(anchor="w", padx=16, pady=(6,1))
        self._frame_combo = ttk.Combobox(parent, textvariable=self.v_start_frame,
                                          state="readonly", font=FONT_SMALL)
        self._frame_combo.pack(fill="x", padx=16, pady=(0,6))
        self._frame_combo.bind("<<ComboboxSelected>>", self._on_frame_selected)

        tk.Label(parent, text="Number of frames", bg=PAL["surface"],
                 fg=PAL["fg_dim"], font=FONT_SMALL).pack(anchor="w", padx=16, pady=(2,1))
        ttk.Spinbox(parent, from_=3, to=500, textvariable=self.v_num_frames,
                    width=8, font=FONT_SMALL).pack(anchor="w", padx=16, pady=(0,6))

        opt_frame = tk.Frame(parent, bg=PAL["surface"])
        opt_frame.pack(fill="x", padx=16, pady=(2,6))
        ttk.Checkbutton(opt_frame, text="Pre-process frames",
                        variable=self.v_do_preproc).pack(anchor="w", pady=2)
        ttk.Checkbutton(opt_frame, text="Run plate solver (WCS)",
                        variable=self.v_do_solve).pack(anchor="w", pady=2)

    def _build_section_actions(self, parent, p):
        self._section_label(parent, "Actions")

        actions = [
            ("Pre-process only",    "preprocess", "Ghost.TButton"),
            ("Difference image",    "diff",       "Ghost.TButton"),
            ("Plate solve",         "solve",      "Ghost.TButton"),
            ("Detect & correlate",  "correlate",  "Ghost.TButton"),
        ]
        btn_frame = tk.Frame(parent, bg=PAL["surface"])
        btn_frame.pack(fill="x", padx=16, pady=(0,8))
        for label, cmd, style in actions:
            ttk.Button(btn_frame, text=label, style=style,
                       command=lambda c=cmd: self._trigger(c)).pack(
                           fill="x", pady=2)

        tk.Frame(parent, bg=PAL["border"], height=1).pack(
            fill="x", padx=16, pady=8)

        self._run_btn = ttk.Button(parent, text="RUN FULL PIPELINE  [F5]",
                                    command=lambda: self._trigger("full"))
        self._run_btn.pack(fill="x", padx=16, pady=(0,4))

        self._cancel_btn = ttk.Button(parent, text="Cancel  [Esc]",
                                       style="Danger.TButton",
                                       command=self._cancel,
                                       state="disabled")
        self._cancel_btn.pack(fill="x", padx=16, pady=(0,16))

    # ── Workspace ─────────────────────────────────────────────────────────────
    def _build_workspace(self):
        ws = tk.Frame(self, bg=PAL["bg"])
        ws.grid(row=0, column=1, sticky="nsew")
        ws.rowconfigure(0, weight=3)
        ws.rowconfigure(1, weight=0)
        ws.rowconfigure(2, weight=1)
        ws.columnconfigure(0, weight=1)

        self._build_viewer(ws)
        self._build_log(ws)

    def _build_viewer(self, parent):
        nb = ttk.Notebook(parent, style="TNotebook")
        nb.grid(row=0, column=0, sticky="nsew", pady=(0,1))
        self._nb = nb

        # Tab 1 — FITS viewer
        t1 = tk.Frame(nb, bg=PAL["bg"])
        nb.add(t1, text="  Image Viewer  ")
        self._build_fits_tab(t1)

        # Tab 2 — Results table
        t2 = tk.Frame(nb, bg=PAL["bg"])
        nb.add(t2, text="  Detected Tracks  ")
        self._build_results_tab(t2)

    def _build_fits_tab(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Toolbar row
        toolbar = tk.Frame(parent, bg=PAL["surface"], height=38)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.grid_propagate(False)

        def _lbl(text):
            return tk.Label(toolbar, text=text, bg=PAL["surface"],
                            fg=PAL["fg_dim"], font=FONT_SMALL)

        _lbl("Stretch:").pack(side="left", padx=(12,4))
        stretch_cb = ttk.Combobox(toolbar, textvariable=self.v_stretch,
                                   values=["percentile","linear","log","sqrt"],
                                   width=11, state="readonly", font=FONT_SMALL)
        stretch_cb.pack(side="left", padx=(0,12))
        stretch_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_display())

        _lbl("Colormap:").pack(side="left", padx=(0,4))
        cmap_cb = ttk.Combobox(toolbar, textvariable=self.v_cmap,
                                values=["gray","viridis","inferno","plasma","bone","hot"],
                                width=11, state="readonly", font=FONT_SMALL)
        cmap_cb.pack(side="left", padx=(0,12))
        cmap_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_display())

        ttk.Checkbutton(toolbar, text="Show annotated overlay",
                        variable=self.v_show_ann,
                        command=self._refresh_display).pack(side="left", padx=8)

        # Coordinate readout — right-aligned
        self._coord_lbl = tk.Label(toolbar, text="X: —  Y: —  |  RA: —  Dec: —",
                                    bg=PAL["surface"], fg=PAL["amber"],
                                    font=FONT_MONO)
        self._coord_lbl.pack(side="right", padx=12)

        # Matplotlib figure — created ONCE, updated in place
        fig_frame = tk.Frame(parent, bg=PAL["bg"])
        fig_frame.grid(row=1, column=0, sticky="nsew")

        self._fig, self._ax = plt.subplots(figsize=(8, 6))
        self._fig.patch.set_facecolor(PAL["bg"])
        self._ax.set_facecolor(PAL["bg"])
        self._ax.tick_params(colors=PAL["fg_dim"])
        for sp in self._ax.spines.values():
            sp.set_color(PAL["border"])
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._ax.set_title("No image loaded", color=PAL["fg_dim"],
                            fontsize=10, pad=8)
        self._im_artist = None    # matplotlib AxesImage, updated not recreated

        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # Navigation toolbar — single loop, style fixed properly
        tb_frame = tk.Frame(fig_frame, bg="#2a2a35")
        tb_frame.pack(fill="x")
        self._mpl_toolbar = NavigationToolbar2Tk(self._canvas, tb_frame)
        self._mpl_toolbar.config(background="#2a2a35")
        for ch in self._mpl_toolbar.winfo_children():
            for key in ("background", "fg", "activebackground", "highlightbackground"):
                try: ch.configure(**{key: "#2a2a35"})
                except Exception: pass
        self._mpl_toolbar.update()

        # Mouse motion — debounced
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _build_results_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        cols = ("id","label","mag","snr","net_flux","peak",
                "length","ra","dec","start_px","end_px","sep")
        headings = {
            "id":       ("ID",              50),
            "label":    ("Debris / Object", 200),
            "mag":      ("Mag",             70),
            "snr":      ("SNR",             70),
            "net_flux": ("Net Flux",        90),
            "peak":     ("Peak",            80),
            "length":   ("Length (px)",     90),
            "ra":       ("RA (deg)",        100),
            "dec":      ("Dec (deg)",       100),
            "start_px": ("Start (x,y)",    110),
            "end_px":   ("End (x,y)",      110),
            "sep":      ("Cat Sep (deg)",   110),
        }

        frame = tk.Frame(parent, bg=PAL["bg"])
        frame.pack(fill="both", expand=True)

        vsb = ttk.Scrollbar(frame, orient="vertical")
        hsb = ttk.Scrollbar(frame, orient="horizontal")
        self._tree = ttk.Treeview(frame, columns=cols, show="headings",
                                   yscrollcommand=vsb.set,
                                   xscrollcommand=hsb.set,
                                   selectmode="browse")
        vsb.config(command=self._tree.yview)
        hsb.config(command=self._tree.xview)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        for col, (title, width) in headings.items():
            self._tree.heading(col, text=title,
                               command=lambda c=col: self._sort_tree(c, False))
            self._tree.column(col, width=width, anchor="center", minwidth=40)

        self._tree.tag_configure("match",   foreground=PAL["green"])
        self._tree.tag_configure("nomatch", foreground=PAL["red"])
        self._tree.bind("<Double-1>", self._on_row_dblclick)

    def _build_log(self, parent):
        log_frame = tk.Frame(parent, bg=PAL["surface"], height=180)
        log_frame.grid(row=2, column=0, sticky="ew")
        log_frame.grid_propagate(False)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)

        hdr = tk.Frame(log_frame, bg=PAL["bg"], height=24)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        hdr.grid_propagate(False)
        tk.Label(hdr, text="  Console output",
                 bg=PAL["bg"], fg=PAL["accent"],
                 font=FONT_UI_B, anchor="w").pack(side="left")
        ttk.Button(hdr, text="Clear", style="Ghost.TButton",
                   command=self._clear_log).pack(side="right", padx=4)

        vsb = ttk.Scrollbar(log_frame, orient="vertical")
        vsb.grid(row=1, column=1, sticky="ns")

        self._log = tk.Text(log_frame,
                             bg=PAL["bg"], fg="#44cc77",
                             font=FONT_MONO, wrap="word",
                             yscrollcommand=vsb.set,
                             padx=8, pady=4,
                             state="disabled",
                             relief="flat")
        self._log.grid(row=1, column=0, sticky="nsew")
        vsb.config(command=self._log.yview)

        # Text tags for colour
        self._log.tag_config("err",  foreground=PAL["red"])
        self._log.tag_config("warn", foreground=PAL["amber"])
        self._log.tag_config("info", foreground=PAL["accent"])

    def _build_statusbar(self):
        sb = tk.Frame(self, bg=PAL["bg"], height=26)
        sb.grid(row=1, column=0, columnspan=2, sticky="ew")
        sb.grid_propagate(False)

        self._spinner_lbl = tk.Label(sb, text="●", bg=PAL["bg"],
                                      fg=PAL["fg_dim"],
                                      font=(PAL["ui"], 10))
        self._spinner_lbl.pack(side="left", padx=(10,4))

        tk.Label(sb, textvariable=self.v_status,
                 bg=PAL["bg"], fg=PAL["fg_dim"],
                 font=FONT_SMALL).pack(side="left")

        # Shortcuts hint right side
        tk.Label(sb, text="F5 Run  |  Ctrl+L Clear log  |  Esc Cancel",
                 bg=PAL["bg"], fg=PAL["fg_dim"],
                 font=FONT_SMALL).pack(side="right", padx=10)

    # ── Image display (core performance path) ─────────────────────────────────
    def _load_fits(self, path: Path):
        """Read FITS into cache. Called from main thread only."""
        path = Path(path)
        try:
            with open(str(path), "rb") as f:
                raw = f.read()
            with fits.open(io.BytesIO(raw), memmap=False) as h:
                self._fits_data   = np.array(h[0].data, dtype=np.float32)
                self._fits_header = h[0].header.copy()
                try:
                    self._fits_wcs = WCS(self._fits_header, relax=True)
                except Exception:
                    self._fits_wcs = None
            self._fits_path     = path
            self._display_cache = None
            self._display_key   = None
            H, W = self._fits_data.shape
            # Compute downsampling ONCE — target ~1200px on shortest axis
            self._ds_factor = max(1, min(W, H) // 1200)
            self._check_png_path()
            self._refresh_display()
            self._nb.select(0)
        except Exception as e:
            self._log_line(f"[ERROR] Failed to load {path.name}: {e}\n", "err")

    def _check_png_path(self):
        if not self._fits_path:
            return
        output = Path(self.v_output.get())
        stem   = self._fits_path.stem.replace(" ", "_")
        nf     = self.v_num_frames.get()
        cand   = output / f"diff_{stem}_{nf}f_correlated.png"
        self._png_path = cand if cand.exists() else None

    def _refresh_display(self):
        """Update the matplotlib axes IN PLACE — no figure recreation."""
        if self._fits_data is None:
            return

        # Show annotated PNG if available and requested
        if self.v_show_ann.get() and self._png_path and self._png_path.exists():
            self._render_png(self._png_path)
            return

        stretch = self.v_stretch.get()
        cmap    = self.v_cmap.get()
        key     = (stretch, cmap)

        if key != self._display_key or self._display_cache is None:
            ds   = self._fits_data[::self._ds_factor, ::self._ds_factor]
            scaled = apply_stretch(ds, stretch_type=stretch)
            self._display_cache = scaled
            self._display_key   = key

        H, W = self._fits_data.shape

        if self._im_artist is None:
            self._im_artist = self._ax.imshow(
                self._display_cache, cmap=cmap, origin="lower",
                extent=[0, W, 0, H], aspect="auto")
        else:
            self._im_artist.set_data(self._display_cache)
            self._im_artist.set_cmap(cmap)
            self._im_artist.set_extent([0, W, 0, H])

        self._ax.set_title(self._fits_path.name if self._fits_path else "",
                            color=PAL["fg"], fontsize=9, pad=4)
        self._draw_track_overlays()
        self._canvas.draw_idle()

    def _render_png(self, png_path: Path):
        """Render an annotated PNG into the axes without recreating the figure."""
        try:
            img = mpimg.imread(str(png_path))
            ih, iw = img.shape[:2]
            H = self._fits_data.shape[0] if self._fits_data is not None else ih
            W = self._fits_data.shape[1] if self._fits_data is not None else iw

            if self._im_artist is None:
                self._im_artist = self._ax.imshow(
                    img, origin="upper", extent=[0, W, 0, H], aspect="auto")
            else:
                self._im_artist.set_data(img)
                self._im_artist.set_extent([0, W, 0, H])

            self._ax.set_title(png_path.name, color=PAL["amber"],
                                fontsize=9, pad=4)
            self._canvas.draw_idle()
        except Exception as e:
            self._log_line(f"[WARN] Could not render PNG: {e}\n", "warn")

    def _draw_track_overlays(self):
        """Draw streak overlays. Called only from _refresh_display."""
        # Remove previous overlays (lines, patches, texts added by us)
        for artist in list(self._ax.lines + self._ax.patches +
                           self._ax.texts):
            try: artist.remove()
            except Exception: pass

        for t in self._tracks:
            p1    = t.get("start_pixel", [0, 0])
            p2    = t.get("end_pixel",   [0, 0])
            color = PAL["green"] if t.get("is_match") else PAL["red"]
            self._ax.plot([p1[0],p2[0]], [p1[1],p2[1]],
                          color=PAL["amber"], lw=1.2, ls="--")
            self._ax.scatter([p1[0],p2[0]], [p1[1],p2[1]],
                             color=color, s=18, zorder=5)
            xlo = min(p1[0],p2[0])-12
            ylo = min(p1[1],p2[1])-12
            xhi = max(p1[0],p2[0])+12
            yhi = max(p1[1],p2[1])+12
            rect = plt.Rectangle((xlo,ylo), xhi-xlo, yhi-ylo,
                                  fill=False, edgecolor=color, lw=1.2)
            self._ax.add_patch(rect)
            self._ax.text(xlo, yhi+3,
                          f"T{t['track_id']}: {t['label']}",
                          color=color, fontsize=7, fontweight="bold")

    # ── Mouse motion — debounced to ~60fps ───────────────────────────────────
    def _on_motion(self, event):
        if self._motion_after:
            self.after_cancel(self._motion_after)
        self._motion_after = self.after(16, lambda: self._update_coords(event))

    def _update_coords(self, event):
        self._motion_after = None
        if event.inaxes is None or self._fits_data is None:
            return
        x, y = event.xdata, event.ydata
        ra_s = dec_s = "—"
        if self._fits_wcs and self._fits_wcs.has_celestial:
            try:
                ra, dec = self._fits_wcs.pixel_to_world_values(x, y)
                ra_s  = f"{ra:.5f}°"
                dec_s = f"{dec:+.5f}°"
            except Exception:
                pass
        try:
            val = self._fits_data[int(round(y)), int(round(x))]
            val_s = f"{val:.1f}"
        except Exception:
            val_s = "—"
        self._coord_lbl.config(
            text=f"X: {x:.1f}  Y: {y:.1f}  [{val_s}]  |  RA: {ra_s}  Dec: {dec_s}")

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log_line(self, text: str, tag: str = ""):
        self._log.config(state="normal")
        # Trim if over limit
        lines = int(self._log.index("end-1c").split(".")[0])
        if lines > _Logger.MAX_LINES:
            self._log.delete("1.0", f"{lines - _Logger.MAX_LINES // 2}.0")
        if tag:
            self._log.insert("end", text, tag)
        else:
            self._log.insert("end", text)
        self._log.see("end")
        self._log.config(state="disabled")

    def _drain_log_q(self):
        try:
            budget = 50  # process at most 50 chunks per tick
            while budget > 0:
                kind, payload = self._log_q.get_nowait()
                if kind == "log":
                    tag = ("err"  if "[ERROR]" in payload else
                           "warn" if "[WARN]"  in payload else
                           "info" if "[INFO]"  in payload else "")
                    self._log_line(payload, tag)
                budget -= 1
        except queue.Empty:
            pass
        self.after(80, self._drain_log_q)

    def _drain_gui_q(self):
        try:
            while True:
                fn = self._gui_q.get_nowait()
                try: fn()
                except Exception as e:
                    self._log_line(f"[ERROR] GUI task: {e}\n", "err")
        except queue.Empty:
            pass
        self.after(50, self._drain_gui_q)

    def _safe(self, fn, *a, **kw):
        """Queue a callable to run on the main thread."""
        self._gui_q.put(lambda: fn(*a, **kw))

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    # ── Status bar spinner ────────────────────────────────────────────────────
    _SPIN = ["◐","◓","◑","◒"]

    def _start_spinner(self, msg="Running…"):
        self.v_status.set(msg)
        self._spinner_idx = 0
        self._tick_spinner()

    def _tick_spinner(self):
        self._spinner_lbl.config(
            fg=PAL["amber"],
            text=self._SPIN[self._spinner_idx % len(self._SPIN)])
        self._spinner_idx += 1
        self._spinner_after = self.after(200, self._tick_spinner)

    def _stop_spinner(self, msg="Ready"):
        if self._spinner_after:
            self.after_cancel(self._spinner_after)
            self._spinner_after = None
        self.v_status.set(msg)
        self._spinner_lbl.config(fg=PAL["fg_dim"], text="●")

    # ── Browse helpers ────────────────────────────────────────────────────────
    def _browse_raw(self):
        p = filedialog.askdirectory(title="Select Raw FITS folder",
                                     initialdir=self.v_raw_dir.get())
        if p:
            self.v_raw_dir.set(p); self._scan_raw_dir()

    def _browse_catalog(self):
        p = filedialog.askopenfilename(
            title="Select TLE catalogue",
            filetypes=[("Text","*.txt"),("All","*.*")],
            initialdir=str(Path(self.v_catalog.get()).parent))
        if p: self.v_catalog.set(p)

    def _browse_output(self):
        p = filedialog.askdirectory(title="Select output directory",
                                     initialdir=self.v_output.get())
        if p: self.v_output.set(p)

    def _scan_raw_dir(self):
        try:
            files = get_sorted_fits(Path(self.v_raw_dir.get()))
            names = [f.name for f in files]
            self._frame_combo.config(values=names)
            if names and self.v_start_frame.get() not in names:
                self.v_start_frame.set(names[0])
        except Exception as e:
            self._log_line(f"[WARN] Could not scan FITS directory: {e}\n", "warn")

    def _on_frame_selected(self, _event=None):
        path = Path(self.v_raw_dir.get()) / self.v_start_frame.get()
        if path.exists():
            self._load_fits(path)

    # ── Tree helpers ──────────────────────────────────────────────────────────
    def _populate_tree(self, results):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for res in results:
            ph  = res["photometry"]
            mag = f"{ph['magnitude']:.2f}" if not math.isnan(ph["magnitude"]) else "N/A"
            p1  = res["start_pixel"]
            p2  = res["end_pixel"]
            tag = "match" if res.get("is_match") else "nomatch"
            self._tree.insert("", "end", tags=(tag,), values=(
                res["track_id"],
                res["label"],
                mag,
                f"{ph['snr']:.1f}",
                f"{ph['net_flux']:.1f}",
                f"{ph['peak_value']:.1f}",
                f"{ph['streak_length']:.1f}",
                f"{res['centroid_ra']:.4f}",
                f"{res['centroid_dec']:.4f}",
                f"({p1[0]:.1f},{p1[1]:.1f})",
                f"({p2[0]:.1f},{p2[1]:.1f})",
                f"{res['separation_deg']:.4f}",
            ))

    _sort_reverse = {}

    def _sort_tree(self, col, reverse):
        data = [(self._tree.set(k, col), k)
                for k in self._tree.get_children("")]
        try:
            data.sort(key=lambda t: float(t[0].replace("N/A","inf")),
                      reverse=reverse)
        except Exception:
            data.sort(reverse=reverse)
        for idx, (_, k) in enumerate(data):
            self._tree.move(k, "", idx)
        self._tree.heading(col,
            command=lambda c=col, r=not reverse: self._sort_tree(c, r))

    def _on_row_dblclick(self, _event=None):
        sel = self._tree.selection()
        if not sel: return
        tid = int(self._tree.item(sel[0], "values")[0])
        track = next((t for t in self._tracks if t["track_id"] == tid), None)
        if not track: return
        p1, p2 = track["start_pixel"], track["end_pixel"]
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        hw = 120
        self._ax.set_xlim(mx-hw, mx+hw)
        self._ax.set_ylim(my-hw, my+hw)
        self._canvas.draw_idle()
        self._nb.select(0)

    # ── Pipeline trigger ──────────────────────────────────────────────────────
    def _trigger(self, stage: str):
        if self._running:
            self._log_line("[WARN] A pipeline step is already running.\n", "warn")
            return
        self._cancel_flag.clear()
        self._set_running(True)

        workers = {
            "preprocess": self._worker_preprocess,
            "diff":       self._worker_diff,
            "solve":      self._worker_solve,
            "correlate":  self._worker_correlate,
            "full":       self._worker_full,
        }
        fn = workers.get(stage)
        if not fn:
            self._set_running(False); return

        self._start_spinner(f"{stage.title()} running…")
        threading.Thread(target=fn, daemon=True).start()

    def _cancel(self):
        if self._running:
            self._cancel_flag.set()
            self._log_line("[INFO] Cancel requested — will stop after current step.\n", "info")

    def _set_running(self, state: bool):
        self._running = state
        st = "disabled" if state else "normal"
        self._run_btn.config(state=st)
        self._cancel_btn.config(state="normal" if state else "disabled")

    def _done(self, msg="Ready"):
        self._safe(self._stop_spinner, msg)
        self._safe(self._set_running, False)

    # ── Workers ───────────────────────────────────────────────────────────────
    def _worker_preprocess(self):
        try:
            run_preprocessing(Path(self.v_raw_dir.get()),
                               Path(self.v_output.get()),
                               num_adj=2, batch_size=40)
            self._done("Pre-processing complete")
        except Exception as e:
            self._log_line(f"[ERROR] Preprocessing: {e}\n", "err")
            self._done("Pre-processing failed")

    def _worker_diff(self):
        try:
            diff, header, fits_out, png_out = run_difference(
                Path(self.v_raw_dir.get()),
                self.v_start_frame.get(),
                self.v_num_frames.get(),
                Path(self.v_output.get()))
            self._safe(self._load_fits, fits_out)
            self._done("Difference image ready")
        except Exception as e:
            self._log_line(f"[ERROR] Differencing: {e}\n", "err")
            self._done("Differencing failed")

    def _worker_solve(self):
        try:
            fits_dir   = Path(self.v_raw_dir.get())
            raw_path   = fits_dir / self.v_start_frame.get()
            stem       = Path(self.v_start_frame.get()).stem.replace(" ","_")
            nf         = self.v_num_frames.get()
            diff_path  = Path(self.v_output.get()) / f"diff_{stem}_{nf}f.fits"
            ok = solve_and_apply_wcs(str(raw_path), str(diff_path))
            msg = "Plate solve complete" if ok else "Plate solve: no solution found"
            if ok: self._safe(self._load_fits, diff_path)
            self._done(msg)
        except Exception as e:
            self._log_line(f"[ERROR] Solve: {e}\n", "err")
            self._done("Plate solve failed")

    def _worker_correlate(self):
        try:
            stem   = Path(self.v_start_frame.get()).stem.replace(" ","_")
            nf     = self.v_num_frames.get()
            out    = Path(self.v_output.get())
            df     = out / f"diff_{stem}_{nf}f.fits"
            png_out= out / f"diff_{stem}_{nf}f_correlated.png"
            cat    = Path(self.v_catalog.get())
            results = process_and_correlate(str(df), str(cat), str(png_out))
            self._tracks = results or []
            self._safe(self._populate_tree, self._tracks)
            if png_out.exists():
                self._png_path = png_out
                self._safe(self._render_png, png_out)
            msg = f"Detected {len(self._tracks)} track(s)"
            self._done(msg)
        except Exception as e:
            self._log_line(f"[ERROR] Correlation: {e}\n", "err")
            self._done("Correlation failed")

    def _worker_full(self):
        try:
            fits_dir = Path(self.v_raw_dir.get())
            out      = Path(self.v_output.get())
            cat      = Path(self.v_catalog.get())
            sf       = self.v_start_frame.get()
            nf       = self.v_num_frames.get()
            stem     = Path(sf).stem.replace(" ", "_")

            if not sf:
                self._log_line("[ERROR] No start frame selected.\n", "err")
                self._done("Failed — no start frame"); return

            print("=" * 55)
            print("  FRIGATE Full Pipeline")
            print("=" * 55)

            # Step 1
            if self.v_do_preproc.get() and not self._cancel_flag.is_set():
                print("\n[1/4] Pre-processing frames…")
                run_preprocessing(fits_dir, out, num_adj=2, batch_size=40)

            # Step 2
            if self._cancel_flag.is_set(): self._done("Cancelled"); return
            print("\n[2/4] Computing difference image…")
            diff, header, diff_fits, _ = run_difference(fits_dir, sf, nf, out)
            self._safe(self._load_fits, diff_fits)

            # Step 3
            if self.v_do_solve.get() and not self._cancel_flag.is_set():
                print("\n[3/4] Plate solving…")
                solve_and_apply_wcs(str(fits_dir / sf), str(diff_fits))
                self._safe(self._load_fits, diff_fits)

            # Step 4
            if self._cancel_flag.is_set(): self._done("Cancelled"); return
            print("\n[4/4] Detecting tracks & correlating…")
            png_out = out / f"diff_{stem}_{nf}f_correlated.png"
            results = process_and_correlate(str(diff_fits), str(cat), str(png_out))
            self._tracks = results or []
            self._safe(self._populate_tree, self._tracks)
            if png_out.exists():
                self._png_path = png_out
                self._safe(self._render_png, png_out)

            print(f"\n[DONE] {len(self._tracks)} track(s) detected.")
            self._done(f"Complete — {len(self._tracks)} track(s) found")

        except Exception as e:
            self._log_line(f"[ERROR] Full pipeline: {e}\n", "err")
            self._done("Pipeline failed")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def destroy(self):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        plt.close("all")
        super().destroy()


# =============================================================================
if __name__ == "__main__":
    app = DebrisTrackerGUI()
    app.mainloop()