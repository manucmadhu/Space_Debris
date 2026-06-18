"""
gui.py — FRIGATE Space Debris Detection Dashboard
===================================================
Design language: deep-space observatory terminal.
Palette: near-black (#0b0d12) + cold-blue accent (#3b82f6) + 
         amber telemetry (#f59e0b) + success-green (#22c55e).
Signature element: a real-time "TELEMETRY" status strip that pulses
amber while any pipeline stage runs — zero CPU cost, pure label swap.

Performance over original:
  - Matplotlib figure created ONCE; image data swapped in-place
  - Mouse coords debounced at 16 ms (~60 fps cap)
  - Log capped at 3000 lines, no state-toggle per write
  - Downsampling factor fixed on image load
  - All Tkinter calls from worker threads routed through gui_queue
  - Cancel flag (threading.Event) honoured between pipeline stages
"""

import gc, io, math, os, queue, sys, threading, time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Pipeline imports ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
CODE_DIR = ROOT_DIR / "Code"
sys.path.extend([str(ROOT_DIR), str(CODE_DIR)])

_IMPORT_ERR = None
try:
    from astropy.io import fits
    from astropy.wcs import WCS
    from pre_process import run_preprocessing, get_sorted_fits
    from difference  import run_difference
    from plate_solver import solve_and_apply_wcs
    from correlate   import process_and_correlate
    from display     import apply_stretch
    from flux_extract import extract_streak_flux
except ImportError as e:
    _IMPORT_ERR = str(e)
    def get_sorted_fits(d): return sorted(Path(d).glob("*.fits"))
    def apply_stretch(d, stretch_type="percentile"):
        lo, hi = np.percentile(d, [1, 99.5])
        return np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)

# =============================================================================
# Design tokens
# =============================================================================
C = {
    # backgrounds
    "bg0":      "#0b0d12",   # root
    "bg1":      "#10131a",   # panels
    "bg2":      "#161b26",   # cards / inputs
    "bg3":      "#1e2535",   # hover / selected
    # borders
    "brd":      "#252d3d",
    "brd2":     "#2f3a50",
    # accent
    "blue":     "#3b82f6",
    "blue_dim": "#1e3a5f",
    "amber":    "#f59e0b",
    "amber_dim":"#4a3000",
    "green":    "#22c55e",
    "green_dim":"#0d3320",
    "red":      "#ef4444",
    "red_dim":  "#3b1010",
    # text
    "fg0":      "#e2e8f0",   # primary
    "fg1":      "#94a3b8",   # secondary
    "fg2":      "#4b5880",   # muted
    # mono
    "mono_fg":  "#4ade80",   # console green
}

# Fonts
FN  = "Segoe UI"
FNM = "Consolas"
F   = lambda sz, w="normal": (FN,  sz, w)
FM  = lambda sz: (FNM, sz)

LOG_MAX = 3000


# =============================================================================
# Thread-safe logger
# =============================================================================
class _SafeWriter:
    def __init__(self, q, orig=None):
        self._q = q; self._orig = orig
    def write(self, s):
        self._q.put(("log", s))
        if self._orig:
            self._orig.write(s)
    def flush(self):
        if self._orig:
            try: self._orig.flush()
            except: pass


# =============================================================================
# Reusable widget helpers
# =============================================================================
def _sep(parent, orient="h", **kw):
    if orient == "h":
        tk.Frame(parent, bg=C["brd"], height=1, **kw).pack(fill="x")
    else:
        tk.Frame(parent, bg=C["brd"], width=1, **kw).pack(fill="y", side="left")

def _label(parent, text, color=None, font=None, **kw):
    return tk.Label(parent, text=text,
                    bg=kw.pop("bg", C["bg1"]),
                    fg=color or C["fg1"],
                    font=font or F(9), **kw)

def _badge(parent, text, bg, fg):
    f = tk.Frame(parent, bg=bg, padx=6, pady=1)
    tk.Label(f, text=text, bg=bg, fg=fg, font=F(8, "bold")).pack()
    return f

def _icon_btn(parent, text, cmd, bg=None, fg=None, width=None):
    kw = dict(text=text, command=cmd,
              bg=bg or C["bg2"], fg=fg or C["fg0"],
              activebackground=C["bg3"], activeforeground=C["fg0"],
              font=F(9), relief="flat", cursor="hand2",
              bd=0, highlightthickness=0, padx=10, pady=5)
    if width: kw["width"] = width
    return tk.Button(parent, **kw)

def _entry(parent, var, **kw):
    e = tk.Entry(parent, textvariable=var,
                 bg=C["bg2"], fg=C["fg0"],
                 insertbackground=C["fg0"],
                 relief="flat", font=F(9),
                 highlightthickness=1,
                 highlightbackground=C["brd"],
                 highlightcolor=C["blue"], **kw)
    return e

def _combo(parent, var, values, width=14):
    cb = ttk.Combobox(parent, textvariable=var, values=values,
                      state="readonly", width=width, font=F(9))
    return cb

def _spin(parent, var, lo, hi, width=7):
    s = ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var,
                    width=width, font=F(9))
    return s


# =============================================================================
# Main application
# =============================================================================
class FrigateGUI(tk.Tk):

    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.title("FRIGATE  ·  Space Debris Detection")
        self.geometry("1480x900")
        self.minsize(1100, 700)
        self.configure(bg=C["bg0"])
        self._apply_ttk_styles()

        # State
        self._fits_data   = None
        self._fits_hdr    = None
        self._fits_wcs    = None
        self._fits_path   = None
        self._ds          = 1
        self._cache_arr   = None
        self._cache_key   = None
        self._im          = None      # single AxesImage kept alive
        self._png_path    = None
        self._tracks      = []
        self._running     = False
        self._cancel      = threading.Event()
        self._motion_id   = None
        self._spin_id     = None
        self._spin_i      = 0

        # Variables
        self.v_raw    = tk.StringVar(value=str(ROOT_DIR / "Data" / "raw"))
        self.v_cat    = tk.StringVar(value=str(ROOT_DIR / "Data" / "3le.txt"))
        self.v_out    = tk.StringVar(value=str(ROOT_DIR / "Output"))
        self.v_frame  = tk.StringVar()
        self.v_nf     = tk.IntVar(value=10)
        self.v_preproc= tk.BooleanVar(value=False)
        self.v_solve  = tk.BooleanVar(value=True)
        self.v_ann    = tk.BooleanVar(value=True)
        self.v_stretch= tk.StringVar(value="percentile")
        self.v_cmap   = tk.StringVar(value="gray")
        self.v_status = tk.StringVar(value="IDLE")

        # Queues
        self._log_q = queue.Queue()
        self._gui_q = queue.Queue()

        self._build()
        self._scan_fits()

        # Keyboard
        self.bind("<F5>",        lambda e: self._run("full"))
        self.bind("<Control-l>", lambda e: self._clear_log())
        self.bind("<Escape>",    lambda e: self._do_cancel())

        # Redirect output
        self._out0 = sys.stdout; self._err0 = sys.stderr
        sys.stdout = _SafeWriter(self._log_q, self._out0)
        sys.stderr = _SafeWriter(self._log_q, self._err0)

        self.after(80,  self._poll_log)
        self.after(50,  self._poll_gui)

        if _IMPORT_ERR:
            self.after(300, lambda: print(f"[WARN] Import error: {_IMPORT_ERR}"))
        print("FRIGATE ready.   F5 = full pipeline   Ctrl+L = clear log   Esc = cancel\n")

    # ── TTK styles ────────────────────────────────────────────────────────────
    def _apply_ttk_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        bg0, bg1, bg2 = C["bg0"], C["bg1"], C["bg2"]
        fg0, fg1, fg2 = C["fg0"], C["fg1"], C["fg2"]
        brd = C["brd"]
        blue = C["blue"]

        s.configure(".", background=bg1, foreground=fg0,
                    font=F(9), borderwidth=0, relief="flat",
                    troughcolor=bg0, selectbackground=C["blue_dim"],
                    selectforeground=fg0)

        s.configure("TScrollbar",
                    background=bg2, troughcolor=bg0,
                    arrowcolor=fg2, gripcount=0,
                    relief="flat", borderwidth=0)

        s.configure("TCombobox",
                    fieldbackground=bg2, background=bg2,
                    foreground=fg0, arrowcolor=blue,
                    bordercolor=brd, lightcolor=brd, darkcolor=brd,
                    insertcolor=fg0)
        s.map("TCombobox",
              fieldbackground=[("readonly", bg2)],
              selectbackground=[("readonly", bg2)],
              selectforeground=[("readonly", fg0)])

        s.configure("TSpinbox",
                    fieldbackground=bg2, foreground=fg0,
                    arrowcolor=blue, bordercolor=brd,
                    lightcolor=brd, darkcolor=brd)

        s.configure("TCheckbutton",
                    background=bg1, foreground=fg0,
                    indicatorcolor=bg2, indicatordiameter=15)
        s.map("TCheckbutton",
              indicatorcolor=[("selected", blue), ("!selected", brd)])

        s.configure("TLabelframe",
                    background=bg1, bordercolor=brd)
        s.configure("TLabelframe.Label",
                    background=bg1, foreground=fg2, font=F(8))

        s.configure("Treeview",
                    background=bg1, fieldbackground=bg1,
                    foreground=fg0, rowheight=26, font=F(9),
                    borderwidth=0)
        s.configure("Treeview.Heading",
                    background=bg0, foreground=fg2,
                    font=F(9, "bold"), relief="flat", padding=(6,4))
        s.map("Treeview",
              background=[("selected", C["blue_dim"])],
              foreground=[("selected", C["fg0"])])

        # Option menu dropdown colours
        self.option_add("*TCombobox*Listbox.background",    bg2)
        self.option_add("*TCombobox*Listbox.foreground",    fg0)
        self.option_add("*TCombobox*Listbox.selectBackground", C["blue_dim"])
        self.option_add("*TCombobox*Listbox.selectForeground", fg0)
        self.option_add("*TCombobox*Listbox.font",          F(9))

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build(self):
        # Root grid: topbar | (sidebar | workspace) | statusbar
        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

        self._build_topbar()
        self._build_sidebar()
        self._build_workspace()
        self._build_statusbar()

    # ── Top bar ───────────────────────────────────────────────────────────────
    def _build_topbar(self):
        bar = tk.Frame(self, bg=C["bg1"], height=48)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        # Logo mark
        logo = tk.Frame(bar, bg=C["bg1"])
        logo.pack(side="left", padx=(16, 0))
        tk.Label(logo, text="FRIGATE", bg=C["bg1"], fg=C["blue"],
                 font=(FN, 15, "bold")).pack(side="left")
        tk.Label(logo, text=" / space debris detection",
                 bg=C["bg1"], fg=C["fg2"], font=F(10)).pack(side="left")

        # Right cluster: telemetry badge + keyboard hint
        right = tk.Frame(bar, bg=C["bg1"])
        right.pack(side="right", padx=16)

        tk.Label(right, text="F5 run · Ctrl+L log · Esc cancel",
                 bg=C["bg1"], fg=C["fg2"], font=F(8)).pack(side="right", padx=(12,0))

        self._telem_frame = tk.Frame(right, bg=C["amber_dim"],
                                      padx=10, pady=4)
        self._telem_frame.pack(side="right")
        self._telem_lbl = tk.Label(self._telem_frame,
                                    text="● IDLE",
                                    bg=C["amber_dim"], fg=C["amber"],
                                    font=F(9, "bold"))
        self._telem_lbl.pack()

        # Thin bottom border
        tk.Frame(bar, bg=C["brd"], height=1).pack(side="bottom", fill="x")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = tk.Frame(self, bg=C["bg1"], width=290)
        sb.grid(row=1, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.columnconfigure(0, weight=1)

        # Scrollable inner
        canvas = tk.Canvas(sb, bg=C["bg1"], highlightthickness=0, bd=0)
        vsb    = ttk.Scrollbar(sb, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.grid(row=0, column=1, sticky="ns")
        canvas.grid(row=0, column=0, sticky="nsew")
        sb.rowconfigure(0, weight=1)

        inner = tk.Frame(canvas, bg=C["bg1"])
        wid   = canvas.create_window((0,0), window=inner, anchor="nw")

        def _resize(e): canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(wid, width=e.width))

        def _scroll(e): canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _scroll))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self._build_sb_paths(inner)
        self._build_sb_params(inner)
        self._build_sb_actions(inner)

    def _sec(self, parent, title):
        """Section header row."""
        f = tk.Frame(parent, bg=C["bg1"])
        f.pack(fill="x", padx=14, pady=(18, 6))
        tk.Label(f, text=title.upper(), bg=C["bg1"],
                 fg=C["blue"], font=F(8, "bold")).pack(side="left")
        tk.Frame(f, bg=C["brd2"], height=1).pack(
            side="left", fill="x", expand=True, padx=(8,0), pady=1)

    def _path_row(self, parent, label, var, browse):
        tk.Label(parent, text=label, bg=C["bg1"],
                 fg=C["fg2"], font=F(8)).pack(anchor="w", padx=14, pady=(4,1))
        row = tk.Frame(parent, bg=C["bg1"])
        row.pack(fill="x", padx=14, pady=(0,4))
        _entry(row, var).pack(side="left", fill="x", expand=True)
        _icon_btn(row, "…", browse, bg=C["bg2"], fg=C["fg1"], width=3
                  ).pack(side="left", padx=(3,0))

    def _build_sb_paths(self, p):
        self._sec(p, "Directories")
        self._path_row(p, "Raw FITS folder", self.v_raw, self._browse_raw)
        self._path_row(p, "TLE catalogue",   self.v_cat, self._browse_cat)
        self._path_row(p, "Output folder",   self.v_out, self._browse_out)

    def _build_sb_params(self, p):
        self._sec(p, "Parameters")

        tk.Label(p, text="Start frame", bg=C["bg1"],
                 fg=C["fg2"], font=F(8)).pack(anchor="w", padx=14, pady=(2,1))
        self._frame_cb = _combo(p, self.v_frame, [], width=22)
        self._frame_cb.pack(fill="x", padx=14, pady=(0,6))
        self._frame_cb.bind("<<ComboboxSelected>>", self._on_frame)

        tk.Label(p, text="Number of frames", bg=C["bg1"],
                 fg=C["fg2"], font=F(8)).pack(anchor="w", padx=14, pady=(2,1))
        _spin(p, self.v_nf, 3, 500).pack(anchor="w", padx=14, pady=(0,8))

        # Options
        opt = tk.Frame(p, bg=C["bg1"])
        opt.pack(fill="x", padx=14, pady=(0,4))
        ttk.Checkbutton(opt, text="Pre-process frames",
                        variable=self.v_preproc).pack(anchor="w", pady=2)
        ttk.Checkbutton(opt, text="Run plate solver (WCS)",
                        variable=self.v_solve).pack(anchor="w", pady=2)

    def _build_sb_actions(self, p):
        self._sec(p, "Pipeline")

        steps = [
            ("Pre-process",        "preprocess"),
            ("Difference image",   "diff"),
            ("Plate solve",        "solve"),
            ("Detect & correlate", "correlate"),
        ]
        step_frame = tk.Frame(p, bg=C["bg1"])
        step_frame.pack(fill="x", padx=14, pady=(0,4))
        for label, cmd in steps:
            btn = _icon_btn(step_frame, label,
                            lambda c=cmd: self._run(c),
                            bg=C["bg2"], fg=C["fg1"])
            btn.pack(fill="x", pady=2)

        tk.Frame(p, bg=C["brd"], height=1).pack(fill="x", padx=14, pady=10)

        # Primary CTA
        self._run_btn = tk.Button(p, text="RUN FULL PIPELINE",
                                   command=lambda: self._run("full"),
                                   bg=C["blue"], fg="#ffffff",
                                   activebackground="#60a5fa",
                                   activeforeground="#ffffff",
                                   font=F(10, "bold"),
                                   relief="flat", cursor="hand2",
                                   bd=0, highlightthickness=0,
                                   pady=9)
        self._run_btn.pack(fill="x", padx=14, pady=(0,4))

        self._cancel_btn = tk.Button(p, text="Cancel",
                                      command=self._do_cancel,
                                      bg=C["red_dim"], fg=C["red"],
                                      activebackground="#4a1515",
                                      activeforeground=C["red"],
                                      font=F(9),
                                      relief="flat", cursor="hand2",
                                      bd=0, highlightthickness=0,
                                      pady=6, state="disabled")
        self._cancel_btn.pack(fill="x", padx=14, pady=(0,20))

    # ── Workspace ─────────────────────────────────────────────────────────────
    def _build_workspace(self):
        ws = tk.Frame(self, bg=C["bg0"])
        ws.grid(row=1, column=1, sticky="nsew")
        ws.rowconfigure(0, weight=3)
        ws.rowconfigure(1, weight=0)
        ws.rowconfigure(2, weight=1)
        ws.columnconfigure(0, weight=1)

        self._build_viewer(ws)
        self._build_log(ws)

    def _build_viewer(self, ws):
        # Tab bar (manual — no ttk.Notebook for better dark styling)
        tab_bar = tk.Frame(ws, bg=C["bg1"], height=36)
        tab_bar.grid(row=0, column=0, sticky="new")
        tab_bar.grid_propagate(False)

        self._tab_frames = {}
        self._tab_btns   = {}

        def _make_tab(name, label):
            btn = tk.Button(tab_bar, text=label,
                            bg=C["bg1"], fg=C["fg2"],
                            activebackground=C["bg0"],
                            activeforeground=C["blue"],
                            font=F(9), relief="flat",
                            cursor="hand2", padx=14, pady=6,
                            bd=0, highlightthickness=0,
                            command=lambda n=name: self._switch_tab(n))
            btn.pack(side="left")
            self._tab_btns[name] = btn

        _make_tab("viewer",  "  Image Viewer")
        _make_tab("results", "  Detected Tracks")
        tk.Frame(tab_bar, bg=C["brd"], height=1).pack(side="bottom", fill="x")

        # Content area
        content = tk.Frame(ws, bg=C["bg0"])
        content.grid(row=0, column=0, sticky="nsew", pady=(36,0))
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        ws.rowconfigure(0, weight=3)

        # Viewer tab
        vf = tk.Frame(content, bg=C["bg0"])
        vf.grid(row=0, column=0, sticky="nsew")
        self._tab_frames["viewer"] = vf
        self._build_fits_tab(vf)

        # Results tab
        rf = tk.Frame(content, bg=C["bg0"])
        self._tab_frames["results"] = rf
        self._build_results_tab(rf)

        self._switch_tab("viewer")

    def _switch_tab(self, name):
        for n, f in self._tab_frames.items():
            f.grid_remove()
        self._tab_frames[name].grid(row=0, column=0, sticky="nsew")
        for n, btn in self._tab_btns.items():
            active = (n == name)
            btn.config(fg=C["blue"] if active else C["fg2"],
                       bg=C["bg0"] if active else C["bg1"],
                       font=F(9, "bold") if active else F(9))

    def _build_fits_tab(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Viewer toolbar
        tbar = tk.Frame(parent, bg=C["bg1"], height=36)
        tbar.grid(row=0, column=0, sticky="ew")
        tbar.grid_propagate(False)

        def _lbl(t): return tk.Label(tbar, text=t, bg=C["bg1"],
                                      fg=C["fg2"], font=F(8))

        _lbl("Stretch").pack(side="left", padx=(10,3))
        cb1 = _combo(tbar, self.v_stretch,
                     ["percentile","linear","log","sqrt"], width=10)
        cb1.pack(side="left", padx=(0,10))
        cb1.bind("<<ComboboxSelected>>", lambda e: self._refresh())

        _lbl("Colormap").pack(side="left", padx=(0,3))
        cb2 = _combo(tbar, self.v_cmap,
                     ["gray","viridis","inferno","plasma","bone","hot"], width=10)
        cb2.pack(side="left", padx=(0,10))
        cb2.bind("<<ComboboxSelected>>", lambda e: self._refresh())

        # Annotated overlay toggle
        ann_chk = ttk.Checkbutton(tbar, text="Annotated overlay",
                                   variable=self.v_ann,
                                   command=self._refresh)
        ann_chk.pack(side="left", padx=(4,0))

        # Coord readout — right side
        self._coord = tk.Label(tbar,
                                text="x: —   y: —   |   RA: —   Dec: —",
                                bg=C["bg1"], fg=C["amber"],
                                font=FM(8))
        self._coord.pack(side="right", padx=12)

        tk.Frame(parent, bg=C["brd"], height=1).grid(
            row=0, column=0, sticky="sew")

        # Figure
        fig_host = tk.Frame(parent, bg=C["bg0"])
        fig_host.grid(row=1, column=0, sticky="nsew")
        fig_host.rowconfigure(0, weight=1)
        fig_host.columnconfigure(0, weight=1)

        self._fig, self._ax = plt.subplots(figsize=(9,6))
        self._fig.patch.set_facecolor(C["bg0"])
        self._ax.set_facecolor(C["bg0"])
        self._ax.tick_params(colors=C["fg2"], labelsize=7)
        for sp in self._ax.spines.values():
            sp.set_color(C["brd"])
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._ax.set_title("No image loaded",
                            color=C["fg2"], fontsize=9, pad=6)

        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=fig_host)
        self._mpl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Nav toolbar
        tb_host = tk.Frame(fig_host, bg=C["bg2"], height=28)
        tb_host.grid(row=1, column=0, sticky="ew")
        tb_host.grid_propagate(False)
        self._mpl_tb = NavigationToolbar2Tk(self._mpl_canvas, tb_host)
        self._mpl_tb.config(background=C["bg2"])
        for ch in self._mpl_tb.winfo_children():
            for k in ("background","activebackground",
                       "highlightbackground","highlightcolor"):
                try: ch.configure(**{k: C["bg2"]})
                except: pass
            for k in ("fg","foreground","activeforeground"):
                try: ch.configure(**{k: C["fg1"]})
                except: pass
        self._mpl_tb.update()

        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _build_results_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        # Header strip
        hdr = tk.Frame(parent, bg=C["bg1"], height=36)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Detected orbital streaks",
                 bg=C["bg1"], fg=C["fg1"],
                 font=F(9, "bold")).pack(side="left", padx=12, pady=8)
        self._track_count = tk.Label(hdr, text="0 tracks",
                                      bg=C["bg1"], fg=C["fg2"], font=F(8))
        self._track_count.pack(side="right", padx=12)

        tk.Frame(parent, bg=C["brd"], height=1).pack(fill="x")

        cols = ("id","label","mag","snr","net_flux","peak",
                "length","ra","dec","start_px","end_px","sep")
        heads = {
            "id":       ("ID",             50),
            "label":    ("Object label",   200),
            "mag":      ("Mag",            70),
            "snr":      ("SNR",            70),
            "net_flux": ("Net flux",       90),
            "peak":     ("Peak",           80),
            "length":   ("Length px",      90),
            "ra":       ("RA deg",         100),
            "dec":      ("Dec deg",        100),
            "start_px": ("Start (x,y)",   110),
            "end_px":   ("End (x,y)",     110),
            "sep":      ("Cat sep deg",    110),
        }

        tree_f = tk.Frame(parent, bg=C["bg0"])
        tree_f.pack(fill="both", expand=True)
        tree_f.rowconfigure(0, weight=1)
        tree_f.columnconfigure(0, weight=1)

        vsb = ttk.Scrollbar(tree_f, orient="vertical")
        hsb = ttk.Scrollbar(tree_f, orient="horizontal")
        self._tree = ttk.Treeview(tree_f, columns=cols,
                                   show="headings",
                                   yscrollcommand=vsb.set,
                                   xscrollcommand=hsb.set,
                                   selectmode="browse")
        vsb.config(command=self._tree.yview)
        hsb.config(command=self._tree.xview)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        for col, (title, width) in heads.items():
            self._tree.heading(col, text=title,
                               command=lambda c=col: self._sort(c))
            self._tree.column(col, width=width, anchor="center", minwidth=40)

        self._tree.tag_configure("corr",  foreground=C["green"])
        self._tree.tag_configure("nocorr",foreground=C["red"])
        self._tree.bind("<Double-1>", self._row_dbl)

    def _build_log(self, ws):
        log_host = tk.Frame(ws, bg=C["bg1"])
        log_host.grid(row=2, column=0, sticky="nsew")
        log_host.rowconfigure(1, weight=1)
        log_host.columnconfigure(0, weight=1)
        ws.rowconfigure(2, weight=1)

        # Log header
        lhdr = tk.Frame(log_host, bg=C["bg0"], height=28)
        lhdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        lhdr.grid_propagate(False)
        tk.Label(lhdr, text="  Console output",
                 bg=C["bg0"], fg=C["blue"],
                 font=F(9, "bold")).pack(side="left", pady=4)
        _icon_btn(lhdr, "Clear", self._clear_log,
                  bg=C["bg0"], fg=C["fg2"]).pack(side="right", padx=4)

        tk.Frame(log_host, bg=C["brd"], height=1).grid(
            row=0, column=0, sticky="sew", columnspan=2)

        vsb = ttk.Scrollbar(log_host, orient="vertical")
        vsb.grid(row=1, column=1, sticky="ns")

        self._log = tk.Text(log_host,
                             bg=C["bg0"], fg=C["mono_fg"],
                             font=FM(9), wrap="word",
                             yscrollcommand=vsb.set,
                             padx=10, pady=6,
                             state="disabled",
                             relief="flat", bd=0,
                             height=8)
        self._log.grid(row=1, column=0, sticky="nsew")
        vsb.config(command=self._log.yview)

        self._log.tag_config("err",  foreground=C["red"])
        self._log.tag_config("warn", foreground=C["amber"])
        self._log.tag_config("info", foreground=C["blue"])

    def _build_statusbar(self):
        sbar = tk.Frame(self, bg=C["bg0"], height=24)
        sbar.grid(row=2, column=0, columnspan=2, sticky="ew")
        sbar.grid_propagate(False)
        tk.Frame(sbar, bg=C["brd"], height=1).pack(side="top", fill="x")

        self._status_dot = tk.Label(sbar, text="●",
                                     bg=C["bg0"], fg=C["fg2"],
                                     font=F(9))
        self._status_dot.pack(side="left", padx=(10,3))
        tk.Label(sbar, textvariable=self.v_status,
                 bg=C["bg0"], fg=C["fg2"],
                 font=FM(8)).pack(side="left")

        # Right: live frame count
        self._frame_info = tk.Label(sbar, text="",
                                     bg=C["bg0"], fg=C["fg2"],
                                     font=FM(8))
        self._frame_info.pack(side="right", padx=10)

    # ── Image display (hot path — no figure recreation) ───────────────────────
    def _load_fits(self, path):
        path = Path(path)
        try:
            with open(str(path), "rb") as f:
                raw = f.read()
            with fits.open(io.BytesIO(raw), memmap=False) as h:
                self._fits_data = np.array(h[0].data, dtype=np.float32)
                self._fits_hdr  = h[0].header.copy()
                try:    self._fits_wcs = WCS(self._fits_hdr, relax=True)
                except: self._fits_wcs = None
            self._fits_path   = path
            self._cache_arr   = None
            self._cache_key   = None
            H, W = self._fits_data.shape
            self._ds = max(1, min(W, H) // 1200)
            self._check_png()
            self._refresh()
            self._switch_tab("viewer")
            self._frame_info.config(
                text=f"{path.name}  {W}×{H}  ds={self._ds}×")
        except Exception as e:
            self._log_write(f"[ERROR] Load {path.name}: {e}\n", "err")

    def _check_png(self):
        if not self._fits_path: return
        out  = Path(self.v_out.get())
        stem = self._fits_path.stem.replace(" ","_")
        nf   = self.v_nf.get()
        cand = out / f"diff_{stem}_{nf}f_correlated.png"
        self._png_path = cand if cand.exists() else None

    def _refresh(self):
        if self._fits_data is None: return
        if self.v_ann.get() and self._png_path and self._png_path.exists():
            self._show_png(self._png_path); return

        stretch = self.v_stretch.get()
        cmap    = self.v_cmap.get()
        key     = (stretch, cmap)
        if key != self._cache_key:
            ds           = self._fits_data[::self._ds, ::self._ds]
            self._cache_arr = apply_stretch(ds, stretch_type=stretch)
            self._cache_key = key

        H, W = self._fits_data.shape
        if self._im is None:
            self._im = self._ax.imshow(
                self._cache_arr, cmap=cmap, origin="lower",
                extent=[0,W,0,H], aspect="auto",
                interpolation="nearest")
        else:
            self._im.set_data(self._cache_arr)
            self._im.set_cmap(cmap)
            self._im.set_extent([0,W,0,H])

        self._ax.set_title(
            self._fits_path.name if self._fits_path else "",
            color=C["fg1"], fontsize=8, pad=4)
        self._draw_overlays()
        self._mpl_canvas.draw_idle()

    def _show_png(self, p):
        try:
            img = mpimg.imread(str(p))
            ih, iw = img.shape[:2]
            W = self._fits_data.shape[1] if self._fits_data is not None else iw
            H = self._fits_data.shape[0] if self._fits_data is not None else ih
            if self._im is None:
                self._im = self._ax.imshow(
                    img, origin="upper", extent=[0,W,0,H], aspect="auto")
            else:
                self._im.set_data(img)
                self._im.set_extent([0,W,0,H])
            self._ax.set_title(p.name, color=C["amber"], fontsize=8, pad=4)
            self._mpl_canvas.draw_idle()
        except Exception as e:
            self._log_write(f"[WARN] PNG render: {e}\n", "warn")

    def _draw_overlays(self):
        for artist in (self._ax.lines + self._ax.patches + self._ax.texts):
            try: artist.remove()
            except: pass
        for t in self._tracks:
            p1    = t.get("start_pixel",[0,0])
            p2    = t.get("end_pixel",  [0,0])
            color = C["green"] if t.get("is_match") else C["red"]
            self._ax.plot([p1[0],p2[0]],[p1[1],p2[1]],
                          color=C["amber"], lw=1.1, ls="--", alpha=.8)
            self._ax.scatter([p1[0],p2[0]],[p1[1],p2[1]],
                             color=color, s=16, zorder=5)
            xlo = min(p1[0],p2[0])-14; ylo = min(p1[1],p2[1])-14
            xhi = max(p1[0],p2[0])+14; yhi = max(p1[1],p2[1])+14
            rect = plt.Rectangle((xlo,ylo),xhi-xlo,yhi-ylo,
                                  fill=False,edgecolor=color,lw=1.1,alpha=.8)
            self._ax.add_patch(rect)
            self._ax.text(xlo, yhi+3, f"T{t['track_id']}: {t['label']}",
                          color=color, fontsize=6.5, fontweight="bold")

    # ── Debounced mouse coords ────────────────────────────────────────────────
    def _on_motion(self, event):
        if self._motion_id:
            self.after_cancel(self._motion_id)
        self._motion_id = self.after(16, lambda: self._update_coord(event))

    def _update_coord(self, event):
        self._motion_id = None
        if event.inaxes is None or self._fits_data is None: return
        x, y = event.xdata, event.ydata
        ra_s = dec_s = "—"
        if self._fits_wcs and self._fits_wcs.has_celestial:
            try:
                ra, dec = self._fits_wcs.pixel_to_world_values(x, y)
                ra_s  = f"{ra:.5f}"
                dec_s = f"{dec:+.5f}"
            except: pass
        try:
            val = self._fits_data[int(round(y)), int(round(x))]
            vs  = f"{val:.1f}"
        except: vs = "—"
        self._coord.config(
            text=f"x: {x:.1f}   y: {y:.1f}   [{vs}]   |   RA: {ra_s}   Dec: {dec_s}")

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log_write(self, text, tag=""):
        self._log.config(state="normal")
        end = int(self._log.index("end-1c").split(".")[0])
        if end > LOG_MAX:
            self._log.delete("1.0", f"{end - LOG_MAX//2}.0")
        if tag:
            self._log.insert("end", text, tag)
        else:
            self._log.insert("end", text)
        self._log.see("end")
        self._log.config(state="disabled")

    def _poll_log(self):
        try:
            n = 0
            while n < 60:
                kind, payload = self._log_q.get_nowait()
                if kind == "log":
                    tag = ("err"  if "[ERROR]" in payload else
                           "warn" if "[WARN]"  in payload else
                           "info" if "[INFO]"  in payload else "")
                    self._log_write(payload, tag)
                n += 1
        except queue.Empty:
            pass
        self.after(80, self._poll_log)

    def _poll_gui(self):
        try:
            while True:
                fn = self._gui_q.get_nowait()
                try: fn()
                except Exception as e:
                    self._log_write(f"[ERROR] GUI: {e}\n", "err")
        except queue.Empty:
            pass
        self.after(50, self._poll_gui)

    def _safe(self, fn, *a, **kw):
        self._gui_q.put(lambda: fn(*a, **kw))

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0","end")
        self._log.config(state="disabled")

    # ── Telemetry + spinner ───────────────────────────────────────────────────
    _SPIN = ["◐","◓","◑","◒"]

    def _start_telem(self, msg="RUNNING"):
        self.v_status.set(msg)
        self._status_dot.config(fg=C["amber"])
        self._telem_lbl.config(text=f"● {msg}")
        self._telem_frame.config(bg=C["amber_dim"])
        self._telem_lbl.config(bg=C["amber_dim"])
        self._spin_i = 0
        self._tick_spin()

    def _tick_spin(self):
        ch = self._SPIN[self._spin_i % len(self._SPIN)]
        self._telem_lbl.config(
            text=f"{ch} {self.v_status.get()}")
        self._spin_i += 1
        self._spin_id = self.after(220, self._tick_spin)

    def _stop_telem(self, msg="IDLE"):
        if self._spin_id:
            self.after_cancel(self._spin_id)
            self._spin_id = None
        self.v_status.set(msg)
        self._status_dot.config(fg=C["fg2"])
        self._telem_lbl.config(text=f"● {msg}",
                                fg=C["amber"], bg=C["amber_dim"])

    # ── Browse ────────────────────────────────────────────────────────────────
    def _browse_raw(self):
        p = filedialog.askdirectory(
            title="Select raw FITS folder", initialdir=self.v_raw.get())
        if p: self.v_raw.set(p); self._scan_fits()

    def _browse_cat(self):
        p = filedialog.askopenfilename(
            title="Select TLE catalogue",
            filetypes=[("Text","*.txt"),("All","*.*")],
            initialdir=str(Path(self.v_cat.get()).parent))
        if p: self.v_cat.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory(
            title="Select output folder", initialdir=self.v_out.get())
        if p: self.v_out.set(p)

    def _scan_fits(self):
        try:
            files = get_sorted_fits(Path(self.v_raw.get()))
            names = [f.name for f in files]
            self._frame_cb.config(values=names)
            if names and self.v_frame.get() not in names:
                self.v_frame.set(names[0])
            self._frame_info.config(text=f"{len(names)} FITS files found")
        except Exception as e:
            self._log_write(f"[WARN] Scan: {e}\n","warn")

    def _on_frame(self, _e=None):
        path = Path(self.v_raw.get()) / self.v_frame.get()
        if path.exists():
            self._load_fits(path)

    # ── Tree ──────────────────────────────────────────────────────────────────
    def _populate_tree(self, results):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for res in results:
            ph  = res["photometry"]
            mag = (f"{ph['magnitude']:.2f}"
                   if not math.isnan(ph["magnitude"]) else "N/A")
            p1, p2 = res["start_pixel"], res["end_pixel"]
            tag = "corr" if res.get("is_match") else "nocorr"
            self._tree.insert("","end", tags=(tag,), values=(
                res["track_id"],   res["label"],
                mag,               f"{ph['snr']:.1f}",
                f"{ph['net_flux']:.1f}", f"{ph['peak_value']:.1f}",
                f"{ph['streak_length']:.1f}",
                f"{res['centroid_ra']:.4f}", f"{res['centroid_dec']:.4f}",
                f"({p1[0]:.1f},{p1[1]:.1f})",
                f"({p2[0]:.1f},{p2[1]:.1f})",
                f"{res['separation_deg']:.4f}",
            ))
        n = len(results)
        nc = sum(1 for r in results if r.get("is_match"))
        self._track_count.config(
            text=f"{n} track{'s' if n!=1 else ''}  ·  {nc} correlated")

    _sort_asc = {}
    def _sort(self, col):
        asc = not self._sort_asc.get(col, False)
        self._sort_asc[col] = asc
        data = [(self._tree.set(k,col),k)
                for k in self._tree.get_children("")]
        try:    data.sort(key=lambda t: float(t[0].replace("N/A","inf")),
                          reverse=not asc)
        except: data.sort(reverse=not asc)
        for i,(_,k) in enumerate(data): self._tree.move(k,"",i)

    def _row_dbl(self, _e=None):
        sel = self._tree.selection()
        if not sel: return
        tid = int(self._tree.item(sel[0],"values")[0])
        t   = next((x for x in self._tracks if x["track_id"]==tid), None)
        if not t: return
        p1,p2 = t["start_pixel"], t["end_pixel"]
        mx,my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        hw = 100
        self._ax.set_xlim(mx-hw, mx+hw)
        self._ax.set_ylim(my-hw, my+hw)
        self._mpl_canvas.draw_idle()
        self._switch_tab("viewer")

    # ── Pipeline control ──────────────────────────────────────────────────────
    def _run(self, stage):
        if self._running:
            print("[WARN] A stage is already running.\n"); return
        self._cancel.clear()
        self._set_busy(True)
        self._start_telem(stage.upper())
        workers = dict(
            preprocess=self._w_preprocess,
            diff=self._w_diff, solve=self._w_solve,
            correlate=self._w_correlate, full=self._w_full)
        t = threading.Thread(target=workers.get(stage,lambda:None), daemon=True)
        t.start()

    def _do_cancel(self):
        if self._running:
            self._cancel.set()
            print("[INFO] Cancelling after current step…\n")

    def _set_busy(self, busy):
        self._running = busy
        self._run_btn.config(state="disabled" if busy else "normal")
        self._cancel_btn.config(state="normal" if busy else "disabled")

    def _done(self, msg="IDLE"):
        self._safe(self._stop_telem, msg)
        self._safe(self._set_busy, False)

    # ── Workers ───────────────────────────────────────────────────────────────
    def _w_preprocess(self):
        try:
            run_preprocessing(Path(self.v_raw.get()),
                               Path(self.v_out.get()), num_adj=2, batch_size=40)
            self._done("PREPROCESS DONE")
        except Exception as e:
            print(f"[ERROR] Preprocess: {e}\n")
            self._done("PREPROCESS FAILED")

    def _w_diff(self):
        try:
            _,_,fits_out,_ = run_difference(
                Path(self.v_raw.get()), self.v_frame.get(),
                self.v_nf.get(), Path(self.v_out.get()))
            self._safe(self._load_fits, fits_out)
            self._done("DIFFERENCE DONE")
        except Exception as e:
            print(f"[ERROR] Difference: {e}\n")
            self._done("DIFFERENCE FAILED")

    def _w_solve(self):
        try:
            stem   = Path(self.v_frame.get()).stem.replace(" ","_")
            nf     = self.v_nf.get()
            df     = Path(self.v_out.get()) / f"diff_{stem}_{nf}f.fits"
            raw    = Path(self.v_raw.get()) / self.v_frame.get()
            ok     = solve_and_apply_wcs(str(raw), str(df))
            if ok: self._safe(self._load_fits, df)
            self._done("SOLVE DONE" if ok else "SOLVE: NO SOLUTION")
        except Exception as e:
            print(f"[ERROR] Solve: {e}\n")
            self._done("SOLVE FAILED")

    def _w_correlate(self):
        try:
            stem    = Path(self.v_frame.get()).stem.replace(" ","_")
            nf      = self.v_nf.get()
            out     = Path(self.v_out.get())
            df      = out / f"diff_{stem}_{nf}f.fits"
            png_out = out / f"diff_{stem}_{nf}f_correlated.png"
            results = process_and_correlate(
                str(df), str(self.v_cat.get()), str(png_out))
            self._tracks = results or []
            self._safe(self._populate_tree, self._tracks)
            if png_out.exists():
                self._png_path = png_out
                self._safe(self._show_png, png_out)
            self._done(f"CORRELATED — {len(self._tracks)} TRACKS")
        except Exception as e:
            print(f"[ERROR] Correlate: {e}\n")
            self._done("CORRELATE FAILED")

    def _w_full(self):
        try:
            raw  = Path(self.v_raw.get())
            out  = Path(self.v_out.get())
            cat  = Path(self.v_cat.get())
            sf   = self.v_frame.get()
            nf   = self.v_nf.get()
            stem = Path(sf).stem.replace(" ","_")

            if not sf:
                print("[ERROR] No start frame selected.\n"); self._done(); return

            print("="*55 + "\n  FRIGATE — Full pipeline\n" + "="*55 + "\n")

            if self.v_preproc.get() and not self._cancel.is_set():
                print("[1/4] Pre-processing…\n")
                self._safe(self._start_telem, "PRE-PROCESS")
                run_preprocessing(raw, out, num_adj=2, batch_size=40)

            if self._cancel.is_set(): self._done("CANCELLED"); return
            print("[2/4] Differencing…\n")
            self._safe(self._start_telem, "DIFFERENCE")
            _,_,df,_ = run_difference(raw, sf, nf, out)
            self._safe(self._load_fits, df)

            if self.v_solve.get() and not self._cancel.is_set():
                print("[3/4] Plate solving…\n")
                self._safe(self._start_telem, "PLATE SOLVE")
                solve_and_apply_wcs(str(raw/sf), str(df))
                self._safe(self._load_fits, df)

            if self._cancel.is_set(): self._done("CANCELLED"); return
            print("[4/4] Detecting & correlating…\n")
            self._safe(self._start_telem, "CORRELATE")
            png_out = out / f"diff_{stem}_{nf}f_correlated.png"
            results = process_and_correlate(str(df), str(cat), str(png_out))
            self._tracks = results or []
            self._safe(self._populate_tree, self._tracks)
            if png_out.exists():
                self._png_path = png_out
                self._safe(self._show_png, png_out)

            n = len(self._tracks)
            print(f"\n[DONE]  {n} track{'s' if n!=1 else ''} detected.\n")
            self._done(f"COMPLETE — {n} TRACKS")
        except Exception as e:
            print(f"[ERROR] Full pipeline: {e}\n")
            self._done("PIPELINE FAILED")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def destroy(self):
        sys.stdout = self._out0
        sys.stderr = self._err0
        plt.close("all")
        super().destroy()


# =============================================================================
if __name__ == "__main__":
    app = FrigateGUI()
    app.mainloop()