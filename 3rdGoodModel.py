#!/usr/bin/env python3
"""
Track & Field — Single-Phone Timing App
========================================
Calibration GUI + YOLO-based processing pipeline with ROI cropping.
Processes pre-recorded 4K 60fps video to automatically time athletes
running 10-100m races using computer vision.

Requirements:
    pip install ultralytics opencv-python pillow
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import numpy as np
from pathlib import Path
from math import sqrt
from collections import deque
import threading
import json
import os
import time

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Pillow is required.\nInstall with: pip install Pillow")

from ultralytics import YOLO

# ── Processing Constants ─────────────────────────────────────────────────────
TRACK_WIDTH_M         = 1.22
CONFIDENCE_THRESHOLD  = 0.30
IMGSZ_START           = 320
IMGSZ_STEP            = 160
DETECTION_CONF        = 0.15

# ── ROI Cropping ─────────────────────────────────────────────────────────────
ROI_SWITCH_IMGSZ      = 640    # full-frame caps here, then ROI for all athletes
CROP_W_MULTIPLIER     = 1.3 # tighter to avoid bystanders near finish line
CROP_H_MULTIPLIER     = 1.3    # tight vertical context
CROP_MIN_PX           = 250
MAX_JUMP_PX           = 200
MAX_JUMP_MULTIPLIER   = 0.5
MAX_JUMP_FLOOR_PX     = 300
MAX_LOST_FRAMES       = 90
MAX_DIST_CHANGE_PER_FRAME = 0.5  # max metres the smoothed distance can change per frame
MAX_FORWARD_DIST_M = 30.0  # reject detections far ahead (smoothing lag grows at distance)
LANE_X_MARGIN = 1.5       # metres of lateral tolerance outside lane bounds (perspective error at distance)

# ── ROI imgsz Escalation ─────────────────────────────────────────────────────
ROI_IMGSZ_START       = 320
ROI_IMGSZ_MAX         = 640
ROI_UPSCALE_BBOX_AREA = 2000  # upscale crop when bbox area is below this
ROI_UPSCALE_FACTOR    = 2     # 2x upscale

# ── Smoothing ────────────────────────────────────────────────────────────────
FOOT_SMOOTHING_ALPHA     = 0.25
DISTANCE_SMOOTHING_ALPHA = 0.25
FINISH_CONFIRM_FRAMES    = 3
TIMER_START_THRESHOLD_M  = 0.0

# ── GUI ──────────────────────────────────────────────────────────────────────
DISTANCE_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ATHLETE_OPTIONS  = [1, 2, 3, 4, 5]
VIDEO_EXTENSIONS = (".MOV", ".mov", ".mp4", ".MP4", ".avi", ".AVI", ".mkv", ".MKV")

VIDEO_DIR  = Path("videosamples")
OUTPUT_DIR = Path("runs/homography")
CALIB_FILE = Path("calibration_data.json")

# ── Dark Theme Colors ────────────────────────────────────────────────────────
BG_DARK    = "#1e1e1e"
BG_PANEL   = "#252526"
BG_ENTRY   = "#333333"
FG_TEXT    = "#cccccc"
FG_BRIGHT  = "#ffffff"
ACCENT     = "#0078d4"
GREEN      = "#4ec9b0"
RED        = "#f44747"
YELLOW     = "#dcdcaa"
POINT_COLORS = ["#ff4444", "#44ff44", "#4444ff", "#ffff44"]


class SimpleKalmanTracker:
    """Lightweight 2D Kalman filter for position + velocity tracking.
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """
    def __init__(self, x, y):
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4) * 100.0
        self.Q = np.eye(4)
        self.Q[0, 0] = 0.5   # position noise — tighter
        self.Q[1, 1] = 0.5
        self.Q[2, 2] = 0.1   # velocity noise — much tighter, velocity is stable for a runner
        self.Q[3, 3] = 0.1
        self.R = np.eye(2) * 4.0
        self.F = np.eye(4)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

    def predict(self):
        """Predict next state. Returns predicted (x, y)."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[0], self.state[1]

    def update(self, x, y):
        """Update with measurement. Returns corrected (x, y)."""
        z = np.array([x, y])
        y_residual = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_residual
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[0], self.state[1]

    def get_velocity(self):
        return self.state[2], self.state[3]

    def get_position(self):
        return self.state[0], self.state[1]

    def clamp_velocity(self, max_vel):
        self.state[2] = max(-max_vel, min(max_vel, self.state[2]))
        self.state[3] = max(-max_vel, min(max_vel, self.state[3]))


def load_model():
    """Load YOLO model, preferring ONNX if available."""
    onnx_path = Path("yolo26n.onnx")
    pt_path = Path("yolo26n.pt")
    if onnx_path.exists():
        print(f"Loading ONNX model: {onnx_path}")
        return YOLO(str(onnx_path))
    elif pt_path.exists():
        print(f"Loading PyTorch model: {pt_path}")
        return YOLO(str(pt_path))
    else:
        raise FileNotFoundError("No model found. Need yolo26n.onnx or yolo26n.pt")


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION APP (GUI)
# ══════════════════════════════════════════════════════════════════════════════

class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Track & Field Timing — Calibration")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1400x900")

        # State
        self.calibrations = {}
        self.load_calibrations()
        self.current_video = None
        self.cap = None
        self.lane_points = [[]]  # list of lists, one per athlete lane
        self.current_lane = 0    # which lane is being calibrated
        self.photo_image = None
        self.original_frame = None
        self.display_frame = None
        self.total_frames = 0
        self.current_frame_idx = 0

        # Zoom/pan state
        self._zoom       = 1.0
        self._off_x      = 0
        self._off_y      = 0
        self._fit_scale  = 1.0
        self._eff_scale  = 1.0
        self._drag_start = None
        self._drag_off0  = (0, 0)
        self._drag_moved = False

        # Processing state
        self.processing = False
        self.stop_requested = False

        self._build_gui()
        self._populate_video_list()

        # Keyboard bindings for frame stepping
        self.root.bind("<Left>", lambda e: self._prev_frame())
        self.root.bind("<Right>", lambda e: self._next_frame())

    # ── Calibration Data Persistence ─────────────────────────────────────────

    def load_calibrations(self):
        if CALIB_FILE.exists():
            try:
                with open(CALIB_FILE, "r") as f:
                    self.calibrations = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.calibrations = {}

    def save_calibrations(self):
        with open(CALIB_FILE, "w") as f:
            json.dump(self.calibrations, f, indent=2)

    # ── GUI Construction ─────────────────────────────────────────────────────

    def _build_gui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top section (3 panels)
        top_frame = tk.Frame(main_frame, bg=BG_DARK)
        top_frame.pack(fill=tk.BOTH, expand=True)

        self._build_left_panel(top_frame)
        self._build_center_panel(top_frame)
        self._build_right_panel(top_frame)

        # Bottom panel — log
        self._build_bottom_panel(main_frame)

    def _build_left_panel(self, parent):
        """Left panel — Video list."""
        left = tk.Frame(parent, bg=BG_PANEL, width=220)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2), pady=5)
        left.pack_propagate(False)

        tk.Label(left, text="Videos", bg=BG_PANEL, fg=FG_BRIGHT,
                 font=("Helvetica", 12, "bold")).pack(pady=(10, 5))

        list_frame = tk.Frame(left, bg=BG_PANEL)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.video_listbox = tk.Listbox(
            list_frame, bg=BG_ENTRY, fg=FG_TEXT, selectbackground=ACCENT,
            selectforeground=FG_BRIGHT, font=("Helvetica", 10),
            yscrollcommand=scrollbar.set, borderwidth=0, highlightthickness=0
        )
        self.video_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.video_listbox.yview)
        self.video_listbox.bind("<<ListboxSelect>>", self._on_video_select)

        self.calib_summary_label = tk.Label(
            left, text="Calibrated: 0 / 0", bg=BG_PANEL, fg=FG_TEXT,
            font=("Helvetica", 9)
        )
        self.calib_summary_label.pack(pady=(5, 10))

    def _build_center_panel(self, parent):
        """Center panel — Video frame display + scrubber."""
        center = tk.Frame(parent, bg=BG_DARK)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=5)

        # Canvas for video display
        self.canvas = tk.Canvas(center, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind canvas events — single left-button handles both pan (drag) and click (place point)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._reset_zoom)
        self.canvas.bind("<MouseWheel>",      self._on_scroll)   # macOS/Windows
        self.canvas.bind("<Button-4>",        self._on_scroll)   # Linux scroll-up
        self.canvas.bind("<Button-5>",        self._on_scroll)   # Linux scroll-down
        self.canvas.bind("<Configure>",       self._on_canvas_resize)

        # Frame scrubber
        scrub_frame = tk.Frame(center, bg=BG_DARK)
        scrub_frame.pack(fill=tk.X, padx=5, pady=(2, 5))

        self.btn_prev = tk.Button(
            scrub_frame, text="◀", bg=BG_ENTRY, fg=FG_TEXT, width=3,
            command=self._prev_frame
        )
        self.btn_prev.pack(side=tk.LEFT)

        self.frame_slider = tk.Scale(
            scrub_frame, from_=0, to=0, orient=tk.HORIZONTAL,
            bg=BG_DARK, fg=FG_TEXT, troughcolor=BG_ENTRY,
            highlightthickness=0, showvalue=False, length=400
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        # Only seek on mouse release, not during drag
        self.frame_slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.frame_slider.bind("<B1-Motion>", self._on_slider_drag)

        self.btn_next = tk.Button(
            scrub_frame, text="▶", bg=BG_ENTRY, fg=FG_TEXT, width=3,
            command=self._next_frame
        )
        self.btn_next.pack(side=tk.LEFT)

        self.frame_label = tk.Label(
            scrub_frame, text="Frame 0 / 0", bg=BG_DARK, fg=FG_TEXT,
            font=("Helvetica", 9), width=18
        )
        self.frame_label.pack(side=tk.LEFT, padx=(10, 0))

    def _build_right_panel(self, parent):
        """Right panel — Controls."""
        right = tk.Frame(parent, bg=BG_PANEL, width=250)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(2, 5), pady=5)
        right.pack_propagate(False)

        tk.Label(right, text="Controls", bg=BG_PANEL, fg=FG_BRIGHT,
                 font=("Helvetica", 12, "bold")).pack(pady=(10, 5))

        # Point status
        pts_frame = tk.LabelFrame(
            right, text="Calibration Points", bg=BG_PANEL, fg=FG_TEXT,
            font=("Helvetica", 9)
        )
        pts_frame.pack(fill=tk.X, padx=10, pady=5)

        self.point_labels = []
        point_names = ["Start-Left", "Start-Right", "Finish-Left", "Finish-Right"]
        for i, name in enumerate(point_names):
            lbl = tk.Label(
                pts_frame, text=f"{name}: ---", bg=BG_PANEL,
                fg=POINT_COLORS[i], font=("Courier", 9), anchor="w"
            )
            lbl.pack(fill=tk.X, padx=5, pady=1)
            self.point_labels.append(lbl)

        # Buttons: Clear Points, Reset Zoom
        btn_row = tk.Frame(right, bg=BG_PANEL)
        btn_row.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(
            btn_row, text="Clear Points", bg=BG_ENTRY, fg=FG_TEXT,
            command=self._clear_points
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        tk.Button(
            btn_row, text="Reset Zoom", bg=BG_ENTRY, fg=FG_TEXT,
            command=self._reset_zoom
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # Race distance dropdown
        tk.Label(right, text="Race Distance (m)", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9)).pack(pady=(10, 2))
        self.distance_var = tk.StringVar(value="100")
        dist_menu = ttk.Combobox(
            right, textvariable=self.distance_var,
            values=[str(d) for d in DISTANCE_OPTIONS],
            state="readonly", width=10
        )
        dist_menu.pack(padx=10)

        # Athletes to track dropdown
        tk.Label(right, text="Athletes to Track", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9)).pack(pady=(10, 2))
        self.athletes_var = tk.StringVar(value="1")
        ath_menu = ttk.Combobox(
            right, textvariable=self.athletes_var,
            values=[str(a) for a in ATHLETE_OPTIONS],
            state="readonly", width=10
        )
        ath_menu.pack(padx=10)
        self.athletes_var.trace_add("write", self._on_athletes_changed)

        # Lane selector dropdown (only visible when athletes > 1)
        self.lane_frame = tk.Frame(right, bg=BG_PANEL)
        self.lane_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
        tk.Label(self.lane_frame, text="Calibrate Lane", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9)).pack(pady=(0, 2))
        self.lane_var = tk.StringVar(value="Athlete 1 Lane")
        self.lane_menu = ttk.Combobox(
            self.lane_frame, textvariable=self.lane_var,
            values=["Athlete 1 Lane"],
            state="readonly", width=16
        )
        self.lane_menu.pack()
        self.lane_menu.bind("<<ComboboxSelected>>", self._on_lane_selected)
        self.lane_frame.pack_forget()  # hidden when single athlete

        # Save Calibration
        tk.Button(
            right, text="Save Calibration", bg=ACCENT, fg=FG_BRIGHT,
            font=("Helvetica", 10, "bold"), command=self._save_calibration
        ).pack(fill=tk.X, padx=10, pady=(15, 5))

        # Separator
        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)

        # Run buttons
        tk.Button(
            right, text="Run Selected Video", bg="#2d8659", fg=FG_BRIGHT,
            font=("Helvetica", 10), command=self._run_selected
        ).pack(fill=tk.X, padx=10, pady=2)

        tk.Button(
            right, text="Run All Videos", bg="#2d8659", fg=FG_BRIGHT,
            font=("Helvetica", 10), command=self._run_all
        ).pack(fill=tk.X, padx=10, pady=2)

        tk.Button(
            right, text="Stop", bg=RED, fg=FG_BRIGHT,
            font=("Helvetica", 10), command=self._stop_processing
        ).pack(fill=tk.X, padx=10, pady=(2, 10))

    def _build_bottom_panel(self, parent):
        """Bottom panel — Processing log."""
        bottom = tk.Frame(parent, bg=BG_PANEL, height=180)
        bottom.pack(fill=tk.X, padx=5, pady=(0, 5))
        bottom.pack_propagate(False)

        tk.Label(bottom, text="Processing Log", bg=BG_PANEL, fg=FG_BRIGHT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=10, pady=(5, 2))

        self.log_text = scrolledtext.ScrolledText(
            bottom, bg=BG_ENTRY, fg=FG_TEXT, font=("Courier", 9),
            wrap=tk.WORD, height=8, borderwidth=0, highlightthickness=0,
            insertbackground=FG_TEXT
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

    # ── Logging ──────────────────────────────────────────────────────────────

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        print(msg)

    # ── Video List ───────────────────────────────────────────────────────────

    def _populate_video_list(self):
        self.video_listbox.delete(0, tk.END)
        self.video_files = []

        if not VIDEO_DIR.exists():
            VIDEO_DIR.mkdir(parents=True, exist_ok=True)

        for f in sorted(VIDEO_DIR.iterdir()):
            if f.suffix in VIDEO_EXTENSIONS:
                self.video_files.append(f.name)
                self.video_listbox.insert(tk.END, f.name)

        self._update_list_colors()
        self._update_calib_summary()

    def _update_list_colors(self):
        for i, name in enumerate(self.video_files):
            if name in self.calibrations:
                self.video_listbox.itemconfig(i, fg=GREEN)
            else:
                self.video_listbox.itemconfig(i, fg=FG_TEXT)

    def _update_calib_summary(self):
        total = len(self.video_files)
        calibrated = sum(1 for v in self.video_files if v in self.calibrations)
        self.calib_summary_label.config(text=f"Calibrated: {calibrated} / {total}")

    # ── Video Loading ────────────────────────────────────────────────────────

    def _on_video_select(self, event):
        sel = self.video_listbox.curselection()
        if not sel:
            return
        name = self.video_files[sel[0]]
        self._load_video(name)

    def _load_video(self, name):
        if self.cap:
            self.cap.release()

        path = VIDEO_DIR / name
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open {name}")
            return

        self.current_video = name
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.frame_slider.config(to=max(0, self.total_frames - 1))

        # Reset zoom/pan
        self._reset_zoom()

        # Clear lane points
        self.current_lane = 0
        self.lane_points = [[]]
        self._update_point_labels()

        # Restore calibration if exists
        if name in self.calibrations:
            cal = self.calibrations[name]
            saved_pts = cal.get("points", [])
            n_athletes = int(cal.get("max_athletes", 1))
            self.distance_var.set(str(cal.get("distance_m", 100)))
            self.athletes_var.set(str(n_athletes))
            # Handle both nested format [[pts], [pts]] and legacy flat format [pt, pt, pt, pt]
            if saved_pts and isinstance(saved_pts[0], (list, tuple)) and len(saved_pts[0]) == 2 and not isinstance(saved_pts[0][0], (list, tuple)):
                # Legacy flat format: single list of 4 points → wrap in one lane
                self.lane_points = [[tuple(p) for p in saved_pts]]
            else:
                # Nested format: list of lanes, each with 4 points
                self.lane_points = [[tuple(p) for p in lane] for lane in saved_pts]
            # Ensure we have enough lanes
            while len(self.lane_points) < n_athletes:
                self.lane_points.append([])
            self.current_lane = 0
            saved_frame = cal.get("frame_idx", 0)
            self.current_frame_idx = saved_frame
            self.frame_slider.set(saved_frame)
            self._update_point_labels()

        self._seek_and_display(self.current_frame_idx)
        self._update_frame_label()

    def _seek_and_display(self, idx):
        if not self.cap:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame_idx = idx
        self.original_frame = frame.copy()
        self._render_frame()

    def _render_frame(self):
        """Render the current frame with zoom/pan, then overlay calibration points as canvas items."""
        if self.original_frame is None:
            return

        frame = self.original_frame
        H, W = frame.shape[:2]
        self.root.update_idletasks()
        cw = max(self.canvas.winfo_width(),  100)
        ch = max(self.canvas.winfo_height(), 100)

        # Scale that fits the whole image in the canvas at zoom=1
        self._fit_scale = min(cw / W, ch / H)
        self._eff_scale = self._fit_scale * self._zoom

        sw = int(W * self._eff_scale)
        sh = int(H * self._eff_scale)

        # Clamp pan so the image never floats away
        if sw <= cw:
            self._off_x = (cw - sw) // 2
        else:
            self._off_x = max(cw - sw, min(0, self._off_x))
        if sh <= ch:
            self._off_y = (ch - sh) // 2
        else:
            self._off_y = max(ch - sh, min(0, self._off_y))

        # Crop only the visible region for performance
        ox0 = max(0, int(-self._off_x / self._eff_scale))
        oy0 = max(0, int(-self._off_y / self._eff_scale))
        ox1 = min(W, ox0 + int(cw / self._eff_scale) + 2)
        oy1 = min(H, oy0 + int(ch / self._eff_scale) + 2)

        crop = frame[oy0:oy1, ox0:ox1]
        if crop.size == 0:
            return

        disp_w = int((ox1 - ox0) * self._eff_scale)
        disp_h = int((oy1 - oy0) * self._eff_scale)
        if disp_w <= 0 or disp_h <= 0:
            return

        scaled = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(scaled)
        self.photo_image = ImageTk.PhotoImage(img)

        img_cx = max(0, self._off_x)
        img_cy = max(0, self._off_y)

        self.canvas.delete("all")
        self.canvas.create_image(img_cx, img_cy, anchor=tk.NW, image=self.photo_image)

        # Helper: original-image coords → canvas coords
        def to_c(ox, oy):
            return (self._off_x + ox * self._eff_scale,
                    self._off_y + oy * self._eff_scale)

        # Per-lane colors: athlete 1 = green, athlete 2 = blue, athlete 3 = orange, etc.
        LANE_COLORS = ["#44ff44", "#4488ff", "#ff8844", "#ff44ff", "#44ffff"]
        POINT_NAMES = ["SL", "SR", "FL", "FR"]
        dot_r = max(6, min(14, int(10 * self._zoom ** 0.3)))
        font_size = max(10, int(13 * self._zoom ** 0.3))

        for lane_idx, pts in enumerate(self.lane_points):
            color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
            is_active = (lane_idx == self.current_lane)
            line_width = 2 if is_active else 1
            dot_outline = "white" if is_active else "#888888"

            # Draw connecting lines
            if len(pts) >= 2:
                x0, y0 = to_c(*pts[0])
                x1, y1 = to_c(*pts[1])
                self.canvas.create_line(x0, y0, x1, y1, fill=color, width=line_width)
            if len(pts) >= 4:
                x2, y2 = to_c(*pts[2])
                x3, y3 = to_c(*pts[3])
                self.canvas.create_line(x2, y2, x3, y3, fill=color, width=line_width)
                x0, y0 = to_c(*pts[0])
                x1, y1 = to_c(*pts[1])
                self.canvas.create_line(x0, y0, x2, y2, fill=color, width=1, dash=(4, 4))
                self.canvas.create_line(x1, y1, x3, y3, fill=color, width=1, dash=(4, 4))

            # Draw calibration dots
            for i, (ox, oy) in enumerate(pts):
                cx, cy = to_c(ox, oy)
                self.canvas.create_oval(cx - dot_r, cy - dot_r,
                                        cx + dot_r, cy + dot_r,
                                        fill=color, outline=dot_outline, width=2)
                label = f"A{lane_idx+1}:{POINT_NAMES[i]}" if len(self.lane_points) > 1 else f"{i+1}:{POINT_NAMES[i]}"
                self.canvas.create_text(cx + dot_r + 5, cy - dot_r,
                                        text=label, fill="white",
                                        font=("Helvetica", font_size, "bold"),
                                        anchor="sw")

        # Zoom indicator (top-right)
        zoom_pct = int(self._zoom * 100)
        self.canvas.create_text(cw - 8, 8, text=f"{zoom_pct}%",
                                fill="white", font=("Helvetica", 10, "bold"),
                                anchor="ne")
        if self._zoom > 1.05:
            self.canvas.create_text(cw // 2, ch - 10,
                                    text="drag to pan \u2022 double-click to reset",
                                    fill="#888", font=("Helvetica", 9))

    # ── Canvas Events ────────────────────────────────────────────────────────

    def _on_press(self, event):
        """Mouse press — record start for pan/click detection."""
        self._drag_start = (event.x, event.y)
        self._drag_off0  = (self._off_x, self._off_y)
        self._drag_moved = False

    def _on_drag(self, event):
        """Mouse drag — pan the view."""
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        if abs(dx) > 3 or abs(dy) > 3:
            self._drag_moved = True
            self._off_x = self._drag_off0[0] + dx
            self._off_y = self._drag_off0[1] + dy
            self._render_frame()

    def _on_release(self, event):
        """Mouse release — if it wasn't a drag, place a calibration point."""
        if not self._drag_moved and self._drag_start is not None:
            pts = self.lane_points[self.current_lane]
            if len(pts) < 4 and self.original_frame is not None:
                H, W = self.original_frame.shape[:2]
                ox = int((event.x - self._off_x) / self._eff_scale)
                oy = int((event.y - self._off_y) / self._eff_scale)
                ox = max(0, min(W - 1, ox))
                oy = max(0, min(H - 1, oy))
                pts.append((ox, oy))
                self._update_point_labels()
                self._render_frame()
        self._drag_start = None
        self._drag_moved = False

    def _on_scroll(self, event):
        """Scroll to zoom — keeps the pixel under the cursor fixed."""
        if self.original_frame is None:
            return
        if event.num == 4:
            factor = 1.15
        elif event.num == 5:
            factor = 1.0 / 1.15
        elif hasattr(event, 'delta') and event.delta > 0:
            factor = 1.15
        elif hasattr(event, 'delta') and event.delta < 0:
            factor = 1.0 / 1.15
        else:
            return

        new_zoom = max(1.0, min(12.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return

        new_eff = self._fit_scale * new_zoom
        orig_x = (event.x - self._off_x) / self._eff_scale
        orig_y = (event.y - self._off_y) / self._eff_scale
        self._off_x = int(event.x - orig_x * new_eff)
        self._off_y = int(event.y - orig_y * new_eff)
        self._zoom = new_zoom
        self._render_frame()

    def _on_canvas_resize(self, event):
        if self.original_frame is not None:
            self._render_frame()

    def _reset_zoom(self, _event=None):
        self._zoom  = 1.0
        self._off_x = 0
        self._off_y = 0
        if self.original_frame is not None:
            self._render_frame()

    # ── Frame Scrubber ───────────────────────────────────────────────────────

    def _on_slider_drag(self, event):
        """During drag, only update the label — don't seek."""
        val = self.frame_slider.get()
        self.frame_label.config(text=f"Frame {val} / {self.total_frames}")

    def _on_slider_release(self, event):
        """On release, seek to the frame."""
        val = self.frame_slider.get()
        self._seek_and_display(val)
        self._update_frame_label()

    def _prev_frame(self):
        if self.cap and self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_slider.set(self.current_frame_idx)
            self._seek_and_display(self.current_frame_idx)
            self._update_frame_label()

    def _next_frame(self):
        if self.cap and self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.frame_slider.set(self.current_frame_idx)
            self._seek_and_display(self.current_frame_idx)
            self._update_frame_label()

    def _update_frame_label(self):
        self.frame_label.config(
            text=f"Frame {self.current_frame_idx} / {self.total_frames}"
        )

    # ── Point Management ─────────────────────────────────────────────────────

    def _update_point_labels(self):
        pts = self.lane_points[self.current_lane]
        lane_prefix = f"L{self.current_lane+1} " if len(self.lane_points) > 1 else ""
        point_names = ["Start-Left", "Start-Right", "Finish-Left", "Finish-Right"]
        for i, name in enumerate(point_names):
            if i < len(pts):
                px, py = pts[i]
                self.point_labels[i].config(text=f"{lane_prefix}{name}: ({px}, {py})")
            else:
                self.point_labels[i].config(text=f"{lane_prefix}{name}: ---")

    def _clear_points(self):
        self.lane_points[self.current_lane] = []
        self._update_point_labels()
        if self.original_frame is not None:
            self._render_frame()

    def _on_athletes_changed(self, *_args):
        """Resize lane_points list when athlete count changes and show/hide lane selector."""
        try:
            n = int(self.athletes_var.get())
        except ValueError:
            return
        # Resize lane_points: grow with empty lists, shrink by truncating
        while len(self.lane_points) < n:
            self.lane_points.append([])
        if len(self.lane_points) > n:
            self.lane_points = self.lane_points[:n]
        # Update lane dropdown
        lane_values = [f"Athlete {i+1} Lane" for i in range(n)]
        self.lane_menu.config(values=lane_values)
        if self.current_lane >= n:
            self.current_lane = 0
        self.lane_var.set(lane_values[self.current_lane])
        # Show/hide lane selector
        if n > 1:
            self.lane_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
        else:
            self.lane_frame.pack_forget()
        self._update_point_labels()
        if self.original_frame is not None:
            self._render_frame()

    def _on_lane_selected(self, _event=None):
        """Switch which lane's points are being edited."""
        val = self.lane_var.get()
        # Extract lane index from "Athlete N Lane"
        try:
            self.current_lane = int(val.split()[1]) - 1
        except (IndexError, ValueError):
            self.current_lane = 0
        self._update_point_labels()
        if self.original_frame is not None:
            self._render_frame()

    # ── Save Calibration ─────────────────────────────────────────────────────

    def _save_calibration(self):
        if not self.current_video:
            messagebox.showwarning("Warning", "No video selected.")
            return
        n_athletes = int(self.athletes_var.get())
        # Validate: each lane must have exactly 4 points
        for i in range(n_athletes):
            if i >= len(self.lane_points) or len(self.lane_points[i]) != 4:
                messagebox.showwarning(
                    "Warning",
                    f"Athlete {i+1} lane needs 4 calibration points "
                    f"(has {len(self.lane_points[i]) if i < len(self.lane_points) else 0})."
                )
                return

        self.calibrations[self.current_video] = {
            "points": [[list(p) for p in lane] for lane in self.lane_points[:n_athletes]],
            "distance_m": float(self.distance_var.get()),
            "max_athletes": n_athletes,
            "frame_idx": self.current_frame_idx
        }
        self.save_calibrations()
        self._update_list_colors()
        self._update_calib_summary()
        self.log(f"Calibration saved for {self.current_video} ({n_athletes} lane(s))")

    # ── Run Processing ───────────────────────────────────────────────────────

    def _run_selected(self):
        if self.processing:
            messagebox.showinfo("Info", "Processing already running.")
            return
        if not self.current_video:
            messagebox.showwarning("Warning", "No video selected.")
            return
        if self.current_video not in self.calibrations:
            messagebox.showwarning("Warning", "Selected video is not calibrated.")
            return
        self._run_videos([self.current_video])

    def _run_all(self):
        if self.processing:
            messagebox.showinfo("Info", "Processing already running.")
            return
        calibrated = [v for v in self.video_files if v in self.calibrations]
        if not calibrated:
            messagebox.showwarning("Warning", "No calibrated videos found.")
            return
        self._run_videos(calibrated)

    def _run_videos(self, video_list):
        self.processing = True
        self.stop_requested = False

        def worker():
            model = load_model()
            all_results = {}  # video_name → list of athlete results
            for video_name in video_list:
                if self.stop_requested:
                    self.log("Processing stopped by user.")
                    break
                cal = self.calibrations[video_name]
                video_path = str(VIDEO_DIR / video_name)
                self.log(f"\n{'='*60}")
                self.log(f"Processing: {video_name}")
                self.log(f"Distance: {cal['distance_m']}m | Athletes: {cal['max_athletes']}")
                self.log(f"{'='*60}")

                results = run_tracking(
                    model=model,
                    video_path=video_path,
                    src_points=cal["points"],
                    distance_m=cal["distance_m"],
                    max_athletes=cal["max_athletes"],
                    log_fn=self.log,
                    stop_check=lambda: self.stop_requested
                )
                if results:
                    all_results[video_name] = results

            # Final timing summary across all videos
            if all_results:
                sep = "\u2550" * 40
                self.log(f"\n{sep}")
                self.log("  TIMING SUMMARY \u2014 ALL VIDEOS")
                self.log(sep)
                for vname, vresults in all_results.items():
                    self.log(f"  Video: {vname}")
                    for r in vresults:
                        tid = r["tid"]
                        if r["time_s"] is not None:
                            self.log(f"    Athlete ID{tid}: {r['time_s']:.3f}s ({r['distance_m']}m)")
                        else:
                            self.log(f"    Athlete ID{tid}: DID NOT FINISH")
                    self.log("")
                self.log(sep)

            self.processing = False
            self.log("\nAll processing complete.")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _stop_processing(self):
        if self.processing:
            self.stop_requested = True
            self.log("Stop requested...")

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def _hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (b, g, r)


# ══════════════════════════════════════════════════════════════════════════════
#  BIRD'S EYE VIEW
# ══════════════════════════════════════════════════════════════════════════════

def generate_birds_eye(frame, H, distance_m, output_path):
    """Generate a top-down warped view of the track and save it."""
    # Create a bird's eye destination size
    scale = 20  # pixels per metre
    dst_w = int(TRACK_WIDTH_M * scale) + 100
    dst_h = int(distance_m * scale) + 100
    offset_x = 50
    offset_y = 50

    # Create mapping for bird's eye
    dst_pts_scaled = np.float32([
        [offset_x, offset_y],
        [offset_x + TRACK_WIDTH_M * scale, offset_y],
        [offset_x, offset_y + distance_m * scale],
        [offset_x + TRACK_WIDTH_M * scale, offset_y + distance_m * scale],
    ])

    # Get source points from the inverse of H
    H_inv = np.linalg.inv(H)

    # Warp the frame to bird's eye view
    src_pts_for_warp = np.float32([
        [0.0, 0.0],
        [TRACK_WIDTH_M, 0.0],
        [0.0, distance_m],
        [TRACK_WIDTH_M, distance_m],
    ])

    # Map world corners back to pixel space
    corners_world = src_pts_for_warp.reshape(-1, 1, 2)
    corners_px = cv2.perspectiveTransform(corners_world, H_inv)

    H_bird, _ = cv2.findHomography(corners_px.reshape(-1, 2), dst_pts_scaled)
    bird_img = cv2.warpPerspective(frame, H_bird, (dst_w, dst_h))

    # Draw distance markers
    if distance_m <= 30:
        interval = 5
    else:
        interval = 10

    for d in range(0, int(distance_m) + 1, interval):
        y = offset_y + int(d * scale)
        cv2.line(bird_img, (offset_x - 30, y),
                 (offset_x + int(TRACK_WIDTH_M * scale) + 30, y),
                 (0, 255, 255), 1)
        label = "START" if d == 0 else ("FINISH" if d == int(distance_m) else f"{d}m")
        cv2.putText(bird_img, label, (5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.imwrite(str(output_path), bird_img)
    return bird_img


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_tracking(model, video_path, src_points, distance_m, max_athletes,
                 log_fn=print, stop_check=lambda: False):
    """Main tracking pipeline."""

    # ── Setup ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_fn(f"ERROR: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log_fn(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = Path(video_path).stem
    out_path = OUTPUT_DIR / f"{base_name}_tracked.mp4"
    bird_path = OUTPUT_DIR / f"{base_name}_bird_eye.png"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # ── Homography ───────────────────────────────────────────────────────────
    dst_pts = np.float32([
        [0.0, 0.0],
        [TRACK_WIDTH_M, 0.0],
        [0.0, distance_m],
        [TRACK_WIDTH_M, distance_m],
    ])

    # Handle both nested format [[lane1_pts], [lane2_pts]] and legacy flat format [pt, pt, pt, pt]
    if src_points and isinstance(src_points[0], (list, tuple)) and len(src_points[0]) == 2 and not isinstance(src_points[0][0], (list, tuple)):
        # Legacy flat format — single set of 4 points
        lane_src_points = [src_points]
    else:
        lane_src_points = src_points

    # Compute per-lane homography matrices
    H_lanes = []  # H_lanes[0] = first lane's homography, etc.
    for lane_idx, lane_pts in enumerate(lane_src_points):
        src_pts = np.float32(lane_pts)
        H_lane, _ = cv2.findHomography(src_pts, dst_pts)
        if H_lane is None:
            log_fn(f"ERROR: Could not compute homography for lane {lane_idx+1}.")
            cap.release()
            out.release()
            return
        H_lanes.append(H_lane)

    H = H_lanes[0]  # default homography (used for track boundary check, bird's eye)
    H_per_athlete = {}  # tid → homography matrix (assigned by lock order)
    lock_order = []  # track IDs in order they were locked

    # Generate bird's eye view from first frame (using lane 1 homography)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        generate_birds_eye(first_frame, H, distance_m, bird_path)
        log_fn(f"Bird's eye view saved: {bird_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Right-Side Cutoff Mask ───────────────────────────────────────────────
    # Black out everything to the RIGHT of the lane's right edge
    # This removes bystanders standing beside the track while keeping
    # the athlete's full body visible (they're on the left side)
    MASK_MARGIN_PX = 0  # no padding — cut right at the lane edge

    # Get the right-edge points: Start-Right and Finish-Right
    # For multi-lane, use the rightmost lane's right edge
    sr = list(lane_src_points[0][1])   # start-right
    fr = list(lane_src_points[0][3])   # finish-right
    for lane_pts in lane_src_points:
        if lane_pts[1][0] > sr[0]: sr = list(lane_pts[1])
        if lane_pts[3][0] > fr[0]: fr = list(lane_pts[3])

    # Shift the line rightward by margin so we don't clip the athlete
    sr[0] = int(sr[0] + MASK_MARGIN_PX)
    fr[0] = int(fr[0] + MASK_MARGIN_PX)

    # Extrapolate the SR→FR line to the top (y=0) and bottom (y=height) of the frame
    # Line equation: given two points, find x at any y
    dx = fr[0] - sr[0]
    dy = fr[1] - sr[1]
    if dy != 0:
        # x at y=0 (top of frame)
        top_x = int(sr[0] + dx * (0 - sr[1]) / dy)
        # x at y=height (bottom of frame)
        bot_x = int(sr[0] + dx * (height - sr[1]) / dy)
    else:
        # Horizontal line — same x everywhere
        top_x = sr[0]
        bot_x = sr[0]

    # Polygon: top of line → bottom of line → bottom-right → top-right
    cutoff_poly = np.array([
        [top_x, 0],
        [bot_x, height],
        [width, height],
        [width, 0],
    ], dtype=np.int32)

    # Mask: white = KEEP, black = HIDE. Start all white, fill the right side black.
    track_mask = np.ones((height, width), dtype=np.uint8) * 255
    cv2.fillPoly(track_mask, [cutoff_poly], 0)
    track_mask_3ch = cv2.merge([track_mask, track_mask, track_mask])
    log_fn(f"Right-side cutoff mask created (SR=({sr[0]},{sr[1]}), FR=({fr[0]},{fr[1]}), margin={MASK_MARGIN_PX}px)")

    # Save debug image
    debug_masked = cv2.bitwise_and(first_frame, track_mask_3ch)
    debug_mask_path = OUTPUT_DIR / f"{base_name}_mask_debug.jpg"
    cv2.imwrite(str(debug_mask_path), debug_masked)
    log_fn(f"Mask debug image saved: {debug_mask_path}")

    # ── Per-athlete tracking state ───────────────────────────────────────────
    multi_athlete = max_athletes > 1
    current_imgsz = IMGSZ_START
    in_roi_mode = False  # only for single athlete

    # Track ID → state
    locked_ids = {}        # tid → True (locked track IDs)
    last_box_full = {}     # tid → (x1, y1, x2, y2) full-frame bbox
    last_conf = {}         # tid → confidence

    # Smoothing state per athlete
    smooth_foot = {}       # tid → (sx, sy)
    smooth_dist = {}       # tid → smoothed world_y
    max_dist_reached = {}  # tid → highest smoothed distance ever reached (never decreases)
    prev_foot = {}         # tid → previous foot position (for direction filter)

    # Timing state per athlete
    race_start_ms = {}     # tid → start timestamp
    race_finish_ms = {}    # tid → finish timestamp
    first_cross_ts = {}    # tid → first crossing timestamp
    finish_count = {}      # tid → consecutive finish frames

    # ROI state per athlete (single athlete only)
    athlete_roi = {}       # tid → {frames_lost, prev_foot, last_crop}
    frames_lost_global = {}  # tid → frames_lost for full-frame mode

    # Last known state for drawing on stale frames
    last_world_pos = {}    # tid → (world_x, world_y)
    last_draw_info = {}    # tid → dict of drawing info

    frame_idx = 0
    processing_start_time = time.time()

    # ── Main Processing Loop (every frame — no skipping) ─────────────────────
    while True:
        if stop_check():
            log_fn("Processing stopped by user.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # Check early stop: all athletes finished
        if locked_ids and all(tid in race_finish_ms for tid in locked_ids):
            log_fn("All athletes finished — stopping early.")
            break

        # ── Processing (every frame) ─────────────────────────────────────────
        if not in_roi_mode:
            # ── Full-Frame Mode ──────────────────────────────────────────────
            # Mask frame to hide bystanders outside the track
            detect_frame = frame
            if track_mask_3ch is not None:
                detect_frame = cv2.bitwise_and(frame, track_mask_3ch)

            results = model.track(
                detect_frame, classes=[0], persist=True, imgsz=current_imgsz,
                conf=DETECTION_CONF, verbose=False
            )


            detections = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                for b in results[0].boxes:
                    if b.id is None:
                        continue
                    tid = int(b.id[0])
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = int(b.xyxy[0][0]), int(b.xyxy[0][1]), \
                                     int(b.xyxy[0][2]), int(b.xyxy[0][3])
                    detections.append((tid, conf, x1, y1, x2, y2))

            # Lock first N track IDs — only if ON the track
            for tid, conf, x1, y1, x2, y2 in detections:
                if tid not in locked_ids and len(locked_ids) < max_athletes:
                    # Check if person is on the track using homography
                    _foot_x = (x1 + x2) / 2
                    _foot_y = float(y2)
                    _foot_px = np.array([[[_foot_x, _foot_y]]], dtype=np.float32)
                    _foot_world = cv2.perspectiveTransform(_foot_px, H)[0][0]
                    _world_x = float(_foot_world[0])

                    x_min = -TRACK_WIDTH_M * (max_athletes - 1) - 0.5
                    x_max = TRACK_WIDTH_M * max_athletes + 0.5

                    if x_min < _world_x < x_max:
                        locked_ids[tid] = True
                        frames_lost_global[tid] = 0
                        # Assign per-athlete homography by lock order
                        lane_idx = len(lock_order)
                        lock_order.append(tid)
                        if lane_idx < len(H_lanes):
                            H_per_athlete[tid] = H_lanes[lane_idx]
                        else:
                            H_per_athlete[tid] = H_lanes[0]  # fallback to lane 1
                        log_fn(f"Locked athlete ID {tid} → lane {lane_idx+1} (x={_world_x:.2f}m)")
                    else:
                        continue  # off track — bystander, skip

            # Process each locked athlete
            for tid in list(locked_ids.keys()):
                if tid in race_finish_ms:
                    continue  # already finished

                # Find this athlete's detection
                det = None
                for d_tid, d_conf, d_x1, d_y1, d_x2, d_y2 in detections:
                    if d_tid == tid:
                        # Proximity filter
                        if tid in last_box_full:
                            lx1, ly1, lx2, ly2 = last_box_full[tid]
                            last_cx = (lx1 + lx2) // 2
                            last_cy = (ly1 + ly2) // 2
                            det_cx = (d_x1 + d_x2) // 2
                            det_cy = (d_y1 + d_y2) // 2
                            jump = sqrt((det_cx - last_cx)**2 + (det_cy - last_cy)**2)
                            if jump > MAX_JUMP_PX:
                                continue
                        det = (d_tid, d_conf, d_x1, d_y1, d_x2, d_y2)
                        break

                if det:
                    _, conf, x1, y1, x2, y2 = det
                    frames_lost_global[tid] = 0
                    last_box_full[tid] = (x1, y1, x2, y2)
                    last_conf[tid] = conf

                    # Foot position
                    raw_fx = (x1 + x2) / 2
                    raw_fy = float(y2)

                    # Smooth foot position
                    if tid in smooth_foot:
                        old_sx, old_sy = smooth_foot[tid]
                        sx = FOOT_SMOOTHING_ALPHA * raw_fx + (1 - FOOT_SMOOTHING_ALPHA) * old_sx
                        sy = FOOT_SMOOTHING_ALPHA * raw_fy + (1 - FOOT_SMOOTHING_ALPHA) * old_sy
                    else:
                        sx, sy = raw_fx, raw_fy
                    smooth_foot[tid] = (sx, sy)

                    # Homography → world coordinates (per-athlete)
                    foot_px = np.array([[[sx, sy]]], dtype=np.float32)
                    H_tid = H_per_athlete.get(tid, H)
                    foot_world = cv2.perspectiveTransform(foot_px, H_tid)[0][0]
                    world_x, world_y = float(foot_world[0]), float(foot_world[1])

                    # Smooth world distance
                    if tid in smooth_dist:
                        old_d = smooth_dist[tid]
                        smoothed_d = DISTANCE_SMOOTHING_ALPHA * world_y + \
                                     (1 - DISTANCE_SMOOTHING_ALPHA) * old_d
                        # Clamp: distance can't change faster than physically possible
                        delta = smoothed_d - old_d
                        if abs(delta) > MAX_DIST_CHANGE_PER_FRAME:
                            smoothed_d = old_d + MAX_DIST_CHANGE_PER_FRAME * (1 if delta > 0 else -1)
                    else:
                        smoothed_d = world_y
                    smooth_dist[tid] = smoothed_d
                    if smoothed_d > max_dist_reached.get(tid, 0):
                        max_dist_reached[tid] = smoothed_d

                    last_world_pos[tid] = (world_x, smoothed_d)

                    # Timing — start
                    if tid not in race_start_ms and smoothed_d > TIMER_START_THRESHOLD_M:
                        race_start_ms[tid] = timestamp_ms
                        log_fn(f"ID{tid} timer started at {timestamp_ms:.0f}ms "
                               f"(dist={smoothed_d:.1f}m)")

                    # Timing — finish
                    if tid in race_start_ms and tid not in race_finish_ms:
                        if smoothed_d >= distance_m:
                            if tid not in first_cross_ts:
                                first_cross_ts[tid] = timestamp_ms
                            finish_count[tid] = finish_count.get(tid, 0) + 1
                            if finish_count[tid] >= FINISH_CONFIRM_FRAMES:
                                race_finish_ms[tid] = first_cross_ts[tid]
                                elapsed = race_finish_ms[tid] - race_start_ms[tid]
                                log_fn(f"ID{tid} FINISHED: {elapsed/1000:.3f}s")
                        else:
                            # Only reset if we drop significantly below finish
                            if smoothed_d < distance_m - 3.0:
                                finish_count[tid] = 0
                                if tid in first_cross_ts:
                                    del first_cross_ts[tid]

                    prev_foot[tid] = (raw_fx, raw_fy)

                    # Update draw info
                    last_draw_info[tid] = {
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "foot": (sx, sy),
                        "world": (world_x, smoothed_d),
                    }

                    # Debug output (every 30 frames to reduce overhead)
                    if frame_idx % 30 == 0:
                        log_fn(f"Frame {frame_idx:5d} | Full  | imgsz={current_imgsz:4d} | "
                               f"ID{tid}:px=({sx:.1f},{sy:.1f}) | det=1 | "
                               f"conf={conf:.3f} | dist={smoothed_d:.1f}m")
                else:
                    frames_lost_global[tid] = frames_lost_global.get(tid, 0) + 1
                    if frames_lost_global[tid] > MAX_LOST_FRAMES:
                        log_fn(f"ID{tid} lost for {MAX_LOST_FRAMES} frames — skipping")

            # Adaptive imgsz — also escalate when nobody is locked yet
            if not locked_ids and current_imgsz < ROI_SWITCH_IMGSZ and frame_idx > 0 and frame_idx % 30 == 0:
                current_imgsz += IMGSZ_STEP
                log_fn(f"imgsz → {current_imgsz} (no lock after {frame_idx} frames)")
            elif detections:
                locked_confs = [c for t, c, *_ in detections if t in locked_ids]
                if locked_confs:
                    min_conf = min(locked_confs)
                    if min_conf < CONFIDENCE_THRESHOLD:
                        if current_imgsz < ROI_SWITCH_IMGSZ:
                            current_imgsz += IMGSZ_STEP
                            log_fn(f"imgsz → {current_imgsz} (conf={min_conf:.3f})")
                        elif current_imgsz >= ROI_SWITCH_IMGSZ:
                            # Switch to ROI for ALL locked athletes
                            in_roi_mode = True
                            for _tid in locked_ids:
                                if _tid not in athlete_roi:
                                    # Initialize Kalman immediately from last known foot position
                                    _init_foot = prev_foot.get(_tid)
                                    _init_kalman = None
                                    if _init_foot is not None:
                                        _init_kalman = SimpleKalmanTracker(_init_foot[0], _init_foot[1])

                                    # Store peak bbox dimensions for crop floor
                                    _peak_w, _peak_h = 0, 0
                                    if _tid in last_box_full:
                                        _bx1, _by1, _bx2, _by2 = last_box_full[_tid]
                                        _peak_w = _bx2 - _bx1
                                        _peak_h = _by2 - _by1

                                    athlete_roi[_tid] = {
                                        "frames_lost": 0,
                                        "prev_foot": _init_foot,
                                        "last_crop": None,
                                        "roi_imgsz": ROI_IMGSZ_START,
                                        "kalman": _init_kalman,
                                        "prev_gray": None,
                                        "flow_point": None,
                                        "flow_crop_origin": (0, 0),
                                        "last_yolo_foot": _init_foot,
                                        "peak_bbox_w": _peak_w,
                                        "peak_bbox_h": _peak_h,
                                    }
                            log_fn(f"Switching to ROI mode for "
                                   f"{len(locked_ids)} athlete(s) (conf={min_conf:.3f})")

        else:
            # ── ROI Mode (All Athletes) ────────────────────────────────────
            for roi_tid in list(locked_ids.keys()):
                if roi_tid in race_finish_ms:
                    continue  # already finished

                if roi_tid not in athlete_roi:
                    continue  # no ROI state

                roi_state = athlete_roi[roi_tid]

                if roi_state["frames_lost"] > MAX_LOST_FRAMES:
                    continue

                # Build ROI crop around foot position
                if roi_tid not in last_box_full:
                    roi_state["frames_lost"] += 1
                    continue

                bx1, by1, bx2, by2 = last_box_full[roi_tid]
                foot_x = (bx1 + bx2) // 2
                foot_y = by2

                bbox_w = bx2 - bx1
                bbox_h = by2 - by1

                # Use peak bbox as floor — prevents crop death spiral from partial detections
                peak_w = roi_state.get("peak_bbox_w", bbox_w)
                peak_h = roi_state.get("peak_bbox_h", bbox_h)
                # Use current bbox with a minimum floor based on peak size
                crop_bbox_w = max(bbox_w, int(peak_w * 0.25))
                crop_bbox_h = max(bbox_h, int(peak_h * 0.25))

                # Use Kalman predicted position as reference (if available)
                # This accounts for athlete movement — bystanders behind fall further away
                kalman = roi_state.get("kalman")
                if kalman is not None:
                    pred_x, pred_y = kalman.predict()
                    foot_x = int(pred_x)
                    foot_y = int(pred_y)

                    # Cap drift: don't let prediction drift too far from last YOLO detection
                    last_yolo = roi_state.get("last_yolo_foot")
                    if last_yolo is not None:
                        max_drift = max(crop_bbox_w, crop_bbox_h, 30)
                        dx = foot_x - last_yolo[0]
                        dy = foot_y - last_yolo[1]
                        if abs(dx) > max_drift:
                            foot_x = int(last_yolo[0] + max_drift * (1 if dx > 0 else -1))
                        if abs(dy) > max_drift:
                            foot_y = int(last_yolo[1] + max_drift * (1 if dy > 0 else -1))

                pad_w = max(int(crop_bbox_w * CROP_W_MULTIPLIER), CROP_MIN_PX)
                pad_h = max(int(crop_bbox_h * CROP_H_MULTIPLIER), CROP_MIN_PX)

                cx1 = max(0, foot_x - pad_w)
                cy1 = max(0, foot_y - pad_h)
                cx2 = min(width, foot_x + pad_w)
                cy2 = min(height, foot_y + pad_h)

                # Guard: skip if crop too small
                if (cx2 - cx1) < 32 or (cy2 - cy1) < 32:
                    roi_state["frames_lost"] += 1
                    continue

                roi_state["last_crop"] = (cx1, cy1, cx2, cy2)
                crop_frame = frame[cy1:cy2, cx1:cx2]

                # Apply the right-side mask to ROI crop too
                if track_mask_3ch is not None:
                    crop_mask = track_mask_3ch[cy1:cy2, cx1:cx2]
                    crop_frame = cv2.bitwise_and(crop_frame, crop_mask)

                roi_imgsz = roi_state.get("roi_imgsz", ROI_IMGSZ_START)

                # SAHI-inspired: upscale crop when athlete is small
                # This makes tiny athletes appear bigger to YOLO
                upscale = 1
                if bbox_w * bbox_h < ROI_UPSCALE_BBOX_AREA:
                    upscale = ROI_UPSCALE_FACTOR
                    h_crop, w_crop = crop_frame.shape[:2]
                    crop_frame_up = cv2.resize(crop_frame,
                                               (w_crop * upscale, h_crop * upscale),
                                               interpolation=cv2.INTER_LINEAR)
                else:
                    crop_frame_up = crop_frame

                results = model.predict(
                    crop_frame_up, classes=[0], verbose=False,
                    imgsz=roi_imgsz, conf=DETECTION_CONF
                )

                # Process ROI detections
                candidates = []
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    for b in results[0].boxes:
                        conf = float(b.conf[0])

                        # Remap to full-frame coordinates (account for upscaling)
                        ff_x1 = int(b.xyxy[0][0] / upscale) + cx1
                        ff_y1 = int(b.xyxy[0][1] / upscale) + cy1
                        ff_x2 = int(b.xyxy[0][2] / upscale) + cx1
                        ff_y2 = int(b.xyxy[0][3] / upscale) + cy1

                        det_foot_x = (ff_x1 + ff_x2) // 2
                        det_foot_y = ff_y2

                        # Filter 1: Radial proximity
                        bbox_diag = sqrt(bbox_w**2 + bbox_h**2)
                        max_jump = max(bbox_diag * MAX_JUMP_MULTIPLIER, MAX_JUMP_FLOOR_PX)
                        d = sqrt((det_foot_x - foot_x)**2 + (det_foot_y - foot_y)**2)
                        if d >= max_jump:
                            if frame_idx % 60 == 0:
                                log_fn(f"  [REJECT proximity] d={d:.0f} >= max_jump={max_jump:.0f}")
                            continue

                        # Filter 3: Size consistency
                        det_area = (ff_x2 - ff_x1) * (ff_y2 - ff_y1)
                        ref_area = max(bbox_w * bbox_h, peak_w * peak_h * 0.04)
                        if ref_area > 0:
                            size_ratio = 3.0 if ref_area < 400 else (2.0 if ref_area < 1000 else 1.5)
                            if det_area > ref_area * size_ratio:
                                if frame_idx % 60 == 0:
                                    log_fn(f"  [REJECT size] det_area={det_area} > ref_area={ref_area:.0f}*{size_ratio}")
                                continue

                        # Filter 4: World-coordinate + pixel-space direction filters
                        # World filters only when bbox is large (homography reliable)
                        # Pixel direction filter when bbox is small (at distance)
                        use_world_filters = (bbox_w * bbox_h > 3000)

                        if use_world_filters:
                            det_fx_w = (ff_x1 + ff_x2) / 2
                            det_fy_w = float(ff_y2)
                            det_px_w = np.array([[[det_fx_w, det_fy_w]]], dtype=np.float32)
                            H_tid = H_per_athlete.get(roi_tid, H)
                            det_world_w = cv2.perspectiveTransform(det_px_w, H_tid)[0][0]
                            det_world_y = float(det_world_w[1])
                            det_world_x = float(det_world_w[0])

                            # 4a: Backward check
                            if roi_tid in max_dist_reached:
                                hwm = max_dist_reached[roi_tid]
                                back_tolerance = 2.0 + hwm * 0.05
                                if det_world_y < hwm - back_tolerance:
                                    if frame_idx % 60 == 0:
                                        log_fn(f"  [REJECT backward] det_y={det_world_y:.1f} < hwm={hwm:.1f}-{back_tolerance:.1f}")
                                    continue

                            # 4b: Forward check
                            if roi_tid in smooth_dist:
                                current_dist = smooth_dist[roi_tid]
                                if det_world_y > current_dist + MAX_FORWARD_DIST_M:
                                    if frame_idx % 60 == 0:
                                        log_fn(f"  [REJECT forward] det_y={det_world_y:.1f} > smooth={current_dist:.1f}+{MAX_FORWARD_DIST_M}")
                                    continue

                            # 4b2: Absolute cap
                            if det_world_y > distance_m + 10.0:
                                if frame_idx % 60 == 0:
                                    log_fn(f"  [REJECT abs_cap] det_y={det_world_y:.1f} > {distance_m}+10.0")
                                continue

                            # 4c: Lateral check
                            if det_world_x < -LANE_X_MARGIN or det_world_x > TRACK_WIDTH_M + LANE_X_MARGIN:
                                if frame_idx % 60 == 0:
                                    log_fn(f"  [REJECT lateral] world_x={det_world_x:.2f} outside [-{LANE_X_MARGIN}, {TRACK_WIDTH_M + LANE_X_MARGIN:.2f}]")
                                continue
                        else:
                            # At distance: pixel-space direction filter
                            # Reject if detection foot is much BELOW (larger y) the
                            # last known position — that means it's behind the athlete
                            last_yolo = roi_state.get("last_yolo_foot")
                            if last_yolo is not None:
                                if det_foot_y > last_yolo[1] + 50:
                                    if frame_idx % 60 == 0:
                                        log_fn(f"  [REJECT pixel_back] foot_y={det_foot_y} > last={last_yolo[1]:.0f}+50")
                                    continue

                        candidates.append({
                            "conf": conf,
                            "bbox": (ff_x1, ff_y1, ff_x2, ff_y2),
                            "foot": (det_foot_x, det_foot_y),
                            "dist": d,
                            "area": det_area,
                        })

                best = None
                if candidates:
                    # Prefer closest to Kalman prediction — prevents jumping to bystanders
                    # Break ties with higher confidence
                    best = min(candidates, key=lambda c: (c["dist"], -c["conf"]))

                if best is not None:
                    # Update state with best detection
                    ff_x1, ff_y1, ff_x2, ff_y2 = best["bbox"]
                    conf = best["conf"]

                    # Reset Kalman velocity if recovering from a long lost streak
                    if roi_state["frames_lost"] >= 3 and kalman is not None:
                        kalman.state[2] = 0.0  # reset vx
                        kalman.state[3] = 0.0  # reset vy

                    roi_state["frames_lost"] = 0
                    roi_state["prev_foot"] = (foot_x, foot_y)
                    last_box_full[roi_tid] = (ff_x1, ff_y1, ff_x2, ff_y2)
                    last_conf[roi_tid] = conf

                    # Update peak bbox (only grow, never shrink)
                    det_w = ff_x2 - ff_x1
                    det_h = ff_y2 - ff_y1
                    if det_w > roi_state.get("peak_bbox_w", 0):
                        roi_state["peak_bbox_w"] = det_w
                    if det_h > roi_state.get("peak_bbox_h", 0):
                        roi_state["peak_bbox_h"] = det_h

                    # Foot position (full-frame)
                    raw_fx = (ff_x1 + ff_x2) / 2
                    raw_fy = float(ff_y2)
                    roi_state["last_yolo_foot"] = (raw_fx, raw_fy)

                    # Smooth foot
                    if roi_tid in smooth_foot:
                        old_sx, old_sy = smooth_foot[roi_tid]
                        sx = FOOT_SMOOTHING_ALPHA * raw_fx + (1 - FOOT_SMOOTHING_ALPHA) * old_sx
                        sy = FOOT_SMOOTHING_ALPHA * raw_fy + (1 - FOOT_SMOOTHING_ALPHA) * old_sy
                    else:
                        sx, sy = raw_fx, raw_fy
                    smooth_foot[roi_tid] = (sx, sy)

                    # Homography (per-athlete)
                    foot_px = np.array([[[sx, sy]]], dtype=np.float32)
                    H_tid = H_per_athlete.get(roi_tid, H)
                    foot_world = cv2.perspectiveTransform(foot_px, H_tid)[0][0]
                    world_x, world_y = float(foot_world[0]), float(foot_world[1])

                    # Smooth distance
                    if roi_tid in smooth_dist:
                        old_d = smooth_dist[roi_tid]
                        smoothed_d = DISTANCE_SMOOTHING_ALPHA * world_y + \
                                     (1 - DISTANCE_SMOOTHING_ALPHA) * old_d
                        # Clamp: distance can't change faster than physically possible
                        delta = smoothed_d - old_d
                        if abs(delta) > MAX_DIST_CHANGE_PER_FRAME:
                            smoothed_d = old_d + MAX_DIST_CHANGE_PER_FRAME * (1 if delta > 0 else -1)
                    else:
                        smoothed_d = world_y
                    smooth_dist[roi_tid] = smoothed_d
                    if smoothed_d > max_dist_reached.get(roi_tid, 0):
                        max_dist_reached[roi_tid] = smoothed_d

                    last_world_pos[roi_tid] = (world_x, smoothed_d)

                    # Timing — start
                    if roi_tid not in race_start_ms and smoothed_d > TIMER_START_THRESHOLD_M:
                        race_start_ms[roi_tid] = timestamp_ms
                        log_fn(f"ID{roi_tid} timer started at {timestamp_ms:.0f}ms "
                               f"(dist={smoothed_d:.1f}m)")

                    # Timing — finish
                    if roi_tid in race_start_ms and roi_tid not in race_finish_ms:
                        if smoothed_d >= distance_m:
                            if roi_tid not in first_cross_ts:
                                first_cross_ts[roi_tid] = timestamp_ms
                            finish_count[roi_tid] = finish_count.get(roi_tid, 0) + 1
                            if finish_count[roi_tid] >= FINISH_CONFIRM_FRAMES:
                                race_finish_ms[roi_tid] = first_cross_ts[roi_tid]
                                elapsed = race_finish_ms[roi_tid] - race_start_ms[roi_tid]
                                log_fn(f"ID{roi_tid} FINISHED: {elapsed/1000:.3f}s")
                        else:
                            # Only reset if we drop significantly below finish
                            if smoothed_d < distance_m - 3.0:
                                finish_count[roi_tid] = 0
                                if roi_tid in first_cross_ts:
                                    del first_cross_ts[roi_tid]

                    # ROI imgsz escalation (2 steps — upscale handles the rest)
                    if conf >= CONFIDENCE_THRESHOLD:
                        roi_state["roi_imgsz"] = ROI_IMGSZ_START  # >= 0.30 → 320
                    else:
                        roi_state["roi_imgsz"] = ROI_IMGSZ_MAX  # < 0.30 → 640

                    # Kalman update
                    kalman = roi_state.get("kalman")
                    if kalman is None:
                        roi_state["kalman"] = SimpleKalmanTracker(raw_fx, raw_fy)
                    else:
                        # predict() already called above for reference point
                        kalman.update(raw_fx, raw_fy)
                        max_vel = max(bbox_w, bbox_h, 30) * 0.2
                        kalman.clamp_velocity(max_vel)

                    # Store for optical flow on next frame
                    gray_crop = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
                    roi_state["prev_gray"] = gray_crop
                    roi_state["flow_point"] = np.array([[[raw_fx - cx1, raw_fy - cy1]]], dtype=np.float32)
                    roi_state["flow_crop_origin"] = (cx1, cy1)

                    # Update draw info
                    last_draw_info[roi_tid] = {
                        "bbox": (ff_x1, ff_y1, ff_x2, ff_y2),
                        "conf": conf,
                        "foot": (sx, sy),
                        "world": (world_x, smoothed_d),
                    }

                    if frame_idx % 30 == 0:
                        det_count = len(candidates)
                        log_fn(f"Frame {frame_idx:5d} | ROI   | ID{roi_tid}:lost=0 "
                               f"px=({sx:.1f},{sy:.1f}) dist={smoothed_d:.1f}m | "
                               f"det={det_count} | conf={conf:.3f} | "
                               f"roi_imgsz={roi_state['roi_imgsz']}")
                else:
                    # No YOLO detection — escalate ROI imgsz to find the athlete
                    if roi_state.get("roi_imgsz", ROI_IMGSZ_START) < ROI_IMGSZ_MAX:
                        roi_state["roi_imgsz"] = min(
                            roi_state.get("roi_imgsz", ROI_IMGSZ_START) + IMGSZ_STEP,
                            ROI_IMGSZ_MAX
                        )

                    # Use Kalman prediction + optical flow fallback
                    kalman = roi_state.get("kalman")
                    recovered = False

                    if kalman is not None:
                        # predict() already called at top of ROI loop — just get current position
                        pred_x, pred_y = kalman.get_position()

                        # Decay velocity during lost frames — but not for small bboxes
                        # At distance, clamped velocity IS the correct velocity
                        if bbox_w * bbox_h > 400:
                            kalman.state[2] *= 0.95  # vx
                            kalman.state[3] *= 0.95  # vy

                        prev_gray = roi_state.get("prev_gray")
                        flow_point = roi_state.get("flow_point")

                        # Skip OF for small bboxes — OF tracks background texture at distance
                        use_of = (prev_gray is not None and flow_point is not None
                                  and bbox_w * bbox_h > 400)

                        if use_of:
                            curr_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

                            # Optical flow requires same-size images — crop may differ due to edge clamping
                            prev_h, prev_w = prev_gray.shape[:2]
                            curr_h, curr_w = curr_gray.shape[:2]
                            if (prev_h, prev_w) != (curr_h, curr_w):
                                curr_gray = cv2.resize(curr_gray, (prev_w, prev_h))

                            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                                prev_gray, curr_gray, flow_point, None,
                                winSize=(15, 15), maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                            )

                            if status is not None and status[0][0] == 1:
                                flow_fx = float(new_pts[0][0][0]) + cx1
                                flow_fy = float(new_pts[0][0][1]) + cy1

                                # Validate: OF must agree with Kalman direction
                                kalman_vx, kalman_vy = kalman.get_velocity()
                                of_dx = flow_fx - foot_x
                                of_dy = flow_fy - foot_y

                                if of_dx * kalman_vx + of_dy * kalman_vy >= 0:
                                    # OF agrees with Kalman — use OF to update
                                    corrected_x, corrected_y = kalman.update(flow_fx, flow_fy)
                                else:
                                    # OF disagrees — ignore OF, use pure Kalman prediction
                                    corrected_x, corrected_y = pred_x, pred_y

                                old_x1, old_y1, old_x2, old_y2 = last_box_full[roi_tid]
                                old_w = old_x2 - old_x1
                                old_h = old_y2 - old_y1

                                # Clamp corrected position to drift cap
                                last_yolo = roi_state.get("last_yolo_foot")
                                if last_yolo is not None:
                                    max_drift = max(old_w, old_h, 30)
                                    if abs(corrected_x - last_yolo[0]) > max_drift:
                                        corrected_x = last_yolo[0] + max_drift * (1 if corrected_x > last_yolo[0] else -1)
                                    if abs(corrected_y - last_yolo[1]) > max_drift:
                                        corrected_y = last_yolo[1] + max_drift * (1 if corrected_y > last_yolo[1] else -1)

                                new_cx = int(corrected_x)
                                new_cy = int(corrected_y)
                                last_box_full[roi_tid] = (
                                    new_cx - old_w // 2, new_cy - old_h,
                                    new_cx + old_w // 2, new_cy
                                )

                                if roi_tid in smooth_foot:
                                    old_sx, old_sy = smooth_foot[roi_tid]
                                    sx = FOOT_SMOOTHING_ALPHA * corrected_x + (1 - FOOT_SMOOTHING_ALPHA) * old_sx
                                    sy = FOOT_SMOOTHING_ALPHA * corrected_y + (1 - FOOT_SMOOTHING_ALPHA) * old_sy
                                    smooth_foot[roi_tid] = (sx, sy)

                                    # Update world distance during lost frames so it doesn't freeze
                                    foot_px_kof = np.array([[[sx, sy]]], dtype=np.float32)
                                    H_tid = H_per_athlete.get(roi_tid, H)
                                    foot_world_kof = cv2.perspectiveTransform(foot_px_kof, H_tid)[0][0]
                                    world_y_kof = float(foot_world_kof[1])
                                    if roi_tid in smooth_dist:
                                        old_d = smooth_dist[roi_tid]
                                        smoothed_d = DISTANCE_SMOOTHING_ALPHA * world_y_kof + \
                                                     (1 - DISTANCE_SMOOTHING_ALPHA) * old_d
                                        delta = smoothed_d - old_d
                                        if abs(delta) > MAX_DIST_CHANGE_PER_FRAME:
                                            smoothed_d = old_d + MAX_DIST_CHANGE_PER_FRAME * (1 if delta > 0 else -1)
                                        smooth_dist[roi_tid] = smoothed_d
                                        last_world_pos[roi_tid] = (float(foot_world_kof[0]), smoothed_d)

                                roi_state["prev_gray"] = curr_gray
                                roi_state["flow_point"] = new_pts
                                roi_state["flow_crop_origin"] = (cx1, cy1)

                                roi_state["frames_lost"] += 1
                                recovered = True

                                if frame_idx % 30 == 0:
                                    log_fn(f"Frame {frame_idx:5d} | ROI   | ID{roi_tid}:"
                                           f"lost={roi_state['frames_lost']} (Kalman+OF) "
                                           f"px=({corrected_x:.1f},{corrected_y:.1f})")

                    if not recovered:
                        roi_state["frames_lost"] = roi_state.get("frames_lost", 0) + 1
                        if kalman is not None:
                            old_x1, old_y1, old_x2, old_y2 = last_box_full[roi_tid]
                            old_w = old_x2 - old_x1
                            old_h = old_y2 - old_y1
                            # Use foot_x/foot_y which were already drift-capped above
                            last_box_full[roi_tid] = (
                                foot_x - old_w // 2, foot_y - old_h,
                                foot_x + old_w // 2, foot_y
                            )

                            # Update smooth_foot/dist even on pure Kalman coast
                            if roi_tid in smooth_foot:
                                old_sx, old_sy = smooth_foot[roi_tid]
                                sx = FOOT_SMOOTHING_ALPHA * foot_x + (1 - FOOT_SMOOTHING_ALPHA) * old_sx
                                sy = FOOT_SMOOTHING_ALPHA * foot_y + (1 - FOOT_SMOOTHING_ALPHA) * old_sy
                                smooth_foot[roi_tid] = (sx, sy)

                                foot_px_coast = np.array([[[sx, sy]]], dtype=np.float32)
                                H_tid = H_per_athlete.get(roi_tid, H)
                                foot_world_coast = cv2.perspectiveTransform(foot_px_coast, H_tid)[0][0]
                                world_y_coast = float(foot_world_coast[1])
                                if roi_tid in smooth_dist:
                                    old_d = smooth_dist[roi_tid]
                                    smoothed_d = DISTANCE_SMOOTHING_ALPHA * world_y_coast + \
                                                 (1 - DISTANCE_SMOOTHING_ALPHA) * old_d
                                    delta = smoothed_d - old_d
                                    if abs(delta) > MAX_DIST_CHANGE_PER_FRAME:
                                        smoothed_d = old_d + MAX_DIST_CHANGE_PER_FRAME * (1 if delta > 0 else -1)
                                    smooth_dist[roi_tid] = smoothed_d
                                    last_world_pos[roi_tid] = (float(foot_world_coast[0]), smoothed_d)

                        if frame_idx % 30 == 0:
                            log_fn(f"Frame {frame_idx:5d} | ROI   | ID{roi_tid}:"
                                   f"lost={roi_state['frames_lost']} | det=0 (coast)")

        # ── Draw overlays ────────────────────────────────────────────────────
        _draw_all_overlays(frame, locked_ids, last_draw_info, last_box_full,
                           smooth_foot, smooth_dist, race_start_ms,
                           race_finish_ms, first_cross_ts, finish_count,
                           timestamp_ms, distance_m, frame_idx, current_imgsz,
                           in_roi_mode, athlete_roi, stale=False)

        out.write(frame)
        frame_idx += 1

        # Progress log every 100 frames
        if frame_idx % 100 == 0:
            log_fn(f"[progress] frame={frame_idx} imgsz={current_imgsz} "
                   f"locked={len(locked_ids)} roi={in_roi_mode}")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    out.release()

    # Print final results
    log_fn(f"\n{'='*40}")
    log_fn("RESULTS")
    log_fn(f"{'='*40}")
    video_results = []
    for tid in locked_ids:
        if tid in race_start_ms and tid in race_finish_ms:
            elapsed = (race_finish_ms[tid] - race_start_ms[tid]) / 1000.0
            log_fn(f"  Athlete ID{tid}: {elapsed:.3f}s ({distance_m}m)")
            video_results.append({"tid": tid, "time_s": elapsed, "distance_m": distance_m})
        elif tid in race_start_ms:
            log_fn(f"  Athlete ID{tid}: DID NOT FINISH")
            video_results.append({"tid": tid, "time_s": None, "distance_m": distance_m})
        else:
            log_fn(f"  Athlete ID{tid}: NEVER STARTED")
            video_results.append({"tid": tid, "time_s": None, "distance_m": distance_m})
    processing_elapsed = time.time() - processing_start_time
    mins, secs = divmod(processing_elapsed, 60)
    log_fn(f"\nProcessing time: {int(mins)}m {secs:.1f}s ({frame_idx} frames)")
    if frame_idx > 0:
        log_fn(f"Average speed:   {frame_idx / processing_elapsed:.1f} fps")
    log_fn(f"\nOutput video: {out_path}")
    log_fn(f"Bird's eye:   {bird_path}")

    return video_results


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING / OVERLAYS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_all_overlays(frame, locked_ids, last_draw_info, last_box_full,
                       smooth_foot, smooth_dist, race_start_ms,
                       race_finish_ms, first_cross_ts, finish_count,
                       timestamp_ms, distance_m, frame_idx, current_imgsz,
                       in_roi_mode, athlete_roi, stale=False):
    """Draw all overlays on a frame."""
    h, w = frame.shape[:2]
    alpha = 0.5 if stale else 1.0

    # Frame counter and mode (top-left)
    mode_str = "ROI" if in_roi_mode else "Full"
    info_text = f"Frame {frame_idx} | {mode_str} | imgsz={current_imgsz}"
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                _dim((255, 255, 255), alpha), 2)

    # Per-athlete overlays
    for tid in locked_ids:
        if tid not in last_draw_info:
            continue

        info = last_draw_info[tid]
        bbox = info.get("bbox")
        conf = info.get("conf", 0)
        foot = info.get("foot")
        world = info.get("world")

        # Assign color per athlete
        colors = [(0, 255, 0), (255, 100, 0), (0, 100, 255),
                  (255, 255, 0), (255, 0, 255)]
        color = colors[list(locked_ids.keys()).index(tid) % len(colors)]

        # Bounding box
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          _dim(color, alpha), 2)

            # ID, confidence, imgsz label
            label = f"ID{tid} {conf:.0%} imgsz={current_imgsz}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        _dim(color, alpha), 1)

        # Red dot at smoothed foot
        if foot:
            fx, fy = int(foot[0]), int(foot[1])
            cv2.circle(frame, (fx, fy), 6, _dim((0, 0, 255), alpha), -1)

        # World position text
        if world:
            wx, wy = world
            pos_text = f"({wx:.1f}m, {wy:.1f}m)"
            if bbox:
                cv2.putText(frame, pos_text, (bbox[0], bbox[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            _dim((0, 255, 255), alpha), 1)

        # Timer
        if tid in race_start_ms:
            if tid in race_finish_ms:
                elapsed = (race_finish_ms[tid] - race_start_ms[tid]) / 1000.0
                timer_text = f"ID{tid}: {elapsed:.3f}s FINISHED"
                timer_color = (0, 255, 0)
            else:
                elapsed = (timestamp_ms - race_start_ms[tid]) / 1000.0
                timer_text = f"ID{tid}: {elapsed:.3f}s"
                timer_color = (255, 255, 255)

                # Show finish confirmation progress
                fc = finish_count.get(tid, 0)
                if fc > 0:
                    timer_text += f" [{fc}/{FINISH_CONFIRM_FRAMES}]"
                    if tid in first_cross_ts:
                        cross_elapsed = (first_cross_ts[tid] - race_start_ms[tid]) / 1000.0
                        timer_text += f" cross@{cross_elapsed:.3f}s"
                    timer_color = (0, 255, 255)

            y_offset = 80 + list(locked_ids.keys()).index(tid) * 50
            cv2.putText(frame, timer_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        _dim(timer_color, alpha), 3)

    # ROI crop rectangle
    if in_roi_mode and athlete_roi:
        for tid, roi_state in athlete_roi.items():
            crop = roi_state.get("last_crop")
            if crop:
                cx1, cy1, cx2, cy2 = crop
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2),
                              _dim((255, 0, 255), alpha), 2)
                cv2.putText(frame, "ROI", (cx1, cy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            _dim((255, 0, 255), alpha), 1)


def _dim(color, alpha):
    """Dim a BGR color by alpha factor."""
    if alpha >= 1.0:
        return color
    return tuple(int(c * alpha) for c in color)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()

    # Apply dark theme to ttk widgets
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".", background=BG_DARK, foreground=FG_TEXT,
                    fieldbackground=BG_ENTRY)
    style.configure("TCombobox", fieldbackground=BG_ENTRY, background=BG_ENTRY,
                    foreground=FG_TEXT)
    style.map("TCombobox",
              fieldbackground=[("readonly", BG_ENTRY)],
              selectbackground=[("readonly", BG_ENTRY)],
              selectforeground=[("readonly", FG_TEXT)])

    app = CalibrationApp(root)
    root.mainloop()
