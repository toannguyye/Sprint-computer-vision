#!/usr/bin/env python3
"""
Track & Field — Single-Phone Timing App
========================================
Calibration GUI + YOLO-based processing pipeline with ROI cropping and CSRT fallback.
Processes pre-recorded 4K 60fps video to automatically time athletes
running 10-100m races using computer vision.

Requirements:
    pip install ultralytics opencv-python opencv-contrib-python pillow
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

# ── Track Constants ─────────────────────────────────────────────────────────
TRACK_WIDTH_M = 1.22

# ── YOLO Full-Frame ─────────────────────────────────────────────────────────
IMGSZ_START = 320
IMGSZ_STEP = 160
ROI_SWITCH_IMGSZ = 960
DETECTION_CONF = 0.15
CONFIDENCE_THRESHOLD = 0.30

# ── ROI Cropping ────────────────────────────────────────────────────────────
CROP_MIN_PX = 80
MAX_LOST_FRAMES = 300
ROI_IMGSZ_START = 320
ROI_IMGSZ_MAX = 1280
ROI_UPSCALE_BBOX_AREA = 2000
ROI_UPSCALE_FACTOR = 2
MAX_JUMP_PX = 200
MAX_FORWARD_DIST_M = 10.0
LANE_X_MARGIN = 1.5

# ── CSRT ────────────────────────────────────────────────────────────────────
CSRT_SWITCH_DIST = 40.0
CSRT_REINIT_INTERVAL = 60
CSRT_RECOVERY_INTERVAL = 30
CSRT_FOOT_SMOOTHING_ALPHA = 0.15
CSRT_DISTANCE_ALPHA = 0.15
CSRT_DIST_CLAMP = 0.12

# ── Smoothing ───────────────────────────────────────────────────────────────
FOOT_SMOOTHING_ALPHA = 0.25
DISTANCE_SMOOTHING_ALPHA = 0.25
MAX_DIST_CHANGE_PER_FRAME = 0.25

# ── Timing ──────────────────────────────────────────────────────────────────
FINISH_CONFIRM_FRAMES = 5
TIMER_START_THRESHOLD_M = 0.0

# ── GUI ─────────────────────────────────────────────────────────────────────
DISTANCE_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ATHLETE_OPTIONS = [1, 2, 3, 4, 5]
VIDEO_EXTENSIONS = (".MOV", ".mov", ".mp4", ".MP4", ".avi", ".AVI", ".mkv", ".MKV")

VIDEO_DIR = Path("videosamples")
OUTPUT_DIR = Path("runs/homography")
CALIB_FILE = Path("calibration_data.json")

# ── Dark Theme ──────────────────────────────────────────────────────────────
BG_DARK = "#1e1e1e"
BG_PANEL = "#252526"
BG_ENTRY = "#333333"
FG_TEXT = "#cccccc"
FG_BRIGHT = "#ffffff"
ACCENT = "#0078d4"
GREEN = "#4ec9b0"
RED = "#f44747"
YELLOW = "#dcdcaa"
POINT_COLORS = ["#ff4444", "#44ff44", "#4444ff", "#ffff44"]
LANE_COLORS = ["#44ff44", "#4488ff", "#ff8844", "#ff44ff", "#44ffff"]
POINT_NAMES = ["SL", "SR", "FL", "FR"]
ATHLETE_COLORS_BGR = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (0, 255, 255), (255, 0, 255)
]


# ═══════════════════════════════════════════════════════════════════════════
#  Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════

class SimpleKalmanTracker:
    """Lightweight 2D Kalman filter: state [x, y, vx, vy], measurement [x, y]."""

    def __init__(self, x, y):
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4) * 100.0
        self.Q = np.eye(4)
        self.Q[0, 0] = 0.5
        self.Q[1, 1] = 0.5
        self.Q[2, 2] = 0.1
        self.Q[3, 3] = 0.1
        self.R = np.eye(2) * 4.0
        self.F = np.eye(4)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0
        self.H_mat = np.zeros((2, 4))
        self.H_mat[0, 0] = 1.0
        self.H_mat[1, 1] = 1.0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.state[0]), float(self.state[1])

    def update(self, mx, my):
        z = np.array([mx, my], dtype=np.float64)
        y = z - self.H_mat @ self.state
        S = self.H_mat @ self.P @ self.H_mat.T + self.R
        K = self.P @ self.H_mat.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H_mat) @ self.P
        # Clamp velocity
        max_vel = 30.0
        self.state[2] = np.clip(self.state[2], -max_vel, max_vel)
        self.state[3] = np.clip(self.state[3], -max_vel, max_vel)
        return float(self.state[0]), float(self.state[1])

    def velocity(self):
        return float(self.state[2]), float(self.state[3])


# ═══════════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model():
    """Load YOLO model, preferring ONNX."""
    onnx_path = Path("yolo26n.onnx")
    pt_path = Path("yolo26n.pt")
    if onnx_path.exists():
        return YOLO(str(onnx_path), task="detect")
    elif pt_path.exists():
        return YOLO(str(pt_path))
    else:
        raise FileNotFoundError("No YOLO model found (yolo26n.onnx or yolo26n.pt)")


# ═══════════════════════════════════════════════════════════════════════════
#  Appearance Matching
# ═══════════════════════════════════════════════════════════════════════════

def compute_color_hist(frame, x1, y1, x2, y2):
    """16-bin HSV histogram on middle 60% of bbox (torso)."""
    h = y2 - y1
    top = y1 + int(h * 0.2)
    bot = y2 - int(h * 0.2)
    if bot <= top or x2 <= x1:
        return None
    roi = frame[max(0, top):bot, max(0, x1):x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def compute_template(frame, x1, y1, x2, y2):
    """32x64 grayscale template for matching."""
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 64), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def appearance_score(ref_hist, ref_template, frame, x1, y1, x2, y2):
    """Combined appearance score: 0.4 * hist + 0.6 * template."""
    score = 0.5
    count = 0
    if ref_hist is not None:
        hist = compute_color_hist(frame, x1, y1, x2, y2)
        if hist is not None:
            h_score = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
            score = h_score * 0.4
            count += 0.4
    if ref_template is not None:
        tmpl = compute_template(frame, x1, y1, x2, y2)
        if tmpl is not None:
            match = cv2.matchTemplate(
                tmpl, ref_template, cv2.TM_CCOEFF_NORMED
            )
            t_score = float(match[0, 0])
            score += t_score * 0.6
            count += 0.6
    return score / count if count > 0 else 0.5


def compute_motion_energy(prev_frame, curr_frame, x1, y1, x2, y2, threshold=25):
    """Fraction of pixels with significant motion within bbox."""
    if prev_frame is None or x2 <= x1 or y2 <= y1:
        return 0.5
    r1 = cv2.cvtColor(prev_frame[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
    r2 = cv2.cvtColor(curr_frame[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
    if r1.shape != r2.shape or r1.size == 0:
        return 0.5
    diff = cv2.absdiff(r1, r2)
    moving = np.count_nonzero(diff > threshold)
    total = max(1, diff.size)
    return min(1.0, moving / (0.3 * total))


# ═══════════════════════════════════════════════════════════════════════════
#  Homography Helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_homography(src_points, distance_m):
    """Compute homography from 4 pixel points to world coordinates."""
    src_pts = np.float32(src_points)
    dst_pts = np.float32([
        [0.0, 0.0],
        [TRACK_WIDTH_M, 0.0],
        [0.0, distance_m],
        [TRACK_WIDTH_M, distance_m],
    ])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H


def pixel_to_world(px, py, H):
    """Convert pixel (px, py) to world (x, y) using homography."""
    pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)[0][0]
    return float(world[0]), float(world[1])


def generate_birds_eye(frame, H, distance_m, output_path):
    """Generate a top-down warped view of the track."""
    scale = 20
    dst_w = int(TRACK_WIDTH_M * scale) + 100
    dst_h = int(distance_m * scale) + 100
    offset_x, offset_y = 50, 50

    dst_pts_scaled = np.float32([
        [offset_x, offset_y],
        [offset_x + TRACK_WIDTH_M * scale, offset_y],
        [offset_x, offset_y + distance_m * scale],
        [offset_x + TRACK_WIDTH_M * scale, offset_y + distance_m * scale],
    ])

    H_inv = np.linalg.inv(H)
    src_pts_for_warp = np.float32([
        [0.0, 0.0], [TRACK_WIDTH_M, 0.0],
        [0.0, distance_m], [TRACK_WIDTH_M, distance_m],
    ]).reshape(-1, 1, 2)
    corners_px = cv2.perspectiveTransform(src_pts_for_warp, H_inv)

    H_bird, _ = cv2.findHomography(corners_px.reshape(-1, 2), dst_pts_scaled)
    bird_img = cv2.warpPerspective(frame, H_bird, (dst_w, dst_h))

    interval = 5 if distance_m <= 30 else 10
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


# ═══════════════════════════════════════════════════════════════════════════
#  Right-Side Cutoff Mask
# ═══════════════════════════════════════════════════════════════════════════

def build_right_mask(lane_src_points, width, height):
    """Build mask that blacks out everything to the right of the track.

    Returns (track_mask, track_mask_3ch) — single and 3-channel masks.
    White (255) = keep, Black (0) = hide.
    """
    # Find rightmost SR and FR across all lanes
    sr = list(lane_src_points[0][1])
    fr = list(lane_src_points[0][3])
    for lane_pts in lane_src_points:
        if lane_pts[1][0] > sr[0]:
            sr = list(lane_pts[1])
        if lane_pts[3][0] > fr[0]:
            fr = list(lane_pts[3])

    # Extrapolate SR->FR line to top and bottom of frame
    dx = fr[0] - sr[0]
    dy = fr[1] - sr[1]
    if dy != 0:
        top_x = int(sr[0] + dx * (0 - sr[1]) / dy)
        bot_x = int(sr[0] + dx * (height - sr[1]) / dy)
    else:
        top_x = sr[0]
        bot_x = sr[0]

    # Polygon covering everything to the RIGHT
    cutoff_poly = np.array([
        [top_x, 0], [bot_x, height], [width, height], [width, 0],
    ], dtype=np.int32)

    track_mask = np.ones((height, width), dtype=np.uint8) * 255
    cv2.fillPoly(track_mask, [cutoff_poly], 0)
    track_mask_3ch = cv2.merge([track_mask, track_mask, track_mask])
    return track_mask, track_mask_3ch


# ═══════════════════════════════════════════════════════════════════════════
#  Debug Overlay Drawing
# ═══════════════════════════════════════════════════════════════════════════

def draw_overlays(frame, locked_ids, lock_order, last_box_full,
                  smooth_foot, smooth_dist, race_start_ms, race_finish_ms,
                  first_cross_ts, finish_count, timestamp_ms, distance_m,
                  frame_idx, current_imgsz, in_roi_mode, athlete_roi):
    """Draw bounding boxes, foot dots, timers, and debug info on frame."""
    height, width = frame.shape[:2]

    # Top-left info
    mode_str = "ROI" if in_roi_mode else "Full"
    info = f"Frame {frame_idx:5d} | {mode_str} | imgsz={current_imgsz}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    timer_y = 70
    for i, tid in enumerate(lock_order):
        color = ATHLETE_COLORS_BGR[i % len(ATHLETE_COLORS_BGR)]

        # Bounding box
        if tid in last_box_full:
            bx1, by1, bx2, by2 = [int(v) for v in last_box_full[tid]]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

            # ID label above bbox
            roi_info = ""
            if tid in athlete_roi:
                rs = athlete_roi[tid]
                roi_info = f" roi={rs.get('roi_imgsz', '?')}"
            label = f"ID{tid}{roi_info}"
            cv2.putText(frame, label, (bx1, by1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Smoothed foot dot
        if tid in smooth_foot:
            sx, sy = smooth_foot[tid]
            cv2.circle(frame, (int(sx), int(sy)), 5, (0, 0, 255), -1)

        # World position
        if tid in smooth_dist and tid in smooth_foot:
            sx, sy = smooth_foot[tid]
            d = smooth_dist[tid]
            cv2.putText(frame, f"{d:.1f}m", (int(sx) + 10, int(sy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Timer
        if tid in race_finish_ms:
            elapsed = (race_finish_ms[tid] - race_start_ms[tid]) / 1000.0
            text = f"ID{tid}: {elapsed:.3f}s FINISHED"
            cv2.putText(frame, text, (10, timer_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif tid in race_start_ms:
            elapsed = (timestamp_ms - race_start_ms[tid]) / 1000.0
            fc = finish_count.get(tid, 0)
            if fc > 0:
                cross_t = first_cross_ts.get(tid)
                cross_str = ""
                if cross_t is not None:
                    cross_str = f" cross@{(cross_t - race_start_ms[tid])/1000:.3f}s"
                text = f"ID{tid}: {elapsed:.3f}s [{fc}/{FINISH_CONFIRM_FRAMES}]{cross_str}"
                cv2.putText(frame, text, (10, timer_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                text = f"ID{tid}: {elapsed:.3f}s"
                cv2.putText(frame, text, (10, timer_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        timer_y += 35

    # ROI crop rectangle
    if in_roi_mode:
        for tid, rs in athlete_roi.items():
            lc = rs.get("last_crop")
            if lc is not None:
                cx1, cy1, cx2, cy2 = lc
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 0, 255), 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Tracking Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_tracking(model, video_path, src_points, distance_m, max_athletes,
                 log_fn=print, stop_check=lambda: False):
    """Main tracking pipeline: Full-Frame YOLO -> ROI YOLO -> ROI CSRT."""

    # ── Video setup ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_fn(f"ERROR: Cannot open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_name = Path(video_path).stem

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{base_name}_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    log_fn(f"Video: {base_name} | {width}x{height} @ {fps:.1f}fps | {total_frames} frames")
    log_fn(f"Distance: {distance_m}m | Athletes: {max_athletes}")

    # ── Normalize calibration points ─────────────────────────────────────
    if isinstance(src_points[0][0], (int, float)):
        lane_src_points = [src_points]
    else:
        lane_src_points = src_points

    # ── Per-lane homography ──────────────────────────────────────────────
    H_lanes = []
    for lane_pts in lane_src_points:
        H_lane = compute_homography(lane_pts, distance_m)
        H_lanes.append(H_lane)
    H = H_lanes[0]  # default

    # ── Right-side cutoff mask ───────────────────────────────────────────
    track_mask, track_mask_3ch = build_right_mask(lane_src_points, width, height)

    # ── Bird's eye view ──────────────────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        bird_path = OUTPUT_DIR / f"{base_name}_bird_eye.png"
        generate_birds_eye(first_frame, H, distance_m, bird_path)
        log_fn(f"Bird's eye saved: {bird_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Tracking state ───────────────────────────────────────────────────
    locked_ids = {}
    lock_order = []
    H_per_athlete = {}
    current_imgsz = IMGSZ_START
    in_roi_mode = False
    frame_idx = 0

    # Per-athlete dicts
    smooth_foot = {}
    smooth_dist = {}
    last_confirmed_dist = {}
    max_dist_reached = {}
    last_box_full = {}
    race_start_ms = {}
    race_finish_ms = {}
    first_cross_ts = {}
    finish_count = {}
    lost_frames = {}
    athlete_roi = {}
    ref_hist = {}
    ref_template = {}
    ref_area = {}
    prev_frame_raw = None


    video_results = []
    start_time = time.time()

    # ── Main loop ────────────────────────────────────────────────────────
    while True:
        if stop_check():
            log_fn("Processing stopped by user.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # ════════════════════════════════════════════════════════════════
        #  PHASE 1: Full-Frame YOLO
        # ════════════════════════════════════════════════════════════════
        if not in_roi_mode:
            detect_frame = cv2.bitwise_and(frame, track_mask_3ch)
            results = model.track(
                detect_frame, classes=[0], persist=True,
                imgsz=current_imgsz, conf=DETECTION_CONF, verbose=False
            )

            detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    tid_tensor = box.id
                    if tid_tensor is None:
                        continue
                    tid = int(tid_tensor.item())
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append((tid, conf, x1, y1, x2, y2))

            # ── Athlete locking ──────────────────────────────────────
            for tid, conf, x1, y1, x2, y2 in detections:
                if tid in locked_ids:
                    continue
                if len(locked_ids) >= max_athletes:
                    break
                foot_x = (x1 + x2) / 2
                foot_y = float(y2)
                wx, wy = pixel_to_world(foot_x, foot_y, H)

                x_min = -TRACK_WIDTH_M * (max_athletes - 1) - 0.5
                x_max = TRACK_WIDTH_M * max_athletes + 0.5
                if x_min < wx < x_max:
                    locked_ids[tid] = True
                    lane_idx = len(lock_order)
                    lock_order.append(tid)
                    H_per_athlete[tid] = (
                        H_lanes[lane_idx] if lane_idx < len(H_lanes) else H_lanes[0]
                    )
                    lost_frames[tid] = 0
                    log_fn(f"  Locked athlete ID{tid} (lane {lane_idx + 1})")

            # ── Process locked athletes ──────────────────────────────
            for tid, conf, x1, y1, x2, y2 in detections:
                if tid not in locked_ids:
                    continue

                H_a = H_per_athlete[tid]
                foot_x = (x1 + x2) / 2
                foot_y = float(y2)

                # Proximity filter
                if tid in smooth_foot:
                    old_fx, old_fy = smooth_foot[tid]
                    jump = sqrt((foot_x - old_fx) ** 2 + (foot_y - old_fy) ** 2)
                    if jump > MAX_JUMP_PX:
                        continue

                # Smooth foot
                if tid in smooth_foot:
                    old_sx, old_sy = smooth_foot[tid]
                    sx = FOOT_SMOOTHING_ALPHA * foot_x + (1 - FOOT_SMOOTHING_ALPHA) * old_sx
                    sy = FOOT_SMOOTHING_ALPHA * foot_y + (1 - FOOT_SMOOTHING_ALPHA) * old_sy
                else:
                    sx, sy = foot_x, foot_y
                smooth_foot[tid] = (sx, sy)

                # World coordinates from smoothed foot
                wx, wy = pixel_to_world(sx, sy, H_a)

                # Smooth distance
                if tid in smooth_dist:
                    old_d = smooth_dist[tid]
                    new_d = DISTANCE_SMOOTHING_ALPHA * wy + (1 - DISTANCE_SMOOTHING_ALPHA) * old_d
                    delta = new_d - old_d
                    if abs(delta) > MAX_DIST_CHANGE_PER_FRAME:
                        new_d = old_d + MAX_DIST_CHANGE_PER_FRAME * (1 if delta > 0 else -1)
                    smooth_dist[tid] = new_d
                else:
                    smooth_dist[tid] = wy
                last_confirmed_dist[tid] = smooth_dist[tid]
                max_dist_reached[tid] = max(max_dist_reached.get(tid, 0), smooth_dist[tid])

                # Update bbox, appearance refs
                last_box_full[tid] = (x1, y1, x2, y2)
                lost_frames[tid] = 0
                bbox_area = (x2 - x1) * (y2 - y1)
                ref_area[tid] = bbox_area

                # Appearance update
                h = compute_color_hist(frame, int(x1), int(y1), int(x2), int(y2))
                t = compute_template(frame, int(x1), int(y1), int(x2), int(y2))
                alpha_app = 0.20 if bbox_area > 2000 else 0.05
                if tid not in ref_hist or ref_hist[tid] is None:
                    ref_hist[tid] = h
                elif h is not None:
                    ref_hist[tid] = alpha_app * h + (1 - alpha_app) * ref_hist[tid]
                if tid not in ref_template or ref_template[tid] is None:
                    ref_template[tid] = t
                elif t is not None:
                    ref_template[tid] = alpha_app * t + (1 - alpha_app) * ref_template[tid]

                # Timing: start
                smoothed_d = smooth_dist[tid]
                if tid not in race_start_ms and smoothed_d > TIMER_START_THRESHOLD_M:
                    race_start_ms[tid] = timestamp_ms
                    log_fn(f"  ID{tid} race started at {timestamp_ms:.0f}ms")

                # Timing: finish (use smoothed_d for both cross and confirm)
                if tid in race_start_ms and tid not in race_finish_ms:
                    if smoothed_d >= distance_m:
                        if tid not in first_cross_ts:
                            first_cross_ts[tid] = timestamp_ms
                        finish_count[tid] = finish_count.get(tid, 0) + 1
                        if finish_count[tid] >= FINISH_CONFIRM_FRAMES:
                            _finish_ts = first_cross_ts.get(tid, timestamp_ms)
                            race_finish_ms[tid] = _finish_ts
                            elapsed = (race_finish_ms[tid] - race_start_ms[tid]) / 1000.0
                            log_fn(f"  ID{tid} FINISHED: {elapsed:.3f}s")
                    else:
                        if smoothed_d < distance_m - 5.0:
                            finish_count[tid] = 0
                            if tid in first_cross_ts:
                                del first_cross_ts[tid]

            # Mark lost for locked athletes not detected this frame
            detected_tids = {d[0] for d in detections}
            for tid in locked_ids:
                if tid not in detected_tids:
                    lost_frames[tid] = lost_frames.get(tid, 0) + 1

            # ── imgsz escalation / ROI switch ────────────────────────
            if not locked_ids and frame_idx > 0 and frame_idx % 30 == 0:
                if current_imgsz < ROI_SWITCH_IMGSZ:
                    current_imgsz = min(current_imgsz + IMGSZ_STEP, ROI_SWITCH_IMGSZ)
            elif locked_ids:
                locked_confs = [c for t, c, *_ in detections if t in locked_ids]
                any_lost = any(lost_frames.get(t, 0) >= 10 for t in locked_ids)
                if locked_confs:
                    min_conf = min(locked_confs)
                    if min_conf < CONFIDENCE_THRESHOLD:
                        if current_imgsz < ROI_SWITCH_IMGSZ:
                            current_imgsz = min(current_imgsz + IMGSZ_STEP, ROI_SWITCH_IMGSZ)
                        else:
                            in_roi_mode = True
                elif any_lost:
                    if current_imgsz < ROI_SWITCH_IMGSZ:
                        current_imgsz = min(current_imgsz + IMGSZ_STEP, ROI_SWITCH_IMGSZ)
                    else:
                        in_roi_mode = True

            # Switch to ROI mode
            if in_roi_mode and not athlete_roi:
                log_fn(f"  Switching to ROI mode at frame {frame_idx}, imgsz={current_imgsz}")
                for tid in locked_ids:
                    foot = smooth_foot.get(tid, (width // 2, height // 2))
                    bbox = last_box_full.get(tid, (0, 0, 50, 100))
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    athlete_roi[tid] = {
                        "frames_lost": 0,
                        "prev_foot": foot,
                        "last_crop": None,
                        "roi_imgsz": ROI_IMGSZ_START,
                        "kalman": SimpleKalmanTracker(foot[0], foot[1]),
                        "prev_gray": None,
                        "flow_point": None,
                        "flow_crop_origin": (0, 0),
                        "last_yolo_foot": foot,
                        "last_real_yolo_foot": foot,
                        "peak_bbox_w": max(bw, 30),
                        "peak_bbox_h": max(bh, 30),
                        "csrt_tracker": None,
                        "csrt_initialized": False,
                        "csrt_confidence": 0.0,
                        "csrt_fail_count": 0,
                        "csrt_good_frames": 0,
                    }

            # Log every 30 frames
            if frame_idx % 30 == 0:
                for tid in lock_order:
                    if tid in smooth_dist:
                        d = smooth_dist[tid]
                        lf = lost_frames.get(tid, 0)
                        log_fn(f"Frame {frame_idx:5d} | Full | ID{tid} "
                               f"dist={d:.1f}m | imgsz={current_imgsz} | lost={lf}")

        # ════════════════════════════════════════════════════════════════
        #  PHASE 2 & 3: ROI YOLO + CSRT Fallback
        # ════════════════════════════════════════════════════════════════
        else:
            for roi_tid in list(athlete_roi.keys()):
                if roi_tid in race_finish_ms:
                    continue
                if stop_check():
                    break

                roi_state = athlete_roi[roi_tid]
                H_a = H_per_athlete.get(roi_tid, H)
                _lost = roi_state["frames_lost"]

                if _lost >= MAX_LOST_FRAMES:
                    log_fn(f"  ID{roi_tid} lost for {MAX_LOST_FRAMES} frames, abandoning")
                    continue

                # ── Crop building ────────────────────────────────────
                kalman = roi_state["kalman"]
                pred_x, pred_y = kalman.predict()
                last_yolo = roi_state["last_yolo_foot"]

                # Freeze crop position after 3 lost frames
                if _lost > 3 and last_yolo is not None:
                    foot_x = int(last_yolo[0])
                    foot_y = int(last_yolo[1])
                else:
                    foot_x = int(pred_x)
                    foot_y = int(pred_y)

                # Bbox dimensions (use peak as floor)
                bbox = last_box_full.get(roi_tid, (0, 0, 50, 100))
                bbox_w = int(bbox[2] - bbox[0])
                bbox_h = int(bbox[3] - bbox[1])
                peak_w = roi_state["peak_bbox_w"]
                peak_h = roi_state["peak_bbox_h"]
                crop_bbox_w = max(bbox_w, int(peak_w * 0.25))
                crop_bbox_h = max(bbox_h, int(peak_h * 0.25))

                # Padding
                _lost_expand = min(_lost * 3, 150)
                pad_w = max(crop_bbox_w * 3, CROP_MIN_PX) + _lost_expand
                pad_h = max(crop_bbox_h * 3, CROP_MIN_PX) + _lost_expand

                cx1 = int(max(0, foot_x - pad_w))
                cy1 = int(max(0, foot_y - pad_h))
                cx2 = int(min(width, foot_x + pad_w))
                cy2 = int(min(height, foot_y + pad_h))
                roi_state["last_crop"] = (cx1, cy1, cx2, cy2)

                # Raw crop (unmasked) for CSRT, masked crop for YOLO
                raw_crop = frame[cy1:cy2, cx1:cx2]
                crop_frame = raw_crop.copy()
                if track_mask_3ch is not None:
                    crop_mask = track_mask_3ch[cy1:cy2, cx1:cx2]
                    crop_frame = cv2.bitwise_and(crop_frame, crop_mask)

                if raw_crop.size == 0:
                    roi_state["frames_lost"] += 1
                    continue

                # ── CSRT crop (large, unmasked — 6× bbox for search region) ──
                csrt_pad = max(max(bbox_w, bbox_h, 30) * 6, CROP_MIN_PX)
                csrt_cx1 = int(max(0, foot_x - csrt_pad))
                csrt_cy1 = int(max(0, foot_y - csrt_pad))
                csrt_cx2 = int(min(width, foot_x + csrt_pad))
                csrt_cy2 = int(min(height, foot_y + csrt_pad))
                csrt_crop = frame[csrt_cy1:csrt_cy2, csrt_cx1:csrt_cx2]

                # ── Upscaling ────────────────────────────────────────
                upscale = 1
                bbox_area = bbox_w * bbox_h
                if _lost > 30 or bbox_area < 500:
                    upscale = 3
                elif bbox_area < ROI_UPSCALE_BBOX_AREA:
                    upscale = ROI_UPSCALE_FACTOR

                if upscale > 1:
                    ch, cw = crop_frame.shape[:2]
                    crop_frame_up = cv2.resize(
                        crop_frame, (cw * upscale, ch * upscale),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    crop_frame_up = crop_frame

                # ── ROI imgsz ────────────────────────────────────────
                roi_imgsz = roi_state["roi_imgsz"]
                _roi_conf = 0.08 if bbox_area < 2000 else DETECTION_CONF

                # ── YOLO predict on crop ─────────────────────────────
                results = model.predict(
                    crop_frame_up, classes=[0], verbose=False,
                    imgsz=roi_imgsz, conf=_roi_conf
                )

                # ── Parse and remap detections ───────────────────────
                candidates = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        box = boxes[i]
                        conf = float(box.conf.item())
                        dx1, dy1, dx2, dy2 = box.xyxy[0].tolist()

                        # Remap from upscaled crop to full-frame
                        if upscale > 1:
                            dx1 /= upscale
                            dy1 /= upscale
                            dx2 /= upscale
                            dy2 /= upscale
                        dx1 += cx1
                        dy1 += cy1
                        dx2 += cx1
                        dy2 += cy1

                        det_foot_x = (dx1 + dx2) / 2
                        det_foot_y = float(dy2)
                        det_area = (dx2 - dx1) * (dy2 - dy1)

                        # ── Proximity filter ─────────────────────
                        diag = sqrt(bbox_w ** 2 + bbox_h ** 2)
                        prox_thresh = diag * 2 + _lost * 2
                        if bbox_area < 800:
                            prox_thresh = max(prox_thresh, 150)
                        elif bbox_area < 2000:
                            prox_thresh = max(prox_thresh, 100)

                        dist_to_pred = sqrt(
                            (det_foot_x - foot_x) ** 2 +
                            (det_foot_y - foot_y) ** 2
                        )
                        if dist_to_pred > prox_thresh:
                            continue

                        # ── Size filter ──────────────────────────
                        _ref_a = ref_area.get(roi_tid, 5000)
                        size_ratio = 4.0 if bbox_area < 800 else (3.0 if bbox_area < 2000 else 1.5)
                        if det_area > _ref_a * size_ratio:
                            continue

                        # ── World-coordinate filters (large bbox) ─
                        _conf_d = last_confirmed_dist.get(roi_tid, 0)
                        _hwm = max_dist_reached.get(roi_tid, 0)
                        det_wx, det_wy = pixel_to_world(det_foot_x, det_foot_y, H_a)

                        if bbox_area > 5000:
                            if det_wy < _hwm - (2.0 + _hwm * 0.05):
                                continue
                            if det_wy > _conf_d + MAX_FORWARD_DIST_M:
                                continue
                            if det_wy > distance_m + 10.0:
                                continue
                            if det_wx < -LANE_X_MARGIN or det_wx > TRACK_WIDTH_M + LANE_X_MARGIN:
                                continue
                        else:
                            # Pixel-direction filters (small bbox)
                            back_tol = 30 + _lost * 3
                            if det_foot_y > foot_y + back_tol:
                                continue
                            lat_tol = max(bbox_w * 2.0, 20) + _lost * 2
                            if abs(det_foot_x - foot_x) > lat_tol:
                                continue

                        # ── Appearance score ─────────────────────
                        app = appearance_score(
                            ref_hist.get(roi_tid), ref_template.get(roi_tid),
                            frame, int(dx1), int(dy1), int(dx2), int(dy2)
                        )

                        # ── Motion energy ────────────────────────
                        motion = compute_motion_energy(
                            prev_frame_raw, frame,
                            int(dx1), int(dy1), int(dx2), int(dy2)
                        )

                        candidates.append({
                            "conf": conf,
                            "bbox": (dx1, dy1, dx2, dy2),
                            "foot": (det_foot_x, det_foot_y),
                            "area": det_area,
                            "dist": dist_to_pred,
                            "app_score": app,
                            "motion": motion,
                            "world_x": det_wx,
                            "world_y": det_wy,
                        })

                # ── Select best candidate ────────────────────────────
                best = None
                if candidates:
                    # Reject low appearance
                    candidates = [c for c in candidates if c["app_score"] >= 0.15]
                    if candidates:
                        if bbox_area > 2000:
                            best = max(candidates,
                                       key=lambda c: (c["app_score"], -c["dist"], c["conf"]))
                        else:
                            _max_d = max(c["dist"] for c in candidates) or 1.0
                            _app_w = max(0.1, min(0.4, bbox_area / 5000.0))
                            _motion_w = 0.3
                            _dist_w = 1.0 - _app_w - _motion_w
                            best = min(candidates, key=lambda c: (
                                _dist_w * (c["dist"] / _max_d) +
                                _app_w * (1.0 - c["app_score"]) +
                                _motion_w * (1.0 - c.get("motion", 0.5))
                            ))

                        # Distance sanity check (scaled: generous near, tight far)
                        if best is not None:
                            _sim_wy = best["world_y"]
                            _cd = last_confirmed_dist.get(roi_tid, 0)
                            _fwd_tol = min(MAX_FORWARD_DIST_M, max(5.0, 15.0 - _cd * 0.1))
                            if _sim_wy < _cd - 3.0 or _sim_wy > _cd + _fwd_tol:
                                best = None

                # ── CSRT tracking ────────────────────────────────────
                _csrt = roi_state.get("csrt_tracker")
                _csrt_ok = False
                _csrt_foot = None

                # Late-init CSRT
                if not roi_state.get("csrt_initialized", False) and roi_tid in last_box_full:
                    _cbx1, _cby1, _cbx2, _cby2 = [int(v) for v in last_box_full[roi_tid]]
                    _cw = _cbx2 - _cbx1
                    _ch = _cby2 - _cby1
                    _crop_bx = max(0, _cbx1 - csrt_cx1)
                    _crop_by = max(0, _cby1 - csrt_cy1)
                    if _cw > 3 and _ch > 3:
                        try:
                            _csrt_init = cv2.TrackerCSRT_create()
                            _csrt_init.init(csrt_crop, (_crop_bx, _crop_by, _cw, _ch))
                            roi_state["csrt_tracker"] = _csrt_init
                            roi_state["csrt_initialized"] = True
                            roi_state["csrt_confidence"] = 0.7
                            _csrt = _csrt_init
                        except cv2.error:
                            pass

                # Update CSRT
                if _csrt is not None:
                    try:
                        _csrt_ok, _csrt_bbox = _csrt.update(csrt_crop)
                    except cv2.error:
                        _csrt_ok = False
                        roi_state["csrt_initialized"] = False
                        roi_state["csrt_tracker"] = None
                        _csrt = None

                    if _csrt_ok:
                        _cx, _cy, _cw, _ch = [int(v) for v in _csrt_bbox]
                        _csrt_ff_x1 = _cx + csrt_cx1
                        _csrt_ff_y1 = _cy + csrt_cy1
                        _csrt_ff_x2 = _csrt_ff_x1 + _cw
                        _csrt_ff_y2 = _csrt_ff_y1 + _ch
                        _csrt_foot_x = (_csrt_ff_x1 + _csrt_ff_x2) / 2
                        _csrt_foot_y = float(_csrt_ff_y2)

                        # Validate CSRT with world coordinates
                        _cf_wx, _cf_wy = pixel_to_world(
                            _csrt_foot_x, _csrt_foot_y, H_a
                        )
                        _cd = last_confirmed_dist.get(roi_tid, 0)
                        _csrt_fwd = min(15.0, max(5.0, 12.0 - _cd * 0.08))
                        _csrt_valid = (
                            _cf_wy > 0
                            and -LANE_X_MARGIN <= _cf_wx <= TRACK_WIDTH_M + LANE_X_MARGIN
                            and _cf_wy >= _cd - 2.0
                            and _cf_wy <= _cd + _csrt_fwd
                            and _cf_wy <= distance_m + 5.0
                        )
                        if _csrt_valid:
                            _csrt_foot = (_csrt_foot_x, _csrt_foot_y)
                            roi_state["csrt_confidence"] = min(
                                1.0, roi_state.get("csrt_confidence", 0.5) + 0.02
                            )
                        else:
                            _csrt_ok = False
                            roi_state["csrt_fail_count"] = roi_state.get("csrt_fail_count", 0) + 1
                            # Reinit from last known good bbox
                            if roi_tid in last_box_full:
                                _rb = last_box_full[roi_tid]
                                _rbw = int(_rb[2] - _rb[0])
                                _rbh = int(_rb[3] - _rb[1])
                                _rbx = max(0, int(_rb[0]) - csrt_cx1)
                                _rby = max(0, int(_rb[1]) - csrt_cy1)
                                if (_rbw > 3 and _rbh > 3 and
                                        _rbx + _rbw <= csrt_crop.shape[1] and
                                        _rby + _rbh <= csrt_crop.shape[0]):
                                    try:
                                        _csrt_fix = cv2.TrackerCSRT_create()
                                        _csrt_fix.init(csrt_crop, (_rbx, _rby, _rbw, _rbh))
                                        roi_state["csrt_tracker"] = _csrt_fix
                                        roi_state["csrt_initialized"] = True
                                        roi_state["csrt_confidence"] = 0.3
                                    except cv2.error:
                                        pass
                    else:
                        roi_state["csrt_fail_count"] = roi_state.get("csrt_fail_count", 0) + 1
                        # Reinit from last known good bbox on reject
                        if roi_tid in last_box_full:
                            _rb = last_box_full[roi_tid]
                            _rbw = int(_rb[2] - _rb[0])
                            _rbh = int(_rb[3] - _rb[1])
                            _rbx = max(0, int(_rb[0]) - csrt_cx1)
                            _rby = max(0, int(_rb[1]) - csrt_cy1)
                            if (_rbw > 3 and _rbh > 3 and
                                    _rbx + _rbw <= csrt_crop.shape[1] and
                                    _rby + _rbh <= csrt_crop.shape[0]):
                                try:
                                    _csrt_fix = cv2.TrackerCSRT_create()
                                    _csrt_fix.init(csrt_crop, (_rbx, _rby, _rbw, _rbh))
                                    roi_state["csrt_tracker"] = _csrt_fix
                                    roi_state["csrt_initialized"] = True
                                    roi_state["csrt_confidence"] = 0.3
                                except cv2.error:
                                    pass

                # Destroy CSRT after too many failures
                if roi_state.get("csrt_fail_count", 0) >= 10:
                    roi_state["csrt_initialized"] = False
                    roi_state["csrt_tracker"] = None
                    roi_state["csrt_fail_count"] = 0

                # ── Determine final detection ────────────────────────
                # Priority: YOLO > CSRT (only when YOLO fails)
                use_csrt_result = False

                if best is not None:
                    # YOLO detection accepted
                    raw_fx, raw_fy = best["foot"]
                    det_bbox = best["bbox"]

                    # Reinit CSRT from YOLO detection
                    _ybx1, _yby1, _ybx2, _yby2 = [int(v) for v in det_bbox]
                    _yw = _ybx2 - _ybx1
                    _yh = _yby2 - _yby1
                    _crop_yx = max(0, _ybx1 - csrt_cx1)
                    _crop_yy = max(0, _yby1 - csrt_cy1)
                    if (_yw > 3 and _yh > 3 and
                            _crop_yx + _yw <= csrt_crop.shape[1] and
                            _crop_yy + _yh <= csrt_crop.shape[0]):
                        try:
                            _csrt_new = cv2.TrackerCSRT_create()
                            _csrt_new.init(csrt_crop, (_crop_yx, _crop_yy, _yw, _yh))
                            roi_state["csrt_tracker"] = _csrt_new
                            roi_state["csrt_initialized"] = True
                            roi_state["csrt_confidence"] = 0.8
                            roi_state["csrt_fail_count"] = 0
                        except cv2.error:
                            pass

                elif _csrt_ok and _csrt_foot is not None:
                    # CSRT fallback (only at distance)
                    raw_fx, raw_fy = _csrt_foot
                    use_csrt_result = True
                    det_bbox = (_csrt_ff_x1, _csrt_ff_y1, _csrt_ff_x2, _csrt_ff_y2)

                else:
                    # Both failed — coast
                    roi_state["frames_lost"] += 1

                    # YOLO recovery attempt during long lost periods
                    if _lost >= CSRT_RECOVERY_INTERVAL and _lost % CSRT_RECOVERY_INTERVAL == 0:
                        ch, cw = raw_crop.shape[:2]
                        if cw > 10 and ch > 10:
                            recovery_up = cv2.resize(
                                raw_crop, (cw * 3, ch * 3),
                                interpolation=cv2.INTER_LINEAR
                            )
                            rec_results = model.predict(
                                recovery_up, classes=[0], verbose=False,
                                imgsz=640, conf=0.08
                            )
                            if (rec_results and len(rec_results) > 0 and
                                    rec_results[0].boxes is not None and
                                    len(rec_results[0].boxes) > 0):
                                # Pick closest to predicted position
                                rec_boxes = rec_results[0].boxes
                                best_rec = None
                                best_rec_dist = float("inf")
                                for ri in range(len(rec_boxes)):
                                    rb = rec_boxes[ri]
                                    rx1, ry1, rx2, ry2 = rb.xyxy[0].tolist()
                                    rx1 /= 3; ry1 /= 3; rx2 /= 3; ry2 /= 3
                                    rx1 += cx1; ry1 += cy1; rx2 += cx1; ry2 += cy1
                                    rfx = (rx1 + rx2) / 2
                                    rfy = float(ry2)
                                    # World-coordinate validation
                                    _rec_wx, _rec_wy = pixel_to_world(rfx, rfy, H_a)
                                    _rec_cd = last_confirmed_dist.get(roi_tid, 0)
                                    if _rec_wy <= 0:
                                        continue
                                    if _rec_wx < -LANE_X_MARGIN or _rec_wx > TRACK_WIDTH_M + LANE_X_MARGIN:
                                        continue
                                    if _rec_wy < _rec_cd - 3.0 or _rec_wy > _rec_cd + 15.0:
                                        continue
                                    if _rec_wy > distance_m + 5.0:
                                        continue
                                    rd = sqrt((rfx - foot_x) ** 2 + (rfy - foot_y) ** 2)
                                    if rd < best_rec_dist:
                                        best_rec_dist = rd
                                        best_rec = (rfx, rfy, rx1, ry1, rx2, ry2)
                                if best_rec is not None:
                                    raw_fx, raw_fy = best_rec[0], best_rec[1]
                                    det_bbox = best_rec[2:]
                                    best = {"foot": (raw_fx, raw_fy), "bbox": det_bbox}
                                    log_fn(f"  ID{roi_tid} YOLO recovery at frame {frame_idx}")
                                    # Fall through to detection processing below
                                else:
                                    # Kalman velocity decay
                                    kalman.state[2] *= 0.9
                                    kalman.state[3] *= 0.9
                                    prev_frame_raw = frame.copy()
                                    continue
                            else:
                                kalman.state[2] *= 0.9
                                kalman.state[3] *= 0.9
                                prev_frame_raw = frame.copy()
                                continue
                    else:
                        # Distance FROZEN during lost frames — just decay Kalman velocity
                        kalman.state[2] *= 0.9
                        kalman.state[3] *= 0.9
                        prev_frame_raw = frame.copy()
                        continue

                # If we got here via YOLO recovery, reset best for processing
                if best is not None and "foot" in best:
                    raw_fx, raw_fy = best["foot"]
                    det_bbox = best["bbox"]
                    use_csrt_result = False

                # ── Process accepted detection ───────────────────────
                _alpha_foot = CSRT_FOOT_SMOOTHING_ALPHA if use_csrt_result else FOOT_SMOOTHING_ALPHA
                _alpha_dist = CSRT_DISTANCE_ALPHA if use_csrt_result else DISTANCE_SMOOTHING_ALPHA
                _clamp_dist = CSRT_DIST_CLAMP if use_csrt_result else MAX_DIST_CHANGE_PER_FRAME

                # Asymmetric clamp: forward generous, backward resistant at far distance
                _clamp_back = _clamp_dist  # default: symmetric
                _cd_for_clamp = last_confirmed_dist.get(roi_tid, 0)
                if _cd_for_clamp > 70.0:
                    if use_csrt_result:
                        _clamp_dist = 0.15   # CSRT forward: ride upward spikes
                        _clamp_back = 0.01   # CSRT backward: ratchet — barely retreat
                    else:
                        _clamp_dist = 0.20   # YOLO forward: accurate, allow big jumps
                        _clamp_back = 0.04   # YOLO backward: light resistance

                # Smooth foot
                if roi_tid in smooth_foot:
                    old_sx, old_sy = smooth_foot[roi_tid]
                    sx = _alpha_foot * raw_fx + (1 - _alpha_foot) * old_sx
                    sy = _alpha_foot * raw_fy + (1 - _alpha_foot) * old_sy
                else:
                    sx, sy = raw_fx, raw_fy
                smooth_foot[roi_tid] = (sx, sy)

                # World coordinates — use RAW foot for homography
                # (pixel EMA lag gets amplified by homography at far distance)
                raw_wx, raw_wy = pixel_to_world(raw_fx, raw_fy, H_a)
                wx, wy = raw_wx, raw_wy

                # Smooth distance
                _prev_smooth_d = smooth_dist.get(roi_tid, None)
                if roi_tid in smooth_dist:
                    old_d = smooth_dist[roi_tid]
                    new_d = _alpha_dist * wy + (1 - _alpha_dist) * old_d
                    delta = new_d - old_d
                    if delta > _clamp_dist:
                        new_d = old_d + _clamp_dist
                    elif delta < -_clamp_back:
                        new_d = old_d - _clamp_back
                    smooth_dist[roi_tid] = new_d
                else:
                    smooth_dist[roi_tid] = wy

                if smooth_dist[roi_tid] > last_confirmed_dist.get(roi_tid, 0):
                    last_confirmed_dist[roi_tid] = smooth_dist[roi_tid]
                max_dist_reached[roi_tid] = max(
                    max_dist_reached.get(roi_tid, 0), smooth_dist[roi_tid]
                )

                smoothed_d = smooth_dist[roi_tid]

                # Far-distance diagnostic logging
                if smoothed_d > 70.0 and (frame_idx % 5 == 0 or use_csrt_result):
                    _raw_wy_log = wy
                    _delta_d = smoothed_d - _prev_smooth_d if _prev_smooth_d is not None else 0
                    _speed_mps = abs(_delta_d) * fps if fps > 0 else 0
                    src = "CSRT" if use_csrt_result else "YOLO"
                    log_fn(f"  [FAR_DIAG] F{frame_idx} {src} | raw_wy={_raw_wy_log:.1f} "
                           f"smooth_d={smoothed_d:.1f} delta={_delta_d:+.3f} "
                           f"speed={_speed_mps:.1f}m/s | foot=({sx:.0f},{sy:.0f}) "
                           f"bbox=({int(det_bbox[0])},{int(det_bbox[1])},{int(det_bbox[2])},{int(det_bbox[3])})")

                # Update tracking state
                roi_state["frames_lost"] = 0
                roi_state["prev_foot"] = (raw_fx, raw_fy)
                roi_state["last_yolo_foot"] = (raw_fx, raw_fy)
                if not use_csrt_result:
                    roi_state["last_real_yolo_foot"] = (raw_fx, raw_fy)
                kalman.update(raw_fx, raw_fy)

                # Reset Kalman velocity on reacquisition
                if _lost >= 3:
                    kalman.state[2] = 0.0
                    kalman.state[3] = 0.0

                # Update bbox
                if isinstance(det_bbox, tuple) and len(det_bbox) == 4:
                    last_box_full[roi_tid] = det_bbox
                    bw = det_bbox[2] - det_bbox[0]
                    bh = det_bbox[3] - det_bbox[1]
                    roi_state["peak_bbox_w"] = max(roi_state["peak_bbox_w"], bw)
                    roi_state["peak_bbox_h"] = max(roi_state["peak_bbox_h"], bh)
                    if not use_csrt_result:
                        ref_area[roi_tid] = bw * bh

                # Appearance update (YOLO only, not CSRT)
                if not use_csrt_result and isinstance(det_bbox, tuple) and len(det_bbox) == 4:
                    _dx1, _dy1, _dx2, _dy2 = [int(v) for v in det_bbox]
                    _ba = (_dx2 - _dx1) * (_dy2 - _dy1)
                    _a_alpha = 0.20 if _ba > 2000 else 0.05
                    _curr_app = 0.5  # default — may be overwritten below
                    h = compute_color_hist(frame, _dx1, _dy1, _dx2, _dy2)
                    t = compute_template(frame, _dx1, _dy1, _dx2, _dy2)
                    if h is not None:
                        if roi_tid not in ref_hist or ref_hist[roi_tid] is None:
                            ref_hist[roi_tid] = h
                        else:
                            # Gate: only update if appearance matches
                            _curr_app = appearance_score(
                                ref_hist.get(roi_tid), ref_template.get(roi_tid),
                                frame, _dx1, _dy1, _dx2, _dy2
                            )
                            if _curr_app > 0.2:
                                ref_hist[roi_tid] = _a_alpha * h + (1 - _a_alpha) * ref_hist[roi_tid]
                    if t is not None:
                        if roi_tid not in ref_template or ref_template[roi_tid] is None:
                            ref_template[roi_tid] = t
                        else:
                            if _curr_app > 0.2:
                                ref_template[roi_tid] = _a_alpha * t + (1 - _a_alpha) * ref_template[roi_tid]

                # ROI imgsz escalation
                if not use_csrt_result and best is not None:
                    _best_conf = best.get("conf", 0)
                    if _best_conf < CONFIDENCE_THRESHOLD and roi_imgsz < ROI_IMGSZ_MAX:
                        roi_state["roi_imgsz"] = min(roi_imgsz + IMGSZ_STEP, ROI_IMGSZ_MAX)
                    elif _best_conf >= CONFIDENCE_THRESHOLD and roi_imgsz > ROI_IMGSZ_START:
                        roi_state["roi_imgsz"] = max(roi_imgsz - IMGSZ_STEP, ROI_IMGSZ_START)

                # CSRT periodic reinit
                if not use_csrt_result:
                    roi_state["csrt_good_frames"] = roi_state.get("csrt_good_frames", 0) + 1
                    if roi_state["csrt_good_frames"] % CSRT_REINIT_INTERVAL == 0:
                        if isinstance(det_bbox, tuple) and len(det_bbox) == 4:
                            _rx1, _ry1, _rx2, _ry2 = [int(v) for v in det_bbox]
                            _rw = _rx2 - _rx1
                            _rh = _ry2 - _ry1
                            _rcx = max(0, _rx1 - csrt_cx1)
                            _rcy = max(0, _ry1 - csrt_cy1)
                            if (_rw > 3 and _rh > 3 and
                                    _rcx + _rw <= csrt_crop.shape[1] and
                                    _rcy + _rh <= csrt_crop.shape[0]):
                                try:
                                    _csrt_r = cv2.TrackerCSRT_create()
                                    _csrt_r.init(csrt_crop, (_rcx, _rcy, _rw, _rh))
                                    roi_state["csrt_tracker"] = _csrt_r
                                    roi_state["csrt_initialized"] = True
                                    roi_state["csrt_confidence"] = 0.7
                                    roi_state["csrt_fail_count"] = 0
                                except cv2.error:
                                    pass

                # ── Timing ───────────────────────────────────────────
                if roi_tid not in race_start_ms and smoothed_d > TIMER_START_THRESHOLD_M:
                    race_start_ms[roi_tid] = timestamp_ms
                    log_fn(f"  ID{roi_tid} race started at {timestamp_ms:.0f}ms")

                if roi_tid in race_start_ms and roi_tid not in race_finish_ms:
                    if smoothed_d >= distance_m:
                        if roi_tid not in first_cross_ts:
                            first_cross_ts[roi_tid] = timestamp_ms
                        finish_count[roi_tid] = finish_count.get(roi_tid, 0) + 1
                        if finish_count[roi_tid] >= FINISH_CONFIRM_FRAMES:
                            _finish_ts = first_cross_ts.get(roi_tid, timestamp_ms)
                            race_finish_ms[roi_tid] = _finish_ts
                            elapsed = (race_finish_ms[roi_tid] - race_start_ms[roi_tid]) / 1000.0
                            log_fn(f"  ID{roi_tid} FINISHED: {elapsed:.3f}s")
                    else:
                        if smoothed_d < distance_m - 5.0:
                            finish_count[roi_tid] = 0
                            if roi_tid in first_cross_ts:
                                del first_cross_ts[roi_tid]

                # Logging
                if frame_idx % 30 == 0 or (use_csrt_result and frame_idx % 10 == 0):
                    src = "CSRT" if use_csrt_result else "YOLO"
                    log_fn(f"Frame {frame_idx:5d} | ROI  | ID{roi_tid} "
                           f"dist={smoothed_d:.1f}m | {src} | lost={_lost}")

            # Check if all athletes finished — early exit
            if locked_ids and all(t in race_finish_ms for t in locked_ids):
                log_fn("All athletes finished — stopping early.")
                # Draw final overlay and write remaining
                draw_overlays(frame, locked_ids, lock_order, last_box_full,
                              smooth_foot, smooth_dist, race_start_ms, race_finish_ms,
                              first_cross_ts, finish_count, timestamp_ms, distance_m,
                              frame_idx, current_imgsz, in_roi_mode, athlete_roi)
                out.write(frame)
                break

        # ── Draw overlays and write frame ────────────────────────────────
        draw_overlays(frame, locked_ids, lock_order, last_box_full,
                      smooth_foot, smooth_dist, race_start_ms, race_finish_ms,
                      first_cross_ts, finish_count, timestamp_ms, distance_m,
                      frame_idx, current_imgsz, in_roi_mode, athlete_roi)
        out.write(frame)
        prev_frame_raw = frame.copy()
        frame_idx += 1

        # Progress logging
        if frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            elapsed_t = time.time() - start_time
            fps_proc = frame_idx / elapsed_t if elapsed_t > 0 else 0
            log_fn(f"  Progress: {frame_idx}/{total_frames} ({pct:.0f}%) | {fps_proc:.1f} fps")

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    out.release()

    elapsed_total = time.time() - start_time
    log_fn(f"\nProcessing complete: {elapsed_total:.1f}s | Output: {out_path}")

    # ── Results ──────────────────────────────────────────────────────────
    for tid in lock_order:
        if tid in race_finish_ms and tid in race_start_ms:
            t = (race_finish_ms[tid] - race_start_ms[tid]) / 1000.0
            video_results.append({"tid": tid, "time_s": t, "distance_m": distance_m})
            log_fn(f"  ID{tid}: {t:.3f}s")
        else:
            d = smooth_dist.get(tid, 0)
            video_results.append({"tid": tid, "time_s": None, "distance_m": d})
            log_fn(f"  ID{tid}: DID NOT FINISH (reached {d:.1f}m)")

    return video_results


# ═══════════════════════════════════════════════════════════════════════════
#  CSRT Debug Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_csrt_debug(model, video_path, src_points, distance_m, max_athletes,
                   log_fn=print, stop_check=lambda: False):
    """CSRT debug pipeline — same as run_tracking but with extra CSRT debug overlay."""
    # For now, delegate to run_tracking. A dedicated debug build can be added later.
    log_fn("Running CSRT debug mode (using main pipeline with extra logging)")
    return run_tracking(model, video_path, src_points, distance_m, max_athletes,
                        log_fn=log_fn, stop_check=stop_check)


# ═══════════════════════════════════════════════════════════════════════════
#  Calibration GUI
# ═══════════════════════════════════════════════════════════════════════════

class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Track & Field Timing — Calibration")
        self.root.geometry("1400x900")
        self.root.configure(bg=BG_DARK)

        # State
        self.calibrations = {}
        self.current_video = None
        self.cap = None
        self.original_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.video_files = []
        self.photo_image = None

        # Multi-lane
        self.lane_points = [[]]
        self.current_lane = 0

        # Zoom/pan
        self._zoom = 1.0
        self._off_x = 0
        self._off_y = 0
        self._fit_scale = 1.0
        self._eff_scale = 1.0
        self._drag_start = None
        self._drag_off0 = (0, 0)
        self._drag_moved = False

        # Processing
        self.processing = False
        self.stop_requested = False

        self.load_calibrations()
        self._build_gui()
        self._populate_video_list()

        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self._prev_frame())
        self.root.bind("<Right>", lambda e: self._next_frame())

    # ── Persistence ──────────────────────────────────────────────────────

    def load_calibrations(self):
        if CALIB_FILE.exists():
            try:
                self.calibrations = json.loads(CALIB_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                self.calibrations = {}

    def save_calibrations(self):
        CALIB_FILE.write_text(json.dumps(self.calibrations, indent=2))

    # ── GUI Building ─────────────────────────────────────────────────────

    def _build_gui(self):
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = tk.Frame(main_frame, bg=BG_DARK)
        top_frame.pack(fill=tk.BOTH, expand=True)

        self._build_left_panel(top_frame)
        self._build_center_panel(top_frame)
        self._build_right_panel(top_frame)
        self._build_bottom_panel(main_frame)

    def _build_left_panel(self, parent):
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
        center = tk.Frame(parent, bg=BG_DARK)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=5)

        self.canvas = tk.Canvas(center, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse events
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>", self._on_scroll)
        self.canvas.bind("<Button-5>", self._on_scroll)
        self.canvas.bind("<Double-Button-1>", lambda e: self._reset_zoom())
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Scrubber row
        scrub_frame = tk.Frame(center, bg=BG_DARK)
        scrub_frame.pack(fill=tk.X, pady=(3, 0))

        tk.Button(scrub_frame, text="<", bg=BG_ENTRY, fg=FG_TEXT, width=3,
                  command=self._prev_frame).pack(side=tk.LEFT, padx=2)

        self.frame_slider = tk.Scale(
            scrub_frame, from_=0, to=0, orient=tk.HORIZONTAL,
            bg=BG_DARK, fg=FG_TEXT, troughcolor=BG_ENTRY,
            highlightthickness=0, showvalue=False, length=400
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.frame_slider.bind("<B1-Motion>", self._on_slider_drag)

        tk.Button(scrub_frame, text=">", bg=BG_ENTRY, fg=FG_TEXT, width=3,
                  command=self._next_frame).pack(side=tk.LEFT, padx=2)

        self.frame_label = tk.Label(
            scrub_frame, text="Frame 0 / 0", bg=BG_DARK, fg=FG_TEXT,
            font=("Helvetica", 9), width=18
        )
        self.frame_label.pack(side=tk.LEFT, padx=5)

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG_PANEL, width=250)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(2, 5), pady=5)
        right.pack_propagate(False)

        tk.Label(right, text="Controls", bg=BG_PANEL, fg=FG_BRIGHT,
                 font=("Helvetica", 12, "bold")).pack(pady=(10, 5))

        # Point status
        pts_frame = tk.LabelFrame(right, text="Calibration Points", bg=BG_PANEL,
                                  fg=FG_TEXT, font=("Helvetica", 9))
        pts_frame.pack(fill=tk.X, padx=10, pady=5)
        self.point_labels = []
        for i, name in enumerate(POINT_NAMES):
            lbl = tk.Label(pts_frame, text=f"{name}: --",
                           bg=BG_PANEL, fg=FG_TEXT, font=("Courier", 9))
            lbl.pack(anchor="w", padx=5)
            self.point_labels.append(lbl)

        # Lane selector (hidden for single athlete)
        self.lane_frame = tk.Frame(right, bg=BG_PANEL)
        self.lane_frame.pack(fill=tk.X, padx=10, pady=3)
        tk.Label(self.lane_frame, text="Lane:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.lane_var = tk.StringVar(value="1")
        self.lane_combo = ttk.Combobox(
            self.lane_frame, textvariable=self.lane_var, values=["1"],
            state="readonly", width=5
        )
        self.lane_combo.pack(side=tk.LEFT, padx=5)
        self.lane_combo.bind("<<ComboboxSelected>>", self._on_lane_selected)
        self.lane_frame.pack_forget()  # hidden by default

        # Clear / Reset buttons
        btn_frame = tk.Frame(right, bg=BG_PANEL)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(btn_frame, text="Clear Points", bg=BG_ENTRY, fg=FG_TEXT,
                  command=self._clear_points).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Reset Zoom", bg=BG_ENTRY, fg=FG_TEXT,
                  command=self._reset_zoom).pack(side=tk.LEFT, padx=2)

        # Distance dropdown
        dist_frame = tk.Frame(right, bg=BG_PANEL)
        dist_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(dist_frame, text="Distance (m):", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.distance_var = tk.StringVar(value="100")
        self.distance_combo = ttk.Combobox(
            dist_frame, textvariable=self.distance_var,
            values=[str(d) for d in DISTANCE_OPTIONS],
            state="readonly", width=5
        )
        self.distance_combo.pack(side=tk.LEFT, padx=5)

        # Athletes dropdown
        ath_frame = tk.Frame(right, bg=BG_PANEL)
        ath_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(ath_frame, text="Athletes:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.athletes_var = tk.StringVar(value="1")
        self.athletes_combo = ttk.Combobox(
            ath_frame, textvariable=self.athletes_var,
            values=[str(a) for a in ATHLETE_OPTIONS],
            state="readonly", width=5
        )
        self.athletes_combo.pack(side=tk.LEFT, padx=5)
        self.athletes_combo.bind("<<ComboboxSelected>>", self._on_athletes_changed)

        # Save button
        tk.Button(right, text="Save Calibration", bg=ACCENT, fg=FG_BRIGHT,
                  font=("Helvetica", 10, "bold"),
                  command=self._save_calibration).pack(fill=tk.X, padx=10, pady=10)

        # Run buttons
        tk.Button(right, text="Run Selected", bg=GREEN, fg="#000000",
                  font=("Helvetica", 10),
                  command=self._run_selected).pack(fill=tk.X, padx=10, pady=3)
        tk.Button(right, text="Run All Calibrated", bg=GREEN, fg="#000000",
                  font=("Helvetica", 10),
                  command=self._run_all).pack(fill=tk.X, padx=10, pady=3)
        tk.Button(right, text="CSRT Debug", bg=YELLOW, fg="#000000",
                  font=("Helvetica", 10),
                  command=self._run_csrt_debug).pack(fill=tk.X, padx=10, pady=3)

        # Stop button
        tk.Button(right, text="Stop", bg=RED, fg=FG_BRIGHT,
                  font=("Helvetica", 10, "bold"),
                  command=self._stop_processing).pack(fill=tk.X, padx=10, pady=10)

    def _build_bottom_panel(self, parent):
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

    # ── Logging ──────────────────────────────────────────────────────────

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        print(msg)

    # ── Video List ───────────────────────────────────────────────────────

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
            color = GREEN if name in self.calibrations else FG_TEXT
            self.video_listbox.itemconfig(i, fg=color)

    def _update_calib_summary(self):
        total = len(self.video_files)
        cal = sum(1 for v in self.video_files if v in self.calibrations)
        self.calib_summary_label.config(text=f"Calibrated: {cal} / {total}")

    # ── Video Selection ──────────────────────────────────────────────────

    def _on_video_select(self, event):
        sel = self.video_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        name = self.video_files[idx]
        if name != self.current_video:
            self._load_video(name)

    def _load_video(self, name):
        self.current_video = name
        if self.cap:
            self.cap.release()

        path = str(VIDEO_DIR / name)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.log(f"ERROR: Cannot open {path}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_slider.config(to=max(0, self.total_frames - 1))

        # Restore calibration if exists
        if name in self.calibrations:
            cal = self.calibrations[name]
            pts = cal.get("points", [])
            # Handle legacy flat format
            if pts and isinstance(pts[0][0], (int, float)):
                self.lane_points = [[tuple(p) for p in pts]]
            else:
                self.lane_points = [[tuple(p) for p in lane] for lane in pts]
            self.distance_var.set(str(cal.get("distance_m", 100)))
            self.athletes_var.set(str(cal.get("max_athletes", 1)))
            frame_idx = cal.get("frame_idx", 0)

            n_ath = int(self.athletes_var.get())
            while len(self.lane_points) < n_ath:
                self.lane_points.append([])
            self._show_lane_selector(n_ath)
        else:
            self.lane_points = [[]]
            self.current_lane = 0
            frame_idx = 0

        self.current_lane = 0
        self.lane_var.set("1")
        self._seek_and_display(frame_idx)
        self.frame_slider.set(frame_idx)
        self._update_frame_label()
        self._update_point_labels()
        self.log(f"Loaded: {name} ({self.total_frames} frames)")

    # ── Frame Navigation ─────────────────────────────────────────────────

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

    def _on_slider_drag(self, event):
        val = self.frame_slider.get()
        self.frame_label.config(text=f"Frame {val} / {self.total_frames}")

    def _on_slider_release(self, event):
        val = self.frame_slider.get()
        self._seek_and_display(val)
        self._update_frame_label()

    def _prev_frame(self):
        if self.cap and self.current_frame_idx > 0:
            self._seek_and_display(self.current_frame_idx - 1)
            self.frame_slider.set(self.current_frame_idx)
            self._update_frame_label()

    def _next_frame(self):
        if self.cap and self.current_frame_idx < self.total_frames - 1:
            self._seek_and_display(self.current_frame_idx + 1)
            self.frame_slider.set(self.current_frame_idx)
            self._update_frame_label()

    def _update_frame_label(self):
        self.frame_label.config(
            text=f"Frame {self.current_frame_idx} / {self.total_frames}"
        )

    # ── Zoom / Pan ───────────────────────────────────────────────────────

    def _on_press(self, event):
        self._drag_start = (event.x, event.y)
        self._drag_off0 = (self._off_x, self._off_y)
        self._drag_moved = False

    def _on_drag(self, event):
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
        if not self._drag_moved and self._drag_start is not None:
            # Click — place calibration point
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

    def _reset_zoom(self):
        self._zoom = 1.0
        self._off_x = 0
        self._off_y = 0
        if self.original_frame is not None:
            self._render_frame()

    # ── Frame Rendering ──────────────────────────────────────────────────

    def _render_frame(self):
        if self.original_frame is None:
            return
        frame = self.original_frame
        H, W = frame.shape[:2]
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)

        self._fit_scale = min(cw / W, ch / H)
        self._eff_scale = self._fit_scale * self._zoom

        sw = int(W * self._eff_scale)
        sh = int(H * self._eff_scale)

        # Clamp pan
        if sw <= cw:
            self._off_x = (cw - sw) // 2
        else:
            self._off_x = max(cw - sw, min(0, self._off_x))
        if sh <= ch:
            self._off_y = (ch - sh) // 2
        else:
            self._off_y = max(ch - sh, min(0, self._off_y))

        # Crop only visible region for performance
        ox0 = max(0, int(-self._off_x / self._eff_scale))
        oy0 = max(0, int(-self._off_y / self._eff_scale))
        ox1 = min(W, ox0 + int(cw / self._eff_scale) + 2)
        oy1 = min(H, oy0 + int(ch / self._eff_scale) + 2)

        crop = frame[oy0:oy1, ox0:ox1]
        disp_w = max(1, int((ox1 - ox0) * self._eff_scale))
        disp_h = max(1, int((oy1 - oy0) * self._eff_scale))
        scaled = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(scaled)
        self.photo_image = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(
            max(0, self._off_x), max(0, self._off_y),
            anchor=tk.NW, image=self.photo_image
        )

        # Draw calibration overlays
        def to_c(ox, oy):
            return (self._off_x + ox * self._eff_scale,
                    self._off_y + oy * self._eff_scale)

        dot_r = max(6, min(14, int(10 * self._zoom ** 0.3)))
        font_size = max(10, int(13 * self._zoom ** 0.3))

        for lane_idx, pts in enumerate(self.lane_points):
            color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
            is_active = (lane_idx == self.current_lane)
            line_width = 2 if is_active else 1
            dot_outline = "white" if is_active else "#888888"

            # Lines
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

            # Dots + labels
            for i, (ox, oy) in enumerate(pts):
                cx, cy = to_c(ox, oy)
                self.canvas.create_oval(
                    cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
                    fill=color, outline=dot_outline, width=2
                )
                label = (f"A{lane_idx + 1}:{POINT_NAMES[i]}"
                         if len(self.lane_points) > 1 else f"{i + 1}:{POINT_NAMES[i]}")
                self.canvas.create_text(
                    cx + dot_r + 5, cy - dot_r,
                    text=label, fill="white",
                    font=("Helvetica", font_size, "bold"), anchor="sw"
                )

    # ── Point Labels ─────────────────────────────────────────────────────

    def _update_point_labels(self):
        pts = self.lane_points[self.current_lane] if self.current_lane < len(self.lane_points) else []
        for i, lbl in enumerate(self.point_labels):
            if i < len(pts):
                x, y = pts[i]
                lbl.config(text=f"{POINT_NAMES[i]}: ({x}, {y})")
            else:
                lbl.config(text=f"{POINT_NAMES[i]}: --")

    def _clear_points(self):
        if self.current_lane < len(self.lane_points):
            self.lane_points[self.current_lane] = []
        self._update_point_labels()
        if self.original_frame is not None:
            self._render_frame()

    # ── Multi-Lane ───────────────────────────────────────────────────────

    def _on_athletes_changed(self, event=None):
        n = int(self.athletes_var.get())
        while len(self.lane_points) < n:
            self.lane_points.append([])
        while len(self.lane_points) > n:
            self.lane_points.pop()
        self.current_lane = min(self.current_lane, n - 1)
        self._show_lane_selector(n)
        self._update_point_labels()
        if self.original_frame is not None:
            self._render_frame()

    def _show_lane_selector(self, n):
        if n > 1:
            self.lane_combo.config(values=[str(i + 1) for i in range(n)])
            self.lane_var.set(str(self.current_lane + 1))
            self.lane_frame.pack(fill=tk.X, padx=10, pady=3)
        else:
            self.lane_frame.pack_forget()

    def _on_lane_selected(self, event=None):
        self.current_lane = int(self.lane_var.get()) - 1
        self._update_point_labels()
        if self.original_frame is not None:
            self._render_frame()

    # ── Save Calibration ─────────────────────────────────────────────────

    def _save_calibration(self):
        if not self.current_video:
            messagebox.showwarning("Warning", "No video selected.")
            return
        n_athletes = int(self.athletes_var.get())
        for i in range(n_athletes):
            if i >= len(self.lane_points) or len(self.lane_points[i]) != 4:
                count = len(self.lane_points[i]) if i < len(self.lane_points) else 0
                messagebox.showwarning(
                    "Warning",
                    f"Athlete {i + 1} lane needs 4 calibration points (has {count})."
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

    # ── Run Buttons ──────────────────────────────────────────────────────

    def _run_selected(self):
        if self.processing:
            messagebox.showinfo("Info", "Processing already running.")
            return
        if not self.current_video:
            messagebox.showwarning("Warning", "No video selected.")
            return
        if self.current_video not in self.calibrations:
            messagebox.showwarning("Warning", "Video not calibrated.")
            return
        self._run_videos([self.current_video])

    def _run_all(self):
        if self.processing:
            messagebox.showinfo("Info", "Processing already running.")
            return
        calibrated = [v for v in self.video_files if v in self.calibrations]
        if not calibrated:
            messagebox.showwarning("Warning", "No calibrated videos.")
            return
        self._run_videos(calibrated)

    def _run_csrt_debug(self):
        if self.processing:
            messagebox.showinfo("Info", "Processing already running.")
            return
        if not self.current_video:
            messagebox.showwarning("Warning", "No video selected.")
            return
        if self.current_video not in self.calibrations:
            messagebox.showwarning("Warning", "Video not calibrated.")
            return
        self._run_videos([self.current_video], csrt_debug=True)

    def _run_videos(self, video_list, csrt_debug=False):
        self.processing = True
        self.stop_requested = False

        def worker():
            try:
                model = load_model()
                self.log(f"Model loaded: {type(model)}")
            except FileNotFoundError as e:
                self.log(f"ERROR: {e}")
                self.processing = False
                return

            all_results = {}
            for video_name in video_list:
                if self.stop_requested:
                    self.log("Processing stopped by user.")
                    break

                cal = self.calibrations[video_name]
                video_path = str(VIDEO_DIR / video_name)
                self.log(f"\n{'='*60}")
                self.log(f"Processing: {video_name}")
                self.log(f"{'='*60}")

                _tracking_fn = run_csrt_debug if csrt_debug else run_tracking
                results = _tracking_fn(
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

            # Summary table
            if all_results:
                self.log(f"\n{'='*60}")
                self.log("RESULTS SUMMARY")
                self.log(f"{'='*60}")
                for vname, res_list in all_results.items():
                    self.log(f"\n{vname}:")
                    for r in res_list:
                        if r["time_s"] is not None:
                            self.log(f"  ID{r['tid']}: {r['time_s']:.3f}s ({r['distance_m']}m)")
                        else:
                            self.log(f"  ID{r['tid']}: DNF (reached {r['distance_m']:.1f}m)")

            self.processing = False
            self.log("\nAll processing complete.")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _stop_processing(self):
        if self.processing:
            self.stop_requested = True
            self.log("Stop requested...")

    # ── Utility ──────────────────────────────────────────────────────────

    @staticmethod
    def _hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (b, g, r)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Entry
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
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
