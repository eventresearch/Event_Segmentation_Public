# 🧭 DSEC Dataset – Geometry & Alignment Guide

> A practical overview of camera frames, calibration files, and alignment logic for RGB, event, and segmentation data in the **DSEC dataset**.  
> Official reference: **https://dsec.ifi.uzh.ch/data-format/**

---

## 📁 1) Dataset Structure Overview

DSEC includes **two stereo pairs**, each composed of one **RGB** and one **event** camera.

| Side | RGB Camera | Event Camera | Typical Resolution | Notes |
|------|-------------|---------------|--------------------|--------|
| **Left**  | `cam1` | `cam0` | 1440×1080 (RGB), 640×480 (events) | Main reference pair |
| **Right** | `cam2` | `cam3` | 1440×1080 / 640×480 | Right stereo counterpart |

Each camera also has a **rectified** version:

- `camRect0`, `camRect3` → rectified **event** cameras  
- `camRect1`, `camRect2` → rectified **RGB** cameras

---

## ⚙️ 2) Key Calibration & Mapping Files

### 🔹 `cam_to_cam.yaml`

Holds the full calibration between cameras:

- **Intrinsics**: `intrinsics.camX.camera_matrix` → `[fx, fy, cx, cy]`  
- **Rectification rotations**: `extrinsics.R_rectX` → 3×3 rotation used to define rectified frames (`camRectX`)  
- **Extrinsics**: `extrinsics.T_01`, `T_02`, … → full 4×4 transforms (rotation **and** translation) between cameras

Use these to convert 3D points between cameras and to project between different rectified frames.

### 🔹 `rectify_map.h5` (per **event** camera)

For each event camera (e.g., left event = `cam0`), DSEC provides a **forward** per-pixel map:
```
rectify_map  # shape: (H, W, 2)
```
At each raw event pixel `(x, y)` in `cam0`, it stores rectified coordinates `(x_rect, y_rect)` in **`camRect0`**.  
Use it once to bring **any event representation** (frames, voxel grids, time surfaces) from `cam0` → `camRect0`.

---

## 🧩 3) Camera Frames — What They Mean

| Frame | Type | Description | Resolution |
|--------|------|-------------|-------------|
| `cam0` | Event-L **raw** | Distorted coordinates straight from the event sensor | 640×480 |
| `camRect0` | Event-L **rectified** | Undistorted & rectified (epipolar-aligned for event pair) | ≈1440×1080 |
| `cam1` | RGB-L **raw** | Distorted | — |
| `camRect1` | RGB-L **rectified** | Undistorted & rectified (epipolar-aligned for RGB pair) | 1440×1080 |
| `cam2`, `camRect2` | RGB-R raw/rectified | Right stereo RGB | 1440×1080 |
| `cam3`, `camRect3` | Event-R raw/rectified | Right stereo event | 640×480 / 1440×1080 |

> **Important:** “Rectified” is **pair-specific**. RGBs are rectified for the **RGB stereo** geometry; events are rectified for the **event stereo** geometry.  
> Therefore **`camRect0` ≠ `camRect1`** (different intrinsics/rotations).

---

## 🎨 4) What Lives in Which Frame

| Data Type | Native Frame | Notes |
|-----------|--------------|-------|
| **Events** | `cam0` | Raw distorted event pixels |
| **Rectified Events** | `camRect0` | Apply `rectify_map.h5` once |
| **RGB Images** | `camRect1` | Already rectified in the dataset |
| **Semantic Labels (DSEC‑Semantic)** | `camRect0` | Warped from RGB → **rectified event** frame; **cropped to 640×440** |

---

## ⚠️ 5) CRITICAL POINTS (Read Twice)

### ⭐ Why RGB frames are called **“rectified”**
- They are **undistorted & epipolar-aligned for the RGB pair** (`cam1`–`cam2`).  
- This **does not** put them in the event rectified frame.  
- Hence **`camRect1` (RGB)** does **not** directly overlay with **`camRect0` (event)**.

### ⭐ Why you **must rectify events**
- Raw events (`cam0`) are **distorted**.  
- To align with labels (which live in `camRect0`) and to work with rectified geometry, you **must** map events to `camRect0` using `rectify_map.h5` (once, on your event frame/representation).

### ⭐ Why you should **NOT warp/rectify labels**
- DSEC‑Semantic labels were already created **in `camRect0`**.  
- Re-warping them introduces errors and breaks pixel accuracy.  
- **Leave labels as-is** and bring other modalities to **`camRect0`**.

---

## 🧮 6) The “Perfect Alignment” Pipeline (Pixel-Accurate)

**Target frame:** `camRect0` (rectified left event)

1. **Events (`cam0`) → `camRect0`:**  
   Apply `rectify_map.h5` (forward map). Do this once for any event frame/representation.

2. **Labels (already `camRect0`):**  
   Use directly. Remember masks are **640×440** (bottom 40 px were cropped).

3. **RGBs (`camRect1`) → `camRect0`:**  
   Reproject using **per‑pixel disparity** + full extrinsics (`T_01`) and rectification rotations (`R_rect0`, `R_rect1`) with intrinsics (`K_rect0`, `K_rect1`).  
   There is **no single 2D homography** across the baseline; you need depth/disparity for exactness.

**Depth-aware reprojection (camRect1 → camRect0):**  
For each pixel \((u_1,v_1)\) in **camRect1** with disparity \(d\):
\`\`\`text
Z   = f1 * b / d(u1, v1)
X1r = Z * inv(K_r1) * [u1, v1, 1]^T
X1  = R_rect1^(-1) * X1r
X0  = R_01 * X1 + t_01
X0r = R_rect0 * X0
[u0, v0, 1]^T = K_r0 * X0r
\`\`\`
- Use **nearest-neighbor** for masks; **linear** (or cubic) for RGB values.

---

## 🕒 7) Timing & Practical Notes

- **Event time correction:** apply the sequence’s `t_offset` from the H5 (`t_corrected = t_raw + t_offset`).  
- **Integration window:** to match a specific label/image time, prefer **±5–10 ms** over long windows (e.g., 50 ms) to reduce motion smear.  
- **Label crop:** masks are **640×440**; crop or pad your rectified event frame accordingly for overlays.  
- **Axis order:** index images as `img[v, u]` (y, x). Swapping x/y produces consistent mismatches.  
- **Interpolation rules:** NN for labels; no resizing artifacts; linear/cubic for RGB only.  
- **Disparity validity:** mask out invalid (d ≤ 0) to avoid speckles on reprojected RGB/masks.

---

## 🔍 8) Minimal Practical Workflow (Copy/Paste Checklist)

1. Aggregate events → frame (e.g., 10–50 ms).  
2. **Rectify event frame** to `camRect0` using `rectify_map.h5`.  
3. Load **labels** (already `camRect0`), overlay directly.  
4. If RGB needed, **reproject camRect1 → camRect0** using disparity + calibration.  
5. Visualize / train with everything now living in **`camRect0`**.

---

## 🧩 9) TL;DR Cheatsheet

| Data | Native | Target | Action |
|------|--------|--------|--------|
| Events | `cam0` | `camRect0` | **Rectify** using `rectify_map.h5` |
| Labels | `camRect0` | `camRect0` | **Use directly** (don’t warp) |
| RGB-L | `camRect1` | `camRect0` | **Reproject** with disparity + extrinsics |
| Right pair | `camRect2`/`camRect3` | analogous | Apply same logic |
| Calibration | `cam_to_cam.yaml` | — | Source of K, R_rect, T_01, baseline |
| Rectify map | `rectify_map.h5` | — | Forward map raw→rectified for events |

---

## 🔗 Official DSEC Docs

- Data format, timestamps, and file conventions: **https://dsec.ifi.uzh.ch/data-format/**

