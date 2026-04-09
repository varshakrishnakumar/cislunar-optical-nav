from __future__ import annotations

import shutil
import subprocess
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from cv.camera import Intrinsics
from cv.pointing import camera_dcm_from_boresight


_BG     = "#080B14"
_PANEL  = "#0E1220"
_BORDER = "#1C2340"
_TEXT   = "#DCE0EC"
_DIM    = "#5A6080"
_CYAN   = "#22D3EE"
_AMBER  = "#F59E0B"
_GREEN  = "#10B981"
_VIOLET = "#8B5CF6"
_RED    = "#F43F5E"
_ORANGE = "#FB923C"
_PINK   = "#F472B6"

_R_MOON_ND   = 0.00452
_TEXTURE_URL = (
    "https://s3-us-west-2.amazonaws.com/s.cdpn.io/17271/lroc_color_poles_1k.jpg"
)

_LM_LONS = np.array([2.6, 2.0, 3.2, 1.5, 3.8, 2.8, 1.8, 3.5, 2.3, 4.0])
_LM_LATS = np.array([0.4, -0.3, 0.6, 0.2, -0.5, -0.2, 0.7, 0.1, -0.6, 0.3])
_N_LM    = len(_LM_LONS)
_TRAIL_LEN = 12

_LM_COLORS  = [_GREEN, _RED, _AMBER, _ORANGE, _CYAN,
               _VIOLET, _PINK, _GREEN, _RED, _AMBER]



def _load_texture(cache_path: Path) -> np.ndarray:
    if not cache_path.exists():
        print(f"Downloading Moon texture → {cache_path} ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_TEXTURE_URL, str(cache_path))
        print("  Done.")
    img = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not load texture: {cache_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0



def _landmark_positions_frame(r_moon: np.ndarray) -> np.ndarray:
    lons, lats = _LM_LONS, _LM_LATS
    ux = np.cos(lats) * np.cos(lons)
    uy = np.cos(lats) * np.sin(lons)
    uz = np.sin(lats)
    dirs = np.column_stack([ux, uy, uz])
    return r_moon[None, :] + _R_MOON_ND * dirs


def _project_landmarks(
    lm_pos: np.ndarray,
    r_sc: np.ndarray,
    r_moon: np.ndarray,
    R_cam: np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
    cam_w: int, cam_h: int,
) -> list[tuple[float, float] | None]:
    result = []
    for i in range(_N_LM):
        p      = lm_pos[i]
        normal = (p - r_moon) / _R_MOON_ND
        to_sc  = r_sc - p
        if np.dot(normal, to_sc) <= 0:
            result.append(None); continue
        p_cam = R_cam @ (p - r_sc)
        if p_cam[2] <= 0:
            result.append(None); continue
        u = float(fx * p_cam[0] / p_cam[2] + cx)
        v = float(fy * p_cam[1] / p_cam[2] + cy)
        if not (0 <= u < cam_w and 0 <= v < cam_h):
            result.append(None); continue
        result.append((u, v))
    return result



def _render_moon(
    *,
    cam_w: int, cam_h: int,
    u_c: float, v_c: float, r_px: float,
    R_cam: np.ndarray,
    r_body: np.ndarray, r_sc: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    tex: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    frame = np.zeros((cam_h, cam_w, 3), dtype=np.float32)
    s = rng.random((cam_h, cam_w)) < 0.0006
    frame[s] = rng.uniform(0.4, 1.0, (s.sum(), 3))

    if r_px < 0.8:
        ui, vi = int(round(u_c)), int(round(v_c))
        if 0 <= ui < cam_w and 0 <= vi < cam_h:
            frame[vi, ui] = [1.0, 1.0, 1.0]
        frame += rng.normal(0.0, 0.008, (cam_h, cam_w, 3)).astype(np.float32)
        return cv2.cvtColor(
            (np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )

    th, tw = tex.shape[:2]
    pad = max(4, int(r_px * 0.05))
    u0 = max(0, int(u_c - r_px) - pad); u1 = min(cam_w, int(u_c + r_px) + pad + 1)
    v0 = max(0, int(v_c - r_px) - pad); v1 = min(cam_h, int(v_c + r_px) + pad + 1)

    bore      = r_body - r_sc
    moon_cam  = R_cam @ bore
    r_sph     = _R_MOON_ND
    sun_cam   = R_cam @ np.array([-1.0, 0.0, 0.0])
    sun_cam  /= np.linalg.norm(sun_cam)

    uu, vv = np.meshgrid(np.arange(u0, u1, dtype=np.float32),
                          np.arange(v0, v1, dtype=np.float32))
    rdx = (uu - cx) / fx; rdy = (vv - cy) / fy
    rdz = np.ones_like(rdx)
    rlen = np.sqrt(rdx**2 + rdy**2 + rdz**2)
    rdx /= rlen; rdy /= rlen; rdz /= rlen

    mc = moon_cam.astype(np.float32)
    b  = -2.0 * (rdx * mc[0] + rdy * mc[1] + rdz * mc[2])
    c_ = float(mc[0]**2 + mc[1]**2 + mc[2]**2 - r_sph**2)
    disc = b**2 - 4.0 * c_
    hit  = disc >= 0.0
    if not hit.any():
        frame += rng.normal(0.0, 0.008, (cam_h, cam_w, 3)).astype(np.float32)
        return cv2.cvtColor(
            (np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )

    sq  = np.where(hit, np.sqrt(np.maximum(disc, 0.0)), 0.0)
    t   = np.where(hit, (-b - sq) / 2.0, 0.0)
    t   = np.where((t < 0) & hit, (-b + sq) / 2.0, t)
    hit = hit & (t > 0)

    hx = np.where(hit, rdx * t, 0.0)
    hy = np.where(hit, rdy * t, 0.0)
    hz = np.where(hit, rdz * t, 0.0)

    nx = np.where(hit, (hx - mc[0]) / r_sph, 0.0)
    ny = np.where(hit, (hy - mc[1]) / r_sph, 0.0)
    nz = np.where(hit, (hz - mc[2]) / r_sph, 0.0)

    diff    = np.clip(nx * sun_cam[0] + ny * sun_cam[1] + nz * sun_cam[2], 0, 1)
    shading = 0.12 + 0.88 * diff

    lon = np.where(hit, np.arctan2(ny, nx), 0.0)
    lat = np.where(hit, np.arcsin(np.clip(nz, -1, 1)), 0.0)
    tu  = np.clip(((lon / (2 * np.pi) + 0.5) * (tw - 1)).astype(int), 0, tw - 1)
    tv  = np.clip(((0.5 - lat / np.pi) * (th - 1)).astype(int), 0, th - 1)

    color = np.where(hit[:, :, None], tex[tv, tu] * shading[:, :, None], 0.0)
    frame[v0:v1, u0:u1] = color
    frame += rng.normal(0.0, 0.012, (cam_h, cam_w, 3)).astype(np.float32)
    return cv2.cvtColor(
        (np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )



def _detect(frame_bgr: np.ndarray) -> dict:
    gray = cv2.GaussianBlur(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0
    )
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False}
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 3:
        return {"found": False}
    M = cv2.moments(best)
    if abs(M["m00"]) < 1e-9:
        return {"found": False}
    (ecx, ecy), er = cv2.minEnclosingCircle(best)
    x, y, w, h = cv2.boundingRect(best)
    return {
        "found":    True,
        "centroid": (M["m10"] / M["m00"], M["m01"] / M["m00"]),
        "contour":  best,
        "bbox":     (x, y, w, h),
        "circle":   (float(ecx), float(ecy), float(er)),
    }



def _draw_combined_panel(
    ax: plt.Axes,
    frame_bgr: np.ndarray,
    det: dict,
    lm_uv: list,
    lm_prev_uv: list,
    trails: list,
    t: float,
    r_px: float,
    fx: float,
    cam_w: int, cam_h: int,
) -> None:
    ax.set_facecolor("#050709")
    ax.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), origin="upper",
               extent=[0, cam_w, cam_h, 0])
    ax.set_xlim(0, cam_w); ax.set_ylim(cam_h, 0); ax.set_axis_off()

    if det["found"]:
        cx2, cy2     = det["centroid"]
        ecx, ecy, er = det["circle"]
        x, y, w, h   = det["bbox"]

        pts = det["contour"].reshape(-1, 2).astype(float)
        ax.plot(pts[:, 0], pts[:, 1], color=_AMBER, lw=1.1, alpha=0.85, zorder=4)

        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                    edgecolor=_GREEN, lw=0.9, ls="--", zorder=5))

        ax.add_patch(plt.Circle((ecx, ecy), er, fill=False,
                                 edgecolor=_VIOLET, lw=1.1, alpha=0.85, zorder=5))

        ax.scatter([cx2], [cy2], s=40, color=_CYAN, zorder=6)
        ax.plot([cx2-6, cx2+6], [cy2, cy2], color=_CYAN, lw=1.0, zorder=6)
        ax.plot([cx2, cx2], [cy2-6, cy2+6], color=_CYAN, lw=1.0, zorder=6)

        ax.annotate("", xy=(ecx+er, ecy), xytext=(ecx, ecy),
                    arrowprops=dict(arrowstyle="-", color=_VIOLET, lw=0.8), zorder=5)

        alpha_rad = er / fx
        lx = min(ecx + er + 4, cam_w - 90)
        ax.text(lx, ecy,
                f"α = {alpha_rad*1e3:.1f} mrad\n"
                f"ρ ≈ {_R_MOON_ND/max(alpha_rad,1e-9):.3f} ND",
                color=_RED, fontsize=8, family="monospace", va="center",
                bbox=dict(facecolor=(0,0,0,0.65), edgecolor="none", pad=3), zorder=8)

    for i in range(_N_LM):
        col   = _LM_COLORS[i]
        trail = trails[i]

        if len(trail) >= 2:
            trail_arr = np.array(trail)
            n = len(trail_arr)
            for j in range(1, n):
                alpha_t = 0.12 + 0.70 * (j / n)
                ax.plot(trail_arr[j-1:j+1, 0], trail_arr[j-1:j+1, 1],
                        color=col, lw=1.0, alpha=alpha_t, zorder=4)

        uv = lm_uv[i]
        if uv is not None:
            u, v = uv
            ax.scatter([u], [v], s=50, facecolors="none", edgecolors=col,
                        linewidths=1.5, zorder=7)

            prev = lm_prev_uv[i]
            if prev is not None and r_px > 25:
                du = u - prev[0]
                dv = v - prev[1]
                speed = np.hypot(du, dv)
                if speed > 0.5:
                    scale = min(3.0, 8.0 / max(speed, 1e-3))
                    ax.annotate("",
                                xy=(u + du * scale, v + dv * scale),
                                xytext=(u, v),
                                arrowprops=dict(
                                    arrowstyle="-|>",
                                    color=col, lw=0.9,
                                    mutation_scale=6,
                                ),
                                zorder=8)

    n_vis = sum(1 for uv in lm_uv if uv is not None)
    lm_str = f"  |  {n_vis}/{_N_LM} landmarks" if n_vis > 0 else ""
    ax.text(5, 13, f"t = {t:.3f} ND{lm_str}",
             color=_AMBER, fontsize=9, family="monospace",
             bbox=dict(facecolor=(0,0,0,0.55), edgecolor="none", pad=4))

    ax.set_title(
        "Next Steps: α-ranging + landmark tracking (SLAM preview)\n"
        "u, v, α  →  LOS + range proxy  ·  tracked features  →  multiple LOS vectors",
        color=_TEXT, fontsize=9, pad=4,
    )



def _make_frame_fig(
    frame_bgr: np.ndarray,
    det: dict,
    lm_uv: list,
    lm_prev_uv: list,
    trails: list,
    t: float,
    r_px: float,
    fx: float,
    cam_w: int, cam_h: int,
) -> plt.Figure:
    dpi   = 100
    fig_w = (cam_w + 20) / dpi
    fig_h = (cam_h + 72) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=_BG)
    ax  = fig.add_axes([0.0, 0.09, 1.0, 0.85])

    _draw_combined_panel(
        ax, frame_bgr, det, lm_uv, lm_prev_uv, trails,
        t, r_px, fx, cam_w, cam_h,
    )

    fig.legend(handles=[
        Line2D([0],[0], color=_CYAN,  lw=2, label="centroid (u,v)"),
        Line2D([0],[0], color=_AMBER, lw=2, label="blob contour"),
        Line2D([0],[0], color=_VIOLET,lw=1.5, label="enclosing circle (α)"),
        Line2D([0],[0], color=_RED,   lw=0, marker="s", markersize=7,
               label="ρ ≈ R☾/α"),
        Line2D([0],[0], color=_GREEN, lw=2, label="landmark trails"),
        Line2D([0],[0], color=_ORANGE,lw=0, marker=">", markersize=7,
               label="optical flow"),
    ], loc="lower center", ncol=6, fontsize=7.5,
       facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT,
       bbox_to_anchor=(0.5, 0.0))
    return fig



def _run_ffmpeg(frames_dir: Path, out: Path, fps: int) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



def main() -> None:
    plots_dir  = Path("results/plots");  plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = Path("results/videos"); videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = videos_dir / "08_feature_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tex = _load_texture(Path("results/moon_texture.jpg"))

    mu     = 0.0121505856
    model  = CR3BP(mu=mu)
    L1x    = model.lagrange_points()["L1"][0]
    x0     = np.array([L1x - 1.2e-3, 2.5e-4, 0.0, 0.0, 0.045, 0.0], dtype=float)
    r_moon = np.asarray(model.primary2, dtype=float)

    fps    = 20
    SLOWMO = 6.0
    dt     = (1.0 / fps) / SLOWMO
    t_arr  = np.arange(0.0, 14.0 + 1e-12, dt)
    N      = len(t_arr)

    cam_w, cam_h = 640, 480
    fx = fy = 700.0
    cx, cy  = cam_w / 2.0, cam_h / 2.0
    intr    = Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=cam_w, height=cam_h)
    rng     = np.random.default_rng(42)

    res = propagate(model.eom, (0.0, 14.0), x0, t_eval=t_arr)
    if not res.success:
        raise RuntimeError(f"Propagation failed: {res.message}")
    r_true = res.x[:, :3]

    lm_pos_frame = _landmark_positions_frame(r_moon)

    trails: list[deque] = [deque(maxlen=_TRAIL_LEN) for _ in range(_N_LM)]
    lm_prev: list = [None] * _N_LM

    print(f"Rendering {N} frames ...")

    strip_frames: list[tuple] = []
    STRIP_N = 8
    stride  = max(1, N // STRIP_N)

    for k in range(N):
        t_k  = float(t_arr[k])
        r_sc = r_true[k]
        bore = r_moon - r_sc
        R_cam = camera_dcm_from_boresight(bore, camera_forward_axis="+z")
        lc    = R_cam @ bore

        if lc[2] <= 0:
            fb = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            u_c = v_c = cx; r_px = 0.0
        else:
            u_c   = float(fx * lc[0] / lc[2] + cx)
            v_c   = float(fy * lc[1] / lc[2] + cy)
            rng_nd = float(np.linalg.norm(bore))
            r_px   = (_R_MOON_ND / max(rng_nd, 1e-6)) * fx
            fb     = _render_moon(
                cam_w=cam_w, cam_h=cam_h,
                u_c=u_c, v_c=v_c, r_px=r_px,
                R_cam=R_cam, r_body=r_moon, r_sc=r_sc,
                fx=fx, fy=fy, cx=cx, cy=cy,
                tex=tex, rng=rng,
            )

        det   = _detect(fb)
        lm_uv = _project_landmarks(
            lm_pos_frame, r_sc, r_moon, R_cam,
            fx, fy, cx, cy, cam_w, cam_h,
        )

        for i, uv in enumerate(lm_uv):
            if uv is not None:
                trails[i].append(uv)

        fig = _make_frame_fig(
            fb, det, lm_uv, lm_prev, trails, t_k, r_px, fx, cam_w, cam_h
        )
        fig.savefig(frames_dir / f"frame_{k:05d}.png", dpi=100, facecolor=_BG)
        plt.close(fig)

        lm_prev = [uv for uv in lm_uv]

        if k % stride == 0 and det["found"] and det["circle"][2] > 2.0:
            if len(strip_frames) < STRIP_N:
                strip_frames.append((
                    fb.copy(), det,
                    list(lm_uv),
                    [deque(tr) for tr in trails],
                    t_k, r_px,
                ))

    if shutil.which("ffmpeg"):
        out_mp4 = videos_dir / "08_feature_tracking.mp4"
        _run_ffmpeg(frames_dir, out_mp4, fps=fps)
        print(f"Wrote video: {out_mp4}")
    else:
        print("ffmpeg not found — skipping video.")

    if strip_frames:
        n = len(strip_frames)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.0))
        fig.patch.set_facecolor(_BG)
        if n == 1: axes = [axes]

        for col, (fb, det, lm_uv, tr, t_k, rpx) in enumerate(strip_frames):
            ax = axes[col]
            _draw_combined_panel(
                ax, fb, det, lm_uv, [None]*_N_LM,
                tr, t_k, rpx, fx, cam_w, cam_h,
            )
            n_vis = sum(1 for uv in lm_uv if uv is not None)
            ax.set_title(
                f"t = {t_k:.1f} ND\n"
                + (f"α={det['circle'][2]/fx*1e3:.0f} mrad  |  {n_vis} lm"
                   if det["found"] else "—"),
                color=_TEXT, fontsize=7, pad=3,
            )

        fig.legend(handles=[
            Line2D([0],[0], color=_CYAN,  lw=2, label="centroid (u,v)"),
            Line2D([0],[0], color=_AMBER, lw=2, label="blob contour"),
            Line2D([0],[0], color=_VIOLET,lw=1.5, label="enclosing circle (α)"),
            Line2D([0],[0], color=_RED,   lw=0, marker="s", markersize=7,
                   label="ρ ≈ R☾/α"),
            Line2D([0],[0], color=_GREEN, lw=2, label="landmark trails"),
        ], loc="lower center", ncol=5, fontsize=8,
           facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT,
           bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(
            "Next Steps: α-ranging + SLAM preview  |  Earth-Moon CR3BP",
            color=_TEXT, fontsize=11, y=1.02,
        )

        strip_path = plots_dir / "08_feature_tracking_strip.png"
        fig.savefig(strip_path, dpi=200, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
        print(f"Wrote strip: {strip_path}")

    print("08 feature tracking demo complete.")


if __name__ == "__main__":
    main()
