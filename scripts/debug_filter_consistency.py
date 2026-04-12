from __future__ import annotations

from _common import ensure_src_on_path

ensure_src_on_path()

import numpy as np
from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_predict_measurement, bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing
from cv.camera import Intrinsics
from cv.sim_measurements import simulate_pixel_measurement
from cv.pointing import camera_dcm_from_boresight

mu    = 0.0121505856
t0    = 0.0
dt    = 0.02
sigma_px = 1.5
seed  = 7

model  = CR3BP(mu=mu)
r_body = np.array([1.0 - mu, 0.0, 0.0], dtype=float)
x0_nom = np.array([1.02, 0.0, 0.0, 0.0, -0.18, 0.0], dtype=float)

x0_true = x0_nom + np.array([1e-4, -1e-4, 0.0, 0.0, 0.0, 0.0])
xhat    = x0_nom + np.array([1e-4,  1e-4, 0.0, 0.0, 0.0, 0.0])
P       = np.diag([1e-6]*6).astype(float)

intr = Intrinsics(fx=400., fy=400., cx=320., cy=240., width=640, height=480)
from cv.pointing import camera_dcm_from_boresight
boresight_nom = r_body - x0_nom[:3]
R_cam = camera_dcm_from_boresight(boresight_nom, camera_forward_axis="+z")
R_frame_from_cam = R_cam.T
rng = np.random.default_rng(seed)

t_meas = np.arange(t0, t0 + 10*dt + 1e-12, dt)
res = propagate(model.eom, (t0, t_meas[-1]), x0_true, t_eval=t_meas)
xs_true = res.x

print(f"{'k':>4} {'pos_err':>12} {'P_pos_1sig':>12} "
      f"{'sigma_theta':>14} {'meas_angle':>14} {'NIS':>10} {'accepted':>10}")
print("-" * 80)

t_prev = t0
for k in range(1, len(t_meas)):
    t_k = float(t_meas[k])
    x_true_k = xs_true[k]

    xhat, P, _ = ekf_propagate_cr3bp_stm(
        mu=mu, x=xhat, P=P, t0=t_prev, t1=t_k, q_acc=1e-14
    )
    t_prev = t_k

    meas = simulate_pixel_measurement(
        r_sc=x_true_k[:3], r_body=r_body,
        intrinsics=intr, R_cam_from_frame=R_cam,
        sigma_px=sigma_px, rng=rng, t=t_k,
        dropout_p=0.0, out_of_frame="drop", behind="drop",
    )

    pos_err   = float(np.linalg.norm(xhat[:3] - x_true_k[:3]))
    P_pos_sig = float(np.sqrt(P[0, 0]))

    if not (meas.valid and np.isfinite(meas.u_px)):
        print(f"{k:>4} {pos_err:>12.3e} {P_pos_sig:>12.3e} "
              f"{'(no meas)':>14} {'':>14} {'':>10} {'':>10}")
        continue

    u_meas, sigma_theta = pixel_detection_to_bearing(
        meas.u_px, meas.v_px, sigma_px, intr, R_frame_from_cam
    )
    u_pred   = bearing_predict_measurement(xhat, r_body)
    angle    = float(np.arccos(np.clip(np.dot(u_meas, u_pred), -1., 1.)))

    upd = bearing_update_tangent(xhat, P, u_meas, r_body, sigma_theta)
    nis = upd.nis

    print(f"{k:>4} {pos_err:>12.3e} {P_pos_sig:>12.3e} "
          f"{sigma_theta:>14.3e} {angle:>14.3e} {nis:>10.2f} {str(upd.accepted):>10}")

    if upd.accepted:
        xhat, P = upd.x_upd, upd.P_upd

print("\nKey checks:")
print(f"  sigma_theta should be sigma_px/fx = {sigma_px/400:.6e}")
print(f"  meas_angle should be << sigma_theta for a consistent filter")
print(f"  NIS should be ~2 for a consistent filter")
print(f"  P_pos_1sig should be >= pos_err for a consistent filter")
