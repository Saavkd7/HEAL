"""Quantifies how much using qcar-52775's REAL calibrated extrinsic
(R_CB_QCAR52775, T_CB_QCAR52775 -- 19-point PnP fit) vs. the manufacturer's
NOMINAL value (R_CB_NOMINAL, FRONT_MOUNT_XYZ -- Quanser hardware manual p.8)
actually changes on THIS project's real captured data.

We only have real calibration for agent "1" (qcar-52775/.198); agent "2"
(qcar-52776/.158) still runs on the nominal value with no calibrated
alternative to compare against -- see build_coop_train_val_dataset.py's EXTRINSIC_PARAMS
and the module docstring there. So this can only test agent "1"'s camera,
using every OTHER vehicle (agent "2", both static targets) across all 9 real
trajectories as targets.

Method: reuse build_coop_train_val_dataset.py directly (load_frames, pair_frames_n,
STATIC_TARGETS, apply_marker_offset, project_world_to_camera,
truly_visible_to_agent) -- never reimplements the geometry -- and for every
real paired frame, computes each target's projection and visibility TWICE:
once with the real calibrated extrinsic already in EXTRINSIC_PARAMS["1"],
once with EXTRINSIC_PARAMS["1"] temporarily swapped to the nominal value
(restored immediately after). Reports:

  - pixel-space shift of the projected point (only where both configs agree
    it's in front of the camera, so the comparison is meaningful)
  - how many frames FLIP visibility (calibrated says visible, nominal says
    not, or vice versa) -- this is what would actually change in scene_gt /
    the training yaml's vehicles{} if agent "1" still used the nominal value

    python compare_extrinsic_nominal_vs_calibrated.py \
      --onboard-root <path to qcar_onboard/ExperimentNo1>
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(os.path.dirname(HERE), "build", "cooperative")
sys.path.insert(0, BUILD_DIR)
import build_coop_train_val_dataset as bcd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

_parser = argparse.ArgumentParser(add_help=False, parents=[bcd._parser])
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    _parser.print_help()
    raise SystemExit(0)

R_NOMINAL = bcd.R_CB_NOMINAL
T_NOMINAL = np.array(bcd.FRONT_MOUNT_XYZ)
R_CALIBRATED, T_CALIBRATED = bcd.EXTRINSIC_PARAMS["1"]

AGENT = "1"  # only agent "1" (qcar-52775) has a real calibrated alternative to compare


def project_point(point_world, agent_pose, R_cb, t_cb):
    bcd.EXTRINSIC_PARAMS[AGENT] = (R_cb, t_cb)
    uv, depth = bcd.project_world_to_camera(
        point_world.reshape(1, 3), agent_pose["x"], agent_pose["y"], agent_pose["z"],
        agent_pose["yaw_rad"], AGENT, bcd.CAM_PARAMS[AGENT]["K"])
    return uv[0], depth[0]


def visible(target_x, target_y, target_z, target_yaw_deg, agent_pose, R_cb, t_cb):
    bcd.EXTRINSIC_PARAMS[AGENT] = (R_cb, t_cb)
    return bcd.truly_visible_to_agent(target_x, target_y, target_z, target_yaw_deg, agent_pose, AGENT)


def main():
    print("Calibrated R/t (agent 1, qcar-52775):\n%s\nt=%s" % (R_CALIBRATED, T_CALIBRATED))
    print("Nominal R/t (Quanser manual):\n%s\nt=%s\n" % (R_NOMINAL, T_NOMINAL))

    pixel_shifts = []
    flips = 0
    both_front = 0
    total_checked = 0

    for idx in bcd.common_indices():
        frames_a = bcd.load_frames(bcd.EGO_ID, idx)
        frames_b = bcd.load_frames(bcd.PEER_ID, idx)
        groups = bcd.pair_frames_n({bcd.EGO_ID: frames_a, bcd.PEER_ID: frames_b})
        pairs = [(g[bcd.EGO_ID], g[bcd.PEER_ID]) for g in groups]
        for fa, fb in pairs:
            raw, corrected, _ = bcd.build_scene(fa, fb)
            agent_pose = raw["1"]
            targets = [("2", corrected["2"]), ("3", corrected["3"]), ("4", corrected["4"])]
            for vid, tgt in targets:
                # Real per-frame wall occlusion (user-decided 2026-09-22:
                # always go by real geometry, not the retired fixed
                # WALL_BLOCKED fact) -- the wall doesn't care which
                # extrinsic is used, so a wall-blocked frame is never
                # informative for this comparison either way.
                if bcd.wall_geometry.blocked_by_wall(
                        (agent_pose["x"], agent_pose["y"]), (raw[vid]["x"], raw[vid]["y"]), bcd.WALLS):
                    continue
                point = np.array([tgt["x"], tgt["y"], tgt["z"]])
                yaw_deg = np.degrees(tgt["yaw_rad"])

                uv_cal, depth_cal = project_point(point, agent_pose, R_CALIBRATED, T_CALIBRATED)
                uv_nom, depth_nom = project_point(point, agent_pose, R_NOMINAL, T_NOMINAL)
                total_checked += 1

                if depth_cal > 0.05 and depth_nom > 0.05:
                    both_front += 1
                    pixel_shifts.append(float(np.linalg.norm(uv_cal - uv_nom)))

                vis_cal = visible(tgt["x"], tgt["y"], tgt["z"], yaw_deg, agent_pose, R_CALIBRATED, T_CALIBRATED)
                vis_nom = visible(tgt["x"], tgt["y"], tgt["z"], yaw_deg, agent_pose, R_NOMINAL, T_NOMINAL)
                if vis_cal != vis_nom:
                    flips += 1

    bcd.EXTRINSIC_PARAMS[AGENT] = (R_CALIBRATED, T_CALIBRATED)  # restore real config

    pixel_shifts = np.array(pixel_shifts)
    print("Real (agent 1) vs nominal extrinsic, agent 1's own camera, all 9 trajectories:")
    print("  target-checks total:      %d" % total_checked)
    print("  both configs in front:    %d" % both_front)
    if len(pixel_shifts):
        print("  pixel-space shift:        mean=%.1fpx  median=%.1fpx  p95=%.1fpx  max=%.1fpx"
              % (pixel_shifts.mean(), np.median(pixel_shifts), np.percentile(pixel_shifts, 95), pixel_shifts.max()))
    print("  visibility FLIPS (calibrated vs nominal disagree): %d / %d (%.1f%%)"
          % (flips, total_checked, 100.0 * flips / total_checked if total_checked else 0.0))


if __name__ == "__main__":
    main()
