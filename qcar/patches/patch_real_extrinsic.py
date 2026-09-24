"""Monkeypatch OPV2VBaseDataset.get_ext_int so HEAL's LSS geometry uses the
REAL camera-to-vehicle rotation instead of HEAL's own hardcoded CARLA/UE4-
to-OpenCV fix-up matrix -- generalized for any ego pose, not just the
origin (PretrainFront keeps ego at [0,0,0,0,0,0]; the two-agent cooperative
dataset has each agent at its own real, moving Vicon pose).

WHY THIS EXISTS
----------------
HEAL's stock get_ext_int() does:
    camera_to_lidar = x1_to_x2(camera_coords, lidar_pose_clean) @ FIX
where FIX = [[0,0,1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]] converts CARLA/UE4's
left-handed pose convention to OpenCV's right-handed one -- correct only
if the rotation feeding into it already comes from that convention. Our
qcar_real cords never encode a camera-relative tilt (camera's own cords
yaw is always set equal to its ego's yaw), so the x1_to_x2 rotation before
FIX is always Identity regardless of ego's world pose (both poses share
the same yaw by construction) -- verified both algebraically and against
HEAL's own x1_to_x2 for the ego-at-origin case, 2026-09-14. The result is
FIX itself, which is a REFLECTION (det=-1: left-right mirror), not the true
camera-to-vehicle rotation (det=+1, verified against the Quanser manual and
then refined against 19 real image/Vicon correspondences from qcar-52775 --
mean reprojection error 8.8px across 0.46-2.9m and the full yaw range).

GENERAL DERIVATION (any ego pose, not just the origin)
---------------------------------------------------------
Because camera cords' yaw always equals its own ego's yaw, and roll/pitch
are always 0 for both, the pre-FIX rotation is Identity for any ego pose
(x_to_world(cords)'s rotation and x_to_world(lidar_pose)'s rotation are
identical, so one cancels the other exactly). The pre-FIX translation is
R_ego_yaw^T @ (camera_world_pos - ego_world_pos): the camera's mounting
offset expressed in ego's own body frame. This patch computes that
translation directly (bypassing x1_to_x2's general machinery, since the
rotation part is already known to be Identity) and applies the REAL,
data-refined rotation instead of FIX.

WHAT'S HARD-CODED HERE
-------------------------
R_CB / T_CB below are for qcar-52775's (192.168.1.198) front camera,
refined 2026-09-14 from 19 image/Vicon click correspondences spanning
0.46-2.9m and the full yaw range (median reprojection error 7.7px), plus a
verified body-frame marker-to-physical-center offset (measured on the
vehicle: +0.0343 m forward, -0.0209 m lateral; independently recovered
from the same 19-point fit as +0.0334/-0.0178 -- the two agree). qcar-52776
(192.168.1.158) has not been separately calibrated this way; it currently
reuses the same manual-nominal rotation (not yet the refined one) via
R_CB_NOMINAL, applied per-camera by qcar_id below. Extend CAMERA_PARAMS
before trusting qcar-52776 to the same precision.
"""
import numpy as np

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import (
    OPV2VBaseDataset,
)
from opencood.utils.transformation_utils import x_to_world

R_CB_NOMINAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

# qcar-52775 (192.168.1.198), refined 2026-09-14 (19-point fit, outliers dropped).
R_CB_QCAR52775 = np.array([
    [-0.00168635, -0.07682121, 0.99704346],
    [-0.99920993, -0.03946071, -0.00473042],
    [0.03970744, -0.9962637, -0.07669397],
], dtype=np.float64)
T_CB_QCAR52775 = np.array([0.21306336, -0.00169975, 0.11930449], dtype=np.float64)

CAMERA_PARAMS = {
    "1": (R_CB_QCAR52775, T_CB_QCAR52775),   # qcar-52775, refined
    "2": (R_CB_NOMINAL, np.array([0.1930, 0.0, 0.0953])),  # qcar-52776, nominal only
}


def get_ext_int_qcar_real(self, params, camera_id):
    lidar_pose = params["lidar_pose_clean"]
    x, y, z, roll, yaw, pitch = lidar_pose
    if abs(roll) > 1e-6 or abs(pitch) > 1e-6:
        raise NotImplementedError(
            "patch_real_extrinsic assumes zero roll/pitch on lidar_pose "
            "(true for every qcar_real dataset built so far)."
        )
    camera_coords = np.array(
        params["camera%d" % camera_id]["cords"]).astype(np.float64)
    cam_yaw = camera_coords[4]
    if abs(cam_yaw - yaw) > 1e-3:
        raise NotImplementedError(
            "patch_real_extrinsic assumes camera cords' yaw always equals "
            "its own ego's yaw (true for every qcar_real build script so "
            "far) -- got camera yaw=%.3f vs ego yaw=%.3f" % (cam_yaw, yaw)
        )

    cav_id = params.get("qcar_provenance", {}).get("cav_id") or params.get("_qcar_cav_id")
    # PretrainFront's yaml predates the cav_id field and is 100% qcar-52775,
    # so default to its refined (v2) geometry: extrinsic + marker-to-
    # physical-center offset, both verified 2026-09-14.
    R_cb, t_cb = CAMERA_PARAMS.get(str(cav_id), CAMERA_PARAMS["1"])

    ego_yaw_rad = np.radians(yaw)
    c, s = np.cos(ego_yaw_rad), np.sin(ego_yaw_rad)
    R_ego = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    t_ego = np.array([x, y, z], dtype=np.float64)
    t_cam_world = camera_coords[:3]

    translation_in_ego_frame = R_ego.T @ (t_cam_world - t_ego)

    camera_to_lidar = np.eye(4, dtype=np.float32)
    camera_to_lidar[:3, :3] = R_cb
    camera_to_lidar[:3, 3] = translation_in_ego_frame
    camera_intrinsic = np.array(
        params["camera%d" % camera_id]["intrinsic"]).astype(np.float32)
    return camera_to_lidar, camera_intrinsic


OPV2VBaseDataset.get_ext_int = get_ext_int_qcar_real
print("[patch_real_extrinsic] OPV2VBaseDataset.get_ext_int -> real, "
      "data-refined camera-to-vehicle geometry (was HEAL's CARLA/UE4 mirror)")
