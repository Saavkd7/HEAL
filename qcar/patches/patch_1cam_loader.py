"""Opt-in monkeypatch: make OPV2VBaseDataset.find_camera_files() return ONLY
camera0 (front), instead of the hardcoded camera0..camera3.

WHY A MONKEYPATCH AND NOT AN EDIT TO THE VENDORED SOURCE
----------------------------------------------------------
`find_camera_files` is a fixed 4-path @staticmethod with no config hook
anywhere (verified: grep for `Ncams` across opencood/data_utils/ and
opencood/models/ returns zero matches -- HEAL's own dataloader and model
code never read it; camera count is purely "how many files this one
function finds"). Editing opv2v_basedataset.py directly would silently
change behaviour for the STOCK 4-camera OPV2V dataset too. This patch is
applied only by scripts that explicitly `import` it, keeping the vendored
HEAL source untouched -- the same "port, don't modify" discipline used for
the rest of this project's QuantV2X/HEAL forks.

Usage: `import qcar.patches.patch_1cam_loader` before building the
dataset. Importing is the entire activation -- no function call needed.

DANGER -- THIS IS A PROCESS-WIDE MONKEYPATCH, NOT PER-DATASET
------------------------------------------------------------------
`find_camera_files` is a class attribute, not an instance one. Once this
module is imported ANYWHERE in a running process (including a long-lived
Jupyter kernel), EVERY OPV2VBaseDataset built afterwards -- even one whose
own yaml says Ncams:4 and points at a real 4-camera directory like
opv2v_ready/ -- silently loads only camera0. Verified: building
camera_attfuse_qcar.yaml (Ncams:4) in the same process as this import still
returned len(train)=90 with no error, because camera0.png always exists;
the other three files were simply never looked for.

If a single script/notebook needs both a 4-camera and a 1-camera dataset,
either do not import this module for the 4-camera one, or run them in
separate processes. Never assume Ncams in a yaml controls this -- it
doesn\'t (grep confirms zero reads of Ncams anywhere in the dataloader).
"""
import os
from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset


def _find_camera_files_1cam(cav_path, timestamp, sensor="camera"):
    return [os.path.join(cav_path, timestamp + "_%s0.png" % sensor)]


OPV2VBaseDataset.find_camera_files = staticmethod(_find_camera_files_1cam)
print("[patch_1cam_loader] OPV2VBaseDataset.find_camera_files -> 1-camera (front only)")
