"""Windows-only monkeypatch: make HEAL's '/'-based path parsing work with
Windows paths. A no-op on Linux/macOS.

WHY
----
HEAL parses paths with `path.split("/")[-1]` in the places that matter at
inference time, while building those paths with os.path.join -- which on
Windows inserts backslashes. The results (seen live 2026-09-25 on the lab's
Windows workstation, running qcar/realtime/run.py):

  - OPV2VBaseDataset.reinitialize: scenario_name becomes
    'qcar_realtime\\heal\\qcar_live' instead of 'qcar_live'
    -> KeyError in the modality assignment lookup.
  - OPV2VBaseDataset.extract_timestamps: the "timestamp" keeps the whole
    relative tail of the path -> wrong camera/yaml file paths.
  - train_utils.load_saved_model: the best-epoch number is eval()'d from
    the split filename -> fails before returning the loaded model.

FIX (without touching opencood/)
----------------------------------
  - scenario folders are normalized to '/' before reinitialize() uses them
    (Windows accepts '/' everywhere);
  - timestamps come from os.path.basename;
  - train_utils sees glob results with '/' (its own `glob` reference is
    swapped for a shim, the global glob module is left alone).

Imported automatically by qcar/realtime on Windows; importing it elsewhere
(e.g. as a plugin) is safe on any OS.
"""
import glob as _glob
import os
import sys
import types

if sys.platform.startswith("win"):
    from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
    from opencood.tools import train_utils

    _original_reinitialize = OPV2VBaseDataset.reinitialize

    def _reinitialize(self):
        self.scenario_folders = [p.replace("\\", "/") for p in self.scenario_folders]
        return _original_reinitialize(self)

    def _extract_timestamps(yaml_files):
        return [os.path.basename(f).replace(".yaml", "") for f in yaml_files]

    OPV2VBaseDataset.reinitialize = _reinitialize
    OPV2VBaseDataset.extract_timestamps = staticmethod(_extract_timestamps)
    train_utils.glob = types.SimpleNamespace(
        glob=lambda pattern: [p.replace("\\", "/") for p in _glob.glob(pattern)])
    print("[patch_windows_paths] HEAL path parsing made Windows-safe")
