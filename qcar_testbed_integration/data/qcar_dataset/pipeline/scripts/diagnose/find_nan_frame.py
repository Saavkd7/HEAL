"""Find which validate frame(s) produce NaN/Inf in the loss, one frame at a
time (batch size 1), on CPU so it does not compete with the live GPU
training run. Uses the same hypes/model/loss as camera_attfuse_pretrain.yaml
and the epoch-1 checkpoint (the last known-good one, before NaN first
appeared at epoch 2).
"""
import argparse
import glob
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

# opencood is importable directly once `python setup.py develop` has been run
# from the HEAL repo root (see its own CLAUDE.md) -- no path hack needed.
import qcar.patches.patch_1cam_loader  # noqa: F401
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

HERE = os.path.dirname(os.path.abspath(__file__))
# diagnose/ -> scripts/ -> pipeline/ -> qcar_dataset/ -> data/ -> qcar_testbed_integration/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
DEFAULT_CONF = os.path.join(HERE, "conf.json")


class _Conf(dict):
    """conf.json contents. It is THE source of every default -- nothing is
    hardcoded in this script; a CLI flag only overrides a key for one run.
    A missing key is a clear error, never a silent fallback."""

    def __init__(self, path, data):
        dict.__init__(self, data)
        self.path = path

    def __missing__(self, key):
        raise SystemExit("%s has no %r key -- add it there" % (self.path, key))

    def path_of(self, key):
        """Path-valued key: relative paths resolve against the
        qcar_testbed_integration/ root (REPO_ROOT); null stays None."""
        v = self[key]
        return v if not v or os.path.isabs(v) else os.path.join(REPO_ROOT, v)


def _load_conf(path):
    if not os.path.exists(path):
        raise SystemExit("conf.json not found: %s" % path)
    with open(path) as f:
        return _Conf(path, json.load(f))


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=DEFAULT_CONF)
_conf = _load_conf(_pre.parse_known_args()[0].conf)

# --- CLI flags, see qcar_dataset/pipeline/README.md's "Flag convention" ---
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--conf", default=DEFAULT_CONF,
    help="conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
_parser.add_argument(
    "--heal-root", default=_conf.path_of("heal_root"),
    help="HEAL repo root, so opencood/hypes_yaml/... paths resolve (default: conf.json's heal_root)")
_parser.add_argument(
    "--hypes", default=_conf["hypes"],
    help="Hypes yaml, relative to --heal-root (default: conf.json's hypes)")
_parser.add_argument(
    "--checkpoint-glob", default=_conf["checkpoint_glob"],
    help="Glob (relative to --heal-root) matching the run dir to scan; the "
         "lexicographically-last match is used (default: conf.json's checkpoint_glob)")
_parser.add_argument(
    "--checkpoint-name", default=_conf["checkpoint_name"],
    help="Checkpoint file inside that run dir (default: conf.json's checkpoint_name)")
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    _parser.print_help()
    raise SystemExit(0)
_args, _ = _parser.parse_known_args()

HYPES = os.path.join(_args.heal_root, _args.hypes)
CKPT_DIR = sorted(glob.glob(os.path.join(_args.heal_root, _args.checkpoint_glob)))[-1]
CKPT = os.path.join(CKPT_DIR, _args.checkpoint_name)


class Args:
    hypes_yaml = HYPES
    model_dir = ""


def main():
    # yaml root_dir/validate_dir are relative to the HEAL repo root (resolved
    # via its qcar_dataset symlink) -- must chdir there for build_dataset()
    # to find them, same as running any other opencood/tools/*.py script.
    os.chdir(_args.heal_root)
    hypes = yaml_utils.load_yaml(HYPES, Args())
    hypes["train_params"]["batch_size"] = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = build_dataset(hypes, visualize=False, train=False)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0,
                             collate_fn=val_ds.collate_batch_train, shuffle=False)

    model = train_utils.create_model(hypes)
    state = torch.load(CKPT, map_location=device)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    criterion = train_utils.create_loss(hypes)

    print(f"Checkpoint: {CKPT}")
    print(f"Scanning {len(val_ds)} validate frames one at a time on CPU...")

    bad = []
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = model(batch_data["ego"])
            loss = criterion(output_dict, batch_data["ego"]["label_dict"])
            val = loss.item()
            if val != val or val in (float("inf"), float("-inf")):  # NaN check
                scenario_index = 0
                for si, ele in enumerate(val_ds.len_record):
                    if i < ele:
                        scenario_index = si
                        break
                folder = val_ds.scenario_folders[scenario_index]
                ts_idx = i if scenario_index == 0 else i - val_ds.len_record[scenario_index - 1]
                cav_content = list(val_ds.scenario_database[scenario_index].values())[0]
                ts_key = val_ds.return_timestamp_key(cav_content, ts_idx)
                bad.append((i, val, folder, ts_key))
                print(f"  BAD frame index {i}: loss={val}  scenario={folder}  frame={ts_key}")
            if i % 200 == 0:
                print(f"  ...checked {i}/{len(val_ds)}", flush=True)

    print(f"\nDone. {len(bad)} bad frame(s) out of {len(val_ds)}.")
    for idx, val, folder, ts_key in bad:
        print(f"  index {idx}: loss={val}  scenario={folder}  frame={ts_key}")


if __name__ == "__main__":
    main()
