"""conf.json loading, CLI/config precedence, and plugin import for qcar/.

Every default of train.py / eval.py / visualize.py lives in qcar/conf.json --
the scripts hardcode none. Precedence, lowest to highest, for anything that
selects a module or a setting:

    qcar/conf.json  <  experiment yaml (or a run's resolved_hypes.json)  <  CLI flag

Relative paths in conf.json resolve against the HEAL repo root (the parent of
qcar/); relative paths given on the CLI resolve against the caller's cwd, like
any other command-line tool.
"""
import importlib
import importlib.util
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_CONF = os.path.join(HERE, "conf.json")


class Conf(dict):
    """conf.json contents. A missing key is a clear error, never a silent
    fallback -- add the key to conf.json instead of hardcoding it."""

    def __init__(self, path, data):
        dict.__init__(self, data)
        self.path = path

    def __missing__(self, key):
        raise SystemExit("%s has no %r key -- add it there" % (self.path, key))

    def path_of(self, key):
        """Path-valued key, resolved against the HEAL repo root."""
        return repo_path(self[key])


def load_conf(path=DEFAULT_CONF):
    if not os.path.exists(path):
        raise SystemExit("conf.json not found: %s" % path)
    with open(path) as f:
        return Conf(path, json.load(f))


def conf_path_from_argv(argv=None):
    """--conf is read before the full parser is built, because every other
    flag's default comes from it."""
    argv = sys.argv[1:] if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == "--conf" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
        if arg.startswith("--conf="):
            return os.path.abspath(arg.split("=", 1)[1])
    return DEFAULT_CONF


def repo_path(value):
    if not value or os.path.isabs(value):
        return value
    return os.path.join(REPO_ROOT, value)


def cli_path(value):
    """CLI paths are relative to the caller's cwd; make them absolute before
    the script chdirs to the repo root."""
    return os.path.abspath(value) if value else value


def split_list(value):
    """'a,b' or ['a', 'b'] -> ['a', 'b']; '' -> []."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [v for v in value if v]
    return [v.strip() for v in value.split(",") if v.strip()]


# --------------------------------------------------------------------------
# plugins
# --------------------------------------------------------------------------

def import_plugin(spec):
    """Import one plugin: a dotted module name (qcar.patches.patch_1cam_loader)
    or a path to a .py file (relative paths resolve against the repo root).
    Importing is the whole activation -- a plugin applies itself on import."""
    if spec.endswith(".py") or os.sep in spec:
        path = repo_path(spec)
        if not os.path.isfile(path):
            raise SystemExit("plugin file not found: %s" % path)
        name = "qcar_plugin_" + os.path.splitext(os.path.basename(path))[0]
        if name in sys.modules:
            return sys.modules[name]
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


def load_plugins(specs):
    for spec in specs:
        import_plugin(spec)
    return list(specs)


def pick(cli_value, hypes, hypes_key, conf, conf_key):
    """CLI > hypes > conf.json, for list-valued module selections."""
    if cli_value is not None:
        return split_list(cli_value)
    if hypes_key in hypes:
        return split_list(hypes[hypes_key])
    return split_list(conf[conf_key])


def resolve_modules(opt, hypes, conf):
    """Decide the plugin list and model/loss search packages for this run and
    record them in hypes, so resolved_hypes.json replays exactly the same
    modules at eval/visualize time."""
    hypes["_qcar_plugins"] = pick(opt.plugins, hypes, "_qcar_plugins",
                                  conf, "plugins")
    hypes["_qcar_model_packages"] = pick(opt.model_packages, hypes,
                                         "_qcar_model_packages",
                                         conf, "model_packages")
    hypes["_qcar_loss_packages"] = pick(opt.loss_packages, hypes,
                                        "_qcar_loss_packages",
                                        conf, "loss_packages")
    load_plugins(hypes["_qcar_plugins"])
    print("[qcar] plugins: %s" % (hypes["_qcar_plugins"] or "none"))


# --------------------------------------------------------------------------
# hypes overrides
# --------------------------------------------------------------------------

def set_by_path(hypes, dotted_key, value):
    node = hypes
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(hypes, conf_overrides, cli_sets):
    """conf.json 'overrides' first, then each --set KEY=VALUE (VALUE parsed as
    yaml, so 3, 0.5, true, [1,2] and plain strings all work).

    Note: this runs AFTER load_yaml's yaml_parser, so it does not re-derive
    anchors/grid sizes -- override cav_lidar_range/voxel_size in the yaml."""
    applied = {}
    for key, value in (conf_overrides or {}).items():
        set_by_path(hypes, key, value)
        applied[key] = value
    for item in cli_sets or []:
        if "=" not in item:
            raise SystemExit("--set expects KEY=VALUE, got %r" % item)
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        set_by_path(hypes, key.strip(), value)
        applied[key.strip()] = value
    for key, value in applied.items():
        print("[qcar] override %s = %r" % (key, value))
    return applied


def add_module_args(ap):
    """Flags shared by every qcar script for picking modules at the CLI.
    Defaults are None so 'not given' falls through to yaml, then conf.json."""
    ap.add_argument("--conf", default=DEFAULT_CONF,
                    help="conf.json with this script's defaults")
    ap.add_argument("--plugins", default=None,
                    help="comma list of modules or .py files to import before "
                         "building the dataset; '' = none "
                         "(default: yaml _qcar_plugins, then conf.json plugins)")
    ap.add_argument("--model_packages", default=None,
                    help="comma list of packages searched for model.core_method "
                         "(default: yaml, then conf.json model_packages)")
    ap.add_argument("--loss_packages", default=None,
                    help="comma list of packages searched for loss.core_method "
                         "(default: yaml, then conf.json loss_packages)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one hypes key by dotted path, e.g. "
                         "--set model.args.fusion_method=max (repeatable)")
