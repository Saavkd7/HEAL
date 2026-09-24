"""Model / loss factories that can load a class from ANY package, not only
opencood.models / opencood.loss.

HEAL's own train_utils.create_model/create_loss hardcode the package prefix.
These keep HEAL's naming rule (class whose lowercase name equals core_method
with '_' stripped) but take the search packages from config, and also accept:

    core_method: heter_model_baseline              -> searched in each package
    core_method: my_pkg.my_models.custom_fusion    -> that exact module
    core_method: my_pkg.my_models.custom_fusion:MyNet  -> that exact class

Fusion inside HEAL's heterogeneous baseline is chosen by
model.args.fusion_method (max, att, disconet, v2vnet, v2xvit, cobevt,
where2comm, who2com), so swapping fusion is a yaml / --set change.
"""
import importlib


def _find_class(module, target):
    target = target.replace("_", "").lower()
    found = None
    for name, obj in module.__dict__.items():
        if name.lower() == target:
            found = obj
    return found


def resolve_class(core_method, packages, kind):
    if ":" in core_method:
        module_name, class_name = core_method.split(":", 1)
        module = importlib.import_module(module_name)
        if not hasattr(module, class_name):
            raise SystemExit("%s: %s has no class %s" % (kind, module_name, class_name))
        return getattr(module, class_name)

    candidates = []
    if "." in core_method:
        candidates.append((core_method, core_method.rsplit(".", 1)[1]))
    candidates += [("%s.%s" % (pkg, core_method), core_method) for pkg in packages]

    tried = []
    for module_name, short_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as err:
            # only swallow "this candidate module does not exist", never an
            # import error raised from inside a module that does exist
            if err.name and module_name.startswith(err.name):
                tried.append(module_name)
                continue
            raise
        cls = _find_class(module, short_name)
        if cls is not None:
            return cls
        tried.append("%s (module found, no class %r)" %
                     (module_name, short_name.replace("_", "")))
    raise SystemExit("%s %r not found. Tried: %s" % (kind, core_method, ", ".join(tried)))


def create_model(hypes):
    cls = resolve_class(hypes["model"]["core_method"],
                        hypes.get("_qcar_model_packages", ["opencood.models"]),
                        "model")
    return cls(hypes["model"]["args"])


def create_loss(hypes):
    cls = resolve_class(hypes["loss"]["core_method"],
                        hypes.get("_qcar_loss_packages", ["opencood.loss"]),
                        "loss")
    return cls(hypes["loss"]["args"])
