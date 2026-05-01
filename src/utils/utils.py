import importlib
import inspect
import torch

def load_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def get_all_init_params(cls):
    """
    Will get all parameters accepted by __init__ of a class, including those inherited from parent classes.
    If any __init__ accepts **kwargs, return None (means: accept everything).
    """
    accepted = set()

    for base in cls.__mro__:
        if base is object:
            continue

        if "__init__" not in base.__dict__:
            continue

        sig = inspect.signature(base.__init__)

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            # if param.kind == inspect.Parameter.VAR_KEYWORD:
            #     # **kwargs present → no filtering should be applied
            #     return None

            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                accepted.add(name)

    return accepted

def filter_class_config(cls, **config):
    accepted = get_all_init_params(cls)

    if accepted is None:
        # class (or one of its parents) accepts **kwargs
        return dict(config)

    return {k: v for k, v in config.items() if k in accepted}

def load_model(modelClass, modelpath, vocab, device="cpu"):
    checkpoint = torch.load(modelpath, map_location=device)

    model = modelClass(
        vocab=vocab,
        **checkpoint["config"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model

def load_tokens(tokenpath):
    with open(tokenpath, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]