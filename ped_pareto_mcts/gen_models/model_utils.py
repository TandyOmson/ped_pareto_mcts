import re
import selfies as sf
import torch
from pathlib import Path
import json

def tokenize_smiles(smiles_list, use_selfies=False):
    tokenized_smiles_list = []
    unique_token_set = set()
    for smi in smiles_list:
        try:
            tokenized_smiles = selfies_tokenizer_from_smiles(smi) if use_selfies else smi_tokenizer(smi)
            unique_token_set |= set(tokenized_smiles)
            tokenized_smiles_list.append(tokenized_smiles)
        except:
            continue
    return sorted(list(unique_token_set)), tokenized_smiles_list

def smi_tokenizer(smi):
    """
    This function is based on https://github.com/pschwllr/MolecularTransformer#pre-processing
    Modified by Shoichi Ishida
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens)
    return tokens

def selfies_tokenizer_from_smiles(smi):
    if "[*]" in smi:
        # Because SELFIES (v2.1.0) currently does not support a wildcard (*) representation.
        smi = smi.replace("[*]", "[Lr]")
    slfs = sf.encoder(smi)
    tokens = list(sf.split_selfies(slfs))
    assert slfs == "".join(tokens)
    return tokens

def save_model(model, path, class_path, model_config=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_class_path": class_path,
            "config": model_config if model_config is not None else {},
        },
        path,
    )

def save_tokens(tokens, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

# logging tools
def estimate_model_memory(model, input_size, batch_size=1, dtype_bytes=4, optimizer="adam"):
    # parameter memory
    params = sum(p.numel() for p in model.parameters())
    param_mem = params * dtype_bytes

    # gradients
    grad_mem = params * dtype_bytes

    # optimizer
    if optimizer.lower() == "adam":
        opt_mem = params * dtype_bytes * 2
    else:  # SGD etc.
        opt_mem = 0

    # rough activation estimate (very crude!)
    activations_mem = 0
    hooks = []

    def hook_fn(module, inp, out):
        nonlocal activations_mem
        if isinstance(out, tuple):
            out = out[0]
        if hasattr(out, "numel"):
            activations_mem += out.numel() * dtype_bytes

    for m in model.modules():
        hooks.append(m.register_forward_hook(hook_fn))

    model.eval()
    x = torch.randn((batch_size, *input_size))
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    total = param_mem + grad_mem + opt_mem + activations_mem

    return {
        "params_MB": param_mem / 1024**2,
        "grads_MB": grad_mem / 1024**2,
        "optimizer_MB": opt_mem / 1024**2,
        "activations_MB": activations_mem / 1024**2,
        "total_MB": total / 1024**2,
    }
