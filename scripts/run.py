import torch
import json

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