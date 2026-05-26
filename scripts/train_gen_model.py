import argparse
from pathlib import Path
import yaml
import logging
import sys
import pprint
import torch

from ped_pareto_mcts.gen_models.model_utils import tokenize_smiles, save_model, save_tokens, estimate_model_memory
from ped_pareto_mcts.base.gen_model import GenModelTrainer
from ped_pareto_mcts.utils.utils import load_class

log = logging.getLogger(__name__)

def setup_logging(logfile):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    # stdout (debug level)
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)
    log.addHandler(console)

    # training log (info level)
    run_handler = logging.FileHandler(Path(logfile))
    run_handler.setLevel(logging.INFO)
    log.addHandler(run_handler)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles")
    parser.add_argument("--model_out")
    parser.add_argument("--vocab_out")
    parser.add_argument("--model_config")
    parser.add_argument("--log_file", default="train.log")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run training on",
    )

    args = parser.parse_args()

    setup_logging(args.log_file)

    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model_config"]
    train_config = config.get("train_config", {"training_method": "teacher_forcing"})

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    smis = [i.strip() for i in open(Path(args.smiles)).readlines()]
    log.info(f"tokenizing {len(smis)} smiles")
    all_tokens, sequences = tokenize_smiles(smis, use_selfies=True)
    log.info(all_tokens)
    log.info(f"vocab_size: {len(all_tokens)}")

    log.info(f"loading class from {config['model_class_path']}")
    modelClass = load_class(Path(config["model_class_path"]))
    model = modelClass(all_tokens, **model_config)
    model.to(device)

    model_memory = estimate_model_memory(model, input_size=(1, model.max_len), vocab_size=len(all_tokens), batch_size=1, dtype_bytes=4, optimizer="adam")
    log.info(pprint.pformat(model_memory))

    train_method = train_config.get("training_method", "teacher_forcing")
    if train_method == "teacher_forcing":
        GenModelTrainer.fit(model, sequences, epochs=250, train_config=train_config)
    elif train_method == "scheduled_sampling":
        GenModelTrainer.fit_scheduled_sampling(model, sequences, epochs=250, train_config=train_config)
    else:
        raise ValueError(f"Unknown training method: {train_method}")

    save_model(model, Path(args.model_out), Path(config["model_class_path"]), model_config=model_config)
    save_tokens(model.vocab.itos, Path(args.vocab_out))
