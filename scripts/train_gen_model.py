import argparse
from pathlib import Path
import yaml
import logging
import sys
import pprint

from ped_pareto_mcts.gen_models.model_utils import tokenize_smiles, save_model, save_tokens, estimate_model_memory
from ped_pareto_mcts.base.gen_model import GenModelTrainer
from ped_pareto_mcts.utils.utils import load_class

log = logging.getLogger(__name__)

def setup_logging(log_dir):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    # stdout (debug level)
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)
    log.addHandler(console)

    # training log (info level)
    run_handler = logging.FileHandler(Path(log_dir) / "train.log")
    run_handler.setLevel(logging.INFO)
    log.addHandler(run_handler)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles")
    parser.add_argument("--model_out")
    parser.add_argument("--vocab_out")
    parser.add_argument("--model_config")
    parser.add_argument("--log_dir", default="train.log")

    args = parser.parse_args()

    setup_logging(args.log_dir)

    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model_config"]
    if "train_config" not in config.keys():
        train_config = {"training_method": "teacher_forcing"}
    else:
        train_config = config["train_config"]

    smis = [i.strip() for i in open(Path(args.smiles)).readlines()]
    all_tokens, sequences = tokenize_smiles(smis, use_selfies=True)
    print(all_tokens)

    modelClass = load_class(Path(config["model_class_path"]))
    model = modelClass(all_tokens, **model_config)

    model_memory = estimate_model_memory(model, input_size=(1, model.max_len), batch_size=1, dtype_bytes=4, optimizer="adam")
    log.info(pprint.pformat(model_memory))

    if train_config.get("training_method") is not None:
        if train_config["training_method"] == "teacher_forcing":
            GenModelTrainer.fit(model, sequences, epochs=250, train_config=train_config)
        if train_config["training_method"] == "scheduled_sampling":
            GenModelTrainer.fit_scheduled_sampling(model, sequences, epochs=250, train_config=train_config)
        else:
            raise ValueError(f"Unknown training method: {train_config['training_method']}")
    else:
        GenModelTrainer.fit(model, sequences, epochs=250, train_config=train_config)

    print("vocab_size:", len(model.vocab.itos))
    save_model(model, Path(args.model_out), Path(config["model_class_path"]), model_config=model_config)
    save_tokens(model.vocab.itos, Path(args.vocab_out))
