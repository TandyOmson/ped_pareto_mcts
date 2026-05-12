import argparse
from pathlib import Path
import yaml

from ped_pareto_mcts.gen_models.model_utils import tokenize_smiles, save_model, save_tokens
from ped_pareto_mcts.base.gen_model import GenModelTrainer
from ped_pareto_mcts.utils.utils import load_class

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles")
    parser.add_argument("--model_out")
    parser.add_argument("--vocab_out")
    parser.add_argument("--model_config")
    parser.add_argument("--train_config", default=None)

    args = parser.parse_args()

    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    if args.train_config:
        with open(args.train_config, "r") as f:
            train_config = yaml.safe_load(f)
            # training method and its config (if I add options other than .fit)
            method = train_config.pop("method")

    smis = [i.strip() for i in open(Path(args.smiles)).readlines()]
    all_tokens, sequences = tokenize_smiles(smis, use_selfies=True)
    print(all_tokens)

    model_class_path = Path(model_config.pop("model_class_path"))
    modelClass = load_class(model_class_path)
    model = modelClass(all_tokens, **model_config)

    if method == "fit":
        GenModelTrainer.fit(model, sequences, epochs=250)
    else:
        raise ValueError(f"Unknown training method: {method}")

    print("vocab_size:", len(model.vocab.itos))
    save_model(model, Path(args.model_out), model_class_path, model_config=model_config)
    save_tokens(model.vocab.itos, Path(args.vocab_out))
