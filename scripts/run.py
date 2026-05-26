""" Runscript for MCTS
"""
import argparse
import yaml
import logging
import sys
import pprint
from pathlib import Path
import csv
import torch

from ped_pareto_mcts.utils.utils import load_model, load_tokens, load_class

def normalise_str(v):
    if isinstance(v, str):
        return v.replace("\n", "\\n")
    return v

class MCTS:
    """ MCTS Callable
    """
    def __init__(self, config):
        self.config = config

        # Stopping criteria
        self.total_gen_mols = 0

    def __call__(self):        
        vocab = load_tokens(Path(self.config["gen_model"]["vocab_file"]))
        device = self.config.get("device", "cpu")
        model = load_model(Path(self.config["gen_model"]["model_file"]), vocab, device=device)
        
        cyclerClass = load_class(Path(self.config["MCTS"]["class_path"]))
        cycler = cyclerClass(model, self.config)

        
        with open(Path(self.config["outdir"]) / "log.csv", "w", newline="") as f:
            writer = None
            while self.total_gen_mols < self.config["MCTS"]["max_gen_mols"]:
                step_log = cycler.step()
                self.total_gen_mols += 1
            
                log.info(f"STEP {self.total_gen_mols}")
                log.debug(pprint.pformat(step_log, compact=True))

                if step_log:
                    step_dict = {"gen_num" : self.total_gen_mols} | {"smiles": step_log["molecule"]} | {i:k for i,k in step_log["reward"].items()}
                    step_dict = {k: normalise_str(v) for k, v in step_dict.items()}
                    
                    if writer is None:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=step_dict.keys(),
                        )
                        writer.writeheader()
                    
                    writer.writerow(step_dict)

        # key result
        return cycler.archive.front
    
def setup_logging(log_dir):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    try:
        log_dir.mkdir(parents=False, exist_ok=False)
    except:
        raise Exception(f"Log directory {log_dir} already exists. Exiting...")
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout (debug level) 
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    log.addHandler(console)

    # run log 
    run_handler = logging.FileHandler(log_dir / "run.log", mode="w")
    run_handler.setLevel(logging.INFO)
    run_handler.setFormatter(formatter)
    log.addHandler(run_handler)

    # detail log 
    detail_handler = logging.FileHandler(log_dir / "detail.log", mode="w")
    detail_handler.setLevel(logging.DEBUG)
    detail_handler.setFormatter(formatter)
    log.addHandler(detail_handler)
    
    return

log = logging.getLogger(__name__)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on",
    )
    args = parser.parse_args()

    with open(args.config, "r") as fr:
        config = yaml.safe_load(fr)

    config["device"] = args.device

    setup_logging(Path(config["outdir"]))
    log.info(f"Using device: {config['device']}")
    
    log.info("STARTING MCTS:")
    final_pareto_front = MCTS(config)()

    log.info("FINAL RESULT:")
    log.info(pprint.pformat(final_pareto_front, compact=True))
