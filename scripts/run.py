""" Runscript for MCTS
"""
import argparse
import yaml
import logging
import sys
import pprint
from pathlib import Path

from ped_pareto_mcts.utils.utils import load_model, load_tokens, load_class

class MCTS:
    """ MCTS Callable
    """
    def __init__(self, config):
        self.config = config

        # Stopping criteria
        self.total_gen_mols = 0

    def __call__(self):        
        modelClass = load_class(Path(self.config["gen_model"]["class_path"]))
        vocab = load_tokens(Path(self.config["gen_model"]["vocab_file"]))
        model = load_model(modelClass, Path(self.config["gen_model"]["model_file"]), vocab)
        
        cyclerClass = load_class(Path(self.config["MCTS"]["class_path"]))
        cycler = cyclerClass(model, self.config)

        while self.total_gen_mols < self.config["MCTS"]["max_gen_mols"]:
            step_log = cycler.step()
            self.total_gen_mols += 1

            log.info(f"STEP {self.total_gen_mols}")
            log.debug(pprint.pformat(step_log, width=2))

        # key result
        return self.cycler.pareto_front
    
def setup_logging(log_dir):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    # try:
    #     log_dir.mkdir(parents=False, exist_ok=False)
    # except:
    #     raise Exception(f"Log directory {log_dir} already exists. Exiting...")
    
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
    args = parser.parse_args()

    with open(args.config, "r") as fr:
        config = yaml.safe_load(fr)

    setup_logging(Path(config["outdir"]))
    
    log.info("STARTING MCTS:")
    final_pareto_front = MCTS(config)()

    log.info("FINAL RESULT:")
    log.info(pprint.pformat(final_pareto_front, width=1))