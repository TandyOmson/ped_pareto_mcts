import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from ped_pareto_mcts.utils.utils import load_class
from joblib import Parallel, delayed
from rdkit import Chem
import pandas as pd
from tqdm import tqdm

def safe_evaluate(obj, smi, scale=False):
    try:
        reward = obj.evaluate(smi)
        if scale and hasattr(obj, "reward_mean") and hasattr(obj, "reward_std"):
            reward = obj.scale_reward(reward)
        return reward
    except Exception:
        return obj.failure_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smifile", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scale", type=bool, default=False)

    args = parser.parse_args()

    with open(args.config, "r") as fr:
        config = yaml.safe_load(fr)

    print("loading smiles")
    smis = [i.strip() for i in open(Path(args.smifile), "r").readlines()]
    print("canonicalising smiles")
    canon_smis = []
    for i in smis:
        try:
            smi = Chem.CanonSmiles(i)
            canon_smis.append(smi)
        except:
            continue
    smis = canon_smis
    print(f"loaded {len(smis)} smiles")

    print("loading objective functions")
    objective_functions = []
    for objective in config["objective_functions"]:
        print(f"Loading objective: {objective}")
        objective_config = config["objective_functions"][objective]
        objectiveClass = load_class(objective_config["class_path"])
        objective_args = objective_config["kwargs"]

        obj = objectiveClass(objective, **objective_args)
        if "mean" in objective_config and "std" in objective_config:
            obj.reward_mean = objective_config["mean"]
            obj.reward_std = objective_config["std"]
        objective_functions.append(obj)

    all_objectives = {}
    for obj in objective_functions:
        print(f"evaluating objective: {obj.name}")
        objective_values = Parallel(
            n_jobs=11,
            backend="loky"
        )(
            delayed(safe_evaluate)(obj, smi, scale=args.scale)
            for smi in tqdm(smis)
        )
        all_objectives[obj.name] = objective_values

    print("writing to file")
    df = pd.DataFrame(all_objectives)
    df["smiles"] = smis
    df.to_csv(Path(args.outfile), index=False)
