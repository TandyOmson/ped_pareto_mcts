""" Generate random moleculess using the trained model
"""
import argparse
from pathlib import Path

from networkx import display

from ped_pareto_mcts.utils.utils import load_model, load_tokens, load_class
from ped_pareto_mcts.policies.generative_rollout import GenerativeRollout
from ped_pareto_mcts.base.tree import Node
from ped_pareto_mcts.gen_models.model_utils import tokenize_smiles
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--gen_model_class_path", type=str, required=True)
    parse.add_argument("--model_file", type=str, required=True)
    parse.add_argument("--vocab_file", type=str, required=True)
    parse.add_argument("--num_mols", type=int, default=10)
    parse.add_argument("--max_len", type=int, default=30)
    parse.add_argument("--prefix", type=str, required=False)
    parse.add_argument("--outimages", type=str, required=False)

    args = parse.parse_args()
    
    modelClass = load_class(args.gen_model_class_path)
    all_tokens = load_tokens(Path(args.vocab_file))
    model = load_model(modelClass, Path(args.model_file), all_tokens)

    def get_max_prob_sequence():
        if not args.prefix:
            root_node = Node(token="<bos>", parent=None)
            rollout = GenerativeRollout(args.max_len, model)
            path = rollout.simulate(root_node)

        if args.prefix:
            tokenized_prefix = tokenize_smiles([args.prefix], use_selfies=True)[1][0]

            root_node = Node(token="<bos>", parent=None)
            current_node = root_node
            for token in tokenized_prefix:
                new_node = Node(token=token, parent=current_node)
                current_node = new_node
            rollout = GenerativeRollout(args.max_len, model)
            path = rollout.simulate(current_node)

        return path
    
    mol_images = []
    for _ in range(args.num_mols):
        new_seq = get_max_prob_sequence()
        print("".join(new_seq), len(new_seq)-2)
        try:
            mol = Chem.MolFromSmiles(sf.decoder("".join(new_seq[1:-1])))
            if mol is None:
                raise Exception
            mol_images.append(mol)
        except:
            print("Invalid molecule:", "".join(new_seq[1:-1]))

    # save sample of generated molecules
    if args.outimages is not None:
        img = Draw.MolsToGridImage(mol_images[:25], molsPerRow=5)
        img.save(args.outimages)
