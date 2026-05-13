""" Generate random moleculess using the trained model
"""
import argparse
from pathlib import Path

from networkx import display

from ped_pareto_mcts.utils.utils import load_model, load_tokens
from ped_pareto_mcts.policies.generative_rollout import GenerativeRollout
from ped_pareto_mcts.base.tree import Node
from ped_pareto_mcts.gen_models.model_utils import tokenize_smiles
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_file", type=str, required=True)
    parse.add_argument("--vocab_file", type=str, required=True)
    parse.add_argument("--num_mols", type=int, default=10)
    parse.add_argument("--max_len", type=int, default=30)
    parse.add_argument("--prefix", type=str, required=False)
    parse.add_argument("--outimages", type=str, required=False)

    args = parse.parse_args()

    all_tokens = load_tokens(Path(args.vocab_file))
    model = load_model(Path(args.model_file), load_tokens(Path(args.vocab_file)))

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
    new_seqs = []
    new_seq_lens = []
    new_smiles = []
    for _ in range(args.num_mols):
        new_seq = get_max_prob_sequence()
        new_seq_lens.append(len(new_seq)-2)
        new_seqs.append("".join(new_seq))
        new_smi = sf.decoder("".join(new_seq[1:-1]))
        new_smiles.append(new_smi)
        try:
            mol = Chem.MolFromSmiles(new_smi)
            if mol is None:
                raise Exception
            mol_images.append(mol)
        except:
            print("Invalid molecule:", "".join(new_seq[1:-1]))

    print("raw sequences")
    for i, j in zip(new_seqs, new_seq_lens):
        print(i, j)
    print("generated smiles")
    for i in new_smiles:
        print(i)
    # save sample of generated molecules
    if args.outimages is not None:
        img = Draw.MolsToGridImage(mol_images[:25], molsPerRow=5)
        img.save(args.outimages)
