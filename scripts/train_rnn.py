import argparse
from ped_pareto_mcts.gen_models.model_utils import tokenize_smiles, save_model, save_tokens
from ped_pareto_mcts.gen_models.rnn import GRUNextTokenLM, rnnTrainer
from pathlib import Path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles")
    parser.add_argument("--model_out")
    parser.add_argument("--vocab_out")
    
    args = parser.parse_args()

    smis = [i.strip() for i in open(Path(args.smiles)).readlines()]
    all_tokens, sequences = tokenize_smiles(smis)
    print(all_tokens)
    exit()

    model = GRUNextTokenLM(all_tokens, embed_dim=128, hidden_dim=128)
    rnnTrainer.fit(model, sequences, epochs=500)

    # probs = model.get_all_prob_next_symbol(["C", "C"])
    # print(probs)

    print("vocab_size:", len(model.vocab.itos))
    save_model(model, Path(args.model_out))
    save_tokens(model.vocab.itos, Path(args.vocab_out))
