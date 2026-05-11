import argparse
from ped_pareto_mcts.gen_models.model_utils import tokenize_smiles, save_model, save_tokens
from ped_pareto_mcts.gen_models.transformer import TransformerNextTokenLM, TransformerTrainer
from ped_pareto_mcts.gen_models.rnn import LSTMNextTokenLM, LSTMTrainer
from pathlib import Path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles")
    parser.add_argument("--model_out")
    parser.add_argument("--vocab_out")
    
    args = parser.parse_args()

    smis = [i.strip() for i in open(Path(args.smiles)).readlines()]
    all_tokens, sequences = tokenize_smiles(smis, use_selfies=True)
    print(all_tokens)

    model = LSTMNextTokenLM(all_tokens)
    LSTMTrainer.fit(model, sequences, epochs=5)

    print("vocab_size:", len(model.vocab.itos))
    save_model(model, Path(args.model_out))
    save_tokens(model.vocab.itos, Path(args.vocab_out))
