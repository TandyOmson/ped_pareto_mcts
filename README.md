# ped_pareto_mcts

Basic MCTS Algorithm:
1. Selection: Navigates from root R to a leaf L using tree policy. A leaf is a node that is not terminal and not yet fully explored (i.e. there remain next tokens with cumulative probability below the threshold). A single leaf node is chosen based on policy values (i.e. visit count) and cumulative reward.

2. Expansion: A single child node C is added by sampling or selecting one next token according to the conditional distribution given by the autoregressive generative model (i.e. Transformer, RNN, LSTM) at node L.

3. Simulation: The single child node C undergoes one or multiple parallel simulations. Each simulation consists of completing the molecule using the autoregressive generative model, and simulating rewards for each. The rewards are then aggregated.

4. Backpropagation:
Aggregated rewards are assigned only to C and L through R. These nodes also have policy values updated.

Pareto:
Multiple objective functions calculate a reward vector. A global pareto archive maintains the pareto front (non dominated complete molecules) is updated during rollout. During rollout, the simulated molecule is added to the pareto pool if nno dominated and may evict a dominated molecule. Reward is no longer a scalar, but a vector, affected by both the rollout and the Pareto pool. Cumulative reward reflects how likely a node's decendants are to contribute to or improve the global pareto front.

INSTALLATION INSTRUCTIONS:
# Create conda environment
conda create --name ped_pareto_mcts python==3.11
# Install pytorch
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# alternatively for CPU only
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
# Install torch-scatter and torch-cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
# Install remaining required modules from environemnet
conda env update --file environment.yaml --prune
