from ped_pareto_mcts.base.reward import ObjectiveFunc
from ped_pareto_mcts.utils.rdmol_utils import smiles_to_mol
from ped_pareto_mcts.objectives.sascorer import calculateScore as sascore
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.SpacialScore import SPS

class logpObjective(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        m = smiles_to_mol(smi)
        return MoleculeDescriptors.MolecularDescriptorCalculator(["MolLogP"]).CalcDescriptors(m)[0]

class qedObjective(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        m = smiles_to_mol(smi)
        return MoleculeDescriptors.MolecularDescriptorCalculator(["qed"]).CalcDescriptors(m)[0]
    
class tpsaObjective(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        m = smiles_to_mol(smi)
        return MoleculeDescriptors.MolecularDescriptorCalculator(["TPSA"]).CalcDescriptors(m)[0]
    
class molwtObjective(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        m = smiles_to_mol(smi)
        return MoleculeDescriptors.MolecularDescriptorCalculator(["MolWt"]).CalcDescriptors(m)[0]

class spsObjective(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        m = smiles_to_mol(smi)
        sps = SPS(m, normalize=True)
        return -sps
    
class saObjective(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        m = smiles_to_mol(smi)
        score =  sascore(m)
        return -score
