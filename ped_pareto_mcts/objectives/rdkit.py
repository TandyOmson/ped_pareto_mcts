from ped_pareto_mcts.base.reward import ObjectiveFunc
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors

class logpObjective(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        m = Chem.MolFromSmiles(smi)
        return MoleculeDescriptors.MolecularDescriptorCalculator(["MolLogP"]).CalcDescriptors(m)[0]

class qedObjective(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        m = Chem.MolFromSmiles(smi)
        return MoleculeDescriptors.MolecularDescriptorCalculator(["qed"]).CalcDescriptors(m)[0]
    
class tpsaObjective(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        m = Chem.MolFromSmiles(smi)
        return MoleculeDescriptors.MolecularDescriptorCalculator(["TPSA"]).CalcDescriptors(m)[0]
    
class molwtObjective(ObjectiveFunc):
    def __init__(self, name):
        super().__init__(name)

    def evaluate(self, smi):
        m = Chem.MolFromSmiles(smi)
        return MoleculeDescriptors.MolecularDescriptorCalculator(["MolWt"]).CalcDescriptors(m)[0]