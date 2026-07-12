from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

def get_ECFP4(smi, nbits=2048):
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(smi))
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)

    return fpgen.GetFingerprint(mol)
