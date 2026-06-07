from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess as sp

def rdkit_robust_embed(mol):
    if mol == None:
        raise Exception

    res = AllChem.EmbedMolecule(mol)
    if res == -1:
        res = AllChem.EmbedMolecule(mol, useBasicKnowledge=False)
        if res == -1:
            raise Exception
        else:
            AllChem.MMFFOptimizeMolecule(mol)

    return mol

def openbabel_embed(smi):
    obabel_cmd = 'echo "{}" | obabel -ismi -osdf --gen3d best --minimize'
    res = sp.run(obabel_cmd.format(smi),
                 shell=True,
                 capture_output=True,
                 text=True,
                 )
    
    mol = Chem.MolFromMolBlock(res.stdout, removeHs=False)
    if mol == None:
        print(smi)
        print(res.stderr)
        raise Exception

    return mol

def embed_mol(mol):
    try:
        mol = rdkit_robust_embed(mol)
    except:
        try:
            mol = openbabel_embed(Chem.MolToSmiles(mol))
        except:
            raise Exception

    return mol
