from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess as sp

def add_nitrogen_charges(m):
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType()=='AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(1)
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0:
                bondcount = 0
                for b in at.GetBonds():
                    bondcount += b.GetBondTypeAsDouble()
                if int(bondcount) > 3:
                    at.SetFormalCharge(1)
                
    Chem.SanitizeMol(m)
    return m

def rdkit_robust_embed(mol):
    if mol == None:
        raise Exception
    
    mol = Chem.AddHs(mol)

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

def embed_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    
    try:
        mol = rdkit_robust_embed(mol)
    except:
        try:
            mol = openbabel_embed(smi)
        except:
            mol = None

    return mol