from rdkit import Chem
from rdkit.Chem import rdmolfiles


def bond_order_sum(atom):
    return sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())


def sanitize_for_charges(m):
    m.UpdatePropertyCache(strict=False)
    problems = Chem.DetectChemistryProblems(m)

    if not problems:
        m = Chem.AddHs(m)
        Chem.SanitizeMol(m)
        return m

    for p in problems:
        if p.GetType() != 'AtomValenceException':
            continue

        at = m.GetAtomWithIdx(p.GetAtomIdx())
        Z = at.GetAtomicNum()
        q = at.GetFormalCharge()
        bos = bond_order_sum(at)

        # --------------------
        # POSITIVE CHARGES
        # --------------------

        # Nitrogen: ammonium / pyridinium
        if Z == 7 and q == 0:
            # typical neutral N valence ≤ 3
            if bos > 3:
                at.SetFormalCharge(1)

        # Phosphorus: phosphonium
        elif Z == 15 and q == 0:
            # neutral P typically valence 3 or 5
            if bos > 5:
                at.SetFormalCharge(1)
            elif bos > 3:
                at.SetFormalCharge(1)

        # Sulfur: sulfonium
        elif Z == 16 and q == 0:
            # neutral S typically 2 or 6
            if bos > 2 and bos <= 4:
                at.SetFormalCharge(1)

        # --------------------
        # NEGATIVE CHARGES
        # --------------------

        # Oxygen: carboxylate, phenolate, phosphate
        elif Z == 8 and q == 0:
            # single-bonded O with one neighbour
            if bos == 1:
                at.SetFormalCharge(-1)

        # Nitrogen anion (rare, but occurs)
        elif Z == 7 and q == 0:
            if bos <= 2:
                at.SetFormalCharge(-1)

        # Sulfur anion: thiolate
        elif Z == 16 and q == 0:
            if bos == 1:
                at.SetFormalCharge(-1)

        # Halides
        elif Z in (9, 17, 35, 53) and q == 0:
            if bos == 0:
                at.SetFormalCharge(-1)

    Chem.SanitizeMol(m)
    return m


def smiles_to_mol(smi, allow_charges=True):
    if not allow_charges:
        m = Chem.MolFromSmiles(smi)
        m = Chem.AddHs(m)

    else:
        params = rdmolfiles.SmilesParserParams()
        params.removeHs = False
        params.sanitize = False
        
        m = Chem.MolFromSmiles(smi, params)
        m = sanitize_for_charges(m)
        
    if m is None:
        raise ValueError("SMILES parse failed")

    return m
