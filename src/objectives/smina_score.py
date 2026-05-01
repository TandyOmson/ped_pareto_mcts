import subprocess as sp
import tempfile 
from pathlib import Path
from rdkit import Chem

from base.reward import ObjectiveFunc
from objectives.objecitve_utils import embed_mol

class VinaScore(ObjectiveFunc):
    def __init__(self, receptor_file):
        super().__init__("vina_score")
        
        self.receptor_file = Path(receptor_file)

    def evaluate(self, smi):
        with tempfile.TemporaryDirectory() as rundir:
            mol = embed_mol(smi)
            Chem.MolToMolFile(rundir / "ligand.sdf")

            smina_cmd_output = rundir / "vina.log"
            launch_args = ["smina", 
                           "-r", self.receptor_file, 
                           "-l", rundir / "ligand.sdf", 
                           "--center_x", "0.0",
                           "--center_y", "0.0",
                           "--center_z", "0.0",
                           "--size_x", "30",
                           "--size_y", "30",
                           "--size_z", "30",
                           "--seed", "1000", 
                           "--exhaustiveness", "9", 
                           " >> ", smina_cmd_output, 
                           " 2>&1",
                           ]
            launch_string = ' '.join(launch_args)

            p = sp.Popen(launch_string, shell=True, stdout=sp.PIPE)
            p.communicate()

            affinity = 500
            with open(smina_cmd_output, 'r') as f:
                for lines in f.readlines():
                    lines = lines.split()
                    if len(lines) == 4 and lines[0] == '1':
                        affinity = float(lines[1])
            p = sp.Popen('rm -rf ' + smina_cmd_output, shell=True, stdout=sp.PIPE)
            p.communicate()
            p = sp.Popen('rm -rf ' + rundir / "ligand.sdf", shell=True, stdout=sp.PIPE)
            p.communicate()

        return affinity
