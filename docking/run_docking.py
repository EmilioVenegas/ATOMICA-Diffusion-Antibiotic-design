import glob
import subprocess
import os
import time
import sys

def run_docking():
    ligands = glob.glob('docking/ligand_*.pdbqt')
    ligands.sort()
    
    print(f"Found {len(ligands)} ligands to dock.")
    
    start_time = time.time()
    
    for i, ligand in enumerate(ligands):
        base_name = os.path.basename(ligand).replace('.pdbqt', '')
        out_file = f"docking/{base_name}_out.pdbqt"
        log_file = f"docking/{base_name}_log.txt"
        
        print(f"[{i+1}/{len(ligands)}] Docking {base_name}...")
        
        # Find vina in the same directory as the python executable
        vina_path = os.path.join(os.path.dirname(sys.executable), 'vina')
        
        cmd = [
            vina_path,
            '--config', 'docking/config.txt',
            '--ligand', ligand,
            '--out', out_file
        ]
        
        try:
            with open(log_file, 'w') as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error docking {base_name}: {e}")
            
    end_time = time.time()
    print(f"Docking completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_docking()
