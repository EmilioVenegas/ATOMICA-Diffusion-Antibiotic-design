import sys
import os
import subprocess
import argparse

def create_complex(ligand_id):
    # Handle input like "ligand_43" or just "43"
    if "ligand_" in ligand_id:
        ligand_name = ligand_id
    else:
        ligand_name = f"ligand_{ligand_id}"

    docked_pdbqt = f"docking/{ligand_name}_out.pdbqt"
    docked_pdb = f"docking/{ligand_name}_out.pdb"
    receptor_pdb = "3PBQ.pdb"
    output_pdb = f"docking/complex_{ligand_name}.pdb"

    if not os.path.exists(docked_pdbqt):
        print(f"Error: Docked file {docked_pdbqt} not found.")
        return

    # Convert docked PDBQT to PDB
    # We assume obabel is in the path or we find it relative to python
    obabel_path = os.path.join(os.path.dirname(sys.executable), 'obabel')
    if not os.path.exists(obabel_path):
        # Fallback to just 'obabel' if not found in env bin
        obabel_path = 'obabel'

    cmd = [obabel_path, '-ipdbqt', docked_pdbqt, '-opdb', '-O', docked_pdb]
    
    print(f"Converting {docked_pdbqt} to PDB...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting ligand: {e}")
        return

    # Merge files
    print(f"Creating complex {output_pdb}...")
    with open(output_pdb, 'w') as outfile:
        # Write receptor
        with open(receptor_pdb, 'r') as infile:
            for line in infile:
                if not line.startswith('END') and not line.startswith('CONECT'):
                    outfile.write(line)
        
        outfile.write("TER\n")
        
        # Write ligand
        with open(docked_pdb, 'r') as infile:
            for line in infile:
                if not line.startswith('END') and not line.startswith('CONECT'):
                    outfile.write(line)
        
        outfile.write("END\n")

    print(f"Done. File saved to {output_pdb}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a receptor-ligand complex PDB.')
    parser.add_argument('ligand_id', type=str, help='Ligand ID (e.g., 43 or ligand_43)')
    args = parser.parse_args()
    
    create_complex(args.ligand_id)
