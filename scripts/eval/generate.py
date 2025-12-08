# Run DiffSBDD to sample ligands.
# Convert the best ligand from SDF into SMILES or CCD (RDKit can give you the SMILES).
# Generate a Boltz YAML using that ligand ID and the protein sequences/structures for the template pocket. (The helper we wrote, scripts/eval/boltz_yamls/gen_yaml.py, can automate this step by inserting the ligand entry and pointing to templates/constraints.)
# Run the Boltz CLI in affinity mode (properties.affinity set) to get affinity_pred_value and affinity_probability_binary.

ef run_diffsbdd():
    pass

def convert_ligand_to_smiles():
    pass

def generate_boltz_yaml():
    pass

def main():
    pass

if __name__ == "__main__":
    main()

# Run DiffSBDD to sample ligands.
# Convert the best ligand from SDF into SMILES or CCD (RDKit can give you the SMILES).
# Generate a Boltz YAML using that ligand ID and the protein sequences/structures for the template pocket. (The helper we wrote, scripts/eval/boltz_yamls/gen_yaml.py, can automate this step by inserting the ligand entry and pointing to templates/constraints.)
# Run the Boltz CLI in affinity mode (properties.affinity set) to get affinity_pred_value and affinity_probability_binary.

d