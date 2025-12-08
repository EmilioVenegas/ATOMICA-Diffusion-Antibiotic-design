#!/usr/bin/env python3
"""Generate CSV files for CDK2 binders and random molecules."""

import csv
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Data for CDK2 Binders
cdk2_binders = [
    ["smiles", "id"],  # Match existing format
    ["CC(C)C1=NC(=NC(=N1)NCC2=CC=CC=C2)N[C@@H](CO)CC3=CC=CC=C3", "Roscovitine"],
    ["CN1CCC(CC1)C(C2=CC(=C(C=C2)O)O)C3=C(C(=O)C4=C(C=C(C=C4O3)Cl)O)O", "Flavopiridol"],
    ["CCOc1nc(nc2c1cnn2C3CCCC3)N4CCC(CC4)N5CCN(CC5)CCO", "Dinaciclib"],
    ["CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=C4C5=CC=CC=C5NC4=O", "SNS-032"],
    ["CNC1=NC(=NC(=C1)C2=C(C=C(S2)Cl)F)NC3=CC=C(C=C3)CN4CCN(CC4)CC(=O)N", "AT7519"],
    ["CN1CCN(CC1)C(=O)NC2=CC=C(C=C2)C3=C4C=C(C=CC4=NN3)C5=CC=CC=C5", "Milciclib"],
    ["C1=CC=C2C(=C1)C3=C(C2=O)NC4=C3C=C(C=C4)Br", "Kenpaullone"],
    ["ON=C1c2ccccc2NC(=O)C1=C3Nc4ccccc4C3=O", "Indirubin-3'-monoxime"],
    ["CC1=C(C(N)=NC(=N1)NC2=CC=CC=C2)N3CCCC3", "Olomoucine"],
    ["CC(C)C1=NC(=NC(=N1)NC2=CC=C(C=C2)O)NC(CC3=CC=CC=C3)C(=O)O", "Purvalanol A"],
    ["CC(C)C1=NC(=NC(=N1)NC2=CC=C(C=C2)O)NC(CC3=CC=CC=C3)C(=O)OC", "Purvalanol B"],
    ["CS(=O)(=O)C1=CC=C(C=C1)NC2=NC(=NC3=C2C=CN3)NC4=CC=C(C=C4)S(=O)(=O)C5=CC=CC=C5", "R547"],
    ["CNC(=O)C1=CN(N=C1)C2=CC3=C(C=C2)N=C(N3)NC4=CC=C(C=C4)N(C)CC5=CC=CC=C5", "AZD5438"],
    ["CC1=C(C(=O)N(C2=CC(=C(C=C21)Cl)C3=CN(N=C3)C)C4=CC=C(C=C4)N5CCN(CC5)C)C(=O)C", "Palbociclib"],
    ["CN(C)C(=O)C1=CC=C(C=C1)NC2=NC=C(C(=N2)N3CCCC3)C4=CN(C5=CC=CC=C54)C", "Ribociclib"],
    ["CCN1CCN(CC1)CC2=C(C=C(C=C2)NC3=NC=C(C(=N3)C)F)C4=C(NC5=C4C=C(C=C5)F)C", "Abemaciclib"],
    ["CC1=CC(=C(C=C1)S(=O)(=O)N)NC(=O)CN2C=C(C3=CC=CC=C32)C4=CC=NC=C4", "SU9516"],
    ["CC(C)N(C[C@@H]1CC[C@@H](CC1)O)C2=NC3=C(C(=N2)N)N(C=N3)C4=C(C=CC(=C4)Cl)Cl", "NU6102"],
    ["COC1=C(C=C2C(=C1)N=CN2C3=CC(=C(C=C3)Cl)Cl)NC4=CC=C(C=C4)N5CCN(CC5)C", "CVT-313"],
    ["CN1C(=O)C(=C(C1=O)C2=CN(C3=CC=CC=C32)C4CCNCC4)C5=C(C=C(C=C5)Cl)Cl", "Ro-3306"],
    ["CC1=CC(=C(S1)C2=CN3C(=N2)C=C(N=C3N)Br)C4=CC=C(C=C4)CN5CCN(CC5)CCO", "MK-8776"],
    ["CC(C)C1=CC(=C(C=C1)C2=CN=C(N=C2)N)C3=CC=C(C=C3)Cl", "A-674563"],
    ["CC1=CN(N=C1)C2=CC(=NC(=N2)NC3=CC(=C(C=C3)S(=O)(=O)C(C)C)F)N4CCOCC4", "PF-07104091"],
    ["CCN(CC)CCNC(=O)C1=C(C=C(C=N1)NC2=CC=C(C=C2)OC3=CC=CC=N3)C", "PF-06873600"],
    ["CN(C)CC=CC(=O)NC1=CC2=C(C=C1)N=C(N2)NC3=CC=C(C=C3)N(=O)=O", "JNJ-7706621"],
    ["C1=CC=C(C=C1)S(=O)(=O)NC(=O)C2=CC=C(C=C2)Cl", "Indisulam"],
    ["C1=CC(=CC=C1N=NC2=C(NC3=C2C=C(C=C3)Br)O)S(=O)(=O)N", "CDK2 Inhibitor II"],
    ["CC1=CC(=NO1)C2=CC3=C(C=C2)N=C(N3)NC4=CC=C(C=C4)N5CCN(CC5)C", "AT-9283"],
    ["CC(C)C1=NC(=NC(=N1)N)NC2=CC=C(C=C2)S(=O)(=O)N", "AG-024322"],
    ["COC1=C(C=C(C=C1)CN2CCN(CC2)C(=O)C3=CC=C(C=C3)C4=CC=NO4)OC", "SNS-595"]
]

# Data for Random Molecules
random_molecules = [
    ["smiles", "name"],  # Lowercase to match format
    ["CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"],
    ["CC(=O)NC1=CC=C(C=C1)O", "Paracetamol"],
    ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen"],
    ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"],
    ["CCO", "Ethanol"],
    ["C(C1C(C(C(C(O1)O)O)O)O)O", "Glucose"],
    ["C1=CC=CC=C1", "Benzene"],
    ["CC1=CC=CC=C1", "Toluene"],
    ["CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C", "Cholesterol"],
    ["CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "Testosterone"],
    ["C1=CC(=C(C=C1CCN)O)O", "Dopamine"],
    ["C1=CC2=C(C=C1O)C(=CN2)CCN", "Serotonin"],
    ["CNC[C@H](C1=CC(=C(C=C1)O)O)O", "Adrenaline"],
    ["C(C(C1C(=C(C(=O)O1)O)O)O)O", "Ascorbic Acid"],
    ["CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=O)C)C", "Retinal"],
    ["CC1=C(C(=C(C2=C1OC(CC2)(C)C)CCC(=C)CCC(=C)CCC(=C)C)C)O", "Vitamin E"],
    ["CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C", "Penicillin G"],
    ["CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)O)C", "Ampicillin"],
    ["CN(C)C(=N)NC(=N)N", "Metformin"],
    ["CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", "Atorvastatin"],
    ["CC1=CN=C(C=C1OC)C(C)S(=O)C2=NC3=C(N2)C=C(C=C3)OC", "Omeprazole"],
    ["CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3N4N=NN=C4)Cl)CO", "Losartan"],
    ["COC(=O)C1=C(C(=C(C(=C1)Cl)C(=O)OC)C)N", "Nifedipine"],
    ["CC(C)NCC(COC1=CC=C(C=C1)CC(=O)N)O", "Atenolol"],
    ["CCC(C)(C)C(=O)O[C@H]1C[C@@H](C=C2[C@H]1[C@H]([C@H](C=C2)C)CCC3=CC(=O)OC3)C", "Simvastatin"],
    ["CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O", "Albuterol"],
    ["C1C(CC(CC1CC(=O)O)CC(=O)O)N", "Gabapentin"],
    ["C1=CC(=C(C=C1S(=O)(=O)N)Cl)S(=O)(=O)NC2=CN=CC=C2", "Chlorothiazide"],
    ["C1=CC=C(C=C1)CC2=C(C=CC(=C2)S(=O)(=O)N)N", "Furosemide"],
    ["C1=CC(=CC(=C1)CC(C(=O)O)N)OC2=CC(=C(C(=C2)I)O)I", "Levothyroxine"]
]

# Function to write data to CSV
def write_csv(filename, data):
    filepath = SCRIPT_DIR / filename
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"File '{filepath}' created successfully.")


if __name__ == "__main__":
    # Generate the files in cdk2_test_data folder
    write_csv('binders.csv', cdk2_binders)
    write_csv('random_molecules.csv', random_molecules)
    print("\nAll CSV files generated in cdk2_test_data folder.")

