#!/usr/bin/env python3
"""Generate a CSV of drug-like decoys for the CDK2 benchmark."""

import csv
from pathlib import Path


# Data for Drug-Like Decoys (Non-Kinase Inhibitors)
drug_like_decoys = [
    ["smiles", "id"],
    ["CC(C)C1=NC(=NC(=N1)NCC2=CC=CC=C2)N[C@@H](CO)CC3=CC=CC=C3", "Atorvastatin"],
    ["CCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3N4N=NN=C4)Cl)CO", "Losartan"],
    ["CC1=CN=C(C=C1OC)C(C)S(=O)C2=NC3=C(N2)C=C(C=C3)OC", "Omeprazole"],
    ["COC(=O)C1=C(C(=C(C(=C1)Cl)C(=O)OC)C)N", "Amlodipine"],
    ["CCC(C)(C)C(=O)O[C@H]1C[C@@H](C=C2[C@H]1[C@H]([C@H](C=C2)C)CCC3=CC(=O)OC3)C", "Simvastatin"],
    ["CNCCC(C1=CC=C(C=C1)OC(F)(F)F)OC2=CC=CC=C2", "Fluoxetine"],
    ["CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", "Celecoxib"],
    ["CC(C)NCC(COC1=CC=C(C=C1)CCOC)O", "Metoprolol"],
    ["CCOC(=O)N1CCC(=C1)C2=C(C=CC(=C2)Cl)C3=CC=CC=N3", "Loratadine"],
    ["CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C", "Sildenafil"],
    ["COC(=O)C1=C(C=CC=C1)NC2=C(C(=C(C=C2)Cl)C(=O)O)C", "Diclofenac"],
    ["CC(C)(C)C(=O)OCOP(=O)(COC1=CC=C(C=C1)C2=CN=C(C=N2)N)COC(=O)C(C)(C)C", "Tenofovir Disoproxil"],
    ["C1CN(CCC1(C2=CC=C(C=C2)Cl)O)CCCC(=O)C3=CC=C(C=C3)F", "Haloperidol"],
    ["CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=O)C)C", "Isotretinoin"],
    ["CN1C2=C(C=CC(=C2)Cl)C3=C(C1=O)C=C(C=N3)O", "Oxamniquine"],
    ["CN(C)CCC=C1C2=CC=CC=C2CCC3=CC=CC=C31", "Amitriptyline"],
    ["C1=CC(=CC=C1C(=O)CC2=C(C=CC(=C2)Cl)O)Cl", "Bupropion"],
    ["CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O", "Salbutamol"],
    ["C1=C(N(C(=O)N1)C2=C(C=C(C=C2)Cl)F)C3=CC(=C(C=C3)N4CCN(CC4)C5=CC=C(C=C5)OC(C)COC6=CC=C(C=C6)N)", "Itraconazole"],
    ["CN1CCN(CC1)C2=C3C=CC=CC3=NC4=C2C=C(C=C4)C", "Olanzapine"],
    ["C1CC(CCC1(C(=O)O)CC(=O)O)CN", "Gabapentin"],
    ["C1=CC=C(C=C1)C(C2=CC=CC=C2)C3=CN=CN3", "Clotrimazole"],
    ["CC(=O)NC1=CC=C(C=C1)O", "Acetaminophen"],
    ["COC1=C(C=C(C=C1)C(C2=CC=CC=C2)O)OC", "Papaverine"],
    ["CC1=CC=C(C=C1)S(=O)(=O)NC(=O)C2=CC=C(C=C2)Cl", "Glibenclamide"],
    ["CCN(CC)CCCC(C)NC1=C2C=C(C=CC2=NC=C1)Cl", "Chloroquine"],
    ["CC1=CN=C(C(=N1)C2=CC=CC=C2)N", "Rimantadine"],
    ["CC12CC3CC(C1)(CC(C3)(C2)O)C(=O)N", "Amantadine"],
    ["C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3)OH", "Trityl Alcohol"],
    ["C1=CC=C(C=C1)C(=O)OC2=CC=CC=C2C(=O)O", "Aspirin"],
]


def write_csv(filename: str, data: list[list[str]]) -> None:
    out_path = Path(__file__).parent / filename
    with out_path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"File '{out_path}' created successfully.")


if __name__ == "__main__":
    write_csv("decoys.csv", drug_like_decoys)

#!/usr/bin/env python3
import csv
from pathlib import Path

# Data for Drug-Like Decoys (Non-Kinase Inhibitors)
drug_like_decoys = [
    ["smiles", "id"],  # match existing header style
    ["CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", "Atorvastatin"],
    ["CCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3N4N=NN=C4)Cl)CO", "Losartan"],
    ["CC1=CN=C(C=C1OC)C(C)S(=O)C2=NC3=C(N2)C=C(C=C3)OC", "Omeprazole"],
    ["COC(=O)C1=C(C(=C(C(=C1)Cl)C(=O)OC)C)N", "Amlodipine"],
    ["CCC(C)(C)C(=O)O[C@H]1C[C@@H](C=C2[C@H]1[C@H]([C@H](C=C2)C)CCC3=CC(=O)OC3)C", "Simvastatin"],
    ["CNCCC(C1=CC=C(C=C1)OC(F)(F)F)OC2=CC=CC=C2", "Fluoxetine"],
    ["CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", "Celecoxib"],
    ["CC(C)NCC(COC1=CC=C(C=C1)CCOC)O", "Metoprolol"],
    ["CCOC(=O)N1CCC(=C1)C2=C(C=CC(=C2)Cl)C3=CC=CC=N3", "Loratadine"],
    ["CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C", "Sildenafil"],
    ["COC(=O)C1=C(C=CC=C1)NC2=C(C(=C(C=C2)Cl)C(=O)O)C", "Diclofenac"],
    ["CC(C)(C)C(=O)OCOP(=O)(COC1=CC=C(C=C1)C2=CN=C(C=N2)N)COC(=O)C(C)(C)C", "Tenofovir Disoproxil"],
    ["C1CN(CCC1(C2=CC=C(C=C2)Cl)O)CCCC(=O)C3=CC=C(C=C3)F", "Haloperidol"],
    ["CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=O)C)C", "Isotretinoin"],
    ["CN1C2=C(C=CC(=C2)Cl)C3=C(C1=O)C=C(C=N3)O", "Oxamniquine"],
    ["CN(C)CCC=C1C2=CC=CC=C2CCC3=CC=CC=C31", "Amitriptyline"],
    ["C1=CC(=CC=C1C(=O)CC2=C(C=CC(=C2)Cl)O)Cl", "Bupropion"],
    ["CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O", "Salbutamol"],
    ["C1=C(N(C(=O)N1)C2=C(C=C(C=C2)Cl)F)C3=CC(=C(C=C3)N4CCN(CC4)C5=CC=C(C=C5)OC(C)COC6=CC=C(C=C6)N)", "Itraconazole"],
    ["CN1CCN(CC1)C2=C3C=CC=CC3=NC4=C2C=C(C=C4)C", "Olanzapine"],
    ["C1CC(CCC1(C(=O)O)CC(=O)O)CN", "Gabapentin"],
    ["C1=CC=C(C=C1)C(C2=CC=C(C=C2)Cl)C3=CN=CN3", "Clotrimazole"],
    ["CC(=O)NC1=CC=C(C=C1)O", "Acetaminophen"],
    ["COC1=C(C=C(C=C1)C(C2=CC=CC=C2)O)OC", "Papaverine"],
    ["CC1=CC=C(C=C1)S(=O)(=O)NC(=O)C2=CC=C(C=C2)Cl", "Glibenclamide"],
    ["CCN(CC)CCCC(C)NC1=C2C=C(C=CC2=NC=C1)Cl", "Chloroquine"],
    ["CC1=CN=C(C(=N1)C2=CC=CC=C2)N", "Rimantadine"],
    ["CC12CC3CC(C1)(CC(C3)(C2)O)C(=O)N", "Amantadine"],
    ["C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3)OH", "Trityl Alcohol"],
    ["C1=CC=C(C=C1)C(=O)OC2=CC=CC=C2C(=O)O", "Aspirin"],
]

def write_csv(filename: str, data: list[list[str]]) -> None:
    out_path = Path(__file__).parent / filename
    with out_path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"File '{out_path}' created successfully.")

if __name__ == "__main__":
    write_csv("decoys.csv", drug_like_decoys)