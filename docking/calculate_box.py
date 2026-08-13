import sys
import numpy as np

def calculate_box(sdf_file, padding=10.0):
    coords = []
    with open(sdf_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # SDF atom lines usually have length > 30 and contain coords in first 3 columns
            # Standard SDF atom line: x(10.4) y(10.4) z(10.4) ...
            # Simple heuristic: split line, check if first 3 are floats
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    # Check if it looks like a coordinate line (often followed by atom symbol)
                    # In V2000, 4th col is symbol. In V3000 it's different. Assuming V2000 from openbabel.
                    # V2000 atom block: x y z symbol ...
                    # Let's just try to parse first 3 as floats.
                    coords.append([x, y, z])
                except ValueError:
                    continue

    if not coords:
        print("No coordinates found!")
        sys.exit(1)

    coords = np.array(coords)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    center = (min_coords + max_coords) / 2
    size = (max_coords - min_coords) + padding

    print(f"Center: {center}")
    print(f"Size: {size}")

    with open('docking/config.txt', 'w') as f:
        f.write(f"receptor = docking/receptor.pdbqt\n")
        f.write(f"center_x = {center[0]:.3f}\n")
        f.write(f"center_y = {center[1]:.3f}\n")
        f.write(f"center_z = {center[2]:.3f}\n")
        f.write(f"size_x = {size[0]:.3f}\n")
        f.write(f"size_y = {size[1]:.3f}\n")
        f.write(f"size_z = {size[2]:.3f}\n")
        f.write(f"exhaustiveness = 8\n")
        f.write(f"num_modes = 9\n")
        f.write(f"energy_range = 3\n")

if __name__ == "__main__":
    calculate_box('generated_ligands.sdf')
