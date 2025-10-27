import argparse
import torch
from pathlib import Path

def inspect_pt_file(file_path):
    """
    Loads a .pt file and prints the keys, types, and shapes of its contents.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Inspecting File: {file_path.name} ---")
    
    try:
        # Load the data. Assumes it was saved to the CPU.
        # If you saved on GPU, you might need: torch.load(file_path, map_location='cpu')
        data = torch.load(file_path)
        
        if not isinstance(data, dict):
            print(f"Data is not a dictionary. Type: {type(data)}")
            if torch.is_tensor(data):
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
            return

        print(f"Found {len(data.keys())} keys:")
        
        for key, value in data.items():
            print(f"\n  Key: '{key}'")
            print(f"    ├─ Type: {type(value)}")
            
            # Check if it's a tensor and print its details
            if torch.is_tensor(value):
                print(f"    ├─ Shape: {value.shape}")
                print(f"    └─ Dtype: {value.dtype}")
            
            # If it's a string (like 'name'), just print it
            elif isinstance(value, str):
                print(f"    └─ Value: '{value}'")
            
            # Handle other types like lists or numbers
            elif isinstance(value, (list, int, float)):
                 # Truncate long lists
                val_str = str(value)
                if len(val_str) > 75:
                    val_str = val_str[:75] + "..."
                print(f"    └─ Value: {val_str}")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a .pt file from the processing script.")
    parser.add_argument("file", type=str, help="Path to the .pt file to inspect.")
    
    args = parser.parse_args()
    
    inspect_pt_file(Path(args.file))