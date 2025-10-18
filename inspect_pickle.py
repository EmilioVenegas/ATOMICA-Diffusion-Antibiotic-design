import pickle
import json
import sys
import numpy as np


DATA_FILE_PATH = 'ATOMICA/data/example/example_outputs.pkl'  
item_num = 1


def inspect_data(filepath, item_num):
    """
    Loads the data file and inspects the structure of the first item
    to help find the pocket/interface mask key.
    """
    print(f"--- Loading data from: {filepath} ---")

    try:
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print("Successfully loaded pickle file.\n")
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            print("Successfully loaded json file.\n")
        else:
            print(f"Error: Unknown file type for '{filepath}'. Only .pkl and .json are supported.")
            return
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # --- Inspect the data structure ---
    if not isinstance(data, list):
        print(f"Loaded data is not a list (it's a {type(data)}). Stopping.")
        return

    if len(data) == 0:
        print("Data list is empty. Cannot inspect structure.")
        return

    print(f"Data is a list with {len(data)} items.")
    print("--- Inspecting the first item (data[0]) ---")
   



    n_item = data[item_num]
    if not isinstance(n_item, dict):
        print(f"First item is not a dictionary (it's a {type(n_item)}). Stopping.")
        return

    print(f"First item's top-level keys: {list(n_item.keys())}")

    # --- Find the core data dictionary ---
    data_key = None
    if 'prot_data' in n_item:
        data_key = 'prot_data'
    elif 'data' in n_item:
        data_key = 'data'
    
    if data_key is None:
        print("\nError: Could not find 'data' or 'prot_data' key in the first item.")
        print("Please inspect the top-level keys above to find your main data dictionary.")
        return

    print(f"\n--- Looking inside 'n_item[{data_key}]' for the mask ---")
    
    core_data = n_item[data_key]
    if not isinstance(core_data, dict):
        print(f"'n_item[{data_key}]' is not a dictionary. Stopping.")
        return

    # Loop through all keys in the core data dict and print info
    for key, value in core_data.items():
        print(f"\nKey: '{key}'")
        value_type = type(value)
        print(f"  ├─ Type: {value_type}")
        
        if isinstance(value, list) or isinstance(value, np.ndarray):
            length = len(value)
            print(f"  ├─ Length: {length}")
            if length > 10:
                print(f"  └─ Sample: {value[:5]} ... {value[-5:]}")
            else:
                print(f"  └─ Sample: {value}")
        else:
            print(f"  └─ Value: {str(value)[:100]}...") # Print a snippet if it's not a list

# --- Run the inspection ---
if __name__ == "__main__":
           
    inspect_data(DATA_FILE_PATH, item_num)
