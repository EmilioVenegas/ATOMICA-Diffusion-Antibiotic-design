import pickle
import numpy as np

def inspect_pickle_structure(file_path: str):
    """
    Loads a pickle file and prints the structure of the first element.
    """
    print(f"--- Analyzing file: {file_path} ---\n")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    if not isinstance(data, list):
        print(f"Data is not a list. It is of type: {type(data)}")
        return
        
    if not data:
        print("The list in the pickle file is empty. No structure to analyze.")
        return

    print(f"The file contains a list of {len(data)} items.")
    print("Analyzing the structure of the first item (index 0):\n")
    
    first_item = data[0]
    
    if not isinstance(first_item, dict):
        print(f"The items in the list are not dictionaries. The first item is of type: {type(first_item)}")
        return

    # Iterate through the dictionary to print details for each key
    for key, value in first_item.items():
        print(f"Key: '{key}'")
        value_type = type(value)
        print(f"  ├─ Type: {value_type}")
        
        if isinstance(value, np.ndarray):
            print(f"  ├─ Shape: {value.shape}")
            print(f"  └─ Dtype: {value.dtype}")
        elif isinstance(value, list):
            list_len = len(value)
            print(f"  ├─ Length: {list_len}")
            if list_len > 0:
                # Also print the type of the first element in the list
                element_type = type(value[0])
                print(f"  ├─ Contains elements of type: {element_type}")
                # Show a small sample of the list's content
                sample = value[:3] # Show up to the first 3 elements
                print(f"  └─ Sample: {sample} {'...' if list_len > 3 else ''}")
        else:
            # For other types like str, int, etc.
            print(f"  └─ Value: {value}")
        print("-" * 20)

if __name__ == '__main__':
    input_file = 'ATOMICA/data/example/example_outputs_embedded.pkl'
    inspect_pickle_structure(input_file)