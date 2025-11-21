#!/usr/bin/env python3
"""
Display contents of a PyTorch .pt file
"""
import torch
import sys
import argparse
import numpy as np

def display_pt_file(file_path):
    """Load and display contents of a .pt file"""
    try:
        data = torch.load(file_path, map_location='cpu')
        print(f"\n{'='*60}")
        print(f"File: {file_path}")
        print(f"{'='*60}\n")
        
        if isinstance(data, dict):
            print(f"Type: Dictionary with {len(data)} keys\n")
            for key, value in data.items():
                print(f"Key: '{key}'")
                if isinstance(value, torch.Tensor):
                    print(f"  Type: torch.Tensor")
                    print(f"  Shape: {tuple(value.shape)}")
                    print(f"  Dtype: {value.dtype}")
                    print(f"  Device: {value.device}")
                    if value.numel() <= 10:  # Show values if small
                        print(f"  Values:\n{value}")
                    elif value.numel() <= 100:  # Show summary if medium
                        print(f"  Min: {value.min().item():.6f}")
                        print(f"  Max: {value.max().item():.6f}")
                        print(f"  Mean: {value.mean().item():.6f}")
                        print(f"  Sample (first 5): {value.flatten()[:5]}")
                    else:  # Just stats if large
                        print(f"  Min: {value.min().item():.6f}")
                        print(f"  Max: {value.max().item():.6f}")
                        print(f"  Mean: {value.mean().item():.6f}")
                        print(f"  Std: {value.std().item():.6f}")
                elif isinstance(value, (str, int, float)):
                    print(f"  Type: {type(value).__name__}")
                    print(f"  Value: {value}")
                else:
                    print(f"  Type: {type(value)}")
                    print(f"  Value: {value}")
                print()
        elif isinstance(data, torch.Tensor):
            print(f"Type: torch.Tensor")
            print(f"Shape: {tuple(data.shape)}")
            print(f"Dtype: {data.dtype}")
            print(f"Device: {data.device}")
            if data.numel() <= 20:
                print(f"\nValues:\n{data}")
            else:
                print(f"\nMin: {data.min().item():.6f}")
                print(f"Max: {data.max().item():.6f}")
                print(f"Mean: {data.mean().item():.6f}")
                print(f"Std: {data.std().item():.6f}")
        else:
            print(f"Type: {type(data)}")
            print(f"Value: {data}")
            
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display contents of a PyTorch .pt file")
    parser.add_argument("file", type=str, help="Path to the .pt file")
    args = parser.parse_args()
    
    display_pt_file(args.file)