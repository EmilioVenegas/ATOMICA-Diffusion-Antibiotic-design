#!/usr/bin/env python3
"""
DrugBank Reference Comparison for Generated Ligands

This script compares generated ligands (from scored CSV files) against the DrugBank
reference set (2,579 approved drugs) using ADMET-AI's built-in plotting functionality.
This allows visualization of how generated molecules compare to known approved drugs
in ADMET property space.

Usage:
    # Compare against all DrugBank drugs
    python drugbank_reference.py \
        --input results/generated_from_refactor_scored.csv \
        --x_property "Human Intestinal Absorption" \
        --y_property "Clinical Toxicity" \
        --output plots/hia_vs_clintox_all.svg

    # Compare against antibiotics only (ATC J01)
    python drugbank_reference.py \
        --input results/generated_from_refactor_scored.csv \
        --atc_filter J01 \
        --x_property "Human Intestinal Absorption" \
        --y_property "Clinical Toxicity" \
        --output plots/hia_vs_clintox_antibiotics.svg
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# Try to import ADMET-AI components
try:
    from admet_ai.drugbank import get_drugbank
    from admet_ai.plot import plot_drugbank_reference
    ADMET_AVAILABLE = True
except ImportError:
    ADMET_AVAILABLE = False
    print("Error: ADMET-AI not available. Install with: pip install admet-ai")
    print("  This script requires admet-ai >= 0.1.0 with drugbank and plot modules.")


class DrugBankReferenceComparator:
    """Compare generated ligands against DrugBank reference set"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize comparator
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.drugbank_df = None
        
        if not ADMET_AVAILABLE:
            raise ImportError("ADMET-AI is required but not available")
    
    def load_drugbank(self) -> pd.DataFrame:
        """
        Load the full DrugBank reference set (2,579 approved drugs)
        
        Returns:
            DataFrame with DrugBank drugs and their ADMET predictions
        """
        if self.drugbank_df is None:
            if self.verbose:
                print("Loading DrugBank reference set...")
            
            self.drugbank_df = get_drugbank()
            
            if self.verbose:
                print(f"✓ Loaded {len(self.drugbank_df)} DrugBank approved drugs")
                print(f"  Columns: {list(self.drugbank_df.columns)[:10]}...")
        
        return self.drugbank_df
    
    def filter_by_atc(self, drugbank_df: pd.DataFrame, atc_prefix: str) -> pd.DataFrame:
        """
        Filter DrugBank DataFrame by ATC code prefix
        
        Args:
            drugbank_df: Full DrugBank DataFrame
            atc_prefix: ATC code prefix (e.g., "J01" for antibiotics)
            
        Returns:
            Filtered DataFrame containing only drugs with matching ATC codes
        """
        # Check available ATC column names (may vary by ADMET-AI version)
        atc_columns = [col for col in drugbank_df.columns if 'atc' in col.lower()]
        
        if not atc_columns:
            print(f"Warning: No ATC column found. Available columns: {list(drugbank_df.columns)[:20]}")
            print("  Returning full DrugBank set (no filtering applied)")
            return drugbank_df
        
        # Use the first ATC column found
        atc_col = atc_columns[0]
        
        if self.verbose:
            print(f"Filtering by ATC code prefix: {atc_prefix}")
            print(f"  Using column: {atc_col}")
        
        # Filter by ATC prefix
        mask = drugbank_df[atc_col].str.startswith(atc_prefix, na=False)
        filtered_df = drugbank_df[mask]
        
        if self.verbose:
            print(f"✓ Found {len(filtered_df)} drugs with ATC code starting with {atc_prefix}")
        
        return filtered_df
    
    def load_scored_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Load scored CSV file from RL loop
        
        Args:
            csv_path: Path to scored CSV file
            
        Returns:
            DataFrame with generated ligands and ADMET predictions
        """
        if self.verbose:
            print(f"Loading scored CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Ensure SMILES column exists
        if 'smiles' not in df.columns:
            raise ValueError("CSV must contain 'smiles' column")
        
        if self.verbose:
            print(f"✓ Loaded {len(df)} generated ligands")
        
        return df
    
    def map_property_name(self, property_name: str, available_columns: list) -> str:
        """
        Map property name to column name in DataFrame
        
        Maps common ADMET-AI property names to CSV column names:
        - "Human Intestinal Absorption" -> "HIA_Hou"
        - "Clinical Toxicity" -> "ClinTox"
        - "Bioavailability" -> "Bioavailability_Ma"
        - "hERG Block" -> "hERG"
        - etc.
        
        Args:
            property_name: Human-readable property name or CSV column name
            available_columns: List of available column names in DataFrame
            
        Returns:
            Column name that exists in DataFrame
        """
        # Property name mapping (human-readable -> CSV column)
        property_mapping = {
            "Human Intestinal Absorption": "HIA_Hou",
            "Clinical Toxicity": "ClinTox",
            "Bioavailability": "Bioavailability_Ma",
            "hERG Block": "hERG",
            "hERG": "hERG",
            "BBB Penetration": "BBB_Martins",
            "CYP3A4 Inhibition": "CYP3A4_Veith",
            "DILI": "DILI",
            "Solubility": "Solubility_AqSolDB",
            "AMES": "AMES",
            "CYP1A2": "CYP1A2_Veith",
            "CYP2C19": "CYP2C19_Veith",
            "CYP2C9": "CYP2C9_Veith",
            "CYP2D6": "CYP2D6_Veith",
            "CYP3A4": "CYP3A4_Veith",
        }
        
        # Check if exact match exists
        if property_name in available_columns:
            return property_name
        
        # Check if mapped name exists
        mapped_name = property_mapping.get(property_name)
        if mapped_name and mapped_name in available_columns:
            if self.verbose:
                print(f"  Mapped '{property_name}' -> '{mapped_name}'")
            return mapped_name
        
        # Try case-insensitive partial match
        property_lower = property_name.lower()
        for col in available_columns:
            if property_lower in col.lower() or col.lower() in property_lower:
                if self.verbose:
                    print(f"  Matched '{property_name}' -> '{col}' (partial match)")
                return col
        
        # If no match found, return original (will fail in ADMET-AI with clear error)
        return property_name
    
    def create_reference_plot(self,
                             exp_df: pd.DataFrame,
                             drugbank_df: pd.DataFrame,
                             x_property: str,
                             y_property: str,
                             output_path: Path) -> bytes:
        """
        Create DrugBank reference plot using ADMET-AI's plotting function
        
        Args:
            exp_df: DataFrame with experimental/generated ligands
            drugbank_df: DrugBank reference DataFrame (filtered or full)
            x_property: Name of ADMET property for x-axis (human-readable or column name)
            y_property: Name of ADMET property for y-axis (human-readable or column name)
            output_path: Path to save plot (SVG format)
            
        Returns:
            Plot bytes (SVG format)
        """
        # Map property names to actual column names
        x_col = self.map_property_name(x_property, list(exp_df.columns))
        y_col = self.map_property_name(y_property, list(exp_df.columns))
        
        if self.verbose:
            print(f"\nCreating reference plot:")
            print(f"  X-axis: {x_property} ({x_col})")
            print(f"  Y-axis: {y_property} ({y_col})")
            print(f"  DrugBank reference: {len(drugbank_df)} drugs")
            print(f"  Generated ligands: {len(exp_df)} molecules")
        
        # Generate plot using ADMET-AI's built-in function
        # Note: ADMET-AI's plot_drugbank_reference may expect human-readable names
        # or column names depending on version - try both approaches
        try:
            plot_bytes = plot_drugbank_reference(
                preds_df=exp_df,
                drugbank_df=drugbank_df,
                x_property_name=x_property,  # Try human-readable name first
                y_property_name=y_property,
            )
        except (KeyError, ValueError) as e:
            # If that fails, try with column names
            if self.verbose:
                print(f"  Retrying with column names: {x_col}, {y_col}")
            plot_bytes = plot_drugbank_reference(
                preds_df=exp_df,
                drugbank_df=drugbank_df,
                x_property_name=x_col,
                y_property_name=y_col,
            )
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(plot_bytes)
        
        if self.verbose:
            print(f"✓ Plot saved to: {output_path}")
        
        return plot_bytes
    
    def compare_ligands(self,
                       csv_path: Path,
                       x_property: str,
                       y_property: str,
                       output_path: Path,
                       atc_filter: Optional[str] = None) -> None:
        """
        Main comparison workflow
        
        Args:
            csv_path: Path to scored CSV file
            x_property: ADMET property for x-axis
            y_property: ADMET property for y-axis
            output_path: Output plot path
            atc_filter: Optional ATC code prefix to filter DrugBank (e.g., "J01")
        """
        # 1. Load DrugBank reference
        drugbank_df = self.load_drugbank()
        
        # 2. Filter by ATC if specified
        if atc_filter:
            drugbank_df = self.filter_by_atc(drugbank_df, atc_filter)
        
        # 3. Load experimental predictions
        exp_df = self.load_scored_csv(csv_path)
        
        # 4. Create plot
        self.create_reference_plot(
            exp_df=exp_df,
            drugbank_df=drugbank_df,
            x_property=x_property,
            y_property=y_property,
            output_path=output_path
        )


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Compare generated ligands against DrugBank reference set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare against all DrugBank drugs
  python drugbank_reference.py \\
      --input results/generated_from_refactor_scored.csv \\
      --x_property "Human Intestinal Absorption" \\
      --y_property "Clinical Toxicity" \\
      --output plots/hia_vs_clintox_all.svg

  # Compare against antibiotics only (ATC J01)
  python drugbank_reference.py \\
      --input results/generated_from_refactor_scored.csv \\
      --atc_filter J01 \\
      --x_property "Human Intestinal Absorption" \\
      --y_property "Clinical Toxicity" \\
      --output plots/hia_vs_clintox_antibiotics.svg

  # Compare bioavailability vs hERG block
  python drugbank_reference.py \\
      --input results/generated_from_refactor_scored.csv \\
      --x_property "Bioavailability" \\
      --y_property "hERG Block" \\
      --output plots/bioavailability_vs_herg.svg

Common ADMET property names:
  - "Human Intestinal Absorption" (HIA_Hou)
  - "Clinical Toxicity" (ClinTox)
  - "Bioavailability" (Bioavailability_Ma)
  - "hERG Block" (hERG)
  - "BBB Penetration" (BBB_Martins)
  - "CYP3A4 Inhibition" (CYP3A4_Veith)
  - "DILI" (DILI)
  - "Solubility" (Solubility_AqSolDB)
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input scored CSV file (from RL_loop.py)'
    )
    
    parser.add_argument(
        '--x_property',
        type=str,
        required=True,
        help='ADMET property name for x-axis (e.g., "Human Intestinal Absorption")'
    )
    
    parser.add_argument(
        '--y_property',
        type=str,
        required=True,
        help='ADMET property name for y-axis (e.g., "Clinical Toxicity")'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output plot path (SVG format recommended)'
    )
    
    parser.add_argument(
        '--atc_filter',
        type=str,
        default=None,
        help='Optional ATC code prefix to filter DrugBank (e.g., "J01" for antibiotics)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Check ADMET-AI availability
    if not ADMET_AVAILABLE:
        print("Error: ADMET-AI is required but not available.")
        print("  Install with: pip install admet-ai")
        sys.exit(1)
    
    # Create comparator
    comparator = DrugBankReferenceComparator(verbose=not args.quiet)
    
    # Run comparison
    try:
        comparator.compare_ligands(
            csv_path=args.input,
            x_property=args.x_property,
            y_property=args.y_property,
            output_path=args.output,
            atc_filter=args.atc_filter
        )
        
        print(f"\n✓ Comparison complete!")
        print(f"  Plot saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

