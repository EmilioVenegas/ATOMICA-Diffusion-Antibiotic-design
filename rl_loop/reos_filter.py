#!/usr/bin/env python3
"""
REOS Filter for Generated Ligands

Rapid Elimination Of Swill filter - filters out unuseful compounds from HTS screening results.
Based on Waters & Namchuk, Nature Reviews Drug Discovery 2, 259-266 (2003).

Default REOS criteria:
- Molecular weight: 200-500
- LogP: -5.0 to +5.0
- H-bond donors: 0-5
- H-bond acceptors: 0-10
- Formal charge: -2 to +2
- Rotatable bonds: 0-8
- Heavy atom count: 15-50
"""

import argparse
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


class REOSFilter:
    """REOS (Rapid Elimination Of Swill) filter"""
    
    def __init__(self,
                 mw_min: float = 200.0,
                 mw_max: float = 500.0,
                 logp_min: float = -5.0,
                 logp_max: float = 5.0,
                 hbd_min: int = 0,
                 hbd_max: int = 5,
                 hba_min: int = 0,
                 hba_max: int = 10,
                 charge_min: int = -2,
                 charge_max: int = 2,
                 rot_bonds_min: int = 0,
                 rot_bonds_max: int = 8,
                 heavy_atoms_min: int = 15,
                 heavy_atoms_max: int = 50,
                 max_violations: int = 0):
        """
        Initialize REOS filter
        
        Args:
            mw_min, mw_max: Molecular weight range
            logp_min, logp_max: LogP range
            hbd_min, hbd_max: H-bond donor count range
            hba_min, hba_max: H-bond acceptor count range
            charge_min, charge_max: Formal charge range
            rot_bonds_min, rot_bonds_max: Rotatable bond count range
            heavy_atoms_min, heavy_atoms_max: Heavy atom count range
            max_violations: Maximum number of filter violations allowed (0 = strict)
        """
        self.mw_min = mw_min
        self.mw_max = mw_max
        self.logp_min = logp_min
        self.logp_max = logp_max
        self.hbd_min = hbd_min
        self.hbd_max = hbd_max
        self.hba_min = hba_min
        self.hba_max = hba_max
        self.charge_min = charge_min
        self.charge_max = charge_max
        self.rot_bonds_min = rot_bonds_min
        self.rot_bonds_max = rot_bonds_max
        self.heavy_atoms_min = heavy_atoms_min
        self.heavy_atoms_max = heavy_atoms_max
        self.max_violations = max_violations
    
    def check_molecule(self, mol: Chem.Mol) -> tuple[bool, int, List[str]]:
        """
        Check if molecule passes REOS filter
        
        Returns:
            (passes, num_violations, violation_list)
        """
        if mol is None:
            return False, 999, ["Invalid molecule"]
        
        violations = []
        
        # Molecular weight
        mw = Descriptors.MolWt(mol)
        if not (self.mw_min <= mw <= self.mw_max):
            violations.append(f"MW={mw:.1f} (range: {self.mw_min}-{self.mw_max})")
        
        # LogP
        logp = Crippen.MolLogP(mol)
        if not (self.logp_min <= logp <= self.logp_max):
            violations.append(f"LogP={logp:.2f} (range: {self.logp_min}-{self.logp_max})")
        
        # H-bond donors
        hbd = Lipinski.NumHDonors(mol)
        if not (self.hbd_min <= hbd <= self.hbd_max):
            violations.append(f"HBD={hbd} (range: {self.hbd_min}-{self.hbd_max})")
        
        # H-bond acceptors
        hba = Lipinski.NumHAcceptors(mol)
        if not (self.hba_min <= hba <= self.hba_max):
            violations.append(f"HBA={hba} (range: {self.hba_min}-{self.hba_max})")
        
        # Formal charge
        charge = Chem.rdmolops.GetFormalCharge(mol)
        if not (self.charge_min <= charge <= self.charge_max):
            violations.append(f"Charge={charge} (range: {self.charge_min}-{self.charge_max})")
        
        # Rotatable bonds
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        if not (self.rot_bonds_min <= rot_bonds <= self.rot_bonds_max):
            violations.append(f"RotBonds={rot_bonds} (range: {self.rot_bonds_min}-{self.rot_bonds_max})")
        
        # Heavy atom count
        heavy_atoms = mol.GetNumHeavyAtoms()
        if not (self.heavy_atoms_min <= heavy_atoms <= self.heavy_atoms_max):
            violations.append(f"HeavyAtoms={heavy_atoms} (range: {self.heavy_atoms_min}-{self.heavy_atoms_max})")
        
        num_violations = len(violations)
        passes = num_violations <= self.max_violations
        
        return passes, num_violations, violations
    
    def filter_molecules(self, molecules: List[Chem.Mol], verbose: bool = True) -> tuple[List[Chem.Mol], List[dict]]:
        """
        Filter list of molecules using REOS criteria
        
        Returns:
            (filtered_molecules, filter_stats)
        """
        filtered = []
        stats = {
            'total': len(molecules),
            'passed': 0,
            'failed': 0,
            'violations': {}
        }
        
        for i, mol in enumerate(molecules):
            if mol is None:
                stats['failed'] += 1
                continue
            
            passes, num_viol, violations = self.check_molecule(mol)
            
            if passes:
                filtered.append(mol)
                stats['passed'] += 1
            else:
                stats['failed'] += 1
                if verbose and num_viol > 0:
                    print(f"  Molecule {i+1} failed: {', '.join(violations)}")
            
            # Track violation types
            for v in violations:
                v_type = v.split('=')[0]
                stats['violations'][v_type] = stats['violations'].get(v_type, 0) + 1
        
        return filtered, stats


def filter_sdf_file(input_sdf: Path, output_sdf: Path, 
                    max_violations: int = 0,
                    min_atoms: int = 5,
                    verbose: bool = True,
                    **filter_kwargs):
    """
    Filter SDF file using REOS criteria
    
    Args:
        input_sdf: Input SDF file path
        output_sdf: Output SDF file path
        max_violations: Maximum violations allowed (0 = strict)
        min_atoms: Minimum number of atoms required (default: 5)
        verbose: Print progress
        **filter_kwargs: REOS filter parameters
    """
    if verbose:
        print(f"Loading molecules from {input_sdf}...")
    
    # Load molecules
    supplier = Chem.SDMolSupplier(str(input_sdf), sanitize=False)
    molecules = []
    small_molecules_removed = 0
    for mol in supplier:
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                # Filter out molecules with < min_atoms
                if mol.GetNumAtoms() < min_atoms:
                    small_molecules_removed += 1
                    if verbose:
                        print(f"  Removed molecule with {mol.GetNumAtoms()} atoms (< {min_atoms})")
                    continue
                molecules.append(mol)
            except:
                if verbose:
                    print(f"  Warning: Skipped invalid molecule")
    
    if verbose:
        print(f"✓ Loaded {len(molecules)} valid molecules")
        if small_molecules_removed > 0:
            print(f"  Removed {small_molecules_removed} molecules with < {min_atoms} atoms")
    
    # Create filter
    reos = REOSFilter(max_violations=max_violations, **filter_kwargs)
    
    # Filter molecules
    if verbose:
        print(f"\nApplying REOS filter (max violations: {max_violations})...")
    
    filtered, stats = reos.filter_molecules(molecules, verbose=verbose)
    
    # Save filtered molecules
    if verbose:
        print(f"\n{'='*60}")
        print(f"REOS Filter Results")
        print(f"{'='*60}")
        print(f"Total molecules (after size filter): {stats['total']}")
        print(f"Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
        print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
        if stats['violations']:
            print(f"\nViolation breakdown:")
            for v_type, count in sorted(stats['violations'].items(), key=lambda x: -x[1]):
                print(f"  {v_type}: {count}")
        print(f"{'='*60}\n")
    
    # Write filtered SDF
    writer = Chem.SDWriter(str(output_sdf))
    for mol in filtered:
        writer.write(mol)
    writer.close()
    
    if verbose:
        print(f"✓ Saved {len(filtered)} filtered molecules to {output_sdf}")
    
    return filtered, stats


def main():
    parser = argparse.ArgumentParser(
        description="REOS filter for generated ligands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default REOS criteria (strict)
  python reos_filter.py input.sdf output_filtered.sdf

  # Allow 1 violation
  python reos_filter.py input.sdf output_filtered.sdf --max_violations 1

  # Custom criteria
  python reos_filter.py input.sdf output_filtered.sdf \\
    --mw_min 250 --mw_max 450 --logp_max 4.0
        """
    )
    
    parser.add_argument('input_sdf', type=Path, help='Input SDF file')
    parser.add_argument('output_sdf', type=Path, help='Output filtered SDF file')
    
    # Filter parameters
    parser.add_argument('--mw_min', type=float, default=200.0, help='Min molecular weight')
    parser.add_argument('--mw_max', type=float, default=500.0, help='Max molecular weight')
    parser.add_argument('--logp_min', type=float, default=-5.0, help='Min LogP')
    parser.add_argument('--logp_max', type=float, default=5.0, help='Max LogP')
    parser.add_argument('--hbd_min', type=int, default=0, help='Min H-bond donors')
    parser.add_argument('--hbd_max', type=int, default=5, help='Max H-bond donors')
    parser.add_argument('--hba_min', type=int, default=0, help='Min H-bond acceptors')
    parser.add_argument('--hba_max', type=int, default=10, help='Max H-bond acceptors')
    parser.add_argument('--charge_min', type=int, default=-2, help='Min formal charge')
    parser.add_argument('--charge_max', type=int, default=2, help='Max formal charge')
    parser.add_argument('--rot_bonds_min', type=int, default=0, help='Min rotatable bonds')
    parser.add_argument('--rot_bonds_max', type=int, default=8, help='Max rotatable bonds')
    parser.add_argument('--heavy_atoms_min', type=int, default=15, help='Min heavy atoms')
    parser.add_argument('--heavy_atoms_max', type=int, default=50, help='Max heavy atoms')
    parser.add_argument('--max_violations', type=int, default=0, 
                       help='Max violations allowed (0 = strict)')
    parser.add_argument('--min_atoms', type=int, default=5,
                       help='Minimum number of atoms required (default: 5)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    filter_kwargs = {
        'mw_min': args.mw_min,
        'mw_max': args.mw_max,
        'logp_min': args.logp_min,
        'logp_max': args.logp_max,
        'hbd_min': args.hbd_min,
        'hbd_max': args.hbd_max,
        'hba_min': args.hba_min,
        'hba_max': args.hba_max,
        'charge_min': args.charge_min,
        'charge_max': args.charge_max,
        'rot_bonds_min': args.rot_bonds_min,
        'rot_bonds_max': args.rot_bonds_max,
        'heavy_atoms_min': args.heavy_atoms_min,
        'heavy_atoms_max': args.heavy_atoms_max,
    }
    
    filter_sdf_file(
        args.input_sdf,
        args.output_sdf,
        max_violations=args.max_violations,
        min_atoms=args.min_atoms,
        verbose=not args.quiet,
        **filter_kwargs
    )