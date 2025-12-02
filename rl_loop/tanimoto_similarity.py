#!/usr/bin/env python3
"""
Tanimoto Similarity Analysis for Ligand Diversity

This script analyzes molecular diversity using Tanimoto similarity on fingerprints.
It clusters top-k molecules to answer: "Are the top candidates super similar?"

Usage:
    # Analyze existing scored CSV
    python tanimoto_similarity.py \
        --input results/generated_from_refactor_scored.csv \
        --top_k 10 \
        --threshold 0.7

    # Analyze SDF file directly
    python tanimoto_similarity.py \
        --input examples/generated_from_refactor.sdf \
        --top_k 10 \
        --threshold 0.7
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors

# Try to import clustering
try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
    print("  Clustering will use simple threshold-based method instead.")


class TanimotoSimilarityAnalyzer:
    """Analyze molecular diversity using Tanimoto similarity"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize analyzer
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES to RDKit molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None
    
    def compute_fingerprint(self, mol: Chem.Mol, fingerprint_type: str = 'RDKit'):
        """
        Compute molecular fingerprint
        
        Args:
            mol: RDKit molecule
            fingerprint_type: 'RDKit' or 'Morgan'
            
        Returns:
            Fingerprint object
        """
        if fingerprint_type == 'RDKit':
            return Chem.RDKFingerprint(mol)
        elif fingerprint_type == 'Morgan':
            from rdkit.Chem import AllChem
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        else:
            raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")
    
    def compute_similarity_matrix(self, molecules: List[Chem.Mol], 
                                   fingerprint_type: str = 'RDKit') -> np.ndarray:
        """
        Compute pairwise Tanimoto similarity matrix
        
        Args:
            molecules: List of RDKit molecule objects
            fingerprint_type: Type of fingerprint to use
            
        Returns:
            NxN similarity matrix (symmetric, diagonal = 1.0)
        """
        if self.verbose:
            print(f"Computing fingerprints ({fingerprint_type})...")
        
        n = len(molecules)
        fingerprints = []
        
        for i, mol in enumerate(molecules):
            if mol is None:
                if self.verbose:
                    print(f"  Warning: Molecule {i} is None, skipping")
                fingerprints.append(None)
            else:
                fp = self.compute_fingerprint(mol, fingerprint_type)
                fingerprints.append(fp)
        
        if self.verbose:
            print(f"Computing {n}x{n} similarity matrix...")
        
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            if fingerprints[i] is None:
                continue
            similarity_matrix[i, i] = 1.0  # Self-similarity
            
            for j in range(i + 1, n):
                if fingerprints[j] is None:
                    continue
                
                similarity = DataStructs.TanimotoSimilarity(
                    fingerprints[i], 
                    fingerprints[j]
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric
        
        if self.verbose:
            print(f"✓ Similarity matrix computed")
        
        return similarity_matrix
    
    def cluster_by_similarity(self, similarity_matrix: np.ndarray, 
                             threshold: float = 0.7,
                             method: str = 'agglomerative') -> np.ndarray:
        """
        Cluster molecules by similarity threshold
        
        Args:
            similarity_matrix: NxN similarity matrix
            threshold: Similarity threshold for clustering
            method: 'agglomerative' (requires sklearn) or 'threshold'
            
        Returns:
            Array of cluster labels (0-indexed)
        """
        n = similarity_matrix.shape[0]
        
        if method == 'agglomerative' and SKLEARN_AVAILABLE:
            if self.verbose:
                print(f"Clustering using agglomerative clustering (threshold={threshold})...")
            
            # Convert similarity to distance
            distance_matrix = 1.0 - similarity_matrix
            
            # Use AgglomerativeClustering with distance threshold
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1.0 - threshold,
                linkage='average',
                metric='precomputed'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            
        else:
            # Simple threshold-based clustering
            if self.verbose:
                print(f"Clustering using threshold method (threshold={threshold})...")
            
            cluster_labels = np.full(n, -1, dtype=int)
            current_cluster = 0
            
            for i in range(n):
                if cluster_labels[i] != -1:
                    continue
                
                # Start new cluster
                cluster_labels[i] = current_cluster
                
                # Find all molecules similar to this one
                for j in range(i + 1, n):
                    if cluster_labels[j] == -1 and similarity_matrix[i, j] >= threshold:
                        cluster_labels[j] = current_cluster
                
                current_cluster += 1
        
        if self.verbose:
            n_clusters = len(set(cluster_labels))
            print(f"✓ Found {n_clusters} clusters")
        
        return cluster_labels
    
    def calculate_diversity_metrics(self, similarity_matrix: np.ndarray, 
                                    cluster_labels: np.ndarray) -> Dict:
        """
        Calculate diversity metrics
        
        Args:
            similarity_matrix: NxN similarity matrix
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary of metrics
        """
        n = similarity_matrix.shape[0]
        
        # Average pairwise similarity (excluding diagonal)
        upper_triangle = np.triu(similarity_matrix, k=1)
        avg_similarity = np.mean(upper_triangle[upper_triangle > 0])
        
        # Number of clusters
        n_clusters = len(set(cluster_labels))
        
        # Cluster sizes
        cluster_sizes = {}
        for label in cluster_labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        cluster_size_list = sorted(cluster_sizes.values(), reverse=True)
        
        # Max similarity within clusters
        max_intra_cluster_sim = 0.0
        for cluster_id in set(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 1:
                cluster_sim = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
                max_sim = np.max(np.triu(cluster_sim, k=1))
                max_intra_cluster_sim = max(max_intra_cluster_sim, max_sim)
        
        # Min similarity between clusters
        min_inter_cluster_sim = 1.0
        unique_clusters = list(set(cluster_labels))
        for i, c1 in enumerate(unique_clusters):
            for c2 in unique_clusters[i+1:]:
                indices1 = np.where(cluster_labels == c1)[0]
                indices2 = np.where(cluster_labels == c2)[0]
                inter_sim = similarity_matrix[np.ix_(indices1, indices2)]
                if inter_sim.size > 0:
                    min_sim = np.min(inter_sim)
                    min_inter_cluster_sim = min(min_inter_cluster_sim, min_sim)
        
        metrics = {
            'n_molecules': n,
            'n_clusters': n_clusters,
            'avg_pairwise_similarity': avg_similarity,
            'max_intra_cluster_similarity': max_intra_cluster_sim,
            'min_inter_cluster_similarity': min_inter_cluster_sim,
            'cluster_sizes': cluster_size_list,
            'largest_cluster_size': cluster_size_list[0] if cluster_size_list else 0,
        }
        
        return metrics
    
    def load_from_csv(self, csv_path: Path, top_k: Optional[int] = None) -> Tuple[List[Chem.Mol], pd.DataFrame]:
        """
        Load molecules from scored CSV file
        
        Args:
            csv_path: Path to CSV file (must have 'smiles' column)
            top_k: If provided, only load top-k by rank
            
        Returns:
            Tuple of (molecules list, dataframe)
        """
        if self.verbose:
            print(f"Loading from CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Check for required columns
        if 'smiles' not in df.columns:
            raise ValueError("CSV must contain 'smiles' column")
        
        # Sort by rank if available, otherwise by composite_score
        if 'rank' in df.columns:
            df = df.sort_values('rank')
        elif 'composite_score' in df.columns:
            df = df.sort_values('composite_score', ascending=False)
        
        # Take top-k if specified
        if top_k is not None:
            df = df.head(top_k)
            if self.verbose:
                print(f"  Using top {top_k} molecules")
        
        # Convert SMILES to molecules
        molecules = []
        valid_indices = []
        
        for idx, smiles in enumerate(df['smiles']):
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                molecules.append(mol)
                valid_indices.append(idx)
            else:
                if self.verbose:
                    print(f"  Warning: Failed to parse SMILES: {smiles[:50]}...")
        
        # Filter dataframe to valid molecules
        df_valid = df.iloc[valid_indices].copy()
        
        if self.verbose:
            print(f"✓ Loaded {len(molecules)} valid molecules from CSV")
        
        return molecules, df_valid
    
    def load_from_sdf(self, sdf_path: Path, top_k: Optional[int] = None) -> Tuple[List[Chem.Mol], pd.DataFrame]:
        """
        Load molecules from SDF file
        
        Args:
            sdf_path: Path to SDF file
            top_k: If provided, only load first k molecules
            
        Returns:
            Tuple of (molecules list, dataframe with molecule info)
        """
        if self.verbose:
            print(f"Loading from SDF: {sdf_path}")
        
        molecules = []
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        
        for idx, mol in enumerate(supplier):
            if top_k is not None and idx >= top_k:
                break
            
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    molecules.append(mol)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Molecule {idx} failed sanitization: {e}")
        
        # Create dataframe with SMILES
        smiles_list = []
        for mol in molecules:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        
        df = pd.DataFrame({
            'molecule_id': [f"{sdf_path.stem}_{i}" for i in range(len(molecules))],
            'smiles': smiles_list
        })
        
        if self.verbose:
            print(f"✓ Loaded {len(molecules)} valid molecules from SDF")
        
        return molecules, df
    
    def analyze_top_k_diversity(self, input_path: Path, 
                               top_k: int = 10,
                               similarity_threshold: float = 0.7,
                               fingerprint_type: str = 'RDKit',
                               output_path: Optional[Path] = None) -> Dict:
        """
        Analyze diversity of top-k molecules
        
        Args:
            input_path: Path to CSV or SDF file
            top_k: Number of top molecules to analyze
            similarity_threshold: Threshold for clustering
            fingerprint_type: Type of fingerprint
            output_path: Optional path to save results CSV
            
        Returns:
            Dictionary with analysis results
        """
        # Load molecules
        if input_path.suffix.lower() == '.csv':
            molecules, df = self.load_from_csv(input_path, top_k=top_k)
        elif input_path.suffix.lower() == '.sdf':
            molecules, df = self.load_from_sdf(input_path, top_k=top_k)
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        
        if len(molecules) == 0:
            raise ValueError("No valid molecules found")
        
        if len(molecules) == 1:
            print("Warning: Only one molecule, cannot compute similarity")
            return {
                'n_molecules': 1,
                'n_clusters': 1,
                'avg_pairwise_similarity': 1.0,
            }
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(molecules, fingerprint_type)
        
        # Cluster molecules
        cluster_labels = self.cluster_by_similarity(
            similarity_matrix, 
            threshold=similarity_threshold
        )
        
        # Calculate metrics
        metrics = self.calculate_diversity_metrics(similarity_matrix, cluster_labels)
        
        # Add cluster info to dataframe
        df['cluster_id'] = cluster_labels
        df['cluster_size'] = df['cluster_id'].map(
            df['cluster_id'].value_counts()
        )
        
        # Calculate average similarity within each molecule's cluster
        avg_cluster_similarities = []
        for idx, cluster_id in enumerate(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 1:
                cluster_sim = similarity_matrix[idx, cluster_indices]
                avg_sim = np.mean(cluster_sim[cluster_sim < 1.0])  # Exclude self
                avg_cluster_similarities.append(avg_sim)
            else:
                avg_cluster_similarities.append(0.0)  # Singleton cluster
        
        df['avg_similarity_in_cluster'] = avg_cluster_similarities
        
        # Print report
        self._print_diversity_report(metrics, top_k, similarity_threshold)
        
        # Save results if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            if self.verbose:
                print(f"\n✓ Results saved to: {output_path}")
        
        return {
            'metrics': metrics,
            'dataframe': df,
            'similarity_matrix': similarity_matrix,
            'cluster_labels': cluster_labels
        }
    
    def _print_diversity_report(self, metrics: Dict, top_k: int, threshold: float):
        """Print formatted diversity report"""
        print("\n" + "="*60)
        print(f"Top-{top_k} Diversity Analysis")
        print("="*60)
        print(f"Number of molecules: {metrics['n_molecules']}")
        print(f"Number of clusters (threshold={threshold}): {metrics['n_clusters']}")
        print(f"Average pairwise similarity: {metrics['avg_pairwise_similarity']:.3f}")
        print(f"Max intra-cluster similarity: {metrics['max_intra_cluster_similarity']:.3f}")
        print(f"Min inter-cluster similarity: {metrics['min_inter_cluster_similarity']:.3f}")
        print(f"Cluster sizes: {metrics['cluster_sizes']}")
        print(f"Largest cluster size: {metrics['largest_cluster_size']}")
        
        # Diversity assessment
        print("\n" + "-"*60)
        if metrics['n_clusters'] >= top_k * 0.5:  # At least 50% of molecules in separate clusters
            print("✅ Top molecules are DIVERSE")
            print(f"   ({metrics['n_clusters']} clusters found, good diversity)")
        elif metrics['n_clusters'] >= 3:
            print("⚠️  Top molecules show MODERATE diversity")
            print(f"   ({metrics['n_clusters']} clusters found)")
        else:
            print("❌ Top molecules are TOO SIMILAR")
            print(f"   (Only {metrics['n_clusters']} cluster(s) found, low diversity)")
        print("-"*60 + "\n")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Analyze molecular diversity using Tanimoto similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze top-10 from scored CSV
  python tanimoto_similarity.py \\
      --input results/generated_from_refactor_scored.csv \\
      --top_k 10 \\
      --threshold 0.7

  # Analyze SDF file directly
  python tanimoto_similarity.py \\
      --input examples/generated_from_refactor.sdf \\
      --top_k 20 \\
      --threshold 0.6 \\
      --output results/diversity_analysis.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input CSV (scored) or SDF file'
    )
    
    parser.add_argument(
        '--top_k', '-k',
        type=int,
        default=10,
        help='Number of top molecules to analyze (default: 10)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.7,
        help='Similarity threshold for clustering (default: 0.7)'
    )
    
    parser.add_argument(
        '--fingerprint', '-f',
        type=str,
        choices=['RDKit', 'Morgan'],
        default='RDKit',
        help='Fingerprint type (default: RDKit)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output CSV path for cluster assignments (optional)'
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
    
    # Create analyzer
    analyzer = TanimotoSimilarityAnalyzer(verbose=not args.quiet)
    
    # Determine output path
    if args.output is None and args.input.suffix.lower() == '.csv':
        # Auto-generate output path
        output_path = args.input.parent / f"{args.input.stem}_diversity.csv"
    else:
        output_path = args.output
    
    # Run analysis
    try:
        results = analyzer.analyze_top_k_diversity(
            input_path=args.input,
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            fingerprint_type=args.fingerprint,
            output_path=output_path
        )
        
        print(f"\n✓ Analysis complete!")
        if output_path:
            print(f"  Results saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

