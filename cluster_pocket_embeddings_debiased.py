#!/usr/bin/env python3
"""
Cluster pocket ATOMICA embeddings with debiasing (BERT anisotropy fix).
Removes global directions before averaging to improve discriminative power.
"""
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import time

def sample_atom_embeddings(directory, max_samples=100000, random_seed=42):
    """
    Sample atom embeddings from across the dataset to compute global statistics.
    
    Args:
        directory: Path to directory containing complex_*.pt files
        max_samples: Maximum number of atom embeddings to sample
        random_seed: Random seed for reproducibility
    
    Returns:
        sampled_embeddings: numpy array of shape (n_samples, embedding_dim)
    """
    start_time = time.time()
    directory = Path(directory)
    pt_files = sorted(directory.glob("complex_*.pt"))
    
    if not pt_files:
        raise ValueError(f"No complex_*.pt files found in {directory}")
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    all_embeddings = []
    files_sampled = 0
    
    print(f"Sampling atom embeddings from {len(pt_files)} files...")
    print(f"Estimated time: ~{len(pt_files) * 0.01:.1f} seconds (assuming ~0.01s per file)")
    for pt_file in tqdm(pt_files, desc="Sampling"):
        try:
            data = torch.load(pt_file, map_location='cpu')
            
            if 'pocket_atomica_embeddings' not in data:
                continue
            
            embeddings = data['pocket_atomica_embeddings']
            
            # Convert to numpy if tensor
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.numpy()
            
            # embeddings shape: (num_atoms, embedding_dim)
            if embeddings.shape[0] > 0:
                all_embeddings.append(embeddings)
                files_sampled += 1
                
                # Stop if we have enough samples
                total_atoms = sum(e.shape[0] for e in all_embeddings)
                if total_atoms >= max_samples:
                    break
                    
        except Exception as e:
            continue
    
    if not all_embeddings:
        raise ValueError("No atom embeddings found in any files")
    
    # Concatenate all sampled embeddings
    sampled_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # If we have more than max_samples, randomly sample
    if sampled_embeddings.shape[0] > max_samples:
        indices = np.random.choice(sampled_embeddings.shape[0], max_samples, replace=False)
        sampled_embeddings = sampled_embeddings[indices]
    
    elapsed = time.time() - start_time
    print(f"Sampled {sampled_embeddings.shape[0]} atom embeddings from {files_sampled} files")
    print(f"Sampling took {elapsed:.2f} seconds")
    return sampled_embeddings

def compute_global_stats(sampled_embeddings, n_components=3):
    """
    Compute global mean and principal components from sampled embeddings.
    
    Args:
        sampled_embeddings: numpy array of shape (n_samples, embedding_dim)
        n_components: Number of principal components to extract
    
    Returns:
        mu_global: Global mean vector (embedding_dim,)
        pca_components: Principal components (n_components, embedding_dim)
        pca: Fitted PCA object
    """
    start_time = time.time()
    print(f"\nComputing global statistics from {sampled_embeddings.shape[0]} atom embeddings...")
    
    # Step A: Compute global mean
    print("Computing global mean...")
    mu_global = np.mean(sampled_embeddings, axis=0)
    print(f"Global mean computed: shape {mu_global.shape} ({time.time() - start_time:.2f}s)")
    
    # Center embeddings
    print("Centering embeddings...")
    centered_embeddings = sampled_embeddings - mu_global
    
    # Compute PCA
    print(f"Computing PCA with {n_components} components...")
    print(f"Estimated time: ~{sampled_embeddings.shape[0] * sampled_embeddings.shape[1] / 1e6:.1f} seconds")
    pca_start = time.time()
    pca = PCA(n_components=n_components)
    pca.fit(centered_embeddings)
    pca_time = time.time() - pca_start
    
    pca_components = pca.components_  # Shape: (n_components, embedding_dim)
    explained_variance = pca.explained_variance_ratio_
    
    elapsed = time.time() - start_time
    print(f"PCA explained variance ratios: {explained_variance}")
    print(f"Total explained variance: {explained_variance.sum():.4f}")
    print(f"PCA computation took {pca_time:.2f} seconds")
    print(f"Total global stats computation: {elapsed:.2f} seconds")
    
    return mu_global, pca_components, pca

def debias_atom_embedding(x, mu_global, pca_components=None, remove_pcs=True):
    """
    Debias a single atom embedding by removing global mean and optionally top PCs.
    
    Args:
        x: Atom embedding vector (embedding_dim,)
        mu_global: Global mean vector (embedding_dim,)
        pca_components: Principal components (n_components, embedding_dim) or None
        remove_pcs: Whether to remove principal components
    
    Returns:
        x_debiased: Debias
ed atom embedding vector (embedding_dim,)
    """
    # Step B: Subtract global mean
    x_centered = x - mu_global
    
    if remove_pcs and pca_components is not None:
        # Remove top K PCs: x'' = x' - sum_k (v_k^T x') v_k
        x_debiased = x_centered.copy()
        for v_k in pca_components:
            projection = np.dot(v_k, x_centered)
            x_debiased = x_debiased - projection * v_k
        return x_debiased
    else:
        return x_centered

def load_and_debias_pocket_embeddings(directory, mu_global, pca_components, remove_pcs=True):
    """
    Load all pocket embeddings, debias atom-level embeddings, then average.
    
    Args:
        directory: Path to directory containing complex_*.pt files
        mu_global: Global mean vector
        pca_components: Principal components to remove
        remove_pcs: Whether to remove principal components
    
    Returns:
        embeddings_dict: dict mapping file_name -> debiased aggregated embedding
        file_names: list of file names in order
    """
    start_time = time.time()
    directory = Path(directory)
    pt_files = sorted(directory.glob("complex_*.pt"))
    
    embeddings_dict = {}
    file_names = []
    
    print(f"\nLoading and debiasing embeddings from {len(pt_files)} files...")
    print(f"Estimated time: ~{len(pt_files) * 0.05:.1f} seconds (assuming ~0.05s per file)")
    for pt_file in tqdm(pt_files, desc="Processing"):
        try:
            data = torch.load(pt_file, map_location='cpu')
            
            if 'pocket_atomica_embeddings' not in data:
                continue
            
            embeddings = data['pocket_atomica_embeddings']
            
            # Convert to numpy if tensor
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.numpy()
            
            # Step C: Debias each atom embedding, then average
            debiased_atoms = []
            for atom_emb in embeddings:
                atom_debiased = debias_atom_embedding(atom_emb, mu_global, pca_components, remove_pcs)
                debiased_atoms.append(atom_debiased)
            
            # Average debiased atom embeddings
            pocket_mean = np.mean(debiased_atoms, axis=0)
            
            # L2-normalize the pocket mean
            norm = np.linalg.norm(pocket_mean)
            if norm > 0:
                pocket_mean_normalized = pocket_mean / norm
            else:
                pocket_mean_normalized = pocket_mean
            
            embeddings_dict[pt_file.name] = pocket_mean_normalized
            file_names.append(pt_file.name)
            
        except Exception as e:
            print(f"Error processing {pt_file.name}: {e}")
            continue
    
    elapsed = time.time() - start_time
    print(f"Successfully processed {len(embeddings_dict)} pockets")
    print(f"Debiasing took {elapsed:.2f} seconds ({elapsed/len(embeddings_dict)*1000:.2f}ms per pocket)")
    return embeddings_dict, file_names

def compute_similarity_matrix(embeddings_dict, file_names):
    """Compute cosine similarity matrix between debiased pocket embeddings."""
    start_time = time.time()
    embeddings_array = np.array([embeddings_dict[name] for name in file_names])
    n_files = len(file_names)
    print(f"Embeddings array shape: {embeddings_array.shape} (num_files={n_files}, embedding_dim={embeddings_array.shape[1]})")
    
    # Estimate time: O(n^2 * d) where n=files, d=embedding_dim
    estimated_time = (n_files ** 2 * embeddings_array.shape[1]) / 1e8
    print(f"Computing similarity matrix... (estimated: ~{estimated_time:.1f} seconds)")
    
    similarity_matrix = cosine_similarity(embeddings_array)
    elapsed = time.time() - start_time
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity computation took {elapsed:.2f} seconds")
    return similarity_matrix

def cluster_embeddings(embeddings_dict, file_names, method='kmeans', n_clusters=5, **kwargs):
    """Cluster debiased embeddings using specified method."""
    start_time = time.time()
    embeddings_array = np.array([embeddings_dict[name] for name in file_names])
    n_files = len(file_names)
    
    if method == 'kmeans':
        # K-means is O(n * k * d * iterations), typically fast
        estimated_time = (n_files * n_clusters * embeddings_array.shape[1]) / 1e7
        print(f"Running K-means with {n_clusters} clusters... (estimated: ~{estimated_time:.1f} seconds)")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings_array)
    elif method == 'agglomerative':
        linkage = kwargs.get('linkage', 'ward')
        # Agglomerative is O(n^2) or O(n^2 log n), can be slow for large n
        estimated_time = (n_files ** 2) / 1e6
        print(f"Running Agglomerative clustering with {n_clusters} clusters... (estimated: ~{estimated_time:.1f} seconds)")
        print(f"  WARNING: This may be slow for {n_files} samples. Consider using kmeans for large datasets.")
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = clusterer.fit_predict(embeddings_array)
    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        # DBSCAN is typically O(n log n) with spatial indexing
        estimated_time = (n_files * np.log2(max(n_files, 1))) / 1e5
        print(f"Running DBSCAN... (estimated: ~{estimated_time:.1f} seconds)")
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clusterer.fit_predict(embeddings_array)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    elapsed = time.time() - start_time
    print(f"Clustering took {elapsed:.2f} seconds")
    return labels, clusterer

def visualize_clusters(embeddings_dict, file_names, labels, similarity_matrix, output_dir):
    """Create visualizations of the clustering results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_array = np.array([embeddings_dict[name] for name in file_names])
    
    # 1. Similarity heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='viridis', square=True, 
                xticklabels=False, yticklabels=False, cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Cosine Similarity Matrix (Debiased Pocket Embeddings)')
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_heatmap_debiased.png', dpi=300)
    plt.close()
    
    # 2. Cluster size distribution
    cluster_counts = defaultdict(int)
    for label in labels:
        cluster_counts[label] += 1
    
    plt.figure(figsize=(10, 6))
    clusters = sorted(cluster_counts.keys())
    counts = [cluster_counts[c] for c in clusters]
    plt.bar(clusters, counts)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Complexes')
    plt.title('Cluster Size Distribution (Debiased)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_sizes_debiased.png', dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def save_cluster_assignments(file_names, labels, output_file):
    """Save cluster assignments to a text file."""
    with open(output_file, 'w') as f:
        f.write("file_name\tcluster_id\n")
        for file_name, label in zip(file_names, labels):
            f.write(f"{file_name}\t{label}\n")
    print(f"Cluster assignments saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Cluster pocket ATOMICA embeddings with debiasing (removes global directions)")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing complex_*.pt files")
    parser.add_argument("--output_dir", type=str, default="./cluster_results_debiased",
                       help="Directory to save results and visualizations")
    parser.add_argument("--max_samples", type=int, default=100000,
                       help="Maximum number of atom embeddings to sample for global stats")
    parser.add_argument("--n_components", type=int, default=3,
                       help="Number of principal components to remove (default: 3). "
                            "Higher values (5-10) remove more global directions but may over-correct. "
                            "Lower values (1-2) are more conservative. "
                            "Try 1, 3, 5, or 10 to see what works best for your data.")
    parser.add_argument("--remove-pcs", type=int, default=None,
                       help="Alternative way to specify number of PCs to remove. "
                            "If specified, overrides --n_components. "
                            "Use this to easily experiment with removing more/fewer components.")
    parser.add_argument("--no-remove-pcs", action="store_true",
                       help="Only remove global mean, don't remove principal components")
    parser.add_argument("--method", type=str, default="kmeans",
                       choices=['kmeans', 'agglomerative', 'dbscan'],
                       help="Clustering method (default: kmeans)")
    parser.add_argument("--n_clusters", type=int, default=None,
                       help="Number of clusters (for kmeans/agglomerative). "
                            "If None, will use sqrt(n_samples/2) as default (~70-100 for 10k samples)")
    parser.add_argument("--eps", type=float, default=0.5,
                       help="DBSCAN eps parameter (for dbscan method)")
    parser.add_argument("--min_samples", type=int, default=5,
                       help="DBSCAN min_samples parameter (for dbscan method)")
    parser.add_argument("--linkage", type=str, default="ward",
                       choices=['ward', 'complete', 'average'],
                       help="Linkage method for agglomerative clustering")
    
    args = parser.parse_args()
    
    # Handle --remove-pcs override
    if args.remove_pcs is not None:
        args.n_components = args.remove_pcs
        print(f"\nUsing --remove-pcs override: removing {args.n_components} principal components")
    
    # Determine default number of clusters if not specified
    # Rule of thumb: sqrt(n/2) for large datasets, but cap reasonable range
    directory = Path(args.input_dir)
    pt_files = sorted(directory.glob("complex_*.pt"))
    n_files = len(pt_files)
    
    if args.n_clusters is None:
        # Default: sqrt(n/2) but between 10 and 200
        suggested_clusters = int(np.sqrt(n_files / 2))
        args.n_clusters = max(10, min(suggested_clusters, 200))
        print(f"\nAuto-selected {args.n_clusters} clusters based on {n_files} samples")
        print(f"  (Formula: sqrt({n_files}/2) ≈ {suggested_clusters}, clamped to [10, 200])")
        print(f"  For 10k samples, this gives ~{int(np.sqrt(10000/2))} clusters")
        print(f"  Use --n_clusters to override")
    else:
        print(f"\nUsing {args.n_clusters} clusters for {n_files} samples")
        if args.n_clusters < 5:
            print(f"  WARNING: {args.n_clusters} clusters may be too few for {n_files} samples")
        elif args.n_clusters > n_files / 10:
            print(f"  WARNING: {args.n_clusters} clusters may be too many (>{n_files//10} recommended)")
    
    total_start_time = time.time()
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Removing {args.n_components} principal component(s)")
    if args.no_remove_pcs:
        print(f"  WARNING: --no-remove-pcs is set, only removing global mean")
    print(f"  Max samples for global stats: {args.max_samples}")
    print(f"{'='*60}\n")
    
    # Step A: Sample atom embeddings and compute global stats
    sampled_embeddings = sample_atom_embeddings(args.input_dir, args.max_samples)
    mu_global, pca_components, pca = compute_global_stats(sampled_embeddings, args.n_components)
    
    # Step B & C: Load and debias pocket embeddings
    embeddings_dict, file_names = load_and_debias_pocket_embeddings(
        args.input_dir, mu_global, pca_components, remove_pcs=not args.no_remove_pcs)
    
    if len(embeddings_dict) == 0:
        print("No embeddings loaded. Exiting.")
        return
    
    # Compute similarity matrix
    print("\nComputing cosine similarity matrix...")
    similarity_matrix = compute_similarity_matrix(embeddings_dict, file_names)
    
    # Print similarity statistics
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    off_diagonal = similarity_matrix[mask]
    print(f"\nSimilarity Statistics (Debiased):")
    print(f"  Mean: {off_diagonal.mean():.4f}")
    print(f"  Std: {off_diagonal.std():.4f}")
    print(f"  Min: {off_diagonal.min():.4f}")
    print(f"  Max: {off_diagonal.max():.4f}")
    print(f"  Median: {np.median(off_diagonal):.4f}")
    print(f"  25th percentile: {np.percentile(off_diagonal, 25):.4f}")
    print(f"  75th percentile: {np.percentile(off_diagonal, 75):.4f}")
    
    # Perform clustering
    print(f"\nClustering using {args.method}...")
    kwargs = {'eps': args.eps, 'min_samples': args.min_samples, 'linkage': args.linkage}
    labels, clusterer = cluster_embeddings(embeddings_dict, file_names, 
                                          method=args.method, 
                                          n_clusters=args.n_clusters,
                                          **kwargs)
    
    # Print cluster info
    unique_labels = np.unique(labels)
    n_clusters_found = len(unique_labels)
    print(f"\nFound {n_clusters_found} clusters:")
    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"  Cluster {label}: {count} complexes")
    
    # Calculate silhouette scores
    print(f"\nCalculating silhouette scores...")
    embeddings_array = np.array([embeddings_dict[name] for name in file_names])
    
    # Filter out noise points (label == -1) for silhouette calculation
    valid_mask = labels != -1
    if np.sum(valid_mask) < 2:
        print("  WARNING: Not enough valid clusters for silhouette score (need at least 2)")
        silhouette_avg = None
        silhouette_by_cluster = None
    else:
        valid_labels = labels[valid_mask]
        valid_embeddings = embeddings_array[valid_mask]
        
        # Silhouette score requires at least 2 clusters
        if len(np.unique(valid_labels)) < 2:
            print("  WARNING: Need at least 2 clusters for silhouette score")
            silhouette_avg = None
            silhouette_by_cluster = None
        else:
            # Use cosine distance for silhouette (since we're using cosine similarity)
            from sklearn.metrics.pairwise import cosine_distances
            silhouette_avg = silhouette_score(valid_embeddings, valid_labels, 
                                            metric='cosine')
            silhouette_samples_vals = silhouette_samples(valid_embeddings, valid_labels,
                                                        metric='cosine')
            
            print(f"\nSilhouette Scores:")
            print(f"  Overall average: {silhouette_avg:.4f}")
            
            # Per-cluster silhouette scores
            silhouette_by_cluster = {}
            for label in np.unique(valid_labels):
                mask = valid_labels == label
                cluster_silhouette = silhouette_samples_vals[mask].mean()
                silhouette_by_cluster[label] = cluster_silhouette
                count = np.sum(mask)
                print(f"  Cluster {label}: {cluster_silhouette:.4f} ({count} samples)")
            
            # Interpretation
            if silhouette_avg > 0.5:
                print(f"\n  ✓ Good clustering (silhouette > 0.5)")
            elif silhouette_avg > 0.25:
                print(f"\n  ~ Reasonable clustering (silhouette > 0.25)")
            else:
                print(f"\n  ⚠ Weak clustering (silhouette < 0.25) - clusters may not be well-separated")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_cluster_assignments(file_names, labels, output_dir / "cluster_assignments_debiased.txt")
    
    # Save silhouette scores
    if silhouette_avg is not None:
        with open(output_dir / "silhouette_scores.txt", 'w') as f:
            f.write(f"Overall Average Silhouette Score: {silhouette_avg:.6f}\n\n")
            f.write("Per-Cluster Silhouette Scores:\n")
            for label, score in sorted(silhouette_by_cluster.items()):
                count = np.sum(labels == label)
                f.write(f"Cluster {label}: {score:.6f} ({count} samples)\n")
        print(f"\nSilhouette scores saved to {output_dir / 'silhouette_scores.txt'}")
    
    # Create visualizations
    visualize_clusters(embeddings_dict, file_names, labels, similarity_matrix, output_dir)
    
    # Save similarity matrix and global stats
    np.save(output_dir / "similarity_matrix_debiased.npy", similarity_matrix)
    np.save(output_dir / "mu_global.npy", mu_global)
    np.save(output_dir / "pca_components.npy", pca_components)
    
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"Total processing time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"Results saved to {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

