import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

class MinimalDataProcessor:
    def __init__(self, dataset_path, target_col='class', exclude_cols=None, num_samples=1000):
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.exclude_cols = exclude_cols or ["e_magic", "e_crlc","Armadillov1xxv2xx"]
        self.num_samples = num_samples
        self.df = None
        self.feature_2to10 = []
        self.all_features = []  # Store all features for "all" option
    
    def load_and_balance_data(self):
        # Load CSV
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        self.df = pd.read_csv(os.path.join(self.dataset_path, csv_files[0]))
        
        # Drop excluded columns and balance dataset
        self.df = self.df.drop(columns=self.exclude_cols, errors='ignore')
        class_0 = self.df[self.df[self.target_col] == 0].sample(n=self.num_samples//2)
        class_1 = self.df[self.df[self.target_col] == 1].sample(n=self.num_samples//2)
        self.df = pd.concat([class_0, class_1]).sample(frac=1).reset_index(drop=True)
        
        # Store all features (excluding target)
        self.all_features = [col for col in self.df.columns if col != self.target_col]
        
        return self.df
    
    def _get_representative_feature(self, cluster_features, corr_matrix):
        if len(cluster_features) == 1:
            return cluster_features[0]
        scores = {f: np.mean([abs(corr_matrix.loc[f, other]) 
                            for other in cluster_features if other != f]) 
                 for f in cluster_features}
        return max(scores, key=scores.get)
    
    def generate_feature_clusters(self, min_clusters=2, max_clusters=10):
        data = self.df.drop(columns=[self.target_col])
        
        # Remove constant features that cause correlation issues
        data = data.loc[:, data.std() > 1e-6]
        
        correlations = data.corr().abs().fillna(0)
        
        # Clip correlations to valid range [0, 1]
        correlations = correlations.clip(0, 1)
        
        # Create dissimilarity matrix
        dissimilarity = 1 - correlations
        
        # Ensure perfect symmetry and valid distance properties
        dissimilarity = (dissimilarity + dissimilarity.T) / 2
        np.fill_diagonal(dissimilarity.values, 0)  # Distance to self = 0
        
        # Ensure all values are non-negative
        dissimilarity = dissimilarity.clip(0, None)
        
        try:
            Z = linkage(squareform(dissimilarity), 'ward')
        except ValueError:
            # Fallback: use average linkage if ward fails
            Z = linkage(squareform(dissimilarity), 'average')
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            labels = fcluster(Z, n_clusters, criterion='maxclust')
            
            # Group features by cluster
            clusters = {}
            for idx, feature in enumerate(data.columns):
                cluster_id = labels[idx]
                clusters.setdefault(cluster_id, []).append(feature)
            
            # Get representative features
            representatives = [self._get_representative_feature(features, correlations) 
                             for features in clusters.values()]
            self.feature_2to10.append(representatives)
        
        return self.feature_2to10
    
    def _cluster_and_reorder_features(self, features, n_clusters=None):
        """Cluster features and return them ordered by cluster groups"""
        if len(features) <= 1:
            return features
        
        data = self.df[features]
        correlations = data.corr().abs().fillna(0).clip(0, 1)
        dissimilarity = 1 - correlations
        dissimilarity = (dissimilarity + dissimilarity.T) / 2
        np.fill_diagonal(dissimilarity.values, 0)
        dissimilarity = dissimilarity.clip(0, None)
        
        try:
            Z = linkage(squareform(dissimilarity), 'ward')
        except ValueError:
            Z = linkage(squareform(dissimilarity), 'average')
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(10, max(2, len(features) // 5))
        
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Group features by cluster and order them
        cluster_groups = {}
        for feature, cluster_id in zip(features, cluster_labels):
            cluster_groups.setdefault(cluster_id, []).append(feature)
        
        # Create ordered feature list (all cluster 1, then all cluster 2, etc.)
        ordered_features = []
        for cluster_id in sorted(cluster_groups.keys()):
            ordered_features.extend(sorted(cluster_groups[cluster_id]))
        
        return ordered_features
    
    def plot_10d_correlation_heatmap(self, figsize=(12, 10)):
        """Plot correlation heatmap with FORCED cluster grouping for 10D features"""
        if not self.feature_2to10:
            print("Run generate_feature_clusters() first")
            return
        
        # Get 10D features (index 8 = 10 clusters)
        features_10d = self.feature_2to10[8]  # 10D is at index 8 (10-2)
        
        # Force cluster grouping
        ordered_features, cluster_assignments = self._force_cluster_grouping(features_10d, n_clusters=5)
        data_10d = self.df[ordered_features]
        correlations = data_10d.corr()
        
        # Plot heatmap with forced cluster boundaries
        plt.figure(figsize=figsize)
        ax = sns.heatmap(correlations, annot=True, cmap='RdBu', 
                        vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
        
        # Add cluster boundaries based on forced grouping
        cluster_boundaries = self._get_cluster_boundaries(cluster_assignments)
        
        for boundary in cluster_boundaries:
            ax.axhline(y=boundary, color='white', linewidth=3)
            ax.axvline(x=boundary, color='white', linewidth=3)
        
        # Add cluster labels based on forced grouping
        cluster_starts = [0] + cluster_boundaries + [len(cluster_assignments)]
        for i in range(len(cluster_starts) - 1):
            start = cluster_starts[i]
            end = cluster_starts[i + 1]
            mid_point = (start + end) / 2
            cluster_id = cluster_assignments[start]
            
            ax.text(mid_point, -0.5, f'C{cluster_id}', 
                   fontweight='bold', fontsize=12, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
            ax.text(len(correlations) + 0.2, mid_point, f'C{cluster_id}', 
                   fontweight='bold', fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        plt.title('10D Features Heatmap with FORCED Cluster Grouping', 
                 fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
        
        # Print cluster summary
        print(f"\n10D Features FORCED Clustering Summary:")
        print(f"Ordered features (grouped by cluster): {ordered_features}")
        self._print_forced_cluster_summary(ordered_features, cluster_assignments, "10D")
    
    def plot_10d_clustermap(self, figsize=(12, 12)):
        """Plot clustermap for 10D features with FORCED cluster grouping"""
        if not self.feature_2to10:
            print("Run generate_feature_clusters() first")
            return
        
        features_10d = self.feature_2to10[8]
        
        # Force cluster grouping
        ordered_features, cluster_assignments = self._force_cluster_grouping(features_10d, n_clusters=5)
        data_10d = self.df[ordered_features]
        correlations = data_10d.corr()
        
        # Create clustermap with clustering DISABLED to preserve our ordering
        g = sns.clustermap(correlations, method="complete", cmap='RdBu', 
                          annot=True, vmin=-1, vmax=1, figsize=figsize,
                          row_cluster=False, col_cluster=False, square=True)  # Disable clustering
        
        ax_heatmap = g.ax_heatmap
        
        # Add cluster boundaries based on forced grouping
        cluster_boundaries = self._get_cluster_boundaries(cluster_assignments)
        
        for boundary in cluster_boundaries:
            ax_heatmap.axhline(y=boundary, color='white', linewidth=3, alpha=0.9)
            ax_heatmap.axvline(x=boundary, color='white', linewidth=3, alpha=0.9)
        
        # Add forced cluster labels
        self._add_forced_cluster_labels(ax_heatmap, cluster_assignments, cluster_boundaries)
        
        g.fig.suptitle('10D Features with FORCED Cluster Grouping', fontsize=16, y=1.02)
        plt.show()
        
        # Print summary
        self._print_forced_cluster_summary(ordered_features, cluster_assignments, "10D")
    
    def get_features(self, n_clusters):
        """Get representative features for n clusters (2-10) or all features"""
        if n_clusters == "all":
            return self.all_features
        return self.feature_2to10[n_clusters - 2] if 2 <= n_clusters <= 10 else None
    
    def get_subset(self, n_clusters, include_target=True):
        """Get DataFrame with representative features"""
        features = self.get_features(n_clusters)
        if include_target and features:
            return self.df[features + [self.target_col]]
        return self.df[features] if features else None
    
    def _select_features_for_dimensions(self, n_dimensions):
        """Select features based on dimension count with intelligent selection"""
        if n_dimensions == "all":
            return self.all_features
            
        if n_dimensions <= 10:
            # Use feature_2to10 for dimensions 2-10
            return self.feature_2to10[n_dimensions - 2]
        else:
            # For >10 dimensions: use 10D features + add least correlated from each cluster
            base_features = self.feature_2to10[8]  # 10D features (index 8)
            selected_features = base_features.copy()
            
            if n_dimensions <= len(base_features):
                return selected_features[:n_dimensions]
            
            # Need more features - get original clustering data
            data = self.df.drop(columns=[self.target_col])
            data = data.loc[:, data.std() > 1e-6]
            correlations = data.corr().abs().fillna(0).clip(0, 1)
            dissimilarity = 1 - correlations
            dissimilarity = (dissimilarity + dissimilarity.T) / 2
            np.fill_diagonal(dissimilarity.values, 0)
            dissimilarity = dissimilarity.clip(0, None)
            
            try:
                Z = linkage(squareform(dissimilarity), 'ward')
            except ValueError:
                Z = linkage(squareform(dissimilarity), 'average')
            
            # Get 10 clusters to match our base features
            cluster_labels = fcluster(Z, 10, criterion='maxclust')
            
            # Group all features by their 10 clusters
            clusters = {}
            for idx, feature in enumerate(data.columns):
                cluster_id = cluster_labels[idx]
                clusters.setdefault(cluster_id, []).append(feature)
            
            # Map each base feature to its cluster
            feature_to_cluster = {}
            for cluster_id, cluster_features in clusters.items():
                for base_feature in base_features:
                    if base_feature in cluster_features:
                        feature_to_cluster[base_feature] = cluster_id
                        break
            
            # Add additional features from each cluster (least correlated to existing)
            additional_needed = n_dimensions - len(selected_features)
            added_count = 0
            
            # Cycle through clusters to add features evenly
            for cycle in range((additional_needed // len(feature_to_cluster)) + 1):
                if added_count >= additional_needed:
                    break
                    
                for base_feature in base_features:
                    if added_count >= additional_needed:
                        break
                        
                    cluster_id = feature_to_cluster.get(base_feature)
                    if cluster_id is None:
                        continue
                        
                    cluster_features = clusters[cluster_id]
                    available_features = [f for f in cluster_features if f not in selected_features]
                    
                    if not available_features:
                        continue
                    
                    # Find feature least correlated with the base feature from this cluster
                    correlations_with_base = {}
                    for candidate in available_features:
                        corr_val = abs(correlations.loc[base_feature, candidate])
                        correlations_with_base[candidate] = corr_val
                    
                    # Select feature with lowest correlation to base feature
                    least_correlated = min(correlations_with_base.keys(), 
                                         key=correlations_with_base.get)
                    selected_features.append(least_correlated)
                    added_count += 1
            
            return selected_features[:n_dimensions]  # Ensure exact count
    
    def plot_features_clustermap(self, n_dimensions, figsize=(16, 16)):
        """Plot hierarchical clustermap for selected features with cluster boundary lines and FORCED grouped features"""
        if self.df is None:
            print("Load data first")
            return
        
        # Select features based on dimensions
        if n_dimensions != "all" and n_dimensions <= 10 and not self.feature_2to10:
            print("Run generate_feature_clusters() first for dimensions <= 10")
            return
        
        selected_features = self._select_features_for_dimensions(n_dimensions)
        
        if not selected_features:
            print(f"Could not select features for {n_dimensions} dimensions")
            return
        
        print(f"Selected {len(selected_features)} features for {n_dimensions}D visualization:")
        if n_dimensions == "all":
            print(f"Using ALL features: {len(selected_features)} total")
        elif n_dimensions <= 10:
            print(f"Using feature_2to10[{n_dimensions-2}]: {selected_features}")
        else:
            base_features = self.feature_2to10[8]
            additional_features = [f for f in selected_features if f not in base_features]
            print(f"Base 10D features: {base_features}")
            print(f"Additional features (least correlated within clusters): {additional_features}")
        
        # FORCE CLUSTER GROUPING - this is the key fix
        if n_dimensions == "all":
            n_clusters = min(15, max(5, len(selected_features) // 10))  # Adaptive for large datasets
        else:
            n_clusters = min(10, len(selected_features))
        
        ordered_features, cluster_assignments = self._force_cluster_grouping(selected_features, n_clusters=n_clusters)
        
        data_selected = self.df[ordered_features]
        correlations = data_selected.corr().fillna(0).clip(-1, 1)
        correlations = correlations.replace([np.inf, -np.inf], 0)
        correlations = (correlations + correlations.T) / 2
        
        try:
            # Determine annotation and font size based on number of features
            show_annotations = len(ordered_features) <= 20
            annot_fontsize = max(8, 16 - len(ordered_features) // 3)  # Bigger font, adaptive
            
            # Create clustermap but DISABLE clustering to preserve our ordering
            g = sns.clustermap(correlations, method="complete", cmap='RdBu', 
                              annot=show_annotations, 
                              annot_kws={"size": annot_fontsize},  # Bigger correlation numbers
                              fmt='.2f',  # 2 decimal places
                              vmin=-1, vmax=1, figsize=figsize,
                              row_cluster=False, col_cluster=False, square=True)  # DISABLE clustering
            
            ax_heatmap = g.ax_heatmap
            
            # Add cluster boundaries based on our forced grouping
            cluster_boundaries = self._get_cluster_boundaries(cluster_assignments)
            
            for boundary in cluster_boundaries:
                ax_heatmap.axhline(y=boundary, color='white', linewidth=4, alpha=0.9)
                ax_heatmap.axvline(x=boundary, color='white', linewidth=4, alpha=0.9)
            
            # Add cluster labels based on our forced grouping
            self._add_forced_cluster_labels(ax_heatmap, cluster_assignments, cluster_boundaries)
            
            title_suffix = f"ALL ({len(selected_features)})" if n_dimensions == "all" else f"{n_dimensions}D"
            title = f'{title_suffix} Features with FORCED Cluster Grouping ({n_clusters} clusters)'
            g.fig.suptitle(title, fontsize=14, y=1.02)
            plt.show()
            
            # Print cluster summary
            self._print_forced_cluster_summary(ordered_features, cluster_assignments, n_dimensions)
            
        except Exception as e:
            print(f"Error creating clustermap: {e}")
    
    def _force_cluster_grouping(self, features, n_clusters=None):
        """Force features to be grouped by clusters and return ordered features with cluster assignments"""
        if len(features) <= 1:
            return features, [1] * len(features)
        
        data = self.df[features]
        correlations = data.corr().abs().fillna(0).clip(0, 1)
        dissimilarity = 1 - correlations
        dissimilarity = (dissimilarity + dissimilarity.T) / 2
        np.fill_diagonal(dissimilarity.values, 0)
        dissimilarity = dissimilarity.clip(0, None)
        
        try:
            Z = linkage(squareform(dissimilarity), 'ward')
        except ValueError:
            Z = linkage(squareform(dissimilarity), 'average')
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(10, max(2, len(features) // 5))
        
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Group features by cluster and force them to be together
        cluster_groups = {}
        for feature, cluster_id in zip(features, cluster_labels):
            cluster_groups.setdefault(cluster_id, []).append(feature)
        
        # Create ordered feature list (all cluster 1, then all cluster 2, etc.)
        ordered_features = []
        ordered_cluster_assignments = []
        
        for cluster_id in sorted(cluster_groups.keys()):
            cluster_features = sorted(cluster_groups[cluster_id])  # Sort within cluster
            ordered_features.extend(cluster_features)
            ordered_cluster_assignments.extend([cluster_id] * len(cluster_features))
        
        print(f"FORCED grouping: Features ordered by clusters")
        for cluster_id in sorted(cluster_groups.keys()):
            print(f"  Cluster {cluster_id}: {cluster_groups[cluster_id]}")
        
        return ordered_features, ordered_cluster_assignments
    
    def _get_cluster_boundaries(self, cluster_assignments):
        """Get cluster boundary positions from cluster assignments"""
        boundaries = []
        current_cluster = cluster_assignments[0]
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id != current_cluster:
                boundaries.append(i)
                current_cluster = cluster_id
        return boundaries
    
    def _add_forced_cluster_labels(self, ax, cluster_assignments, cluster_boundaries):
        """Add cluster labels based on forced cluster grouping"""
        cluster_starts = [0] + cluster_boundaries + [len(cluster_assignments)]
        
        for i in range(len(cluster_starts) - 1):
            start = cluster_starts[i]
            end = cluster_starts[i + 1]
            mid_point = (start + end) / 2
            cluster_id = cluster_assignments[start]
            
            # Row labels (LEFT)
            ax.text(-1.5, mid_point, f'C{cluster_id}', 
                   fontweight='bold', fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
            
            # Column labels (TOP)
            ax.text(mid_point, -1.5, f'C{cluster_id}', 
                   fontweight='bold', fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    def _print_forced_cluster_summary(self, ordered_features, cluster_assignments, n_dimensions):
        """Print summary of forced cluster grouping"""
        print(f"\nFORCED Cluster Summary for {n_dimensions}:")
        print(f"Features are GUARANTEED to be grouped by cluster (no dendrogram reordering)")
        print(f"Ordered features: {ordered_features}")
        
        # Group features by cluster for summary
        cluster_groups = {}
        for feature, cluster_id in zip(ordered_features, cluster_assignments):
            cluster_groups.setdefault(cluster_id, []).append(feature)
        
        print(f"\nCluster groups (in matrix order):")
        for cluster_id in sorted(cluster_groups.keys()):
            features = cluster_groups[cluster_id]
            print(f"Cluster {cluster_id} ({len(features)} features): {features}")
    
    
    def _find_boundaries(self, cluster_labels):
        """Find cluster boundary positions"""
        boundaries = []
        current_cluster = cluster_labels[0]
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id != current_cluster:
                boundaries.append(i)
                current_cluster = cluster_id
        return boundaries
    
    def _add_external_cluster_labels(self, ax, row_boundaries, col_boundaries, row_labels, col_labels):
        """Add cluster labels OUTSIDE the correlation matrix (to the left and top)"""
        # Row cluster labels on the LEFT (outside the matrix)
        row_starts = [0] + row_boundaries + [len(row_labels)]
        for i in range(len(row_starts) - 1):
            start = row_starts[i]
            end = row_starts[i + 1]
            mid_point = (start + end) / 2
            cluster_id = row_labels[start] if start < len(row_labels) else row_labels[-1]
            
            ax.text(-1.5, mid_point, f'C{cluster_id}', 
                   fontweight='bold', fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        # Column cluster labels at the TOP (outside the matrix)
        col_starts = [0] + col_boundaries + [len(col_labels)]
        for i in range(len(col_starts) - 1):
            start = col_starts[i]
            end = col_starts[i + 1]
            mid_point = (start + end) / 2
            cluster_id = col_labels[start] if start < len(col_labels) else col_labels[-1]
            
            ax.text(mid_point, -1.5, f'C{cluster_id}', 
                   fontweight='bold', fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    def run_all(self, plotd=10):
        """Complete pipeline with customizable dimension plotting"""
        self.load_and_balance_data()
        
        if plotd != "all":
            self.generate_feature_clusters()
            print(f"Generated feature_2to10 with {len(self.feature_2to10)} entries")
            for i, features in enumerate(self.feature_2to10):
                print(f"  {i+2}D: {len(features)} features - {features}")
        else:
            print(f"Processing ALL features: {len(self.all_features)} total")
            # Still generate clustering for consistency but won't use feature_2to10
            self.generate_feature_clusters()
        
        if plotd:
            if plotd == "all":
                print(f"\nGenerating correlation plot for ALL {len(self.all_features)} features...")
            else:
                print(f"\nGenerating {plotd}D correlation plots...")
            
            if plotd == 10:
                # Special case: show both 10D specific plots and clustermap
                self.plot_10d_correlation_heatmap()
                self.plot_10d_clustermap()
            
            # Always show the customizable clustermap
            self.plot_features_clustermap(plotd)
        
        return self.feature_2to10 if plotd != "all" else self.all_features

# Usage:
"""
processor = MinimalDataProcessor(
    dataset_path="/path/to/dataset"
)

# Plot all features with grouped clusters
all_features = processor.run_all(plotd="all")

# Plot 52D with grouped clusters
feature_52d = processor.run_all(plotd=52)

# Plot 10D with special plots
feature_10d = processor.run_all(plotd=10)
"""