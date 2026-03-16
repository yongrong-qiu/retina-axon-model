import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

def sort_mat_by_1pc(X, num_components=1):
    """
    Sort rows of a matrix by their first principal component score.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data matrix whose rows will be sorted.
    num_components : int, optional
        Number of PCA components to compute (default: 1).
        Only the first component is used for sorting.

    Returns
    -------
    mat_srtd : np.ndarray, shape (n_samples, n_features)
        Row-sorted data matrix.
    idx_pc1_sort : np.ndarray, shape (n_samples,)
        Indices that sort the rows by ascending PC1 score.
    """
    
    # Perform PCA
    pca = PCA(n_components=num_components)
    pca.fit(X)
    
    # Get PC scores (projection of each row onto the PCs)
    pc_scores = pca.transform(X)
    
    # Sort rows by their PC1 scores
    idx_pc1_sort = np.argsort(pc_scores[:, 0])
    mat_srtd = X[idx_pc1_sort]
    
    return mat_srtd, idx_pc1_sort

def spca_weights_sort_and_plot(tv, sparse_pca_components, num_pcs, X_pca, alpha):
    """
    Sort sparse PCA components by the time of their first non-zero weight
    and returns the component matrix as a heatmap.

    Parameters
    ----------
    tv : np.ndarray, shape (n_timepoints,)
        Time vector corresponding to feature columns.
    sparse_pca_components : np.ndarray, shape (num_pcs, n_timepoints)
        Sparse PCA component weight matrix.
    num_pcs : int
        Number of sparse PCA components.
    X_pca : np.ndarray, shape (n_samples, num_pcs)
        Projected data (scores) in sparse PCA space.
    alpha : float
        Sparsity parameter used (displayed in the plot title).

    Returns
    -------
    sparse_pca_components_sorted : np.ndarray, shape (num_pcs, n_timepoints)
        Components reordered by first non-zero time index.
    """
    
    # Find the first non-zero entry for each PC
    first_nonzero_idx = np.argmax(sparse_pca_components != 0, axis=1)
    pcs_time_first_nonzero = tv[first_nonzero_idx]
    pcs_order = np.argsort(pcs_time_first_nonzero)
    
    sparse_pca_components_sorted_l = []
    X_pca_sorted_l = []
    for i in range(num_pcs):
        sparse_pca_components_sorted_l.append(sparse_pca_components[pcs_order[i], :])
        X_pca_sorted_l.append(X_pca[:, pcs_order[i]:pcs_order[i]+1])  # Fixed indexing
    sparse_pca_components_sorted = np.vstack(sparse_pca_components_sorted_l)
    X_pca_sorted = np.hstack(X_pca_sorted_l)
    
    return sparse_pca_components_sorted

def correlation_distance(X):
    """
    Compute pairwise correlation distance matrix.
    Distance d_ij = 1 - corr(m_i, m_j)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data matrix where each row is a sample (cluster mean)
    
    Returns:
    --------
    distance_matrix : array, shape (n_samples, n_samples)
        Pairwise correlation distance matrix
    """
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X)
    
    # Convert to distance: d = 1 - correlation
    distance_matrix = 1 - corr_matrix
    
    # Ensure diagonal is zero and matrix is symmetric
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Clip negative values that might arise from numerical issues
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    return distance_matrix

def get_dendrogram(X, cluster_IDs, color_thresh=0.7, reverse_order=False):
    """
    Perform average-linkage hierarchical clustering on correlation distances
    and plot a horizontal dendrogram.

    Parameters
    ----------
    X : np.ndarray, shape (n_clusters, n_features)
        Cluster mean responses (one row per cluster).
    cluster_IDs : np.ndarray, shape (n_clusters,)
        Labels for each cluster (used as leaf labels).
    color_thresh : float, optional
        Fraction of the maximum linkage distance used as the color
        threshold (default: 0.7).
    reverse_order : bool, optional
        If True, invert the y-axis and flip the leaf order (default: False).

    Returns
    -------
    optimal_leaf_order : np.ndarray, shape (n_clusters,)
        Cluster IDs in dendrogram leaf order (top-to-bottom).
    linkage_mat : np.ndarray, shape (n_clusters-1, 4)
        Scipy linkage matrix.
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    n_clusters = X.shape[0]
    
    # Compute correlation distance matrix
    distance_matrix = correlation_distance(X)
    
    # Convert to condensed distance matrix for linkage
    # (scipy linkage expects condensed form)
    condensed_distances = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering with average linkage
    linkage_mat = linkage(condensed_distances, method='average', optimal_ordering=True)

    # Plot dendrogram with optimized leaf order
    fig, ax = plt.subplots(1,1,figsize=(3,15))
    dendrogram_result = dendrogram(
                                    linkage_mat,
                                    labels=cluster_IDs,
                                    leaf_font_size=8,
                                    color_threshold= color_thresh * max(linkage_mat[:, 2]),
                                    leaf_rotation=0,
                                    orientation='left',
                                    ax=ax,
                                    )
    # Reverse the dendrogram if requested
    if reverse_order:
        ax.invert_yaxis()
    
    sns.despine(bottom=True, left=True)
    
    optimal_leaf_order_indices = dendrogram_result['leaves']
    optimal_leaf_order = cluster_IDs[optimal_leaf_order_indices]
    
    # If reversed, flip the optimal_leaf_order array to match visual order
    if reverse_order:
        optimal_leaf_order = np.flip(optimal_leaf_order)
    
    return optimal_leaf_order, linkage_mat, fig, ax

def intra_cluster_correlation(traces, labels):
    """
    Calculate intra-cluster correlation using leave-one-out approach.
    
    For each cell in a cluster, compute correlation with the mean response
    of all other cells in that cluster. Returns mean and std of correlations
    per cluster.
    
    Parameters:
    -----------
    traces : np.ndarray, shape (n_cells, n_timepoints)
        Neural responses (can be concatenated chirp+bar)
    labels : np.ndarray, shape (n_cells,)
        Cluster assignments for each cell
        
    Returns:
    --------
    mean_corr_per_cluster : np.ndarray, shape (n_clusters,)
        Mean correlation per cluster
    std_corr_per_cluster : np.ndarray, shape (n_clusters,)
        Std of correlations per cluster
    """
    cluster_ids = np.unique(labels).astype(int)
    n_clusters = len(cluster_ids)
    
    mean_corr_per_cluster = np.zeros(n_clusters, dtype=float)
    std_corr_per_cluster = np.zeros(n_clusters, dtype=float)
    
    for idx, cluster_id in enumerate(cluster_ids):
        # Get all traces belonging to this cluster
        cluster_mask = labels == cluster_id
        traces_cluster = traces[cluster_mask]
        n_cells_in_cluster = traces_cluster.shape[0]
        
        # Compute leave-one-out correlation for each cell
        correlations = np.zeros(n_cells_in_cluster, dtype=float)
        
        for i in range(n_cells_in_cluster):
            # Average of all cells except cell i
            leave_one_out_mean = np.mean(traces_cluster[np.arange(n_cells_in_cluster) != i], axis=0)
            
            # Correlation between cell i and the leave-one-out mean
            corr_matrix = np.corrcoef(traces_cluster[i], leave_one_out_mean)
            correlations[i] = corr_matrix[0, 1]
            
            # Handle NaN (occurs if variance is zero)
            if np.isnan(correlations[i]):
                correlations[i] = 0
        
        mean_corr_per_cluster[idx] = np.mean(correlations)
        std_corr_per_cluster[idx] = np.std(correlations)
    
    return mean_corr_per_cluster, std_corr_per_cluster