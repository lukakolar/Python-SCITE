import numpy as np
from numba import njit

import SuMoTEDUtils
import TreeUtils
import mcmcse


def get_distances(parent_vector_matrix, tree_distance_type, repetitions):
    """Get distances between trees according to pseudo ESS calculation.

    Args:
        parent_vector_matrix (np.ndarray): Matrix in which rows are parent vectors.
        tree_distance_type (int): 0 - number of nodes with different parents, 1 - SuMoTED.
        repetitions (int): Number of focal trees used in calculation of pseudo ESS.

    Returns:
        ndarray: Matrix of distances. For each focal tree (repetitions) distances to all trees in parent_vector_matrix
            are stored in a row of the matrix.
    """
    if tree_distance_type == 0:
        return get_distances_parent_vector(parent_vector_matrix, repetitions)
    elif tree_distance_type == 1:
        return get_distances_sumoted(parent_vector_matrix, repetitions)
    else:
        raise Exception("Specified tree distance not recognized.")


@njit(cache=True)
def get_distances_parent_vector(parent_vector_matrix, repetitions):
    """Get distances between trees according to pseudo ESS calculation when parent vector distance is selected."""
    num_trees, num_nodes = parent_vector_matrix.shape

    distances = np.empty((repetitions, num_trees,), dtype=np.int32)
    for repetition in range(repetitions):
        focal_tree = parent_vector_matrix[np.random.choice(num_trees), :]

        for tree_index in range(num_trees):
            distances[repetition, tree_index] = TreeUtils.get_tree_distance(focal_tree,
                                                                            parent_vector_matrix[tree_index, :])

    return distances


@njit(cache=True)
def get_distances_sumoted(parent_vector_matrix, repetitions):
    """Get distances between trees according to pseudo ESS calculation when SuMoTED is selected."""
    num_trees, num_nodes = parent_vector_matrix.shape

    distances = np.empty((repetitions, num_trees,), dtype=np.int32)
    dfts = get_dfts(parent_vector_matrix)

    for repetition in range(repetitions):
        focal_index = np.random.choice(num_trees)
        focal_tree = TreeUtils.get_ancestor_matrix(parent_vector_matrix[focal_index, :], dfts[focal_index, :])

        for tree_index in range(num_trees):
            current_tree = TreeUtils.get_ancestor_matrix(parent_vector_matrix[tree_index, :], dfts[tree_index, :])
            distances[repetition, tree_index] = SuMoTEDUtils.distance(focal_tree, current_tree)

    return distances


@njit(cache=True)
def get_dfts(parent_vector_matrix):
    """Get depth first traversals for all specified trees.

    Args:
        parent_vector_matrix (np.ndarray): Matrix in which rows are parent vectors.

    Returns:
        ndarray: Matrix of depth first traversals corresponding to trees in parent_vector_matrix.
    """
    num_trees, num_nodes = parent_vector_matrix.shape
    dfts = np.empty((num_trees, num_nodes + 1), dtype=np.int32)

    for i in range(num_trees):
        dfts[i, :] = TreeUtils.get_depth_first_traversal(parent_vector_matrix[i, :])

    return dfts


def calculate_pseudo_ess_trees(parent_vector_matrix, tree_distance_type, repetitions):
    """Calculate pseudo effective sample size for specified trees.

    Args:
        parent_vector_matrix (np.ndarray): Matrix in which rows are parent vectors.
        tree_distance_type (int): 0 - number of nodes with different parents, 1 - SuMoTED.
        repetitions (int): Number of focal trees used in calculation of pseudo ESS.

    Returns:
        float: Mean of pseudo effective sample sizes, calculated for various focal trees.
    """
    distances = get_distances(parent_vector_matrix, tree_distance_type, repetitions)
    ess_vector = np.empty((repetitions,), dtype=np.float64)

    for repetition in range(repetitions):
        ess_vector[repetition] = mcmcse.ess(distances[repetition, :])

    return np.median(ess_vector)


def calculate_ess_betas(beta_vector):
    """Calculate effective sample size for specified betas.

    Args:
        beta_vector (np.ndarray): Vector of beta values.

    Returns:
        float: Effective sample size for beta.
    """
    return mcmcse.ess(beta_vector)
