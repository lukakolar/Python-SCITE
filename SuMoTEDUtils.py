import numpy as np
from numba import njit

import TreeUtils


@njit(cache=True)
def get_adjacency_matrices(parent_vector_matrix):
    """Construct adjacency matrices corresponding to parent vectors in parent_vector_matrix. An adjacency matrix is
    a binary matrix in which value 1 in a cell at position ij means that node i is a parent (direct ancestor) of node j.
    These type of matrices are needed for the calculation of SuMoTED distance.

    Args:
        parent_vector_matrix (np.ndarray): Parent vector representation of mutation trees.

    Returns:
        np.ndarray: 3D matrix of 2D adjacency matrices corresponding to provided parent vectors.
    """
    num_trees, num_nodes = parent_vector_matrix.shape
    adjacency_matrices = np.zeros((num_trees, num_nodes + 1, num_nodes + 1), dtype=np.int32)

    for i in range(num_trees):
        for j in range(num_nodes):
            adjacency_matrices[i, parent_vector_matrix[i, j], j] = 1

    return adjacency_matrices


@njit(cache=True)
def distance(first_tree_ancestor_matrix, second_tree_ancestor_matrix):
    """Calculate SuMoTED - Subtree Moving Tree Edit Distance for specified trees.

    Args:
        first_tree_ancestor_matrix (np.ndarray): Ancestor matrix representation of first mutation tree.
        second_tree_ancestor_matrix (np.ndarray): Ancestor matrix representation of second mutation tree.

    Returns:
        int: SuMoTED - Subtree Moving Tree Edit Distance.
    """
    first_tree_closure = get_transitive_closure_from_ancestor_matrix(first_tree_ancestor_matrix)
    second_tree_closure = get_transitive_closure_from_ancestor_matrix(second_tree_ancestor_matrix)

    intersection = np.logical_and(first_tree_closure, second_tree_closure).astype(np.int32)
    intersection_closure = intersection_to_closure(intersection)

    s1 = np.sum(first_tree_closure) - np.sum(np.logical_and(first_tree_closure, intersection_closure))
    s2 = np.sum(second_tree_closure) - np.sum(np.logical_and(second_tree_closure, intersection_closure))
    tree_distance = s1 + s2

    return tree_distance


@njit(cache=True)
def get_transitive_closure_from_ancestor_matrix(ancestor_matrix):
    num_nodes = ancestor_matrix.shape[0]

    closure = np.zeros((num_nodes + 1, num_nodes + 1), dtype=np.int32)
    closure[:num_nodes, :num_nodes] = ancestor_matrix
    closure[num_nodes, :] = 1

    for i in range(num_nodes + 1):
        closure[i, i] -= 1

    return closure


@njit(cache=True)
def intersection_to_closure(intersection):
    num_nodes = intersection.shape[0] - 1

    sums = np.sum(intersection, axis=0)

    current_node = num_nodes
    visited_nodes = np.zeros((num_nodes + 1,), dtype=np.int32)
    depths = np.zeros((num_nodes + 1,), dtype=np.int32)

    order = np.empty((num_nodes + 1,), dtype=np.int32)
    order_index = 0
    parent_vector = np.zeros((num_nodes,), dtype=np.int32)

    while True:
        order[order_index] = current_node
        order_index += 1
        visited_nodes[current_node] = 1

        max_ancestor_depth = 0
        parent = -1
        for i in range(num_nodes + 1):
            if intersection[i, current_node] == 1 and depths[i] >= max_ancestor_depth:
                max_ancestor_depth = depths[i]
                parent = i

        depths[current_node] = max_ancestor_depth + 1
        if parent != -1:
            parent_vector[current_node] = parent

        next_node = -1

        for i in range(num_nodes + 1):
            sums[i] -= intersection[current_node, i]
            if sums[i] == 0 and visited_nodes[i] == 0 and next_node == -1:
                next_node = i

        if next_node == -1:
            break
        else:
            current_node = next_node

    return get_transitive_closure_from_ancestor_matrix(TreeUtils.get_ancestor_matrix(parent_vector, order))
