import numpy as np
from numba import njit
from scipy.stats import beta as beta_distribution

import MiscUtils
import TreeUtils

TOLERANCE = 1e-5


@njit(cache=True)
def calculate_beta_distribution_parameters(beta_prior_mean, beta_prior_stdev):
    """Calculate beta distribution parameters for beta.

    Args:
        beta_prior_mean (float): Beta mean prior.
        beta_prior_stdev (float): Beta standard deviation prior.

    Returns:
        list: Beta distribution parameters for beta.
    """
    beta_distribution_alpha_prior = \
        ((1 - beta_prior_mean) * beta_prior_mean ** 2) / beta_prior_stdev ** 2 - beta_prior_mean
    beta_distribution_beta_prior = beta_distribution_alpha_prior * (1 / beta_prior_mean - 1)

    return [beta_distribution_alpha_prior, beta_distribution_beta_prior]


def get_beta_score(value, params):
    """Evaluate beta with beta distribution.

    Args:
        value (float): Beta value.
        params (list): Parameters of beta distribution for beta.

    Returns:
        float: Beta log score.
    """
    # _pdf is used instead of pdf because it works faster and additional checks are not needed
    return np.log(beta_distribution._pdf(value, params[0], params[1]))


@njit(cache=True)
def get_log_scores_matrix(d0e1, d1e0, d2e0, d2e1):
    """Construct log scores matrix in which log probabilities of all error probabilities are provided.

    Args:
        d0e1 (float): Probability P(D_ij = 0|E_ij = 1).
        d1e0 (float): Probability P(D_ij = 1|E_ij = 0).
        d2e0 (float): Probability P(D_ij = 2|E_ij = 0).
        d2e1 (float): Probability P(D_ij = 2|E_ij = 1).

    Returns:
        np.ndarray: Log scores matrix corresponding to error probabilities.
    """
    log_scores_matrix = np.zeros((4, 2), dtype=np.float64)

    # P(D_ij = 0|E_ij = 0)
    log_scores_matrix[0, 0] = np.log(1.0 - d1e0 - d2e0)

    # P(D_ij = 1|E_ij = 0)
    log_scores_matrix[1, 0] = np.log(d1e0)

    # Avoid computation of log(0)
    if d2e0 > 0.0:
        # P(D_ij = 2|E_ij = 0)
        log_scores_matrix[2, 0] = np.log(d2e0)

    # P(D_ij = 3|E_ij = 0)
    # Value N/A = 0.0

    # P(D_ij = 0|E_ij = 1)
    log_scores_matrix[0, 1] = np.log(d0e1)

    # P(D_ij = 1|E_ij = 1)
    log_scores_matrix[1, 1] = np.log(1.0 - d0e1 - d2e1)

    # Avoid computation of log(0)
    if d2e1 > 0.0:
        # P(D_ij = 2|E_ij = 1)
        log_scores_matrix[2, 1] = np.log(d2e1)

    # P(D_ij = 3|E_ij = 1)
    # Value N/A = 0.0

    return log_scores_matrix


@njit(cache=True)
def get_updated_log_scores_matrix(log_scores_matrix, new_beta):
    """Construct updated log scores matrix in which log probabilities of all error probabilities are provided.

    Args:
        log_scores_matrix (np.ndarray): A basis for the updated log scores matrix.
        new_beta (float): New beta, from which some error probabilities will be calculated.

    Returns:
        np.ndarray: Updated log scores matrix.
    """
    new_log_scores_matrix = np.copy(log_scores_matrix)

    # Homozygous mutations are present
    if new_log_scores_matrix[2, 1] != 0:
        # Beta probability is equally distributed among P(D_ij = 0|E_ij = 1) and P(D_ij = 2|E_ij = 1)
        new_log_scores_matrix[0, 1] = np.log(new_beta / 2.0)
        new_log_scores_matrix[2, 1] = np.log(new_beta / 2.0)
    # Homozygous mutations are not present
    else:
        # P(D_ij = 0|E_ij = 1)
        new_log_scores_matrix[0, 1] = np.log(new_beta)

    # P(D_ij=1 | E_ij = 1)
    new_log_scores_matrix[1, 1] = np.log(1 - new_beta)

    return new_log_scores_matrix


@njit(cache=True)
def get_fast_tree_score(mutation_matrix, mask_matrix, parent_vector, dft, log_scores, best_tree_log_score):
    """Quickly evaluate mutation tree. There may be a small discrepancy between fast score and the accurate score.

    Score is calculated in fast matrix for all cells at the same time. Each row represents a mutation and each column
    represents a cell.

    Probability of attachment of a cell to the root node is obtained by summing log probabilities (effectively
    multiplication of probabilities) for each mutation entry in the mutation profile of the cell. In the root node there
    should not be any mutation. Therefore, for root node all probabilities look like P(D_ij = x|E_ij = 0). This means
    that we can use only first column of the log_scores matrix. This way we obtain the probability
    PI(i=1 -> n) P(D_ij|A(T)_i ro_j). This calculation is done in the root_matrix.

    Probabilities for other nodes can be calculated from probabilities of their parent nodes. Parent and child node
    differ only in a single mutation (mutation that is present by the child node). For a child node we can replace the
    probability P(D_ij = x|E_ij = 0) with the probability P(D_ij = x|E_ij = 1) and thus obtain the probability for the
    child node. x is determined by the data point in the mutation matrix (mutation that is present in the child node or
    absence of mutation for the cell for which the probabilities are calculated).

    The change from P(D_ij = x|E_ij = 0) to P(D_ij = x|E_ij = 1) can be achieved by subtracting the former probability
    and adding the latter probability. Same can be achieved by adding (P(D_ij = x|E_ij = 1) - P(D_ij = x|E_ij = 0)).
    This subtraction is represented by log_scores_condensed.

    Cell in fast_matrix at position ij represents attachment of cell j to node i or probability
    PI(i=1 -> n) P(D_ij|A(T)_i ro_j).

    To visit all nodes in such an order that parent is always evaluated before the child, depth first traversal is
    used.

    Args:
        mutation_matrix (np.ndarray): Mutation matrix (D matrix).
        mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
                P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        dft (np.ndarray): Depth first traversal of mutation tree.
        log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.
        best_tree_log_score (float): Best tree log score so far.

    Returns:
        float: Fast score of mutation tree.
        np.ndarray: fast_matrix containing probabilities of type PI(i=1 -> n) P(D_ij|A(T)_i ro_j) that relate to the
            attachment of every cell to every node.
    """
    num_nodes, num_cells = mutation_matrix.shape

    # Condensed log scores reflect the change in probability when transitioning from parent node to child node
    log_scores_condensed = log_scores[:, 1] - log_scores[:, 0]

    # Used for the calculation of probabilities for the root node
    log_scores_first_column = log_scores[:, 0]

    # Matrix that will be used for the calculation of root probabilities
    root_matrix = np.empty((num_nodes, num_cells), dtype=np.float64)

    # Replace each entry in mutation matrix with respective condensed log score to construct fast_matrix and replace
    # each entry in mutation matrix with first column of log_scores_matrix to obtain root_matrix
    fast_matrix = np.empty((num_nodes + 1, num_cells), dtype=np.float64)
    for i in range(num_nodes):
        for j in range(num_cells):
            fast_matrix[i, j] = log_scores_condensed[mutation_matrix[i, j]]
            root_matrix[i, j] = log_scores_first_column[mutation_matrix[i, j]]

    # Obtain root probabilities for all cells by summing log probabilities P(D_ij = x|E_ij = 0), where each probability
    # reflects one mutation in the mutation profile of the cell
    fast_matrix[num_nodes, :] = np.sum(root_matrix, axis=0)

    # Use DFS to visit all nodes
    # Parent will be always evaluated before child, so we can calculate the score of the child from the score of the
    # parent, root is already evaluated so it can be skipped
    for i in dft[1:]:
        for j in range(num_cells):
            fast_matrix[i, j] += fast_matrix[parent_vector[i], j]

    fast_score = get_score_from_fast_matrix(fast_matrix)

    return get_accurate_score_and_fast_matrix_if_needed(mask_matrix,
                                                        parent_vector,
                                                        dft,
                                                        log_scores,
                                                        fast_score,
                                                        fast_matrix,
                                                        best_tree_log_score)


@njit(cache=True)
def get_partial_tree_score_prune_and_reattach(mutation_matrix,
                                              mask_matrix,
                                              parent_vector,
                                              dft,
                                              ancestor_matrix,
                                              log_scores,
                                              fast_matrix,
                                              best_tree_log_score,
                                              node_to_move):
    """Get partial tree score for prune and reattach tree move."""
    return get_partial_tree_score(mutation_matrix,
                                  mask_matrix,
                                  parent_vector,
                                  dft,
                                  log_scores,
                                  fast_matrix,
                                  node_to_move,
                                  -1,
                                  TreeUtils.get_num_descendants(ancestor_matrix, node_to_move),
                                  -1,
                                  best_tree_log_score)


@njit(cache=True)
def get_partial_tree_score_swap_node_labels(mutation_matrix,
                                            mask_matrix,
                                            parent_vector,
                                            dft,
                                            ancestor_matrix,
                                            log_scores,
                                            fast_matrix,
                                            best_tree_log_score,
                                            above_node,
                                            below_node,
                                            same_lineage):
    """Get partial tree score for swap node labels tree move."""
    if same_lineage:
        return get_partial_tree_score(mutation_matrix,
                                      mask_matrix,
                                      parent_vector,
                                      dft,
                                      log_scores,
                                      fast_matrix,
                                      above_node,
                                      -1,
                                      TreeUtils.get_num_descendants(ancestor_matrix, below_node),
                                      -1,
                                      best_tree_log_score)
    else:
        return get_partial_tree_score(mutation_matrix,
                                      mask_matrix,
                                      parent_vector,
                                      dft,
                                      log_scores,
                                      fast_matrix,
                                      above_node,
                                      below_node,
                                      TreeUtils.get_num_descendants(ancestor_matrix, below_node),
                                      TreeUtils.get_num_descendants(ancestor_matrix, above_node),
                                      best_tree_log_score)


@njit(cache=True)
def get_partial_tree_score_swap_subtrees(mutation_matrix,
                                         mask_matrix,
                                         parent_vector,
                                         dft,
                                         ancestor_matrix,
                                         log_scores,
                                         fast_matrix,
                                         best_tree_log_score,
                                         above_node,
                                         below_node,
                                         same_lineage):
    """Get partial tree score for swap subtrees tree move."""
    if same_lineage:
        return get_partial_tree_score(mutation_matrix,
                                      mask_matrix,
                                      parent_vector,
                                      dft,
                                      log_scores,
                                      fast_matrix,
                                      above_node,
                                      -1,
                                      TreeUtils.get_num_descendants(ancestor_matrix, below_node),
                                      -1,
                                      best_tree_log_score)
    else:
        return get_partial_tree_score(mutation_matrix,
                                      mask_matrix,
                                      parent_vector,
                                      dft,
                                      log_scores,
                                      fast_matrix,
                                      above_node,
                                      below_node,
                                      TreeUtils.get_num_descendants(ancestor_matrix, above_node),
                                      TreeUtils.get_num_descendants(ancestor_matrix, below_node),
                                      best_tree_log_score)


@njit(cache=True)
def get_partial_tree_score(mutation_matrix,
                           mask_matrix,
                           parent_vector,
                           dft,
                           log_scores,
                           fast_matrix,
                           first_affected_nodes_root,
                           second_affected_nodes_root,
                           first_affected_nodes_count,
                           second_affected_nodes_count,
                           best_tree_log_score):
    """Evaluate the mutation tree, but only nodes that have been affected by the tree move are reevaluated.
    For detailed explanation of the fast_matrix please see docstring of get_fast_tree_score.

    Args:
        mutation_matrix (np.ndarray): Mutation matrix (D matrix).
        mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
                P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        dft (np.ndarray): Depth first traversal of mutation tree.
        log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.
        fast_matrix (np.ndarray): Fast matrix corresponding to mutation tree and beta before the tree move.
        first_affected_nodes_root (int): Root of first subtree of affected nodes.
        second_affected_nodes_root (int): Root of second subtree of affected nodes (-1 if that root does not exist).
        first_affected_nodes_count (int): Number of nodes in first subtree of affected nodes.
        second_affected_nodes_count (int): Number of nodes in second subtree of affected nodes (if
            second_affected_nodes_root is set to -1, this variable does not mean anything).
        best_tree_log_score (float): Best tree log score so far.

    Returns:
        float: Fast partial_score of mutation tree.
        np.ndarray: fast_matrix containing probabilities of type PI(i=1 -> n) P(D_ij|A(T)_i ro_j) that relate to the
            attachment of every cell to every node.
    """
    fast_matrix_copy = np.copy(fast_matrix)

    num_nodes, num_cells = mutation_matrix.shape

    # Condensed log scores reflect the change in probability when transitioning from parent node to child node
    log_scores_condensed = log_scores[:, 1] - log_scores[:, 0]

    # Nodes left in the subtree of affected nodes
    current_root_num_descendants = 0

    for current_node in dft[1:]:
        # Found root of affected nodes subtree
        if first_affected_nodes_root == current_node:
            current_root_num_descendants = first_affected_nodes_count
        elif second_affected_nodes_root == current_node:
            current_root_num_descendants = second_affected_nodes_count

        # If nodes are affected, reevaluate that part of fast matrix
        if current_root_num_descendants > 0:
            current_root_num_descendants -= 1
            current_node_parent = parent_vector[current_node]
            for j in range(num_cells):
                fast_matrix_copy[current_node, j] = log_scores_condensed[mutation_matrix[current_node, j]]
                fast_matrix_copy[current_node, j] += fast_matrix_copy[current_node_parent, j]

    partial_score = get_score_from_fast_matrix(fast_matrix_copy)

    return get_accurate_score_and_fast_matrix_if_needed(mask_matrix,
                                                        parent_vector,
                                                        dft,
                                                        log_scores,
                                                        partial_score,
                                                        fast_matrix_copy,
                                                        best_tree_log_score)


@njit(cache=True)
def get_mask_matrix(mutation_matrix):
    """Instead of directly subtracting and adding probabilities as in get_fast_tree_score, a 'count of probabilities' of
    type P(D_ij = x|E_ij = 0) and P(D_ij = x|E_ij = 1) is kept. Since there are 8 possible probabilities,
    there is a vector with 8 elements for every attachment of cell to every node. Therefore, instead of fast_matrix,
    mask_matrix is used. At the end counts are multiplied with log probabilities (log_scores_vector) to obtain
    fast_matrix (this is done in get_accurate_tree_score).

    Args:
        mutation_matrix (np.ndarray): Mutation matrix (D matrix).

    Returns:
        np.ndarray: Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and P(D_ij = x|E_ij = 1) from which
            fast_matrix can be obtained if log scores matrix is known.
    """
    num_nodes, num_cells = mutation_matrix.shape

    # Matrix of rows in which each row represents change from P(D_ij = x|E_ij = 0) to P(D_ij = x|E_ij = 1), where row
    # index = x
    # By adding the correct row to the vector of a parent node we can obtain correct vector of counts of the child node
    type_to_mask = np.zeros((4, 8), dtype=np.int32)
    for i in range(4):
        type_to_mask[i, i] = -1
        type_to_mask[i, i + 4] = 1

    # Dimensions n x m x 8
    # Every attachment of a cell to a node is represented by a vector with 8 elements
    mask_matrix = np.empty((mutation_matrix.shape[0] + 1, mutation_matrix.shape[1], 8), dtype=np.int32)

    for j in range(mutation_matrix.shape[1]):
        # For each cell add count of each type of mutation in the mutation profile of the cell
        # Counts are stored in a vector with 8 elements (only first 4 elements are non-zero, because there should not
        # be any mutation present in the root node)
        counts = np.bincount(mutation_matrix[:, j])
        mask_matrix[num_nodes, j, :] = np.hstack((counts, np.zeros(8 - counts.shape[0], dtype=np.int32)))

        # For each node add a mask that represents a change from P(D_ij = x|E_ij = 0) to P(D_ij = x|E_ij = 1)
        for i in range(mutation_matrix.shape[0]):
            mask_matrix[i, j, :] = type_to_mask[mutation_matrix[i, j], :]

    return mask_matrix


@njit(cache=True)
def get_accurate_tree_score(mask_matrix, parent_vector, dft, log_scores):
    """Accurately evaluate mutation tree.

    For detailed explanation of the fast_matrix please see docstring of get_fast_tree_score.

    Args:
        mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
            P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        dft (np.ndarray): Depth first traversal of mutation tree.
        log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.

    Returns:
        float: Accurate score of mutation tree.
        np.ndarray: fast_matrix containing probabilities of type PI(i=1 -> n) P(D_ij|A(T)_i ro_j) that relate to the
            attachment of every cell to every node.
    """
    # Create a copy
    mask_matrix_copy = np.copy(mask_matrix)

    # Use DFS to visit all nodes
    # Parent will be always evaluated before child, so we can calculate the score of the child from the score of the
    # parent, root is already evaluated so it can be skipped
    for i in dft[1:]:
        mask_matrix_copy[i, :, :] += mask_matrix_copy[parent_vector[i], :, :]

    # Transformation of log_scores_matrix to analogous vector representation
    log_scores_vector = np.hstack((log_scores[:, 0], log_scores[:, 1]))

    # Transform counts to probabilities by using log_scores_vector
    fast_matrix = MiscUtils.dot_3d(mask_matrix_copy, log_scores_vector)

    return get_score_from_fast_matrix(fast_matrix), fast_matrix


@njit(cache=True)
def get_score_from_fast_matrix(fast_matrix):
    """Get tree log score from fast matrix. In order to sum over probabilities, exponent has to be used to obtain
    probabilities from log probabilities. Then probabilities are summed to obtain
    SIGMA(ro_j=1 -> n+1) [PI(i=1 -> n) P(D_ij|A(T)_i ro_j)]. Lastly, log and sum are used instead of multiplication to
    obtain PI(j=1 -> m) SIGMA(ro_j=1 -> n+1) [PI(i=1 -> n) P(D_ij|A(T)_i ro_j)].
    For more information on fast matrix please see docstring of get_fast_tree_score.

    Args:
        fast_matrix (np.ndarray): Fast matrix corresponding to mutation tree and beta.

    Returns:
        float: Tree log score corresponding to fast matrix, which summarizes information about current tree and beta.
    """
    tree_log_score = np.sum(np.log(np.sum(np.exp(fast_matrix), axis=0)))

    return tree_log_score


@njit(cache=True)
def check_accept_move(proposed_log_score, current_log_score, neighbourhood_correction=1.0):
    """Accept or reject move according to Metropolis-Hastings algorithm.

    Args:
        proposed_log_score (float): Proposed log score (tree or combined).
        current_log_score (float): Log score of current state (tree or combined).
        neighbourhood_correction (float): correction for moves for which transition probabilities are not always
            symmetric.

    Returns:
        bool: True if move is accepted, False if not.
    """
    return np.random.random() < np.exp(proposed_log_score - current_log_score) * neighbourhood_correction


@njit(cache=True)
def get_accurate_score_and_fast_matrix_if_needed(mask_matrix,
                                                 parent_vector,
                                                 dft,
                                                 log_scores,
                                                 current_tree_log_score,
                                                 current_fast_matrix,
                                                 best_tree_log_score):
    """Calculate accurate score if current tree and beta are the best so far or only slightly worse."""
    if best_tree_log_score != 1 and current_tree_log_score > best_tree_log_score - TOLERANCE:
        accurate_score, _ = get_accurate_tree_score(mask_matrix,
                                                    parent_vector,
                                                    dft,
                                                    log_scores)
        return accurate_score, current_fast_matrix

    return current_tree_log_score, current_fast_matrix


def get_attachment_matrix(mask_matrix, parent_vector, dft, beta, log_scores):
    """Get matrix of pairs (A, B) where A is a mutation (node) and B is cell. Pairs are stored in rows of matrix. For
    each cell B it holds that mutation (node) A is the best attachment point for that cell. However, for some cells
    there may exist multiple equally good best attachment points.

    Args:
        mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
            P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        dft (np.ndarray): Depth first traversal of mutation tree.
        beta (float): Beta value.
        log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.

    Returns:
        np.ndarray: Matrix of pairs (A, B) that represent best attachment points for all cells.
    """
    current_log_scores = get_updated_log_scores_matrix(log_scores, beta)

    # Construct fast_matrix from which best attachment points can be inferred
    _, fast_matrix = get_accurate_tree_score(mask_matrix,
                                             parent_vector,
                                             dft,
                                             current_log_scores)

    # First for every cell the highest attachment probability among all nodes is determined
    # Then it is determined which combinations of nodes and cells have the aforementioned highest probabilities
    return np.argwhere(fast_matrix == np.amax(fast_matrix, axis=0))
