import numpy as np
from numba import njit

import TreeUtils


@njit(cache=True)
def get_move_params_prune_and_reattach(num_nodes, parent_vector, ancestor_matrix):
    """Get move parameters for prune and reattach tree move.

    Args:
        num_nodes (int): Number of nodes in the mutation tree (excluding root node).
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        ancestor_matrix (np.ndarray): Ancestor matrix representation of mutation tree.

    Returns:
        int: Node that will be pruned and reattached together with its descendants.
        int: New attachment point of the subtree to be pruned and reattached.
        int: Current attachment point of the subtree to be pruned and reattached.
    """
    # Pick a random node that is not the root node (choose a node with label on interval [0, num_nodes - 1])
    # Node and its descendants will be pruned and reattached
    node_to_move = np.random.choice(num_nodes)

    # Possible attachment point for the selected subtree
    # Flag can be ignored, since the root node will not be selected and thus non-descendants will always exist
    possible_attachments, _ = TreeUtils.get_non_descendants(ancestor_matrix, node_to_move)

    # Pick a random attachment point among non-descendants
    new_parent = np.random.choice(possible_attachments)

    old_parent = parent_vector[node_to_move]

    return node_to_move, new_parent, old_parent


@njit(cache=True)
def get_move_params_swap_node_labels(num_nodes, ancestor_matrix):
    """Get move parameters for swap node labels tree move.

    Args:
        num_nodes (int): Number of nodes in the mutation tree (excluding root node).
        ancestor_matrix (np.ndarray): Ancestor matrix representation of mutation tree.

    Returns:
        int: Node that will be swapped and will be above below_node after the move or in different branch.
        int: Node that will be swapped and will be below above_node after the move or in different branch.
        bool: True if swapped nodes are in same lineage, False if not.
    """
    # Sample two nodes without replacement that will have labels swapped (root cannot be sampled)
    first_node, second_node = np.random.choice(num_nodes, 2, replace=False)

    same_lineage = False

    # above_node - at the end it will be above
    # below_node - at the end it will be below

    # Inverted logic because we are using the ancestor matrix before the move
    if ancestor_matrix[first_node, second_node] == 1:
        below_node = first_node
        above_node = second_node
        same_lineage = True
    elif ancestor_matrix[second_node, first_node] == 1:
        below_node = second_node
        above_node = first_node
        same_lineage = True
    else:
        # below and above do not really mean anything in this case, since nodes are not in same lineage
        below_node = first_node
        above_node = second_node

    return above_node, below_node, same_lineage


def get_move_params_swap_subtrees(num_nodes, ancestor_matrix):
    """Get move parameters for swap subtrees tree move.

    Args:
        num_nodes (int): Number of nodes in the mutation tree (excluding root node).
        ancestor_matrix (np.ndarray): Ancestor matrix representation of mutation tree.

    Returns:
        int: Root of subtree that will be swapped and will be above below_node after the tree move (if nodes are in same
            lineage).
        int: Root of subtree that will be swapped and will be below above_node after the tree move (if nodes are in same
            lineage).
        bool: True if selected nodes are in same lineage, False if not.
        int: New attachment point of subtree that was above the other subtree before the tree move.
        np.ndarray: Descendants of above_node.
        np.ndarray: Descendants of below_node.
        float: Correction for move when transition probabilities are not symmetric.
    """
    # Sample two nodes without replacement that will be roots of subtrees to be swapped (root cannot be sampled)
    first_node, second_node = np.random.choice(num_nodes, 2, replace=False)

    same_lineage = False

    # above_node - at the end it will be above
    # below_node - at the end it will be below

    # Inverted logic because we are using the ancestor matrix before the move
    if ancestor_matrix[first_node, second_node] == 1:
        below_node = first_node
        above_node = second_node
        same_lineage = True
    elif ancestor_matrix[second_node, first_node] == 1:
        below_node = second_node
        above_node = first_node
        same_lineage = True
    else:
        # below and above do not really mean anything in this case, since nodes are not in same lineage
        below_node = first_node
        above_node = second_node

    # Default value, so that Numba does not have a problem with None value
    new_parent = 0
    neighbourhood_correction = 1.0

    if same_lineage:
        above_node_descendants = TreeUtils.get_descendants(ancestor_matrix, above_node)
        below_node_descendants = TreeUtils.get_descendants(ancestor_matrix, below_node)

        # Select new attachment_point of subtree that was above the other subtree before the tree move
        new_parent = np.random.choice(above_node_descendants)

        # Calculation of neighbourhood correction
        # For more information see supplementary material of Tree inference for single-cell data
        # d(k) + 1 =  |above_node_descendants|
        # d(i) + 1 = |below_node_descendants| - |above_node_descendants|
        neighbourhood_correction = len(above_node_descendants) / \
            float(len(below_node_descendants) - len(above_node_descendants))

    return above_node, below_node, same_lineage, new_parent, neighbourhood_correction


@njit(cache=True)
def get_move_params_remove_and_insert(num_nodes, parent_vector):
    """Get move parameters for remove and insert tree move.

    Args:
        num_nodes (int): Number of nodes in the mutation tree (excluding root node).
        parent_vector (np.ndarray): Parent vector representation of mutation tree.

    Returns:
        int: Selected node that will be removed and inserted.
        int: Parent of selected node before the tree move.
        int: Parent of selected node after the tree move.
        np.ndarray: Children of parent of selected node after the tree move that will be attached to selected node
            instead to its new parent.
        float: Correction for move when transition probabilities are not symmetric.
    """
    # Select node to be removed and inserted
    # It can be any node except root
    node_to_move = np.random.choice(num_nodes)

    # Parent of selected node before the tree move
    old_parent = parent_vector[node_to_move]

    # Parent of selected node after the tree move
    # It can be any node except the selected node itself
    new_parent = np.random.choice(num_nodes)
    if new_parent >= node_to_move:
        new_parent += 1

    # Determine children old_parent after the tree move and children of new_parent before the tree move
    children_of_new_parent = [-1]
    children_of_old_parent = [-1]
    for i in range(parent_vector.size):
        if parent_vector[i] == old_parent and i != node_to_move or parent_vector[i] == node_to_move:
            children_of_old_parent.append(i)
        elif parent_vector[i] == new_parent:
            children_of_new_parent.append(i)

    # Corner case when selected node retains its parent but can change its sons
    if new_parent == old_parent:
        children_of_new_parent = children_of_old_parent.copy()

    # Default value when selected node will not have any children
    new_children = np.array([-1], dtype=np.int32)
    if len(children_of_new_parent) > 1:
        # For each possible child of selected node flip a coin to determine whether it will be an actual child
        new_children_candidates = children_of_new_parent[1:]
        which_sons_mask = np.random.random((len(new_children_candidates),)) < 0.5

        # Store children of selected node or do nothing (return default value) if no child is selected
        how_many_sons = np.sum(which_sons_mask)
        if how_many_sons > 0:
            new_children = np.empty((how_many_sons,), dtype=np.int32)
            new_sons_index = 0
            for i in range(which_sons_mask.size):
                if which_sons_mask[i]:
                    new_children[new_sons_index] = new_children_candidates[i]
                    new_sons_index += 1

    # When new_parent is selected, number of possible trees for that move will be 2^|children(new_parent)| since
    # any subset of children can be attached to selected node
    # Consequently the bigger the |children(new_parent)|, less likely is one specific tree chosen and therefore
    # correction is needed
    # If new_parent and old_parent are same node, also children_of_new_parent and children_of_old_parent are of equal
    # size and thus correction is correctly calculated for the corner case (always 1.0)
    neighbourhood_correction = np.exp2(len(children_of_new_parent) - len(children_of_old_parent))

    return node_to_move, old_parent, new_parent, new_children, neighbourhood_correction


@njit(cache=True)
def prune_and_reattach_parent_vector_in_place(parent_vector, node_to_move, new_parent):
    """Change parent vector representation of tree in-place according to specified parameters for prune and reattach
    tree move.

    Args:
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        node_to_move (int): Node that will be pruned and reattach together with its descendants.
        new_parent (int): New attachment point of the subtree to be pruned and reattached.
    """
    parent_vector[node_to_move] = new_parent


@njit(cache=True)
def prune_and_reattach_parent_vector_revert_in_place(parent_vector, node_to_move, old_parent):
    """Change parent vector representation of tree in-place such that the parent vector will be the same as before
    the prune and reattach tree move.

    Args:
        parent_vector (np.ndarray): Parent vector representation of mutation tree after the prune and reattach tree
            move.
        node_to_move (int): Node that was to be pruned and reattached together with its descendants.
        old_parent (int): Attachment point of the subtree before the tree move.
    """
    parent_vector[node_to_move] = old_parent


@njit(cache=True)
def prune_and_reattach_ancestor_matrix_in_place(ancestor_matrix, node_to_move, new_parent, old_parent):
    """Change ancestor matrix representation of tree in-place according to specified parameters for prune and reattach
    tree move.

    Args:
        ancestor_matrix (np.ndarray): Ancestor matrix representation of mutation tree.
        node_to_move (int): Node that will be pruned and reattached together with its descendants.
        new_parent (int): New attachment point of the subtree to be pruned and reattached.
        old_parent (int): Current attachment point of the subtree to be pruned and reattached.
    """
    ancestors_old_attachment, flag_1 = TreeUtils.get_ancestors(ancestor_matrix, old_parent)
    ancestors_new_attachment, flag_2 = TreeUtils.get_ancestors(ancestor_matrix, new_parent)

    # Mask representing descendants of node_to_move (including self)
    or_mask = ancestor_matrix[node_to_move, :]

    # Mask representing non-descendants of node_to_move
    and_mask = 1 - or_mask

    # Remove descendants of node_to_move (including self) from being descendants of ancestors of old_parent
    if flag_1:
        ancestor_matrix[ancestors_old_attachment, :] = \
            np.logical_and(ancestor_matrix[ancestors_old_attachment, :], and_mask).astype(np.int32)

    # Add descendants of node_to_move (including sel) to also be descendants of ancestors of new_parent
    if flag_2:
        ancestor_matrix[ancestors_new_attachment, :] = \
            np.logical_or(ancestor_matrix[ancestors_new_attachment, :], or_mask).astype(np.int32)


@njit(cache=True)
def swap_node_labels_parent_vector(parent_vector, first_node, second_node):
    """Construct parent vector representation of tree according to specified parameters for swap node labels tree move.

    Args:
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        first_node (int): First node whose label will be swapped.
        second_node (int): Second node whose label will be swapped.

    Returns:
        np.ndarray: Parent vector representation of mutation tree after the tree move.
    """
    proposed_parent_vector = np.copy(parent_vector)

    first_parent = parent_vector[first_node]
    second_parent = parent_vector[second_node]

    # Change parents of children
    for i in range(parent_vector.size):
        if parent_vector[i] == first_node:
            proposed_parent_vector[i] = second_node
        elif parent_vector[i] == second_node:
            proposed_parent_vector[i] = first_node

    # Change parents of swapped nodes
    proposed_parent_vector[first_node] = second_parent
    proposed_parent_vector[second_node] = first_parent

    # Handle case where one node is parent of another
    if proposed_parent_vector[first_node] == first_node:
        proposed_parent_vector[first_node] = second_node

    if proposed_parent_vector[second_node] == second_node:
        proposed_parent_vector[second_node] = first_node

    return proposed_parent_vector


@njit(cache=True)
def swap_node_labels_ancestor_matrix_in_place(ancestor_matrix, above_node, below_node):
    """Change ancestor matrix representation of tree in-place according to specified parameters for swap node labels
    tree move.

    Args:
        ancestor_matrix (np.ndarray): Ancestor matrix representation of mutation tree.
        above_node (int): Node that will be swapped and will be above below_node after the tree move.
        below_node (int): Node that will be swapped and will be below above_node after the tree move.
    """
    # Swap descendants
    temporary_vector = np.copy(ancestor_matrix[above_node, :])
    ancestor_matrix[above_node, :] = ancestor_matrix[below_node, :]
    ancestor_matrix[below_node, :] = temporary_vector

    # Swap ancestors
    temporary_vector = np.copy(ancestor_matrix[:, above_node])
    ancestor_matrix[:, above_node] = ancestor_matrix[:, below_node]
    ancestor_matrix[:, below_node] = temporary_vector


@njit(cache=True)
def swap_subtrees_parent_vector(parent_vector, above_node, below_node, same_lineage, new_parent):
    """Construct parent vector representation of tree according to specified parameters for swap subtrees tree move.

    Args:
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        above_node (int): Root of subtree that will be swapped and will be above below_node after the tree move
            (if nodes are in same lineage).
        below_node (int): Root of subtree that will be swapped and will be below above_node after the tree move
            (if nodes are in same lineage).
        same_lineage (bool): True if selected nodes are in same lineage, False if not.
        new_parent (int): New attachment point of subtree that was above the other subtree before the tree move.

    Returns:
        np.ndarray: Parent vector representation of mutation tree after the tree move.
    """
    # above_node - at the end it will be above
    # below_node - at the end it will be below

    proposed_parent_vector = np.copy(parent_vector)

    if same_lineage:
        # Attach subtree that is currently below the other subtree to where the other subtree is attached
        proposed_parent_vector[above_node] = parent_vector[below_node]

        # Attach other subtree to one of the descendants of the first subtree
        proposed_parent_vector[below_node] = new_parent

    else:
        # Similar to prune and reattach tree move
        proposed_parent_vector[below_node] = parent_vector[above_node]
        proposed_parent_vector[above_node] = parent_vector[below_node]

    return proposed_parent_vector


@njit(cache=True)
def remove_and_insert_parent_vector(parent_vector, node_to_move, old_parent, new_parent, new_children):
    """Construct parent vector representation of tree according to specified parameters for remove and insert tree move.

    Args:
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        node_to_move (int): Selected node that will be removed and inserted.
        old_parent (int): Parent of selected node before the tree move.
        new_parent (int): Parent of selected node after the tree move.
        new_children (np.ndarray): Children of parent of selected node after the tree move that will be attached to
            selected node instead to its new parent.

    Returns:
        np.ndarray: Parent vector representation of mutation tree after the tree move.
    """

    proposed_parent_vector = np.copy(parent_vector)

    # Construct a O(1) lookup to whether a node is in new_children
    is_new_child = np.zeros((parent_vector.size,), dtype=np.int32)

    for new_child in new_children:
        if new_child != -1:
            is_new_child[new_child] = 1

    # Change parent vector
    for i in range(parent_vector.size):
        # Attach children of selected of node to its parent
        if parent_vector[i] == node_to_move and not is_new_child[i]:
            proposed_parent_vector[i] = old_parent

        # Attach some of the children of new parent to selected node
        elif is_new_child[i]:
            proposed_parent_vector[i] = node_to_move

    # Attach selected node to new parent
    proposed_parent_vector[node_to_move] = new_parent

    return proposed_parent_vector
