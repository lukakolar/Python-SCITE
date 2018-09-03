import numpy as np
from numba import njit


@njit(cache=True)
def get_tree_distance(first_parent_vector, second_parent_vector):
    """Calculate distance between two trees as a number of nodes in both trees that do not have the same parent in
    both trees.

    Args:
        first_parent_vector (np.ndarray): Parent vector representation of first mutation tree.
        second_parent_vector (np.ndarray): Parent vector representation of second mutation tree.

    Returns:
        int: Distance between two trees.
    """
    return np.sum(first_parent_vector != second_parent_vector)


@njit(cache=True)
def get_depth_first_traversal(parent_vector):
    """Construct depth first traversal of mutation tree. First, the parent child mapping representation of mutation
    tree is constructed from the parent vector representation. Then, traversal is constructed from the parent child
    mapping.

    Args:
        parent_vector (np.ndarray): Parent vector representation of mutation tree.

    Returns:
        np.ndarray: Depth first traversal of the mutation tree.
    """
    num_nodes = parent_vector.size

    # Construct parent child mapping
    # Index in list = parent node, list at index in list = list of children of parent node (and -1)
    # -1 is added at the beginning so that Numba can infer type of list and is later ignored
    parent_child_mapping = [[-1] for _ in range(num_nodes + 1)]

    for i in range(parent_vector.size):
        parent_child_mapping[parent_vector[i]].append(i)

    # Construct dft
    dft = np.empty((num_nodes + 1,), dtype=np.int32)
    dft_index_insert = 0

    # Construct stack
    stack = np.empty((num_nodes + 1,), dtype=np.int32)
    stack[0] = num_nodes

    # First node is always root node
    stack_index = 0

    while stack_index >= 0:
        current_node = stack[stack_index]
        stack_index -= 1
        dft[dft_index_insert] = current_node
        dft_index_insert += 1
        for i in range(1, len(parent_child_mapping[current_node])):
            stack_index += 1
            stack[stack_index] = parent_child_mapping[current_node][i]

    return dft


@njit(cache=True)
def get_non_descendants(ancestor_matrix, node):
    """Get all nodes that are not descendants of specified node (ancestors and nodes in different branches).

    Args:
        ancestor_matrix (np.ndarray): Ancestor matrix representation of mutation tree.
        node (int): Node for which non-descendants will be determined.

    Returns:
        np.ndarray: Nodes that are not descendants of specified node.
        bool: True if np.ndarray should be used, False if np.ndarray is specified only for Numba to work.
    """
    # Root node case
    if node == ancestor_matrix.shape[0]:
        return np.array([-1], dtype=np.int64), False

    # Default case
    non_descendants = np.where(ancestor_matrix[node, :] == 0)[0]

    # Add the root node
    non_descendants_and_root = np.empty((non_descendants.size + 1), dtype=np.int64)
    non_descendants_and_root[:-1] = non_descendants
    non_descendants_and_root[-1] = ancestor_matrix.shape[0]

    return non_descendants_and_root, True


@njit(cache=True)
def get_descendants(ancestor_matrix, node):
    """Get all nodes that are descendants of specified node (including self).

        Args:
            ancestor_matrix (np.ndarray): Ancestor matrix representation of mutation tree.
            node (int): Node for which descendants will be determined.

        Returns:
            np.ndarray: Nodes that are descendants of specified node (including self).
        """
    # Root node case
    if node == ancestor_matrix.shape[0]:
        return np.arange(0, ancestor_matrix.shape[0])

    # Default case
    return np.where(ancestor_matrix[node, :] == 1)[0]


@njit(cache=True)
def get_num_descendants(ancestor_matrix, node):
    """Get number of descendants (including self) of specified node.

    Args:
        ancestor_matrix (np.ndarray): Ancestor matrix representation of mutation tree.
        node (int): Node for which number of descendants will be determined.

    Returns:
        int: Number of descendants of specified node (including self).
    """
    return np.sum(ancestor_matrix[node, :])


@njit(cache=True)
def get_ancestors(ancestor_matrix, node):
    """Get all nodes that are ancestors of specified node (including self).

    Args:
        ancestor_matrix (np.ndarray): Ancestor matrix representation of mutation tree.
        node (int): Node for which ancestors will be determined.

    Returns:
        np.ndarray: Nodes that are ancestors of specified node (including self).
        bool: True if np.ndarray should be used, False if np.ndarray is specified only for Numba to work.
    """
    # Root node case
    if node == ancestor_matrix.shape[0]:
        return np.array([-1], dtype=np.int64), False

    # Default case
    return np.where(ancestor_matrix[:, node] == 1)[0], True


@njit(cache=True)
def get_ancestor_matrix(parent_vector, dft):
    """Construct ancestor matrix representation of mutation tree. For ancestor matrix A holds that if A_ij has value 1,
    node i is ancestor of node j.

    Args:
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        dft (np.ndarray): Depth first traversal of mutation tree.

    Returns:
        np.ndarray: Ancestor matrix representation of mutation tree.
    """
    num_nodes = parent_vector.size

    # For each pair of nodes, the default value is 0, meaning that first node is not the ancestor of second node
    ancestor_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    for i in range(num_nodes):
        # Go through depth first traversal in reverse order: start with leaf nodes and progress towards the root
        index = num_nodes - i

        current_node = dft[index]
        parent = parent_vector[current_node]

        # Each node is ancestor of itself
        ancestor_matrix[current_node, current_node] = 1

        # Add descendants of child node and child node itself to descendants of parent node
        if parent != num_nodes:
            ancestor_matrix[parent, :] += ancestor_matrix[current_node, :]

    return ancestor_matrix


@njit(cache=True)
def get_number_of_branches(parent_vector_matrix):
    """Get number of branches (number of leaves) in the tree.

    Args:
        parent_vector_matrix (np.ndarray): Matrix in which rows are parent vectors.

    Returns:
        np.ndarray: Vector of numbers of branches in which each number corresponds to a row in parent_vector_matrix.
    """
    num_trees, num_nodes = parent_vector_matrix.shape
    num_branches = np.empty((num_trees,), dtype=np.int32)

    for i in range(num_trees):
        # Count nodes that are never present in the parent vector - those nodes are leaves
        num_branches[i] = num_nodes + 1 - np.unique(parent_vector_matrix[i, :]).size

    return num_branches


@njit(cache=True)
def generate_random_parent_vector(num_nodes):
    """Generate random mutation tree and represent it with a parent vector.
    
    Args:
        num_nodes (int): Number of nodes in the mutation tree (excluding root node).

    Returns:
        np.ndarray: Parent vector representation of a random mutation tree.
    """
    # Root is not included in the num_nodes
    actual_num_nodes = num_nodes + 1

    # Random Pruefer sequence
    # randint - lower bound is included, upper bound is not
    # Sample from interval [1, actual_num_nodes], sample actual_num_nodes - 2 numbers
    # Subtract 1 to obtain a modified Pruefer sequence, that is appropriate for the transformation to parent vector
    random_pruefer_sequence = np.random.randint(1, actual_num_nodes + 1, actual_num_nodes - 2) - 1

    # Convert Pruefer sequence to parent vector
    return _get_parent_vector_from_pruefer_sequence(random_pruefer_sequence)


@njit(cache=True)
def _get_parent_vector_from_pruefer_sequence(sequence):
    """Construct parent vector representation of mutation tree that is represented with Pruefer sequence.

    Args:
        sequence (np.ndarray): Pruefer sequence representation of mutation tree.

    Returns:
        np.ndarray: Parent vector representation of  mutation tree.
    """
    code_length = sequence.size

    # Including the root node
    actual_num_nodes = code_length + 2

    # First node is indexed with 0
    root_node = actual_num_nodes - 1

    parent_vector = np.empty((actual_num_nodes - 1,), dtype=np.int32)

    # For each node get index of last occurrence in the code
    # Index = node, value at index = index of last occurrence in the code
    # If node does not exist in the code or node is root, value -1 is used
    last_occurrences = _get_last_occurrences(sequence)

    # Index = node, value at index = (1 -> all children have been already attached or leaf node,
    # 0 -> this node has some children that have not been attached to it yet)
    queue = _get_initial_queue(sequence)

    # Node that will be used before already selected node, -1 means that such node currently does not exist
    queue_cutter = -1

    # Get first node (lowest label) that has all children attached or is a leaf node
    next_node = _get_next_in_queue(queue, 0)

    # In each iteration add new edge
    for i in range(code_length):
        if queue_cutter != -1:
            parent_vector[queue_cutter] = sequence[i]
            queue_cutter = -1
        else:
            parent_vector[next_node] = sequence[i]
            next_node = _get_next_in_queue(queue, next_node + 1)

        # This is the last occurrence of a node in the code
        if last_occurrences[sequence[i]] == i:
            # Newly discovered node candidate does not interfere with already selected next node
            # It will be used later
            if sequence[i] >= next_node:
                queue[sequence[i]] = 1

            # Newly discovered node candidate needs to be used in the next iteration
            else:
                queue_cutter = sequence[i]

        if queue_cutter >= 0:
            parent_vector[queue_cutter] = root_node
        else:
            parent_vector[next_node] = root_node

    return parent_vector


@njit(cache=True)
def _get_last_occurrences(code):
    code_length = code.size
    last_occurrences = -1 * np.ones((code_length + 2,), dtype=np.int32)
    root_node = code_length + 1
    for i in range(code_length):
        if code[i] != root_node:
            last_occurrences[code[i]] = i

    return last_occurrences


@njit(cache=True)
def _get_initial_queue(code):
    code_length = code.size

    # "Boolean" vector, default value 1
    queue = np.ones((code_length + 2,), dtype=np.int32)

    for i in range(code_length):
        queue[code[i]] = 0

    return queue


@njit(cache=True)
def _get_next_in_queue(queue, from_index):
    candidate_indices = np.nonzero(queue[from_index:])[0]
    if candidate_indices.size == 0:
        return queue.size

    return np.nonzero(queue[from_index:])[0][0] + from_index
