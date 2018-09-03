import numpy as np


def read_mutation_matrix(filename):
    """Read the mutation matrix (D matrix) from file.

    Args:
        filename (str): Path to file that contains the data.

    Returns:
        np.ndarray: Mutation matrix (D matrix).
    """
    data = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line != "":
                line = line.split(" ")
                data.append(line)

    return np.asarray(data, dtype=np.int32)


def get_gene_names(gene_names_filename, num_nodes):
    """Read gene names from a file. Each name must be in its own line in file. Order of names should correspond to the
    order of genes in the mutation matrix.

    Args:
        gene_names_filename (str): Path to file that contains names of genes.
        num_nodes (int): Number of nodes in the mutation tree (excluding root node).

    Returns:
        list: Names of genes, where index in the list represents current label of node.
    """
    names = []

    if gene_names_filename is not None:
        with open(gene_names_filename, "r") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    names.append(line)

    else:
        for i in range(num_nodes):
            names.append(i)

    names.append("Root")

    return names


def output_graph_viz_file(output_filename, parent_vector, gene_names, add_cells_to_tree=False, attachment_matrix=None):
    """Write a mutation tree to a file.

    Args:
        output_filename (str): Path to file in which mutation tree will be stored.
        parent_vector (np.ndarray): Parent vector representation of mutation tree.
        gene_names (list): Names of genes, where index in the list represents current label of node.
        add_cells_to_tree (bool): Visualize best attachment of cells to the tree.
        attachment_matrix (np.ndarray): Specified best attachment points (nodes) for each cell.
    """
    with open(output_filename, "w") as f:
        f.write("digraph G {\n")
        f.write("node [color=deeppink4, style=filled, fontcolor=white];\n")

        for i in range(len(parent_vector)):
            f.write(str(gene_names[parent_vector[i]]) + " -> " + str(gene_names[i]) + ";\n")

        if add_cells_to_tree:
            f.write("node [color=lightgrey, style=filled, fontcolor=black];\n")
            for i in range(attachment_matrix.shape[0]):
                f.write(str(gene_names[attachment_matrix[i, 0]]) + " -> s" + str(attachment_matrix[i, 1]) + ";\n")

        f.write("}\n")
