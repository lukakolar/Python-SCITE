import time

import numpy as np

import ESSUtils
import FileUtils
import MiscUtils
import MoveUtils
import ScoreUtils
import TreeUtils


class MHRunner:
    def __init__(self, mutation_matrix_filename):
        """Initialization. Throughout the class beta means the probability of false negative, which means the
        probability of not observing mutation, even though the mutation is present. Beta consists of probabilities
        P(D_ij = 0|E_ij = 1) and P(D_ij = 2|E_ij = 1). All trees are mutation trees.

        Args:
            mutation_matrix_filename (str): Path to file that contains the data. File should contain numbers
                0, 1, 2, 3; numbers must be separated with spaces. Each line (row) represents a mutation and each
                column represents a cell (mutation profile).
        """
        # Read the mutation matrix from file
        self.mutation_matrix = FileUtils.read_mutation_matrix(mutation_matrix_filename)

        # Construct mask matrix
        self.mask_matrix = ScoreUtils.get_mask_matrix(self.mutation_matrix)

        # Store dimensions
        self.num_nodes, self.num_cells = self.mutation_matrix.shape

    def run_mh(self,
               num_repetitions,
               chain_length,
               d0e1_probability,
               d1e0_probability,
               d2e0_probability=0.0,
               d2e1_probability=0.0,
               burn_in_proportion=0.25,
               sampling_enabled=False,
               beta_move_probability=0.0,
               mh_jump_scaling=10.0,
               beta_prior_stdev=0.1,
               gene_names_filename=None,
               store_best_trees=True,
               max_best_trees_stored=100,
               add_cells_to_best_trees=True,
               prune_reattach_probability=0.55,
               swap_node_labels_probability=0.4,
               swap_subtrees_probability=0.05,
               remove_and_insert_probability=0.0,
               silent=False,
               output_name=""):
        """Run Metropolis-Hastings algorithm with the data provided in the initialization.

        Args:
            num_repetitions (int): Number of repetitions of the Metropolis-Hastings algorithm.
            chain_length (int): Length of chain (number of steps) in each repetition.
            d0e1_probability (float): Probability P(D_ij = 0|E_ij = 1).
            d1e0_probability (float): Probability P(D_ij = 1|E_ij = 0).
            d2e0_probability (float): Probability P(D_ij = 2|E_ij = 0).
            d2e1_probability (float): Probability P(D_ij = 2|E_ij = 1).
            burn_in_proportion (float): Proportion of initial steps that are deemed burn in phase.
            sampling_enabled (bool): Sampling of the posterior distribution. If enabled, in each step, current tree and
                beta are stored.
            beta_move_probability (float): Probability of choosing beta move instead of a tree move.
            mh_jump_scaling (float): Scaling of beta jump standard deviation in relation to beta prior standard
                deviation (beta jump standard deviation = beta prior standard deviation / mh_jump_scaling).
            beta_prior_stdev (float): Beta standard deviation prior.
            gene_names_filename (str): Path to file that contains gene names of genes in the mutation matrix. Each name
                must be in its own line in file. Order of names should correspond to the order of genes in the mutation
                matrix.
            store_best_trees (bool): Store best MAP trees encountered during the algorithm execution. Trees are written
                to separate files in Graphviz format and are stored in folder best_trees.
            max_best_trees_stored (int): Maximum number of best trees stored to folder best_trees if store_best_trees
                is set to True.
            add_cells_to_best_trees: Find best attachment of all cells when the algorithm terminates.
            prune_reattach_probability (float): Probability of choosing prune and reattach tree move when tree move is
                used.
            swap_node_labels_probability (float): Probability of choosing swap node labels tree move when tree move is
                used.
            swap_subtrees_probability (float): Probability of choosing swap subtrees tree move when tree move is used.
            remove_and_insert_probability (float): Probability of choosing remove and insert tree move when tree move is
                used.
            silent (bool): If True, output less intermediate information.
            output_name (str): String used for naming best trees and posterior samples files.
        """
        # Time spent in optimal states after the burn in phase
        time_in_optimal_states_after_burn_in = 0

        # Number of steps that are in the burn-in phase and number of steps that are not in the-burn in phase
        num_burn_in_steps = int(burn_in_proportion * chain_length)
        num_non_burn_in_steps = chain_length - num_burn_in_steps

        # Reserve space for posterior distributions
        posterior_trees = None
        posterior_betas = None
        if sampling_enabled:
            num_posterior_samples = num_repetitions * num_non_burn_in_steps
            posterior_trees = np.empty((num_posterior_samples, self.num_nodes), dtype=np.int32)
            posterior_betas = np.empty((num_posterior_samples,), dtype=np.float64)

        # Check that provided error probabilities are valid
        self.check_error_probabilities(d0e1_probability,
                                       d1e0_probability,
                                       d2e0_probability,
                                       d2e1_probability)

        # Check that provided move probabilities are valid or correct them if they are not
        prune_reattach_probability, swap_node_labels_probability, swap_subtrees_probability, \
            remove_and_insert_probability, beta_move_probability = \
            self.check_move_probabilities(prune_reattach_probability,
                                          swap_node_labels_probability,
                                          swap_subtrees_probability,
                                          remove_and_insert_probability,
                                          beta_move_probability)

        # Estimated beta (false negative probability) mean and standard deviation
        beta_prior_mean = float(d0e1_probability + d2e1_probability)
        beta_prior_stdev = float(beta_prior_stdev)

        # Calculation of beta distribution parameters for beta
        beta_params = ScoreUtils.calculate_beta_distribution_parameters(beta_prior_mean, beta_prior_stdev)

        # Scaling of beta jump standard deviation in relation to beta prior standard deviation
        jump_stdev = beta_prior_stdev / float(mh_jump_scaling)

        # Best results
        # Value 1 means that best score is not yet present (equivalent of None)
        # Integer default value is used so that variables can be used in @njit(cache=True) annotated functions
        best_tree_log_score = 1
        best_combined_log_score = 1
        best_beta = beta_prior_mean

        # Define a variable here so that attachment matrix can be calculated
        log_scores = None

        # Initialize object in which best trees and betas are stored
        best_results = BestResults(max_best_trees_stored, self.num_nodes)

        for repetition in range(num_repetitions):
            # Print information to console regarding current repetition
            print("Repetition: ", repetition + 1)
            if not silent:
                print("{:>25} {:>25} {:>25} {:>25}"
                      .format("num steps", "best_tree_log_score", "best_beta", "best_combined_log_score"))

            # Initialize a tabras
            # Mutation tree is initialized as a random mutation tree
            # Beta is initialized as beta_prior_mean
            tabras = TreeAndBetaRepresentationsAndScores(self.num_nodes, beta_prior_mean)

            # Construct log scores matrix
            # Each entry in the log scores matrix corresponds to one of the log probabilities P(D_ij = x|E_ij = y)
            log_scores = ScoreUtils.get_log_scores_matrix(d0e1_probability,
                                                          d1e0_probability,
                                                          d2e0_probability,
                                                          d2e1_probability)

            # Evaluate initialized tabras
            tabras.calculate_initial_scores(self.mutation_matrix,
                                            self.mask_matrix,
                                            log_scores,
                                            beta_move_probability,
                                            beta_params)

            for step in range(chain_length):
                # Output intermediate information
                if not silent and (step == 1 or step > 0 and step % 10000 == 0):
                    print("{:>25} {:>25.15f} {:>25.15f} {:>25.15f}"
                          .format(str(step), best_tree_log_score, best_beta, best_combined_log_score))

                # Beta move is chosen
                if beta_move_probability > 0 and np.random.random() < beta_move_probability:
                    log_scores = tabras.propose_new_beta(self.mutation_matrix,
                                                         self.mask_matrix,
                                                         log_scores,
                                                         best_tree_log_score,
                                                         jump_stdev,
                                                         beta_params)

                # Tree move is chosen
                else:
                    tabras.propose_new_tree(self.mutation_matrix,
                                            self.mask_matrix,
                                            log_scores,
                                            best_tree_log_score,
                                            prune_reattach_probability,
                                            swap_node_labels_probability,
                                            swap_subtrees_probability)

                # Update optimal trees if current tree is optimal (at least currently)
                if store_best_trees:
                    best_results.update_results(tabras.parent_vector,
                                                tabras.beta,
                                                tabras.combined_log_score,
                                                best_combined_log_score)

                # Store tree and beta for future sampling from posterior distribution
                if sampling_enabled and step >= num_burn_in_steps:
                    posterior_index = repetition * num_non_burn_in_steps + step - num_burn_in_steps
                    posterior_trees[posterior_index, :] = tabras.parent_vector
                    posterior_betas[posterior_index] = tabras.beta

                # Update log scores if current tree and beta are the best until now
                if best_combined_log_score == 1 or tabras.combined_log_score > best_combined_log_score:
                    time_in_optimal_states_after_burn_in = 0
                    best_tree_log_score = tabras.tree_log_score
                    best_combined_log_score = tabras.combined_log_score
                    best_beta = tabras.beta

                if tabras.combined_log_score == best_combined_log_score and step >= num_burn_in_steps:
                    time_in_optimal_states_after_burn_in += 1

            print("{:>25} {:>25.15f} {:>25.15f} {:>25.15f}"
                  .format(chain_length, best_tree_log_score, best_beta, best_combined_log_score))

        print("Number of steps in optimal states after burn-in: {0}".format(time_in_optimal_states_after_burn_in))

        if output_name != "":
            added_string = "_" + output_name
        else:
            added_string = ""

        # Store samples to a file
        if sampling_enabled:
            np.save("posterior_samples/trees" + added_string + ".npy", posterior_trees)
            np.save("posterior_samples/betas" + added_string + ".npy", posterior_betas)

        if store_best_trees:
            # Read gene names
            gene_names = FileUtils.get_gene_names(gene_names_filename, self.num_nodes)

            # Write best trees to files
            best_trees, best_beta = best_results.get_best_results()
            num_best_results = min(best_trees.shape[0], max_best_trees_stored)

            for i in range(num_best_results):
                output_filename = "best_trees/map" + added_string + "_" + str(i) + ".gv"
                attachment_matrix = None

                if add_cells_to_best_trees:
                    dft = TreeUtils.get_depth_first_traversal(best_trees[i])
                    attachment_matrix = ScoreUtils.get_attachment_matrix(self.mask_matrix,
                                                                         best_trees[i],
                                                                         dft,
                                                                         best_beta[i],
                                                                         log_scores)

                FileUtils.output_graph_viz_file(output_filename,
                                                best_trees[i, :],
                                                gene_names,
                                                add_cells_to_best_trees,
                                                attachment_matrix)

    @staticmethod
    def calculate_pseudo_ess_trees(posterior_trees_filename="posterior_samples/trees.npy",
                                   tree_distance_type=0,
                                   repetitions=100,
                                   output_info=True):
        """Calculate ESS for trees in specified .npy file."""
        posterior_trees = np.load(posterior_trees_filename)
        ess_trees = ESSUtils.calculate_pseudo_ess_trees(posterior_trees, tree_distance_type, repetitions)

        if output_info:
            print("Number of trees sampled: {0}, pseudo ESS for trees: {1}".format(posterior_trees.shape[0], ess_trees))
        return ess_trees

    @staticmethod
    def calculate_ess_betas(posterior_betas_filename="posterior_samples/betas.npy", output_info=True):
        posterior_betas = np.load(posterior_betas_filename)
        ess_betas = ESSUtils.calculate_ess_betas(posterior_betas)
        """Calculate ESS for betas in specified .npy file."""
        if output_info:
            print("Number of betas sampled: {0}, ESS for betas: {1}".format(posterior_betas.shape[0], ess_betas))
        return ess_betas

    @staticmethod
    def check_error_probabilities(d0e1, d1e0, d2e0, d2e1):
        """Check that provided error probabilities are valid (sum to 1).

        Args:
            d0e1 (float): Probability P(D_ij = 0|E_ij = 1).
            d1e0 (float): Probability P(D_ij = 1|E_ij = 0).
            d2e0 (float): Probability P(D_ij = 2|E_ij = 0).
            d2e1 (float): Probability P(D_ij = 2|E_ij = 1).
        """
        if d0e1 + d2e1 > 1 or d1e0 + d2e0 > 1:
            raise Exception("Provided false positive / false negative probabilities are larger than 1.")

    @staticmethod
    def check_move_probabilities(prune_reattach, swap_node_labels, swap_subtrees, remove_insert, beta_move):
        """Check that provided move probabilities are valid or correct them if they are not.

        Args:
            prune_reattach (float): Probability of choosing prune and reattach tree move when tree move is used.
            swap_node_labels (float): Probability of choosing swap node labels tree move when tree move is used.
            swap_subtrees (float): Probability of choosing swap subtrees tree move when tree move is used.
            remove_insert (float): Probability of choosing remove and insert tree move when tree move is used.
            beta_move (float): Probability of choosing beta move instead of tree move.

        Returns:
            float: Corrected prune_reattach probability.
            float: Corrected swap_node_labels probability.
            float: Corrected swap_subtrees probability.
            float: Corrected remove_insert probability.
            float: Corrected beta_move probability.
        """
        tree_move_probabilities_sum = float(prune_reattach + swap_node_labels + swap_subtrees + remove_insert)

        if tree_move_probabilities_sum != 1.0 and tree_move_probabilities_sum > 0:
            print("Move probabilities do not sum to 1. Using probabilities as a ratio to calculate probabilities.")

            prune_reattach /= tree_move_probabilities_sum
            swap_node_labels /= tree_move_probabilities_sum
            swap_subtrees /= tree_move_probabilities_sum
            remove_insert /= tree_move_probabilities_sum

            print("\tPrune & reattach: ", prune_reattach)
            print("\tSwap node labels: ", swap_node_labels)
            print("\tSwap subtrees:    ", swap_subtrees)
            print("\tRemove & insert:    ", remove_insert)

        elif tree_move_probabilities_sum != 1:
            raise Exception("All move probabilities are 0.")

        if beta_move > 1:
            print("Beta move probability set to 1.")

            beta_move = 1

        return prune_reattach, swap_node_labels, swap_subtrees, remove_insert, beta_move


class TreeAndBetaRepresentationsAndScores:
    def __init__(self, num_nodes, beta_prior_mean):
        """Initialization of tabras.

        Args:
            num_nodes (int): Number of nodes in the mutation tree (excluding root node).
            beta_prior_mean (float): Beta mean prior.
        """
        self.num_nodes = num_nodes

        # Initialize beta to be its mean prior
        self.beta = beta_prior_mean

        # Generate random mutation tree
        self.parent_vector = TreeUtils.generate_random_parent_vector(num_nodes)

        # Construct other representations of the tree
        self.dft = TreeUtils.get_depth_first_traversal(self.parent_vector)
        self.ancestor_matrix = TreeUtils.get_ancestor_matrix(self.parent_vector, self.dft)

        # Used for faster calculation of score when only a part of mutation tree needs to be reevaluated
        self.fast_matrix = None

        # Scores of current mutation tree and beta
        self.tree_log_score = None
        self.beta_log_score = None
        self.combined_log_score = None

    def calculate_initial_scores(self, mutation_matrix, mask_matrix, log_scores, beta_move_probability, beta_params):
        """Evaluate initial mutation tree and beta.

        Args:
            mutation_matrix (np.ndarray): Mutation matrix (D matrix).
            mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
                P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
            log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.
            beta_move_probability (float): Probability of choosing beta move instead of tree move.
            beta_params (list): Parameters of beta distribution for beta.
        """
        # Use accurate evaluation to obtain accurate log tree score and fast matrix
        self.tree_log_score, self.fast_matrix = ScoreUtils.get_fast_tree_score(mutation_matrix,
                                                                               mask_matrix,
                                                                               self.parent_vector,
                                                                               self.dft,
                                                                               log_scores,
                                                                               1)

        # Beta moves are not used
        if beta_move_probability == 0.0:
            self.beta_log_score = 0.0
        # Beta moves are used
        else:
            # Evaluate current beta to obtain beta log score
            self.beta_log_score = ScoreUtils.get_beta_score(self.beta, beta_params)

        self.update_combined_log_score()

    def update_beta(self, new_beta, new_beta_log_score, new_tree_log_score, fast_matrix):
        """Update beta and log scores."""
        self.beta = new_beta
        self.beta_log_score = new_beta_log_score
        self.tree_log_score = new_tree_log_score
        self.update_combined_log_score()
        self.fast_matrix = fast_matrix

    def update_tree_score(self, new_tree_log_score):
        """Update tree log score and combined log score."""
        self.tree_log_score = new_tree_log_score
        self.update_combined_log_score()

    def update_combined_log_score(self):
        """Update combined log score from tree log score and beta log score."""
        if self.tree_log_score is not None and self.beta_log_score is not None:
            self.combined_log_score = self.tree_log_score + self.beta_log_score

    def propose_new_beta(self,
                         mutation_matrix,
                         mask_matrix,
                         log_scores,
                         best_tree_log_score,
                         jump_stdev,
                         beta_params):
        """Construct a beta move in Metropolis-Hastings algorithm and accept or reject it.

        Args:
            mutation_matrix (np.ndarray): Mutation matrix (D matrix).
            mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
                P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
            log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.
            best_tree_log_score (float): Best tree log score so far.
            jump_stdev (float): Beta jump normal random variable standard deviation.
            beta_params (list): Beta distribution parameters for beta.

        Returns:
            np.ndarray: Log scores matrix corresponding to error probabilities.
        """
        # Find new beta
        proposed_beta = self.beta + np.random.normal(0, jump_stdev)

        # Mirror value if not on interval [0, 1]
        proposed_beta = MiscUtils.get_mirrored_beta(proposed_beta)

        # Calculate score of the proposed beta
        proposed_beta_log_score = ScoreUtils.get_beta_score(proposed_beta, beta_params)

        # Update log scores matrix
        proposed_log_scores = ScoreUtils.get_updated_log_scores_matrix(log_scores, proposed_beta)

        # Calculate score of the mutation tree with the new beta
        proposed_tree_log_score, proposed_fast_matrix = ScoreUtils.get_fast_tree_score(mutation_matrix,
                                                                                       mask_matrix,
                                                                                       self.parent_vector,
                                                                                       self.dft,
                                                                                       proposed_log_scores,
                                                                                       best_tree_log_score)

        # Accept move
        if ScoreUtils.check_accept_move(proposed_beta_log_score + proposed_tree_log_score, self.combined_log_score):
            self.update_beta(proposed_beta, proposed_beta_log_score, proposed_tree_log_score, proposed_fast_matrix)

            return proposed_log_scores

        # Reject move
        return log_scores

    def propose_new_tree(self,
                         mutation_matrix,
                         mask_matrix,
                         log_scores,
                         best_tree_log_score,
                         move_probability_prune_reattach,
                         move_probability_swap_node_labels,
                         move_probability_swap_subtrees):
        """Construct a tree move in Metropolis-Hastings algorithm and accept or reject it.

        Args:
            mutation_matrix (np.ndarray): Mutation matrix (D matrix).
            mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
                P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
            log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.
            best_tree_log_score (float): Best tree log score so far.
            move_probability_prune_reattach (float): Probability of choosing prune and reattach tree move when tree move
                is used.
            move_probability_swap_node_labels (float): Probability of choosing swap node labels tree move when tree move
                is used.
            move_probability_swap_subtrees (float): Probability of choosing swap subtrees tree move when tree move is
                used.
        """
        # Choose tree move
        random_value = np.random.random()

        # Prune and reattach tree move
        if random_value <= move_probability_prune_reattach:
            self.tree_move_prune_and_reattach(mutation_matrix, mask_matrix, log_scores, best_tree_log_score)

        # Swap node labels tree move
        elif random_value <= move_probability_prune_reattach + move_probability_swap_node_labels:
            self.tree_move_swap_node_labels(mutation_matrix, mask_matrix, log_scores, best_tree_log_score)

        # Swap subtrees tree move
        elif random_value <= \
                move_probability_prune_reattach + move_probability_swap_node_labels + move_probability_swap_subtrees:
            self.tree_move_swap_subtrees(mutation_matrix, mask_matrix, log_scores, best_tree_log_score)

        # Remove and insert tree move
        else:
            self.tree_move_remove_and_insert(mutation_matrix, mask_matrix, log_scores, best_tree_log_score)

    def tree_move_prune_and_reattach(self,
                                     mutation_matrix,
                                     mask_matrix,
                                     log_scores,
                                     best_tree_log_score):
        """Construct a prune and reattach tree move in Metropolis-Hastings algorithm and accept or reject it.

        Args:
            mutation_matrix (np.ndarray): Mutation matrix (D matrix).
            mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
                P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
            log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.
            best_tree_log_score (float): Best tree log score so far.
        """
        # Get move parameters
        node_to_move, new_parent, old_parent = \
            MoveUtils.get_move_params_prune_and_reattach(self.num_nodes, self.parent_vector, self.ancestor_matrix)

        # Change parent vector in-place - self.parent_vector represents proposed parent vector
        MoveUtils.prune_and_reattach_parent_vector_in_place(self.parent_vector, node_to_move, new_parent)

        # Calculate dft
        proposed_dft = TreeUtils.get_depth_first_traversal(self.parent_vector)

        # Evaluate the new tree
        proposed_tree_log_score, proposed_fast_matrix = ScoreUtils.get_partial_tree_score_prune_and_reattach(
            mutation_matrix,
            mask_matrix,
            self.parent_vector,
            proposed_dft,
            self.ancestor_matrix,
            log_scores,
            self.fast_matrix,
            best_tree_log_score,
            node_to_move)

        # Accept move
        if ScoreUtils.check_accept_move(proposed_tree_log_score, self.tree_log_score):
            self.dft = proposed_dft
            self.fast_matrix = proposed_fast_matrix
            self.update_tree_score(proposed_tree_log_score)

            # Change ancestor matrix accordingly
            MoveUtils.prune_and_reattach_ancestor_matrix_in_place(self.ancestor_matrix,
                                                                  node_to_move,
                                                                  new_parent,
                                                                  old_parent)

        # Reject move
        else:
            # Revert parent vector
            MoveUtils.prune_and_reattach_parent_vector_revert_in_place(self.parent_vector, node_to_move, old_parent)

    def tree_move_swap_node_labels(self, mutation_matrix, mask_matrix, log_scores, best_tree_log_score):
        """Create a swap node labels tree move in Metropolis-Hastings algorithm and accept or reject it.

        Args:
            mutation_matrix (np.ndarray): Mutation matrix (D matrix).
            mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
                P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
            log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.
            best_tree_log_score (float): Best tree log score so far.
        """
        # Get move parameters
        above_node, below_node, same_lineage = MoveUtils.get_move_params_swap_node_labels(self.num_nodes,
                                                                                          self.ancestor_matrix)

        # Construct parent vector
        proposed_parent_vector = MoveUtils.swap_node_labels_parent_vector(self.parent_vector, above_node, below_node)

        # Construct dft
        proposed_dft = TreeUtils.get_depth_first_traversal(proposed_parent_vector)

        # Evaluate the new tree
        proposed_tree_log_score, proposed_fast_matrix = ScoreUtils.get_partial_tree_score_swap_node_labels(
            mutation_matrix,
            mask_matrix,
            proposed_parent_vector,
            proposed_dft,
            self.ancestor_matrix,
            log_scores,
            self.fast_matrix,
            best_tree_log_score,
            above_node,
            below_node,
            same_lineage)

        # Accept move
        if ScoreUtils.check_accept_move(proposed_tree_log_score, self.tree_log_score):
            self.parent_vector = proposed_parent_vector
            self.dft = proposed_dft
            self.fast_matrix = proposed_fast_matrix
            self.update_tree_score(proposed_tree_log_score)

            # Change ancestor matrix accordingly
            MoveUtils.swap_node_labels_ancestor_matrix_in_place(self.ancestor_matrix,
                                                                above_node,
                                                                below_node)

    def tree_move_swap_subtrees(self, mutation_matrix, mask_matrix, log_scores, best_tree_log_score):
        """Create a swap subtrees tree move in Metropolis-Hastings algorithm and accept or reject it.

        Args:
            mutation_matrix (np.ndarray): Mutation matrix (D matrix).
            mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
                P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
            log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.
            best_tree_log_score (float): Best tree log score so far.
        """
        # Get move parameters
        above_node, below_node, same_lineage, new_parent, nbh = \
            MoveUtils.get_move_params_swap_subtrees(self.num_nodes, self.ancestor_matrix)

        # Construct parent vector
        proposed_parent_vector = MoveUtils.swap_subtrees_parent_vector(self.parent_vector,
                                                                       above_node,
                                                                       below_node,
                                                                       same_lineage,
                                                                       new_parent)

        # Construct dft
        proposed_dft = TreeUtils.get_depth_first_traversal(proposed_parent_vector)

        # Evaluate the new tree
        proposed_tree_log_score, proposed_fast_matrix = ScoreUtils.get_partial_tree_score_swap_subtrees(
            mutation_matrix,
            mask_matrix,
            proposed_parent_vector,
            proposed_dft,
            self.ancestor_matrix,
            log_scores,
            self.fast_matrix,
            best_tree_log_score,
            above_node,
            below_node,
            same_lineage)

        # Accept move
        if ScoreUtils.check_accept_move(proposed_tree_log_score, self.tree_log_score, neighbourhood_correction=nbh):
            self.parent_vector = proposed_parent_vector
            self.dft = proposed_dft
            self.fast_matrix = proposed_fast_matrix
            self.update_tree_score(proposed_tree_log_score)
            self.ancestor_matrix = TreeUtils.get_ancestor_matrix(proposed_parent_vector, proposed_dft)

    def tree_move_remove_and_insert(self, mutation_matrix, mask_matrix, log_scores, best_tree_log_score):
        """Create a remove and insert tree move in Metropolis-Hastings algorithm and accept or reject it.

        Args:
            mutation_matrix (np.ndarray): Mutation matrix (D matrix).
            mask_matrix (np.ndarray): Matrix of 'count of probabilities' of type P(D_ij = x|E_ij = 0) and
                P(D_ij = x|E_ij = 1) from which fast_matrix can be obtained if log scores matrix is known.
            log_scores (np.ndarray): Log scores matrix corresponding to error probabilities.
            best_tree_log_score (float): Best tree log score so far.
       """
        # Get move parameters
        node_to_move, old_attachment, new_attachment, new_children, nbh = MoveUtils.get_move_params_remove_and_insert(
            self.num_nodes,
            self.parent_vector)

        # Construct parent vector
        proposed_parent_vector = MoveUtils.remove_and_insert_parent_vector(self.parent_vector,
                                                                           node_to_move,
                                                                           old_attachment,
                                                                           new_attachment,
                                                                           new_children)

        # Construct dft
        proposed_dft = TreeUtils.get_depth_first_traversal(proposed_parent_vector)

        # Evaluate the new tree
        proposed_score, proposed_fast_matrix = ScoreUtils.get_fast_tree_score(mutation_matrix,
                                                                              mask_matrix,
                                                                              proposed_parent_vector,
                                                                              proposed_dft,
                                                                              log_scores,
                                                                              best_tree_log_score)

        # Accept move
        if ScoreUtils.check_accept_move(proposed_score, self.tree_log_score, neighbourhood_correction=nbh):
            self.parent_vector = proposed_parent_vector
            self.dft = proposed_dft
            self.fast_matrix = proposed_fast_matrix
            self.update_tree_score(proposed_score)
            self.ancestor_matrix = TreeUtils.get_ancestor_matrix(proposed_parent_vector, proposed_dft)

    def update_minimal_distance_to_true_tree(self,
                                             true_parent_vector,
                                             minimal_distance_to_true_tree,
                                             best_combined_log_score):
        """Get updated minimal distance to true tree.

        Args:
            true_parent_vector (np.ndarray): Parent vector representation of true mutation tree.
            minimal_distance_to_true_tree (int): Currently minimal distance to true tree.
            best_combined_log_score (float): Currently best log score of mutation tree and beta.

        Returns:
            int: Updated minimal distance to true tree.
        """
        current_distance_to_true_tree = TreeUtils.get_tree_distance(self.parent_vector, true_parent_vector)

        if minimal_distance_to_true_tree is None:
            return current_distance_to_true_tree

        if self.combined_log_score > best_combined_log_score:
            return current_distance_to_true_tree

        if self.combined_log_score == best_combined_log_score and \
                current_distance_to_true_tree < minimal_distance_to_true_tree:
            return current_distance_to_true_tree

        return minimal_distance_to_true_tree


class BestResults:
    def __init__(self, max_best_results, num_nodes):
        """Initialize object that stores best trees and betas encountered during the execution of the algorithm.

        Args:
            max_best_results (int): Maximum number of best results stored at any given time.
            num_nodes (int): Number of nodes in the mutation tree (excluding root node).
        """
        # Current number of best results stored
        self.best_results_stored = 0

        self.max_best_results = max_best_results

        self.trees = np.empty((max_best_results, num_nodes), dtype=np.int32)
        self.betas = np.empty((max_best_results,), dtype=np.float64)

    def reset_top_results(self):
        self.best_results_stored = 0

    def add_entry(self, parent_vector, beta):
        """Store currently best tree and beta if there are not yet enough trees and betas with same score already
        stored.

        Args:
            parent_vector (np.ndarray): Parent vector representation of mutation tree to be stored.
            beta (float): Beta to be stored.
        """
        if self.best_results_stored < self.max_best_results:
            self.trees[self.best_results_stored, :] = parent_vector
            self.betas[self.best_results_stored] = beta
            self.best_results_stored += 1

    def check_tree_already_stored(self, parent_vector):
        """Check whether the new tree with best combined log score so far is already stored.

        Args:
            parent_vector (np.ndarray): Parent vector representation of mutation tree to be stored.

        Returns:
            bool: True if the new tree is already stored, False if not.
        """
        for i in range(self.best_results_stored):
            if np.all(self.trees[i, :] == parent_vector):
                return True

        return False

    def update_results(self, parent_vector, beta, current_combined_log_score, best_combined_log_score):
        """Update best results if current mutation tree and beta are currently the best among all trees and betas
        encountered until then.

        Args:
            parent_vector (np.ndarray): Parent vector representation of current mutation tree.
            beta (float): Current beta.
            current_combined_log_score (float): Combined log score of current tree and beta.
            best_combined_log_score (float): Combined log score of the currently best tree and beta.
        """
        # New best combined log score
        if best_combined_log_score is None or current_combined_log_score > best_combined_log_score:
            self.reset_top_results()
            self.add_entry(parent_vector, beta)

        # Equally good score, this specific tree was not encountered yet
        elif current_combined_log_score == best_combined_log_score and \
                not self.check_tree_already_stored(parent_vector):
            self.add_entry(parent_vector, beta)

    def get_best_results(self):
        """Get best mutation trees and betas (all have equal score).

        Returns:
            np.ndarray: Parent vectors that represent best mutation trees. One row corresponds to one parent vector.
            np.ndarray: Vector of beta values corresponding to parent vectors.
        """

        return self.trees[:self.best_results_stored, :], self.betas[:self.best_results_stored]


if __name__ == "__main__":
    start_time = time.time()

    # dataHou18
    mcmc = MHRunner("datasets/dataHou18.csv")
    mcmc.run_mh(num_repetitions=1,
                chain_length=10000000,
                d0e1_probability=0.21545,
                d1e0_probability=6.04E-5,
                d2e0_probability=1.299164E-5,
                d2e1_probability=0.21545,
                beta_move_probability=0.1,
                mh_jump_scaling=3,
                gene_names_filename="datasets/dataHou18.geneNames",
                output_name="hou18")

    # dataXu
    mcmc = MHRunner("datasets/dataXu.csv")
    mcmc.run_mh(num_repetitions=10,
                chain_length=10000000,
                d0e1_probability=0.1643,
                d1e0_probability=2.67E-5,
                d2e0_probability=0.0,
                d2e1_probability=0.0,
                beta_move_probability=0.1,
                beta_prior_stdev=0.06,
                gene_names_filename="datasets/dataXu.geneNames",
                output_name="xu")

    # dataNavin
    mcmc = MHRunner("datasets/dataNavin.csv")
    mcmc.run_mh(num_repetitions=20,
                chain_length=10000000,
                d0e1_probability=0.0972,
                d1e0_probability=1.24E-6,
                d2e0_probability=0.0,
                d2e1_probability=0.0,
                beta_move_probability=0.1,
                beta_prior_stdev=0.04,
                gene_names_filename="datasets/dataNavin.geneNames",
                output_name="navin")

    # dataMissionBio (20 x 8954)
    mcmc = MHRunner("datasets/dataMissionBio.csv")
    mcmc.run_mh(num_repetitions=1,
                chain_length=100000,
                d0e1_probability=0.0697,
                d1e0_probability=1E-6,
                d2e0_probability=0.0,
                d2e1_probability=0.0,
                beta_move_probability=0.1,
                output_name="mission_bio")

    print("Time Running: ", time.time() - start_time)
