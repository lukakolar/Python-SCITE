from collections import defaultdict as dd
from unittest import TestCase

import numpy as np

import FileUtils
import MiscUtils
import MoveUtils
import ScoreUtils
import TreeUtils


def get_max_deviation_matrix(matrix_1, matrix_2):
    return np.max(np.max(np.abs(matrix_1 - matrix_2)))


def check_max_variation_matrix(first_matrix, second_matrix, tolerance):
    return check_each_element_matrix(np.abs(first_matrix - second_matrix) < tolerance)


def check_max_variation_array(first_array, second_array, tolerance):
    return np.all(np.abs(first_array - second_array) < tolerance)


def check_each_element_matrix(matrix):
    return np.all(np.all(matrix))


def check_each_element_matrix_3d(matrix_3d):
    return np.all(np.all(np.all(matrix_3d)))


class Scenario:
    def __init__(self, parent_vector, move_parameters, parent_vector_after_move, fail_message):
        self.parent_vector = parent_vector
        self.move_parameters = move_parameters
        self.parent_vector_after_move = parent_vector_after_move
        self.fail_message = fail_message


def get_mock_tree_parent_vector():
    return np.array([2, 2, 3, 8, 8, 7, 7, 8, 15, 11, 11, 12, 14, 14, 15], dtype=np.int32)


def get_other_tree_representations(parent_vector):
    dft = TreeUtils.get_depth_first_traversal(parent_vector)
    ancestor_matrix = TreeUtils.get_ancestor_matrix(parent_vector, dft)

    return dft, ancestor_matrix


def get_mock_tree_representation_and_move_params_prune_and_reattach():
    scenarios = []

    mock_parent_vector = get_mock_tree_parent_vector()

    # 1. new_parent = old_parent
    scenarios.append(Scenario(mock_parent_vector,
                              [2, 3, 3],
                              mock_parent_vector,
                              "same new_parent and old_parent"))

    # 2. new_parent in different lineage than old_parent
    mock_parent_vector_2 = np.copy(mock_parent_vector)
    mock_parent_vector_2[2] = 12

    scenarios.append(Scenario(mock_parent_vector,
                              [2, 12, 3],
                              mock_parent_vector_2,
                              "new_parent and old_parent in different lineage"))

    # 3. new_parent in same lineage as old_parent
    mock_parent_vector_3 = np.copy(mock_parent_vector)
    mock_parent_vector_3[2] = 15

    scenarios.append(Scenario(mock_parent_vector,
                              [2, 15, 3],
                              mock_parent_vector_3,
                              "new_parent and old_parent in same lineage"))

    # 4. new_parent and old_parent are direct descendants
    mock_parent_vector_4 = np.copy(mock_parent_vector)
    mock_parent_vector_4[2] = 8

    scenarios.append(Scenario(mock_parent_vector,
                              [2, 8, 3],
                              mock_parent_vector_4,
                              "new_parent and old_parent direct descendants"))

    return scenarios


def get_mock_tree_representation_and_move_params_swap_node_labels():
    scenarios = []

    mock_parent_vector = get_mock_tree_parent_vector()

    # 1. above_node in different lineage than below_node
    mock_parent_vector_1 = np.copy(mock_parent_vector)
    mock_parent_vector_1[0] = 11
    mock_parent_vector_1[1] = 11
    mock_parent_vector_1[9] = 2
    mock_parent_vector_1[10] = 2
    mock_parent_vector_1[2] = 12
    mock_parent_vector_1[11] = 3

    scenarios.append(Scenario(mock_parent_vector,
                              [2, 11, False],
                              mock_parent_vector_1,
                              "above_node and below_node in different lineage"))

    # 2. above_node in same lineage as below_node
    mock_parent_vector_2 = np.copy(mock_parent_vector)
    mock_parent_vector_2[0] = 8
    mock_parent_vector_2[1] = 8
    mock_parent_vector_2[2] = 15
    mock_parent_vector_2[3] = 2
    mock_parent_vector_2[4] = 2
    mock_parent_vector_2[7] = 2
    mock_parent_vector_2[8] = 3

    scenarios.append(Scenario(mock_parent_vector,
                              [2, 8, True],
                              mock_parent_vector_2,
                              "above_node and below_node in same lineage"))

    # 3. above_node and below_node are direct descendants
    mock_parent_vector_3 = np.copy(mock_parent_vector)
    mock_parent_vector_3[0] = 3
    mock_parent_vector_3[1] = 3
    mock_parent_vector_3[2] = 8
    mock_parent_vector_3[3] = 2

    scenarios.append(Scenario(mock_parent_vector,
                              [2, 3, True],
                              mock_parent_vector_3,
                              "above_node and below_node direct descendants"))

    return scenarios


def get_mock_tree_representation_and_move_params_swap_subtrees():
    scenarios = []

    mock_parent_vector = get_mock_tree_parent_vector()

    # 1. above_node in different lineage than below_node
    mock_parent_vector_1 = np.copy(mock_parent_vector)
    mock_parent_vector_1[2] = 12
    mock_parent_vector_1[11] = 3

    scenarios.append(Scenario(mock_parent_vector,
                              [2, 11, False, 0, None],
                              mock_parent_vector_1,
                              "above_node and below_node in different lineage"))

    # 2. above_node in same lineage as below_node
    mock_parent_vector_2 = np.copy(mock_parent_vector)
    mock_parent_vector_2[5] = 15
    mock_parent_vector_2[8] = 5

    scenarios.append(Scenario(mock_parent_vector,
                              [5, 8, True, 5, None],
                              mock_parent_vector_2,
                              "above_node and below_node in same lineage"))

    # 3. above_node and below_node are direct descendants
    mock_parent_vector_3 = np.copy(mock_parent_vector)
    mock_parent_vector_3[5] = 8
    mock_parent_vector_3[7] = 5

    scenarios.append(Scenario(mock_parent_vector,
                              [5, 7, True, 5, None],
                              mock_parent_vector_3,
                              "above_node and below_node direct descendants"))

    return scenarios


def get_mock_tree_tree_representation_and_move_params_remove_and_insert():
    scenarios = []

    mock_parent_vector = get_mock_tree_parent_vector()

    # 1. new_parent in different lineage
    mock_parent_vector_1 = np.copy(mock_parent_vector)
    mock_parent_vector_1[0] = 3
    mock_parent_vector_1[1] = 3
    mock_parent_vector_1[2] = 12

    scenarios.append(Scenario(mock_parent_vector,
                              [2, 3, 12, np.array([-1]), None],
                              mock_parent_vector_1,
                              "new_parent in different lineage"))

    # 2. new_parent in same lineage
    mock_parent_vector_2 = np.copy(mock_parent_vector)
    mock_parent_vector_2[0] = 3
    mock_parent_vector_2[1] = 3
    mock_parent_vector_2[2] = 8
    mock_parent_vector_2[3] = 2
    mock_parent_vector_2[4] = 2

    scenarios.append(Scenario(mock_parent_vector,
                              [2, 3, 8, np.array([3, 4]), None],
                              mock_parent_vector_2,
                              "new_parent in same lineage"))

    # 3. new_parent is old_parent
    mock_parent_vector_3 = np.copy(mock_parent_vector)
    mock_parent_vector_3[4] = 7
    mock_parent_vector_3[6] = 8

    scenarios.append(Scenario(mock_parent_vector,
                              [7, 8, 8, np.array([4, 5]), None],
                              mock_parent_vector_3,
                              "new_parent is old_parent"))

    return scenarios


class TestMoveUtils(TestCase):
    def test_prune_and_reattach_parent_vector_in_place(self):
        scenarios = get_mock_tree_representation_and_move_params_prune_and_reattach()
        for scenario in scenarios:
            self.scenario_test_prune_and_reattach_parent_vector_in_place(scenario)

    def scenario_test_prune_and_reattach_parent_vector_in_place(self, scenario):
        parent_vector = scenario.parent_vector
        node_to_move, new_parent, _ = scenario.move_parameters

        MoveUtils.prune_and_reattach_parent_vector_in_place(parent_vector, node_to_move, new_parent)

        self.assertTrue(np.all(scenario.parent_vector_after_move == parent_vector),
                        msg=scenario.fail_message)

    def test_prune_and_reattach_parent_vector_revert_in_place(self):
        scenarios = get_mock_tree_representation_and_move_params_prune_and_reattach()
        for scenario in scenarios:
            self.scenario_test_prune_and_reattach_parent_vector_revert_in_place(scenario)

    def scenario_test_prune_and_reattach_parent_vector_revert_in_place(self, scenario):
        parent_vector = scenario.parent_vector
        parent_vector_test = np.copy(parent_vector)
        node_to_move, new_parent, old_parent = scenario.move_parameters

        MoveUtils.prune_and_reattach_parent_vector_in_place(parent_vector, node_to_move, new_parent)
        MoveUtils.prune_and_reattach_parent_vector_revert_in_place(parent_vector, node_to_move, old_parent)

        self.assertTrue(np.all(parent_vector == parent_vector_test), msg=scenario.fail_message)

    def test_prune_and_reattach_ancestor_matrix_in_place(self):
        scenarios = get_mock_tree_representation_and_move_params_prune_and_reattach()
        for scenario in scenarios:
            self.scenario_test_prune_and_reattach_ancestor_matrix_in_place(scenario)

    def scenario_test_prune_and_reattach_ancestor_matrix_in_place(self, scenario):
        _, ancestor_matrix = get_other_tree_representations(scenario.parent_vector)
        node_to_move, new_parent, old_parent = scenario.move_parameters
        MoveUtils.prune_and_reattach_ancestor_matrix_in_place(ancestor_matrix, node_to_move, new_parent, old_parent)
        _, ancestor_matrix_test = get_other_tree_representations(scenario.parent_vector_after_move)

        self.assertTrue(check_each_element_matrix(ancestor_matrix == ancestor_matrix_test))

    def test_swap_node_labels_parent_vector(self):
        scenarios = get_mock_tree_representation_and_move_params_swap_node_labels()
        for scenario in scenarios:
            self.scenario_test_swap_node_labels_parent_vector(scenario)

    def scenario_test_swap_node_labels_parent_vector(self, scenario):
        above_node, below_node, _ = scenario.move_parameters

        parent_vector = MoveUtils.swap_node_labels_parent_vector(scenario.parent_vector, above_node, below_node)

        self.assertTrue(np.all(scenario.parent_vector_after_move == parent_vector),
                        msg=scenario.fail_message)

    def test_swap_node_labels_ancestor_matrix_in_place(self):
        scenarios = get_mock_tree_representation_and_move_params_swap_node_labels()
        for scenario in scenarios:
            self.scenario_test_swap_node_labels_ancestor_matrix_in_place(scenario)

    def scenario_test_swap_node_labels_ancestor_matrix_in_place(self, scenario):
        _, ancestor_matrix = get_other_tree_representations(scenario.parent_vector)
        above_node, below_node, _ = scenario.move_parameters
        MoveUtils.swap_node_labels_ancestor_matrix_in_place(ancestor_matrix, above_node, below_node)
        _, ancestor_matrix_test = get_other_tree_representations(scenario.parent_vector_after_move)

        self.assertTrue(check_each_element_matrix(ancestor_matrix == ancestor_matrix_test))

    def test_swap_subtrees_parent_vector(self):
        scenarios = get_mock_tree_representation_and_move_params_swap_subtrees()
        for scenario in scenarios:
            self.scenario_test_swap_subtrees_parent_vector(scenario)

    def scenario_test_swap_subtrees_parent_vector(self, scenario):
        above_node, below_node, same_lineage, new_parent, _ = scenario.move_parameters
        parent_vector = MoveUtils.swap_subtrees_parent_vector(scenario.parent_vector,
                                                              above_node,
                                                              below_node,
                                                              same_lineage,
                                                              new_parent)

        self.assertTrue(np.all(scenario.parent_vector_after_move == parent_vector), msg=scenario.fail_message)

    def test_remove_and_insert_parent_vector(self):
        scenarios = get_mock_tree_tree_representation_and_move_params_remove_and_insert()
        for scenario in scenarios:
            self.scenario_test_remove_and_insert_parent_vector(scenario)

    def scenario_test_remove_and_insert_parent_vector(self, scenario):
        node_to_move, old_parent, new_parent, new_children, _ = scenario.move_parameters
        parent_vector = MoveUtils.remove_and_insert_parent_vector(scenario.parent_vector,
                                                                  node_to_move,
                                                                  old_parent,
                                                                  new_parent,
                                                                  new_children)

        self.assertTrue(np.all(scenario.parent_vector_after_move == parent_vector), msg=scenario.fail_message)


class TestScoreUtils(TestCase):
    def test_get_fast_tree_score(self):
        mutation_matrix = FileUtils.read_mutation_matrix("datasets/dataHou78.csv")
        mask_matrix = ScoreUtils.get_mask_matrix(mutation_matrix)
        log_scores = np.asarray([-7.33943332981645e-05, -1.53502641492029, -9.7145214530235, -0.563699113373059,
                                 -11.2512044842885, -1.53502641492029, 0, 0]).reshape(4, 2)
        parent_vector = np.asarray([9, 13, 12, 31, 5, 76, 37, 38, 1, 36, 1, 29, 67, 78, 67, 23, 14, 63, 70, 61, 53, 11,
                                    10, 69, 43, 7, 48, 29, 65, 43, 55, 21, 1, 35, 13, 14, 11, 5, 59, 46, 5, 61, 14, 71,
                                    21, 7, 6, 55, 44, 33, 48, 26, 24, 78, 10, 8, 23, 77, 13, 10, 35, 57, 46, 73, 58, 8,
                                    2, 23, 32, 61, 53, 67, 60, 48, 58, 11, 3, 38])
        dft = TreeUtils.get_depth_first_traversal(parent_vector)

        score, _ = ScoreUtils.get_fast_tree_score(mutation_matrix, mask_matrix, parent_vector, dft, log_scores, 1)

        self.assertAlmostEqual(score, -7478.72872196288)

    def test_get_partial_tree_score_prune_and_reattach(self):
        scenarios = get_mock_tree_representation_and_move_params_prune_and_reattach()
        for scenario in scenarios:
            self.scenario_test_get_partial_tree_score_prune_and_reattach(scenario)

    def scenario_test_get_partial_tree_score_prune_and_reattach(self, scenario):
        parent_vector = get_mock_tree_parent_vector()
        mutation_matrix = np.random.randint(0, 4, (parent_vector.shape[0], 1000), dtype=np.int32)
        mask_matrix = ScoreUtils.get_mask_matrix(mutation_matrix)
        log_scores = np.asarray([-7.33943332981645e-05, -1.53502641492029, -9.7145214530235, -0.563699113373059,
                                 -11.2512044842885, -1.53502641492029, 0, 0]).reshape(4, 2)
        dft, ancestor_matrix = get_other_tree_representations(parent_vector)

        _, fast_matrix = ScoreUtils.get_fast_tree_score(mutation_matrix, mask_matrix, parent_vector, dft, log_scores, 1)

        node_to_move, new_parent, old_parent = scenario.move_parameters

        MoveUtils.prune_and_reattach_parent_vector_in_place(parent_vector, node_to_move, new_parent)
        proposed_dft = TreeUtils.get_depth_first_traversal(parent_vector)

        score, fast_matrix = ScoreUtils.get_partial_tree_score_prune_and_reattach(mutation_matrix,
                                                                                  mask_matrix,
                                                                                  parent_vector,
                                                                                  proposed_dft,
                                                                                  ancestor_matrix,
                                                                                  log_scores,
                                                                                  fast_matrix,
                                                                                  1,
                                                                                  node_to_move)

        score_test, fast_matrix_test = ScoreUtils.get_fast_tree_score(mutation_matrix,
                                                                      mask_matrix,
                                                                      parent_vector,
                                                                      proposed_dft,
                                                                      log_scores,
                                                                      1)

        self.assertTrue(check_max_variation_matrix(fast_matrix_test, fast_matrix, 1e-15)
                        and np.abs(score - score_test) < 1e-15, msg=scenario.fail_message)

    def test_get_partial_tree_score_swap_node_labels(self):
        scenarios = get_mock_tree_representation_and_move_params_swap_node_labels()
        for scenario in scenarios:
            self.scenario_test_get_partial_tree_score_swap_node_labels(scenario)

    def scenario_test_get_partial_tree_score_swap_node_labels(self, scenario):
        parent_vector = get_mock_tree_parent_vector()
        mutation_matrix = np.random.randint(0, 4, (parent_vector.shape[0], 1000), dtype=np.int32)
        mask_matrix = ScoreUtils.get_mask_matrix(mutation_matrix)
        log_scores = np.asarray([-7.33943332981645e-05, -1.53502641492029, -9.7145214530235, -0.563699113373059,
                                 -11.2512044842885, -1.53502641492029, 0, 0]).reshape(4, 2)
        dft, ancestor_matrix = get_other_tree_representations(parent_vector)

        _, fast_matrix = ScoreUtils.get_fast_tree_score(mutation_matrix, mask_matrix, parent_vector, dft, log_scores, 1)

        above_node, below_node, same_lineage = scenario.move_parameters

        proposed_parent_vector = MoveUtils.swap_node_labels_parent_vector(parent_vector, above_node, below_node)
        proposed_dft = TreeUtils.get_depth_first_traversal(proposed_parent_vector)

        score, fast_matrix = ScoreUtils.get_partial_tree_score_swap_node_labels(mutation_matrix,
                                                                                mask_matrix,
                                                                                proposed_parent_vector,
                                                                                proposed_dft,
                                                                                ancestor_matrix,
                                                                                log_scores,
                                                                                fast_matrix,
                                                                                1,
                                                                                above_node,
                                                                                below_node,
                                                                                same_lineage)

        score_test, fast_matrix_test = ScoreUtils.get_fast_tree_score(mutation_matrix,
                                                                      mask_matrix,
                                                                      proposed_parent_vector,
                                                                      proposed_dft,
                                                                      log_scores,
                                                                      1)

        self.assertTrue(check_max_variation_matrix(fast_matrix_test, fast_matrix, 1e-15)
                        and np.abs(score - score_test) < 1e-15, msg=scenario.fail_message)

    def test_get_partial_tree_score_swap_subtrees(self):
        scenarios = get_mock_tree_representation_and_move_params_swap_subtrees()
        for scenario in scenarios:
            self.scenario_test_get_partial_tree_score_swap_subtrees(scenario)

    def scenario_test_get_partial_tree_score_swap_subtrees(self, scenario):
        parent_vector = get_mock_tree_parent_vector()
        mutation_matrix = np.random.randint(0, 4, (parent_vector.shape[0], 1000), dtype=np.int32)
        mask_matrix = ScoreUtils.get_mask_matrix(mutation_matrix)
        log_scores = np.asarray([-7.33943332981645e-05, -1.53502641492029, -9.7145214530235, -0.563699113373059,
                                 -11.2512044842885, -1.53502641492029, 0, 0]).reshape(4, 2)
        dft, ancestor_matrix = get_other_tree_representations(parent_vector)

        _, fast_matrix = ScoreUtils.get_fast_tree_score(mutation_matrix, mask_matrix, parent_vector, dft, log_scores, 1)

        above_node, below_node, same_lineage, new_parent, _ = scenario.move_parameters

        proposed_parent_vector = MoveUtils.swap_subtrees_parent_vector(parent_vector,
                                                                       above_node,
                                                                       below_node,
                                                                       same_lineage,
                                                                       new_parent)
        proposed_dft = TreeUtils.get_depth_first_traversal(proposed_parent_vector)

        score, fast_matrix = ScoreUtils.get_partial_tree_score_swap_subtrees(mutation_matrix,
                                                                             mask_matrix,
                                                                             proposed_parent_vector,
                                                                             proposed_dft,
                                                                             ancestor_matrix,
                                                                             log_scores,
                                                                             fast_matrix,
                                                                             1,
                                                                             above_node,
                                                                             below_node,
                                                                             same_lineage)

        score_test, fast_matrix_test = ScoreUtils.get_fast_tree_score(mutation_matrix,
                                                                      mask_matrix,
                                                                      proposed_parent_vector,
                                                                      proposed_dft,
                                                                      log_scores,
                                                                      1)

        self.assertTrue(check_max_variation_matrix(fast_matrix_test, fast_matrix, 1e-15)
                        and np.abs(score - score_test) < 1e-15, msg=scenario.fail_message)

    def test_get_accurate_tree_score(self):
        mutation_matrix = FileUtils.read_mutation_matrix("datasets/dataHou78.csv")
        mask_matrix = ScoreUtils.get_mask_matrix(mutation_matrix)
        log_scores = np.asarray([-7.33943332981645e-05, -1.53502641492029, -9.7145214530235, -0.563699113373059,
                                 -11.2512044842885, -1.53502641492029, 0, 0]).reshape(4, 2)
        parent_vector = np.asarray([9, 13, 12, 31, 5, 76, 37, 38, 1, 36, 1, 29, 67, 78, 67, 23, 14, 63, 70, 61, 53, 11,
                                    10, 69, 43, 7, 48, 29, 65, 43, 55, 21, 1, 35, 13, 14, 11, 5, 59, 46, 5, 61, 14, 71,
                                    21, 7, 6, 55, 44, 33, 48, 26, 24, 78, 10, 8, 23, 77, 13, 10, 35, 57, 46, 73, 58, 8,
                                    2, 23, 32, 61, 53, 67, 60, 48, 58, 11, 3, 38])
        dft = TreeUtils.get_depth_first_traversal(parent_vector)

        score, _ = ScoreUtils.get_accurate_tree_score(mask_matrix, parent_vector, dft, log_scores)

        self.assertAlmostEqual(score, -7478.72872196288)

    def test_get_attachment_matrix(self):
        # Attachment to MAP tree with fixed beta obtained from dataXu.csv
        beta = 0.198
        log_scores = ScoreUtils.get_log_scores_matrix(beta, 2.67E-5, 0, 0)
        mutation_matrix = FileUtils.read_mutation_matrix("datasets/dataXu.csv")
        mask_matrix = ScoreUtils.get_mask_matrix(mutation_matrix)
        parent_vector = np.asarray([5, 35, 0, 10, 2, 27, 9, 32, 11, 25, 17, 33, 15, 23, 34, 3, 14, 1, 13, 24, 19, 28,
                                    16, 30, 7, 8, 22, 6, 31, 26, 12, 20, 29, 18, 4], dtype=np.int32)
        dft = TreeUtils.get_depth_first_traversal(parent_vector)
        attachment_matrix = ScoreUtils.get_attachment_matrix(mask_matrix,
                                                             parent_vector,
                                                             dft,
                                                             beta,
                                                             log_scores)

        attachment_matrix_test_list = [6, 8, 7, 2, 8, 8, 9, 8, 14, 13, 16, 13, 19, 2, 19, 12, 20, 2, 20, 10, 20, 12,
                                       21, 7, 21, 9, 21, 15, 21, 16, 22, 13, 24, 2, 25, 8, 26, 11, 28, 0, 28, 1, 28,
                                       3, 28, 5, 28, 6, 28, 14, 29, 11, 31, 3, 31, 4, 32, 11]
        attachment_matrix_test = np.asarray(attachment_matrix_test_list, dtype=np.int32) \
            .reshape(int(len(attachment_matrix_test_list) / 2), 2)

        self.assertTrue(check_each_element_matrix(attachment_matrix == attachment_matrix_test))


class TestTreeUtils(TestCase):
    @staticmethod
    def get_parent_child_mapping(parent_vector):
        parent_child_mapping = dd(list)
        for child in range(len(parent_vector)):
            parent = parent_vector[child]
            parent_child_mapping[parent].append(child)

        return parent_child_mapping

    def test_get_ancestor_matrix(self):

        parent_vector = np.asarray([2, 2, 5, 5, 5, 10, 8, 8, 9, 10])

        num_nodes = parent_vector.shape[0]
        _, ancestor_matrix = get_other_tree_representations(parent_vector)

        ancestor_matrix_test = np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                           1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                           1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                           0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                                           0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(num_nodes, num_nodes)

        self.assertTrue(check_each_element_matrix(ancestor_matrix == ancestor_matrix_test))

    def test_generate_random_parent_vector(self):
        num_nodes = 100

        parent_vector = TreeUtils.generate_random_parent_vector(num_nodes)
        parent_child_mapping = self.get_parent_child_mapping(parent_vector)

        visited_nodes = set()

        # Cycles should not be present in a tree, so check that they do not exist
        def dfs(pcm, current_node):
            if current_node in visited_nodes:
                # Cycle exists
                self.fail()

            visited_nodes.add(current_node)

            for child in parent_child_mapping[current_node]:
                dfs(pcm, child)

        dfs(parent_child_mapping, num_nodes)


class TestMiscUtils(TestCase):
    def test_dot_3d(self):
        a = np.random.randint(0, 100, (15, 3000, 8), dtype=np.int32)
        b = np.log(np.random.random((8,)))

        dot_product = MiscUtils.dot_3d(a, b)
        dot_product_test = np.dot(a, b)
        self.assertTrue(check_max_variation_matrix(dot_product, dot_product_test, 1e-15))

    def test_get_mirrored_beta(self):
        betas = np.array([0, 0.5, 1, -0.3, 1.3, -2.5, 2.5, -3.7, 3.7, -4.8, 4.8, -5.6, 5.6], dtype=np.float64)
        betas_mirrored = np.empty((betas.size,), dtype=np.float64)
        for beta_index in range(betas.size):
            betas_mirrored[beta_index] = MiscUtils.get_mirrored_beta(betas[beta_index])

        betas_mirrored_test = np.array([0, 0.5, 1, 0.3, 0.7, 0.5, 0.5, 0.3, 0.3, 0.8, 0.8, 0.4, 0.4], dtype=np.float64)

        self.assertTrue(check_max_variation_array(betas_mirrored, betas_mirrored_test, 1e-15))
