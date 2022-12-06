#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os.path

# Testing modules
import multiplex_ms_utils as mms
import unittest
from unittest import mock


TEST_PATH = os.path.dirname(__file__)


class TestMultiplexMsUtilsFunctions(unittest.TestCase):

    def test_create_multiplex_ms_grid(self):

        initial_grid = np.array([[1, 2, 3, 4, 5],
                                 [6, 7, 8, 9, 10],
                                 [11, 12, 13, 14, 15],
                                 [16, 17, 18, 19, 20],
                                 [21, 22, 23, 24, 25]])

        rearranged_grid_solution = np.array([[1, 7, 13, 19, 25],
                                             [12, 18, 24, 5, 6],
                                             [23, 4, 10, 11, 17],
                                             [9, 15, 16, 22, 3],
                                             [20, 21, 2, 8, 14]])

        initial_grid_str = np.array([['a', 'b', 'c', 'd', 'e'],
                                     ['f', 'g', 'h', 'i', 'j'],
                                     ['k', 'l', 'm', 'n', 'o'],
                                     ['p', 'q', 'r', 's', 't'],
                                     ['u', 'v', 'w', 'x', 'y']])

        rearranged_grid_solution_str = np.array([['a', 'g', 'm', 's', 'y'],
                                                 ['l', 'r', 'x', 'e', 'f'],
                                                 ['w', 'd', 'j', 'k', 'q'],
                                                 ['i', 'o', 'p', 'v', 'c'],
                                                 ['t', 'u', 'b', 'h', 'n']])

        # Perform rearrangement
        rearranged_grid = mms.create_multiplex_ms_grid(grid=initial_grid, for_integer=True)
        rearranged_grid_str = mms.create_multiplex_ms_grid(grid=initial_grid_str, for_integer=False)

        np.testing.assert_array_equal(rearranged_grid, rearranged_grid_solution)
        np.testing.assert_array_equal(rearranged_grid_str, rearranged_grid_solution_str)

    def test_revert_multiplex_ms_grid(self):

        initial_grid = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                 23, 24, 25])

        rearranged_grid = np.array([1, 7, 13, 19, 25,
                                    12, 18, 24, 5, 6,
                                    23, 4, 10, 11, 17,
                                    9, 15, 16, 22, 3,
                                    20, 21, 2, 8, 14])

        rearranged_grid_str = np.array(['a', 'g', 'm', 's', 'y',
                                        'l', 'r', 'x', 'e', 'f',
                                        'w', 'd', 'j', 'k', 'q',
                                        'i', 'o', 'p', 'v', 'c',
                                        't', 'u', 'b', 'h', 'n'])

        initial_grid_str = np.array(['a', 'b', 'c', 'd', 'e',
                                     'f', 'g', 'h', 'i', 'j',
                                     'k', 'l', 'm', 'n', 'o',
                                     'p', 'q', 'r', 's', 't',
                                     'u', 'v', 'w', 'x', 'y'])

        reversed_grid = mms.revert_multiplex_ms_grid(rearranged_grid, for_integer=True)
        reversed_grid_str = mms.revert_multiplex_ms_grid(rearranged_grid_str, for_integer=False)

        np.testing.assert_array_equal(reversed_grid, initial_grid)
        np.testing.assert_array_equal(reversed_grid_str, initial_grid_str)

    def test_observed_presence_to_flattened_grid(self):
        row_vector = np.array([1, 0, 1, 0, 0, 0])
        column_vector = np.array([1, 0, 1, 0, 0, 0])

        solution = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.assertIsNone(np.testing.assert_array_equal(solution,
                                                        mms.observed_presence_to_flattened_grid(
                                                            column_vector, row_vector)))

    def test_deconvolute_multiplexed_grid(self):
        def test_deconvolute_multiplexed_grid(self):
            features = ['f1', 'f2', 'f3']
            samples = ['a', 'b', 'c', 'd', 'f', 'g', 'h', 'i']

            index_pooled = ['grid_1_row_1_initial', 'grid_1_row_2_initial', 'grid_1_row_3_initial',
                            'grid_1_column_1_initial', 'grid_1_column_2_initial', 'grid_1_column_3_initial',
                            'grid_1_row_1_rearranged', 'grid_1_row_2_rearranged', 'grid_1_row_3_rearranged',
                            'grid_1_column_1_rearranged', 'grid_1_column_2_rearranged', 'grid_1_column_3_rearranged']

            feature_presence_table = [[1, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0],
                                      [1, 0, 0], [0, 1, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0]]
            feature_presence_table = pd.DataFrame(feature_presence_table, index=index_pooled, columns=features)

            initial_grid = [['a', 'b', 'c'], ['d', 'placeholder', 'f'], ['g', 'h', 'i']]
            initial_grid = pd.DataFrame(initial_grid, index=index_pooled[:3], columns=index_pooled[3:6])

            rearranged_grid = [['a', 'placeholder', 'i'], ['h', 'c', 'd'], ['f', 'g', 'b']]
            rearranged_grid = pd.DataFrame(rearranged_grid, index=index_pooled[6:9], columns=index_pooled[9:])

            solution = np.reshape(np.zeros(24), (3, 8))
            solution[1, 2] = 1
            solution = pd.DataFrame(solution, index=features, columns=samples, dtype='int8')

            self.assertIsNone(
                pd.testing.assert_frame_equal(solution, mms.deconvolute_multiplexed_grid(initial_grid,
                                                                                         rearranged_grid,
                                                                                         feature_presence_table)))


class TestMultiplexMsGUIFunctions(unittest.TestCase):

    def test_read_sample_list(self):
        path = os.path.join(TEST_PATH, 'test_files/sample_names_test.txt')

        sample_list = mms.read_sample_list(path)
        self.assertEqual(sample_list, ['sample1', 'sample2', 'sample3', 'sample5'])

        invalid_suffix = 'wrong_suffix.xlsx'
        self.assertRaises(FileNotFoundError, mms.read_sample_list, invalid_suffix)

    def test_combine_grids(self):

        valid_list = ['grid_1_initial.csv', 'grid_1_rearranged.csv',
                      'grid_2_initial.csv', 'grid_2_rearranged.csv',
                      'grid_3_initial.csv', 'grid_3_rearranged.csv']

        valid_result = [('grid_1_initial.csv', 'grid_1_rearranged.csv'),
                        ('grid_2_initial.csv', 'grid_2_rearranged.csv'),
                        ('grid_3_initial.csv', 'grid_3_rearranged.csv')]

        invalid_list_no_files = []

        invalid_list_uneven_initial = ['grid_1_initial.csv', 'grid_1_rearranged.csv',
                                       'grid_2_initial.csv', 'grid_2_rearranged.csv',
                                       'grid_3_rearranged.csv']

        invalid_list_uneven_rearranged = ['grid_1_initial.csv', 'grid_1_rearranged.csv',
                                          'grid_2_initial.csv', 'grid_2_rearranged.csv',
                                          'grid_3_rearranged.csv']

        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = valid_list
            self.assertEqual(mms.combine_grids(path_to_grids='mock'), valid_result)

        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = invalid_list_no_files
            self.assertRaises(FileNotFoundError, mms.combine_grids, "mock")

        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = invalid_list_uneven_rearranged
            self.assertRaises(FileNotFoundError, mms.combine_grids, "mock")

        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = invalid_list_uneven_initial
            self.assertRaises(FileNotFoundError, mms.combine_grids, "mock")

    def test_distribute_samples_in_grids_with_placeholders(self):
        r = 4
        sample_count = 100
        max_grid_size = r * r
        grid_count = np.ceil(sample_count / max_grid_size)

        solution = [14, 14, 14, 14, 14, 15, 15]

        attempt = mms.distribute_samples_in_grids_with_placeholders(sample_count=int(sample_count),
                                                                    max_grid_size=int(max_grid_size),
                                                                    grid_count=int(grid_count))

        self.assertEqual(solution, attempt)

    def test_create_initial_grid_with_placeholders(self):
        r = 4
        remaining_samples_count = 13
        sample_names = np.array(['sample_' + str(i) for i in range(1, 14)])
        seed = 42

        solution = np.array([['sample_1', 'sample_2', 'placeholder', 'sample_3'],
                             ['sample_4', 'sample_5', 'sample_6', 'sample_7'],
                             ['sample_8', 'sample_9', 'placeholder', 'placeholder'],
                             ['sample_10', 'sample_11', 'sample_12', 'sample_13']])

        self.assertIsNone(
            np.testing.assert_array_equal(solution,
                                          mms.create_initial_grid_with_placeholders(r,
                                                                                    remaining_samples_count,
                                                                                    sample_names,
                                                                                    seed))
        )

