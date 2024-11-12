import unittest
import pandas as pd
import numpy as np

from wolta import data_tools


class TestColTypes(unittest.TestCase):
    def test_none(self):
        with self.assertRaises(TypeError):
            data_tools.col_types(None, print_columns=None)

        print('Test 0-0-0-both is done ✅')

        sasha = pd.read_csv('../test_data/traditional/sasha.csv')

        with self.assertRaises(TypeError):
            data_tools.col_types(sasha, print_columns=None)

        print('Test 0-0-0-print_columns is done ✅')

        with self.assertRaises(TypeError):
            data_tools.col_types(None, print_columns=True)

        print('Test 0-0-0-df is done ✅')

    def test_other_types(self):
        with self.assertRaises(TypeError):
            data_tools.col_types("array", print_columns="false")

        print('Test 0-0-1-both is done ✅')

        sasha = pd.read_csv('../test_data/traditional/sasha.csv')

        with self.assertRaises(TypeError):
            data_tools.col_types(sasha, print_columns="false")

        print('Test 0-0-1-print_columns is done ✅')

        with self.assertRaises(TypeError):
            data_tools.col_types("array", print_columns=False)

        print('Test 0-0-1-df is done ✅')

    def test_empty(self):
        self.assertAlmostEqual(data_tools.col_types(pd.DataFrame()), [])
        print('Test 0-0-3 is done ✅')

    def test_perfect(self):
        sasha = pd.read_csv('../test_data/traditional/sasha.csv')
        result = ['int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64']
        self.assertAlmostEqual(data_tools.col_types(sasha), result)
        print('Test 0-0-2-int is done ✅')