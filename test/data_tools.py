import pandas as pd
import numpy as np

import unittest
import io

from unittest.mock import patch
from wolta import data_tools

# FILE ID: 0


# FUNCTION ID: 0
class TestColTypes(unittest.TestCase):

    # TEST ID: 0
    def test_none(self):
        with self.assertRaises(TypeError):
            data_tools.col_types(None, print_columns=None)

        sasha = pd.read_csv('../test_data/traditional/sasha.csv')

        with self.assertRaises(TypeError):
            data_tools.col_types(sasha, print_columns=None)

        with self.assertRaises(TypeError):
            data_tools.col_types(None, print_columns=True)

    # TEST ID: 1
    def test_other_types(self):
        with self.assertRaises(TypeError):
            data_tools.col_types("array", print_columns="false")

        sasha = pd.read_csv('../test_data/traditional/sasha.csv')

        with self.assertRaises(TypeError):
            data_tools.col_types(sasha, print_columns="false")

        with self.assertRaises(TypeError):
            data_tools.col_types("array", print_columns=False)

    # TEST ID: 3
    def test_empty(self):
        self.assertAlmostEqual(data_tools.col_types(pd.DataFrame()), [])

    # TEST ID: 2
    def test_perfect(self):
        sasha = pd.read_csv('../test_data/traditional/sasha.csv')
        result = ['int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64']
        self.assertEqual(data_tools.col_types(sasha), result)

    # TEST ID: 4
    def test_output(self):
        with patch('sys.stdout', new=io.StringIO()) as console:
            data_tools.col_types(pd.DataFrame(), print_columns=True)
            output = console.getvalue().strip()
            self.assertEqual(output, 'The dataframe is empty!')

        sasha = pd.read_csv('../test_data/traditional/sasha.csv')

        with patch('sys.stdout', new=io.StringIO()) as console:
            data_tools.col_types(sasha, print_columns=True)
            output = console.getvalue().strip()
            self.assertEqual(output, 'discrete1: int64\ndiscrete2: int64\ndiscrete3: int64\ncontinuous1: int64\ncontinuous2: int64\ncontinuous3: int64\ncontinuous4: int64\ncontinuous5: int64\ncontinuous6: int64\ncontinuous7: int64\noutput: int64')