import pandas as pd
import numpy as np

import unittest
import io
import warnings

from unittest.mock import patch
from wolta import data_tools
from loader import get_dataset


class TestColTypes(unittest.TestCase):

    def test_none(self):
        with self.assertRaises(TypeError):
            data_tools.col_types(None, print_columns=None)

        sasha = pd.read_csv(get_dataset('sasha'))

        with self.assertRaises(TypeError):
            data_tools.col_types(sasha, print_columns=None)

        with self.assertRaises(TypeError):
            data_tools.col_types(None, print_columns=True)

    def test_other_types(self):
        with self.assertRaises(TypeError):
            data_tools.col_types("array", print_columns="false")

        sasha = pd.read_csv(get_dataset('sasha'))

        with self.assertRaises(TypeError):
            data_tools.col_types(sasha, print_columns="false")

        with self.assertRaises(TypeError):
            data_tools.col_types("array", print_columns=False)

    def test_empty(self):
        self.assertEqual(data_tools.col_types(pd.DataFrame()), [])

    def test_perfect(self):
        sasha = pd.read_csv(get_dataset('sasha'))
        result = ['int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64']
        self.assertEqual(data_tools.col_types(sasha), result)

    def test_output(self):
        with patch('sys.stdout', new=io.StringIO()) as console:
            data_tools.col_types(pd.DataFrame(), print_columns=True)
            output = console.getvalue().strip()
            self.assertEqual(output, 'The dataframe is empty!')

        sasha = pd.read_csv(get_dataset('sasha'))

        with patch('sys.stdout', new=io.StringIO()) as console:
            data_tools.col_types(sasha, print_columns=True)
            output = console.getvalue().strip()
            self.assertEqual(output, 'discrete1: int64\ndiscrete2: int64\ndiscrete3: int64\ncontinuous1: int64\ncontinuous2: int64\ncontinuous3: int64\ncontinuous4: int64\ncontinuous5: int64\ncontinuous6: int64\ncontinuous7: int64\noutput: int64')


class TestUniqueAmounts(unittest.TestCase):
    def test_none(self):
        with self.assertRaises(TypeError):
            data_tools.unique_amounts(None)

        sasha = pd.read_csv(get_dataset('sasha'))

        with self.assertRaises(TypeError):
            data_tools.unique_amounts(sasha, print_dict=None)

        with self.assertRaises(TypeError):
            data_tools.unique_amounts(None, print_dict=None)

    def test_other_types(self):
        with self.assertRaises(TypeError):
            data_tools.unique_amounts("array", print_dict="false")

        sasha = pd.read_csv(get_dataset('sasha'))

        with self.assertRaises(TypeError):
            data_tools.unique_amounts(sasha, print_dict="false")

        with self.assertRaises(TypeError):
            data_tools.unique_amounts(sasha, strategy="list")

        with self.assertRaises(TypeError):
            data_tools.unique_amounts("array", strategy='list', print_dict='false')

    def test_empty(self):
        self.assertEqual(data_tools.unique_amounts(pd.DataFrame()), {})

    def test_perfect(self):
        sasha = pd.read_csv(get_dataset('sasha'))
        result = {'discrete1': 7, 'discrete2': 3, 'discrete3': 5, 'continuous1': 46, 'continuous2': 963, 'continuous3': 438, 'continuous4': 112, 'continuous5': 49, 'continuous6': 360, 'continuous7': 768, 'output': 4}
        self.assertEqual(data_tools.unique_amounts(sasha), result)

        requested = ['discrete1', 'continuous2', 'discrete3']
        result = {'discrete1': 7, 'discrete3': 5, 'continuous2': 963}
        self.assertEqual(data_tools.unique_amounts(sasha, strategy=requested), result)

    def test_invalid_keys(self):
        sasha = pd.read_csv(get_dataset('sasha'))
        requested = ['discrete1', 'continuous2', 'invalid_key', 'discrete3']
        result = {'discrete1': 7, 'discrete3': 5, 'continuous2': 963}
        self.assertEqual(data_tools.unique_amounts(sasha, strategy=requested), result)

    def test_output(self):
        sasha = pd.read_csv(get_dataset('sasha'))

        with patch('sys.stdout', new=io.StringIO()) as console:
            data_tools.unique_amounts(sasha, print_dict=True)
            output = console.getvalue().strip()
            self.assertEqual(output, 'continuous1: 46 different values\ncontinuous2: 963 different values\ncontinuous3: 438 different values\ncontinuous4: 112 different values\ncontinuous5: 49 different values\ncontinuous6: 360 different values\ncontinuous7: 768 different values\ndiscrete1: 7 different values\ndiscrete2: 3 different values\ndiscrete3: 5 different values\noutput: 4 different values')

        with patch('sys.stdout', new=io.StringIO()) as console:
            data_tools.unique_amounts(sasha, strategy=['discrete1', 'continuous2', 'discrete3'], print_dict=True)
            output = console.getvalue().strip()
            self.assertEqual(output, 'continuous2: 963 different values\ndiscrete1: 7 different values\ndiscrete3: 5 different values')

        with patch('sys.stdout', new=io.StringIO()) as console:
            data_tools.unique_amounts(sasha, strategy=['discrete1', 'continuous2', 'invalid_key', 'discrete3'], print_dict=True)
            output = console.getvalue().strip()
            self.assertEqual(output, 'continuous2: 963 different values\ndiscrete1: 7 different values\ndiscrete3: 5 different values')

    def test_warning(self):
        sasha = pd.read_csv(get_dataset('sasha'))

        with self.assertRaises(Warning) as warn:
            data_tools.unique_amounts(pd.DataFrame(), print_dict=True)

        self.assertEqual(str(warn.exception).strip(), 'The dataframe is empty!')

        with self.assertRaises(Warning) as warn:
            data_tools.unique_amounts(sasha, strategy=['invalid_key'], print_dict=True)

        self.assertEqual(str(warn.exception).strip(), 'There is no such a column as requested.')