# Data Tools Instructions

## col_types

1. Take df as a Pandas Dataframe.
2. Take printed_columns as a Boolean, if there is no passed argument for it, take it False.

```python
def col_types(df, print_columns=False)
```

3. If there is any issue in datatypes of passed arguments, raise an error. Same as well if they are None.

```python
if not isinstance(df, pd.DataFrame):
    raise TypeError('df is not a Pandas Dataframe!')

if not isinstance(print_columns, bool):
    raise TypeError('print_columns is not a Boolean!')
```

4. Create an empty list

```python
types = []
```

5. Get all feature names from df in 'list' format.

```python
columns = list(df.columns)
```

6. In a loop, get the very first item in each column and obtain its datatype information in String format. It will be like this:

```
<class 'datatype'>
```

7. Get an inner string between 8 and -2 indexes, delete if it has a 'numpy.' phrase.

```python
for i in range(len(columns)):
    sub_type = str(type(df.iloc[0, i]))[8: -2].replace('numpy.', '')
    types.append(sub_type)
```

8. If print_columns is True and there are actual data inside, then print it.

```python
if print_columns:
    if len(columns) > 0:
        for i in range(len(columns)):
            print('{}: {}'.format(columns[i], types[i]))

    else:
        print('The dataframe is empty!')
```

9. Return list

```python
return types
```