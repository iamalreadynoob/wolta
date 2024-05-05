# WOLTA DOCUMENTATION

Wolta is designed for making simplify the frequently used processes which includes Pandas, Numpy and Scikit-Learn in Machine Learning.
<br><br>
Currently there are four modules inside the library, which are 'data_tools', 'model_tools', 'progressive_tools' and 'feature_tools'

## Installation

```
pip install wolta
```

## Data Tools

Data Tools was designed for manipulating the data.

### load_by_parts

**Returns**: _pandas dataframe_

**Parameters**:
<br>
* paths, _python list_
* strategy, {'default', 'efficient'}, by default, 'default'
* deleted_columns, _python string list_, by default, None
* print_description, _boolean_, by default, False
* shuffle, _boolean_ , by default, False
* encoding, _string_, by default, 'utf-8'

1. _paths_ holds the locations of data files.
<br>
2. If _strategy_ is 'default', then the datatypes of columns are assigned with maximum bytes (64).
<br>
3. If _strategy_ is 'efficient', then the each column is examined and the min-max values are detected. According to the info, least required byte amount is assigned to each column.
<br>
4. _deleted\_columns_ holds the names of the columns that will be directly from each sub dataframe.
<br>
5. If _print\_description_ is True, then how many paths have been read is printed out in the console.

```python
from wolta.data_tools import load_by_parts
import glob

paths = glob.glob('path/to/dir/*.csv')
df = load_by_parts(paths)
```

***

### col_types

**Returns**: _python string list_

**Parameters**:
<br>
* df, _pandas dataframe_
* print_columns, _boolean_, by default, False
<br>

1. _df_ holds the dataframe and for each datatype for columns are returned.
2. If _print\_columns_ is True, then 'class name: datatype' is printed out for each column.

```python
import pandas as pd
from wolta.data_tools import col_types

df = pd.read_csv('data.csv')
types = col_types(df)
```

***

### make_numerics

**Returns**: 
1. _pandas dataframe column_ which has int64 data inside it
2. If space_requested is True, then dictionary that used in mapping

**Parameters**: 
* column, _pandas dataframe column_
* space_requested, _boolean_, by default, False

```python
import pandas as pd
from wolta.data_tools import make_numerics

df = pd.read_csv('data.csv')
df['output'] = make_numerics(df['output'])
```

***

### make_null

**Returns**: pandas dataframe or numpy array

**Parameters**:

* matrix (pandas dataframe or numpy array)
* replace, string, the text which will converted to null
* type, {'df', 'np'}, by default 'df'

***

### seek_null

**Returns**: boolean

**Parameters**:

* df, pandas dataframe
* print_columns, boolean, False by default

***

### transform_data

**Returns**:
1. transformed X
2. transformed y
3. if strategy ends with 'm', amin_x
4. if strategy ends with 'm', amin_y

**Parameters**:

* X
* y
* strategy, {'log', 'log-m', 'log2', 'log2-m', 'log10', 'log10-m', 'sqrt', 'sqrt-m', 'cbrt'}, by default 'log-m'

If you concern about situations like sqrt(0) or log(0), use strategies which ends with 'm' (means manipulated).

***

### transform_pred

This function is designed for make predictions realistic and handles back-transformation.

**Returns**: back-transformed y

**Parameters**:

* y_pred
* strategy, {'log', 'log-m', 'log2', 'log2-m', 'log10', 'log10-m', 'sqrt', 'sqrt-m', 'cbrt'}, by default 'log-m'
* amin_y, int, by default 0. amin_y is used min y value in transform_data if data was manipulated and it is required if a strategy which ends with 'm' was selected.

***

### make_categorical

It places regression outputs into three sub-classes according to mean and standard deviation. Normal distribution is required.

**Returns**: 
1. y
the other outputs are returned if only normal-extra is selected
2. the min value of array
3. the max value of array
4. the standard deviation of array
5. the mean value of array
6. the result of mean - standard deviation
7. the result of mean + standard deviation

**Parameters**:
* y
* strategy, {'normal', 'normal-extra'}, default by, 'normal'

***

### state_sum

It gives information about features.

**Returns**: dictionary which keys are feature names. (if get_dict param is True)

**Parameters**:
* df, pandas dataframe
* requested, array of string

| value | meaning                     |
| --- |-----------------------------|
| min | minimum value for a feature |
| max | maximum value for a feature |
| width | difference between max and min |
| std | standard |
| mean | mean |
| med | median |
| var | variance |

* only, list of string, it gets results for these features only, by default None. If it is none, function gets results for all features
* exclude, list of string, it gets results for except these features, by default None.
* get_dict, by default False
* verbose, by default True

***

### create_chunks

**Parameters**:

* path, _string_
* sample_amount, _int_, sample amount for each chunk
* target_dir, _string_, directory path to save chunks, by default, None
* print_description, _boolean_, shows the progress in console or not, by default, False
* chunk_name, _string_, general name for chunks, by default, 'part'

```python
from wolta.data_tools import create_chunks
create_chunks('whole_data.csv', 1000000)
```

***

### unique_amounts

**Returns**: dictionary with <string, int> form, <column name, unique value amount>
**Parameters**:
<br>
1. df, _pandas dataframe_
2. strategy, _python string list_, by default, None, it is designed for to hold requested column names
3. print_dict, _boolean_, by default, False

```python
import pandas as pd
from wolta.data_tools import unique_amounts

df = pd.read_csv('data.csv')
amounts = unique_amounts(df)
```

***

### scale_x

**Returns**:
1. X_train
2. X_test

**Parameters**:
1. X_train
2. X_test

It makes Standard Scaling.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from wolta.data_tools import scale_x

df = pd.read_csv('data.csv')

y = df['output']
del df['output']
X = df
del df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = scale_x(X_train, X_test)
```

***

### is_normal

**Returns**: Boolean
<br>
if data has normal distribution returns true, else returns false
<br>
**Parameter**: y

***

### examine_floats

**Returns**: _list_ with full of column names which supplies the requested situation

**Parameters**:
* df, _pandas dataframe_
* float_columns, _string list_, column names which has float data
* get, {'float', 'int'}, by default, 'float'.

1. If _get_ is 'float', then it returns the names of the columns which has float values rather than .0
2. If _get_ is 'int', then it returns the names of the columns which has int values with .0

```python
import pandas as pd
from wolta.data_tools import examine_floats

df = pd.read_csv('data.csv')
float_cols = ['col1', 'col2', 'col3']

float_cols = examine_floats(df, float_cols)
```

***

### calculate_min_max

**Returns**:
1. columns, _list of string_, column names which holds int or float data
2. types, _list of string_, datatypes of these columns
3. max_val, _list of int & float_, holds the maximum value for each column
4. min_val, _list of int & float_, holds the minimum value for each column

**Parameters**:
* paths, _list of string_, paths of dataframes
* deleted_columns, _list of string_, initially excluded column names, by default, None

```python
import glob
from wolta.data_tools import calculate_min_max

paths = glob.glob('path/to/dir/*.csv')
columns, types, max_val, min_val = calculate_min_max(paths)
```

***

### calculate_bounds

**Returns**: list of string, name of types with optimum sizes.

**Parameters**:
* gen_types, _list of string_, holds the default datatypes for each column
* min_val, _list of int & float_, holds the minimum value for each column
* max_val, _list of int & float_, holds the maximum value for each column

```python
import glob
from wolta.data_tools import calculate_bounds, calculate_min_max

paths = glob.glob('path/to/dir/*.csv')
columns, types, max_val, min_val = calculate_min_max(paths)
types = calculate_bounds(types, min_val, max_val)
```

***

## Model Tools

Model Tools was designed for to get some results on models.

### get_score

**Returns**: dictionary with <string, float> form, <score type, value>, also prints out the result by default

**Parameters**:

* y_test, _1D numpy array_
* y_pred, _1D numpy array_
* metrics, _list of string_, this list can only have values the table below:

For 'clf':

| value | full name      |
| --- |----------------|
| acc | accuracy score |
| f1 | f1 score       |
| hamming | hamming loss   |
| jaccard | jaccard score |
| log | log loss |
| mcc | matthews corrcoef |
| precision | precision score |
| recall | recall score |
| zol | zero one loss |

by default, ['acc']
<br>

For 'reg':

| value   | full name                          |
|---------|------------------------------------|
| var     | explained variance                 |
| max     | max error                          |
| abs     | neg mean absolute error            |
| sq      | neg mean squared error             |
| rsq     | neg root mean squared error        |
| log     | neg mean squared log error         |
| rlog    | neg mean squared log error         |
| medabs  | neg median absolute error          |
| poisson | neg mean poisson deviance          |
| gamma   | neg mean gamma deviance            |
| per     | neg mean absolute percentage error |
| d2abs   | d2 absolute error score            |
| d2pin   | d2 pinball score                   |
| d2twe   | d2 tweedie score                   |

by default, ['sq']

* average, string, {'weighted', 'micro', 'macro', 'binary', 'samples'}, by default, 'weighted'
* algo_type, {'clf', 'reg'}, 'clf' by default

```python
import numpy as np
from wolta.model_tools import get_score

y_test = np.load('test.npy')
y_pred = np.load('pred.npy')

scores = get_score(y_test, y_pred, ['acc', 'precision'])
```

***

### get_supported_metrics

It returns the string list of possible score names for _get\_score_ function

```python
from wolta.model_tools import get_supported_metrics

print(get_supported_metrics())
```

***

### get_avg_options

It returns the string list of possible average values for _get\_score_ function

```python
from wolta.model_tools import get_avg_options

print(get_avg_options())
```

***

### compare_models

**Returns**: nothing, just prints out the results

**Parameters**:

* algo_type, {'clf', 'reg'}
* algorithms, list of string, if the first element is 'all' then it gets results for every algorithm.

for 'clf':

| value | full name |
|-------|-----------|
| cat   | catboost  |
| ada | adaboost  |
| dtr | decision tree |
| raf | random forest |
| lbm | lightgbm |
| ext | extra tree |
| log | logistic regression |
| knn | knn |
| gnb | gaussian nb |
| rdg | ridge |
| bnb | bernoulli nb |
| svc | svc |
| per | perceptron |
| mnb | multinomial nb |

for 'reg':

| value | full name         |
|-------|-------------------|
| cat   | catboost          |
| ada   | adaboost          |
| dtr   | decision tree     |
| raf   | random forest     |
| lbm   | lightgbm          |
| ext   | extra tree        |
| lin   | linear regression |
| knn   | knn               |
| svr   | svr               |

* metrics, list of string, its values must be acceptable for get_score method
* X_train
* y_train
* X_test
* y_test

***

### do_combinations

**Returns**: list of the int lists

**Parameters**:
* indexes, _list of int_
* min_item, _int_, it is the minimum amount of index inside a combination
* max_item, _int_, it is the maximum amount of index inside a combination

It creates a list for all possible min_item <= x <= max_item terms combinations

```python
from wolta.model_tools import do_combinations

combinations = do_combinations([0, 1, 2], 1, 3)
```

***

### do_voting

**Returns**: list of 1D numpy arrays

**Parameters**:
* y_pred_list, _list of 1D numpy arrays_
* combinations, _list of int lists_, it holds the indexes from y_pred_list for each combination
* strategy, {'avg', 'mode'}, default by, 'avg'

If 'avg' is selected then this function makes sum of matrices, then divides it the amount of matrices and finally makes whole matrix as int value.
<br>
If 'mode' is selected then for every sample, the predicts are collected and then mode is found one by one.

```python
import numpy as np
from wolta.model_tools import do_voting, do_combinations

y_pred_1 = np.load('one.npy')
y_pred_2 = np.load('two.npy')
y_pred_3 = np.load('three.npy')
y_preds = [y_pred_1, y_pred_2, y_pred_3]

combinations = do_combinations([0, 1, 2], 1, 3)
results = do_voting(y_preds, combinations)
```

***

### WelkinClassification

The Welkin Classification has a very basic idea. It calculates min and max values for each feature for every class according to the training data. Then, in prediction process, it checks every classes one by one, if input features between the range that detected, it gives a score. The class which has the biggest score is become the predict. Ofcourse this is for 'travel' strategy. If the strategy is 'limit', then if m of features has value between those ranges, that becomes the answer and the other possible answers aren't investigated. This strategy is recommended for speed. At this point, feature investigation order becomes crucial so they can be reordered with 'priority' parameter.

**Parameters**:
* strategy, {'travel', 'limit'}, by default, 'travel'
* priority, list of feature indexes, by default, None
* limit, integer, by default, None

This class has those functions:

* fit(X_train, y_train)
* predict(X_test), returns y_pred

***

### DistRegressor

This regression approach provides a hybrid solution for problems which have output space in wide range thanks to normal distribution. 

**Parameters**:
* verbose, boolean, by default, True
* clf_model, classification model class, by default, None (If it is None, CatBoostClassifier is used with default parameters except iterations, iterations has 20 as value)
* clf_params, parameters for classification model in dict form, by default, None
* reg_model, regression model, by default, None (If it is None, CatBoostRegressor is used with default parameters except iterations, iterations has 20 as value)
* reg_params, parameters for regression model in dict form, by default, None

This class has those functions:

* fit(X_train, y_train)
* predict(X_test), returns y_pred

***


***

### examine_time

It calculates the fitting time for a model and also returns the trained model.

**Returns**: 
1. int
2. model

**Parameters**:
* model
* X_train
* y_train

```python
from wolta.model_tools import examine_time
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X_train = np.load('x.npy')
y_train = np.load('y.npy')

model = RandomForestClassifier(random_state=42)
consumed, model = examine_time(model, X_train, y_train)
```

***

## Progressive Tools

This module has been designed for progressive sampling.

### make_run

It was designed to use one loaded numpy array for all sampling trials.

**Returned**:
1. list of int, percentage logging
2. list of dictionaries, metrics logging

**Parameters**:
* model_class
* X_train
* y_train
* X_test
* y_test
* init_per, int, default by, 1, inclusive starting percentage
* limit_per, int, default by, 100, inclusive ending percentage
* increment, int, default by, 1
* metrics, list of string, the values must be recognizable for model_tools.get_score(), default by, ['acc']
* average, string, the value must be recognizable for model_tools.get_score(), default by, 'weighted'
* params, dictionary, if model has parameters, they initialize it here, default by, None

```python
from wolta.progressive_tools import make_run
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('x_test.npy')
y_test = np.load('x_test.npy')

percentage_log, metrics_log = make_run(RandomForestClassifier, X_train, y_train, X_test, y_test)
```

***

### get_best

**Returns**:
1. int, best percentage
2. float, best score

**Parameters**:
* percentage_log, _list of int_
* metrics_log, _list of dictionary_
* requested_metrics, _string_

```python
from wolta.progressive_tools import make_run, get_best
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('x_test.npy')
y_test = np.load('x_test.npy')

percentage_log, metrics_log = make_run(RandomForestClassifier, X_train, y_train, X_test, y_test)
best_per, best_score = get_best(percentage_log, metrics_log, 'acc')
```

***

### path_chain

Unlike make_run, it loads train data from different files every time.

**Returns**: list of dictionary, metrics logging

**Parameters**:
* paths, _list of string_
* model_class
* X_test
* y_test
* output_column, _string_
* metrics, list of string, the values must be recognizable for model_tools.get_score(), default by, ['acc']
* average, string, the value must be recognizable for model_tools.get_score(), default by, 'weighted'
* params, dictionary, if model has parameters, they initialize it here, default by, None

```python
from wolta.progressive_tools import path_chain
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import glob

X_test = np.load('x_test.npy')
y_test = np.load('x_test.npy')

paths = glob.glob('path/to/dir/*.csv')

percentage_log, metrics_log = path_chain(paths, RandomForestClassifier, X_test, y_test, 'output')
```

***

## Feature Tools

This module is about to manipulating features in datasets.

### quest_selection

Prints out suggestions about what feature(s) can be deleted with less loss or maximum gain.

The algorithm works with two steps: Firstly, It removes one feature for each time and compares accuracies between current situation and whole-features case. If new accuracy is the better than whole-feature one or their difference less-equal than flag_one_tol, it passes to the second step.
<br>
The next process 'trials' times creates combinations with random amounts of passed features and they are removed at same time. If new accuracy is the better than whole-feature one or their difference less-equal than fin_tol, it becomes a suggestion.

**Parameters**:
* model_class
* X_train
* y_train
* X_test
* y_test
* features, list of string, holds column names for X.
* flag_one_tol, float
* fin_tol, float
* params, dictionary, if model has parameters, they initialize it here, default by, None
* normal_acc, float, default by, None. If it is None then it is calculated first of all
* trials, int, default by, 100