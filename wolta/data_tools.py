import numpy as np


def col_types(df, print_columns=False):
    types = []

    columns = list(df.columns)
    for i in range(len(columns)):
        sub_type = str(type(df.iloc[0, i]))[8: -2].replace('numpy.', '')
        types.append(sub_type)

    if print_columns:
        for i in range(len(columns)):
            print('{}: {}'.format(columns[i], types[i]))

    return types


def unique_amounts(df, strategy=None, print_dict=False):
    columns = list(df.columns)

    if strategy is None:
        space = {}

        for col in columns:
            amount = len(list(df[col].unique()))
            space[col] = amount

        if print_dict:
            print(space)

        return space

    else:
        space = {}

        for col in columns:
            amount = len(list(df[col].unique()))

            if col in strategy:
                space[col] = amount

        if print_dict:
            print(space)

        return space


def make_numerics(column, space_requested=False):
    unique_vals = list(column.unique())
    space = {}

    for i in range(len(unique_vals)):
        space[unique_vals[i]] = i

    column = column.map(space)

    if space_requested:
        return column, space
    else:
        return column


def scale_x(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()

    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.transform(X_test)

    return X_train, X_test


def examine_floats(df, float_columns, get='float'):
    space = []

    for col in float_columns:
        is_int = df[col].eq(df[col].round(0)).all()

        if get == 'float' and not is_int:
            space.append(col)
        elif get == 'int' and is_int:
            space.append(col)

    return space


def calculate_bounds(gen_types, min_val, max_val):
    types = []

    for i in range(len(gen_types)):
        if gen_types[i].startswith('int'):
            if min_val[i] > -127 and max_val[i] < 128:
                types.append('int8')
            elif min_val[i] > -32768 and max_val[i] < 32767:
                types.append('int16')
            elif min_val[i] > -2147483648 and max_val[i] < 2147483647:
                types.append('int32')
            else:
                types.append('int64')

        elif gen_types[i].startswith('float'):
            if min_val[i] > 1.17549435e-38 and max_val[i] < 3.40282346e+38:
                types.append('float32')
            else:
                types.append('float64')

    return types


def calculate_min_max(paths, deleted_columns=None):
    import pandas as pd

    min_val = []
    max_val = []
    columns = None
    types = []

    for path in paths:
        sub_df = pd.read_csv(path)

        if deleted_columns is not None:
            for col in deleted_columns:
                del sub_df[col]

        if columns is None:
            columns = list(sub_df.columns)

            will_deleted = []

            for i in range(len(columns)):
                sub_type = str(type(sub_df.iloc[0, i]))[8:-2].replace('numpy.', '')

                if sub_type.startswith('int'):
                    types.append(sub_type)
                    max_val.append(0)
                    min_val.append(0)
                elif sub_type.startswith('float'):
                    types.append(sub_type)
                    max_val.append(float(0))
                    min_val.append(float(0))
                else:
                    will_deleted.append(columns[i])

            for i in range(len(will_deleted)):
                columns.remove(will_deleted[i])

            del will_deleted

        for i in range(len(columns)):
            sub_max = sub_df[columns[i]].max()
            sub_min = sub_df[columns[i]].min()

            if sub_max > max_val[i]:
                max_val[i] = sub_max

            if sub_min < min_val[i]:
                min_val[i] = sub_min

    return columns, types, max_val, min_val


def load_by_parts(paths, strategy='default', deleted_columns=None, print_description=False, shuffle=False, encoding='utf-8'):
    import pandas as pd

    if strategy == 'default':
        dfs = []
        loc = 0
        for path in paths:
            df = pd.read_csv(path, encoding=encoding)

            if deleted_columns is not None:
                for col in deleted_columns:
                    del df[col]

            if shuffle:
                df = df.sample(frac=1).reset_index(drop=True)

            dfs.append(df)

            if print_description:
                loc += 1
                print('{} out of {} paths done'.format(str(loc), str(len(paths))))

        main_df = None

        if shuffle:
            import random
            times = len(dfs)
            shuffled_dfs = []

            for i in range(times):
                index = random.randint(0, len(dfs) - 1)
                shuffled_dfs.append(dfs.pop(index))

            del dfs
            main_df = pd.concat(shuffled_dfs)
            del shuffled_dfs

        else:
            main_df = pd.concat(dfs)
            del dfs

        return main_df

    elif strategy == 'efficient':
        numeric_columns, types, max_val, min_val = calculate_min_max(paths, deleted_columns=deleted_columns)
        types = calculate_bounds(types, min_val, max_val)

        dtype = {}
        for i in range(len(numeric_columns)):
            dtype[numeric_columns[i]] = types[i]

        del numeric_columns, max_val, min_val, types

        dfs = []
        loc = 0

        for path in paths:
            df = pd.read_csv(path, dtype=dtype, encoding=encoding)

            if deleted_columns is not None:
                for col in deleted_columns:
                    del df[col]

            if shuffle:
                df = df.sample(frac=1).reset_index(drop=True)

            dfs.append(df)

            if print_description:
                loc += 1
                print('{} out of {} paths done'.format(str(loc), str(len(paths))))

        main_df = None

        if shuffle:
            import random
            times = len(dfs)
            shuffled_dfs = []

            for i in range(times):
                index = random.randint(0, len(dfs) - 1)
                shuffled_dfs.append(dfs.pop(index))

            del dfs
            main_df = pd.concat(shuffled_dfs)
            del shuffled_dfs

        else:
            main_df = pd.concat(dfs)
            del dfs

        return main_df


def create_chunks(path, sample_amount, target_dir=None, print_description=False, chunk_name='part'):
    import os

    with open(path, 'r', newline='') as f:
        lines = f.readlines()
        headers = lines[0]

        part = 0
        loc = 1

        while loc < len(lines):
            sub = [headers]
            times = 0

            while loc < len(lines) and times < sample_amount:
                sub.append(lines[loc])

                loc += 1
                times += 1

            sub_path = '{}-{}.csv'.format(chunk_name, str(part))
            if target_dir is not None:
                sub_path = os.path.join(target_dir, sub_path)

            with open(sub_path, 'w', newline='') as sf:
                sf.writelines(sub)

            if print_description:
                print('part {} is done'.format(str(part)))

            part += 1


def transform_data(X, y, strategy='log-m'):
    import numpy

    if strategy == 'log':
        X = np.log(X)
        y = np.log(y)

        return X, y

    elif strategy == 'log-m':
        amin_x = numpy.amin(X)
        amin_y = numpy.amin(y)

        X = np.log(X + np.abs(amin_x) + 1)
        y = np.log(y + np.abs(amin_y) + 1)

        return X, y, amin_x, amin_y

    elif strategy == 'log2':
        X = np.log2(X)
        y = np.log2(y)

        return X, y

    elif strategy == 'log2-m':
        amin_x = numpy.amin(X)
        amin_y = numpy.amin(y)

        X = np.log2(X + np.abs(amin_x) + 1)
        y = np.log2(y + np.abs(amin_y) + 1)

        return X, y, amin_x, amin_y

    elif strategy == 'log10':
        X = np.log10(X)
        y = np.log10(y)

        return X, y

    elif strategy == 'log10-m':
        amin_x = numpy.amin(X)
        amin_y = numpy.amin(y)

        X = np.log10(X + np.abs(amin_x) + 1)
        y = np.log10(y + np.abs(amin_y) + 1)

        return X, y, amin_x, amin_y

    elif strategy == 'sqrt':
        X = np.sqrt(X)
        y = np.sqrt(y)

        return X, y

    elif strategy == 'sqrt-m':
        amin_x = numpy.amin(X)
        amin_y = numpy.amin(y)

        X = np.sqrt(X + np.abs(amin_x) + 1)
        y = np.sqrt(y + np.abs(amin_y) + 1)

        return X, y, amin_x, amin_y

    elif strategy == 'cbrt':
        X = np.cbrt(X)
        y = np.cbrt(y)

        return X, y


def transform_pred(y_pred, strategy='log-m', amin_y=0):
    amin_y = np.abs(amin_y)

    if strategy == 'log':
        y_pred = np.exp(y_pred)

    elif strategy == 'log-m':
        y_pred = np.exp(y_pred) - amin_y - 1

    elif strategy == 'log2':
        y_pred = 2**y_pred

    elif strategy == 'log2-m':
        y_pred = 2**y_pred - amin_y - 1

    elif strategy == 'log10':
        y_pred = 10**y_pred

    elif strategy == 'log10-m':
        y_pred = 10**y_pred - amin_y - 1

    elif strategy == 'sqrt':
        y_pred = np.power(y_pred, 2)

    elif strategy == 'sqrt-m':
        y_pred = np.power(y_pred, 2) - amin_y - 1

    elif strategy == 'cbrt':
        y_pred = np.power(y_pred, 3)

    elif strategy == 'cbrt-m':
        y_pred = np.power(y_pred, 3) - amin_y - 1

    return y_pred


def make_categorical(y, strategy='normal'):
    import numpy as np

    if strategy == 'normal' or strategy == 'normal-extra':
        amin = np.amin(y)
        amax = np.amax(y)
        std = np.std(y)
        mean = np.mean(y)
        border_one = mean - std
        border_two = mean + std

        if amin >= border_one or amax <= border_two:
            raise ValueError('There is no such a normal distribution!')

        for i in range(y.shape[0]):
            if y[i] <= border_one:
                y[i] = 0
            elif y[i] >= border_two:
                y[i] = 2
            else:
                y[i] = 1

        if strategy == 'normal':
            return y
        else:
            return y, amin, amax, std, mean, border_one, border_two

def is_normal(y):
    amin = np.amin(y)
    amax = np.amax(y)
    std = np.std(y)
    mean = np.mean(y)
    border_one = mean - std
    border_two = mean + std

    if amin >= border_one or amax <= border_two:
        return False
    else:
        return True


def seek_null(df, print_columns=False):
    null_columns = []

    for col in df.columns:
        if df[col].isna().sum() > 0:
            null_columns.append(col)

            if print_columns:
                print('{} has {} null values'.format(col, str(df[col].isna().sum())))

    return null_columns


def make_null(matrix, replace, type='df'):
    if type == 'df':
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[0]):
                if matrix.iloc[j, i] == replace:
                    matrix.iloc[j, i] = None

        return matrix

    elif type == 'np':
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[0]):
                if matrix[j, i] == replace:
                    matrix.iloc[j, i] = None

        return matrix


def stat_sum(df, requested, only=None, exclude=None, get_dict=False, verbose=True):
    process_columns = []

    if only is None:
        process_columns = list(df.columns)
    else:
        process_columns = list(only)

    if exclude is not None:
        for col in exclude:
            if col in process_columns:
                process_columns.remove(col)

    if requested[0] == 'all':
        requested = ['min', 'max', 'width', 'mean', 'std', 'med', 'var']

    gen_results = {}

    for col in process_columns:
        results = {}

        if verbose:
            print(col)

        for req in requested:
            if req == 'min':
                res = np.amin(df[col].values)
                results[req] = res

                if verbose:
                    print('min: {}'.format(str(res)))

            elif req == 'max':
                res = np.amax(df[col].values)
                results[req] = res

                if verbose:
                    print('max: {}'.format(str(res)))

            elif req == 'width':
                res_max = np.amax(df[col].values)
                res_min = np.amin(df[col].values)
                res = res_max - res_min
                results[req] = res

                if verbose:
                    print('width: {}'.format(str(res)))

            elif req == 'mean':
                res = np.mean(df[col].values)
                results[req] = res

                if verbose:
                    print('mean: {}'.format(str(res)))

            elif req == 'std':
                res = np.std(df[col].values)
                results[req] = res

                if verbose:
                    print('std: {}'.format(str(res)))

            elif req == 'med':
                res = np.median(df[col].values)
                results[req] = res

                if verbose:
                    print('median: {}'.format(str(res)))

            elif req == 'var':
                res = np.var(df[col].values)
                results[req] = res

                if verbose:
                    print('variance: {}'.format(str(res)))

        gen_results[col] = results

        if verbose:
            print('***')

    if get_dict:
        return gen_results


def find_deflection(y_test, y_pred, arr=True, avg=False, gap=None, gap_type='num', dif_type='f-i', avg_w_abs=True, success_indexes=False):
    if arr == True or avg == True or gap is not None:
        diffs = []

        first = None
        second = None

        if dif_type == 'f-i':
            first = y_pred
            second = y_test
        elif dif_type == 'i-f':
            first = y_test
            second = y_pred
        else:
            first = y_pred
            second = y_test

        for i in range(y_test.shape[0]):
            diffs.append(first[i] - second[i])

        diffs = np.array(diffs)

        if dif_type == 'abs':
            diffs = np.abs(diffs)

        if arr == True and avg == False and gap is None:
            return diffs

        if avg == True:
            avg_score = 0

            tot_arr = diffs
            if dif_type != 'abs' and avg_w_abs == True:
                tot_arr = np.abs(diffs)

            for i in range(diffs.shape[0]):
                avg_score += tot_arr[i]

            avg_score /= diffs.shape[0]

            if arr == True and avg == True and gap is None:
                return diffs, avg_score

            if arr == False and avg == True and gap is None:
                return avg_score

            indexes = []

            if gap_type == 'exact':
                for i in range(y_test.shape[0]):
                    if y_test[i] == y_pred[i]:
                        indexes.append(i)
            else:
                for i in range(y_test.shape[0]):
                    lowest = None
                    highest = None

                    if gap_type == 'num':
                        lowest = y_test[i] - gap
                        highest = y_test[i] + gap
                    elif gap_type == 'num+':
                        lowest = y_test[i]
                        highest = y_test[i] + gap
                    elif gap_type == 'num-':
                        lowest = y_test[i] - gap
                        highest = y_test[i]
                    elif gap_type == 'per':
                        lowest = y_test[i] * (100 - gap) / 100
                        highest = y_test[i] * (100 + gap) / 100
                    elif gap_type == 'per+':
                        lowest = y_test[i]
                        highest = y_test[i] * (100 + gap) / 100
                    elif gap_type == 'per-':
                        lowest = y_test[i] * (100 - gap) / 100
                        highest = y_test[i]

                    if highest < lowest:
                        temp = lowest
                        lowest = highest
                        highest = temp

                    if lowest <= y_pred[i] <= highest:
                        indexes.append(i)

            if arr == True and avg == True and gap is not None:
                if success_indexes == True:
                    return diffs, avg_score, len(indexes), indexes
                else:
                    return diffs, avg_score, len(indexes)

            if arr == False and avg == True and gap is not None:
                if success_indexes == True:
                    return avg_score, len(indexes), indexes
                else:
                    return avg_score, len(indexes)

        if gap is not None:
            indexes = []

            if gap_type == 'exact':
                for i in range(y_test.shape[0]):
                    if y_test[i] == y_pred[i]:
                        indexes.append(i)
            else:
                for i in range(y_test.shape[0]):
                    lowest = None
                    highest = None

                    if gap_type == 'num':
                        lowest = y_test[i] - gap
                        highest = y_test[i] + gap
                    elif gap_type == 'num+':
                        lowest = y_test[i]
                        highest = y_test[i] + gap
                    elif gap_type == 'num-':
                        lowest = y_test[i] - gap
                        highest = y_test[i]
                    elif gap_type == 'per':
                        lowest = y_test[i] * (100 - gap) / 100
                        highest = y_test[i] * (100 + gap) / 100
                    elif gap_type == 'per+':
                        lowest = y_test[i]
                        highest = y_test[i] * (100 + gap) / 100
                    elif gap_type == 'per-':
                        lowest = y_test[i] * (100 - gap) / 100
                        highest = y_test[i]

                    if highest < lowest:
                        temp = lowest
                        lowest = highest
                        highest = temp

                    if lowest <= y_pred[i] <= highest:
                        indexes.append(i)

            if arr == True and avg == False:
                if success_indexes == True:
                    return diffs, len(indexes), indexes
                else:
                    return diffs, len(indexes)

            if arr == False and avg == False:
                if success_indexes == True:
                    return len(indexes), indexes
                else:
                    return len(indexes)


def extract_float(column, symbols):
    for i in range(column.shape[0]):
        if column[i] != np.nan or column[i] is None:
            for sym in symbols:
                column.values[i] = str(column.values[i]).replace(sym, '')

            column.values[i] = float(column.values[i])

    return column


def list_deletings(df, extra=None, del_null=True, null_tolerance=20, del_single=True, del_almost_single=False, almost_tolerance=50, suggest_extra=True, return_extra=False, unique_tolerance=10):
    will = None

    if extra is not None:
        for ftr in extra:
            del df[ftr]

    if del_null is True:
        tol = df.shape[0] * null_tolerance // 100
        print('The maximum tolerated null value amount is {}'.format(str(tol)))

        will_del = []
        for col in df.columns:
            amnt = df[col].isna().sum()
            if amnt > tol:
                will_del.append(str(col))
                print('{} will be deleted because it has {} null values and this is {} values more than tolerance'.format(str(col), str(amnt), str(amnt - tol)))

        for col in will_del:
            del df[col]

    if del_single is True:
        will_del = []

        for col in df.columns:
            if len(list(df[col].unique())) == 1:
                will_del.append(str(col))
                print('{} will be deleted because it has single value'.format(str(col)))

        for col in will_del:
            del df[col]

    if del_almost_single is True:
        will_del = []
        tol = df.shape[0] * almost_tolerance // 100
        print('The maximum tolerated same value amount is {}'.format(str(tol)))

        for col in df.columns:
            if df[col].value_counts().max() > tol:
                will_del.append(col)
                print('{} will be deleted because it has single (almost) value'.format(str(col)))

        for col in will_del:
            del df[col]

    if suggest_extra is True:
        tol = df.shape[0] * unique_tolerance // 100
        print('The maximum tolerated unique value amount is {} in string data'.format(str(tol)))
        will = []

        for col in df.columns:
            if str(df[col].dtype).__contains__('str') or str(df[col].dtype).__contains__('obj'):
                uniq = len(list(df[col].unique()))
                if uniq > tol:
                    will.append(col)
                    print('{} might be deleted because it has {} unique values and this is {} values more than tolerance'.format(str(col), str(uniq), str(uniq - tol)))

    if return_extra is True:
        return df, will
    else:
        return df


def col_counts(df, exclude=None, only=None):
    if only is not None:
        for col in df.columns:
            if col in only:
                print(df[col].value_counts())
                print('***')

    elif exclude is not None:
        for col in df.columns:
            if not col in exclude:
                print(df[col].value_counts())
                print('***')


def check_similarity(col1, col2):
  similar = True
  connections = {}
  col1 = list(col1)
  col2 = list(col2)

  for i in range(len(col1)):
    if col1[i] in connections and connections[col1[i]] != col2[i]:
      similar = False
      break
    else:
      connections[col1[i]] = col2[i]

  return similar


def find_broke(column, dtype=float, get_indexes=True, get_words=False, verbose=True, verbose_limit=10):
    limit = 0
    indexes = []
    words = []

    column = list(column)

    for i in range(len(column)):
        try:
            temp = dtype(column[i])
        except:
            if verbose is True and limit < verbose_limit:
                print(column[i])
                limit += 1

            if get_indexes is True:
                indexes.append(i)
            if get_words is True:
                words.append(column[i])

    if get_indexes is True and get_words is True:
        return indexes, words
    elif get_indexes is True:
        return indexes
    elif get_words is True:
        return words


def multi_split(df, test_size, output, threshold_set):
    from sklearn.feature_selection import VarianceThreshold as vart
    import pandas as pd

    test_size = test_size * 100

    temp = df.sample(frac=1)
    uniqs = list(df[output].unique())

    X_trains = []
    X_tests = []

    y_train = None
    y_test = None

    train = None
    test = None

    for uniq in uniqs:
        sub = temp[temp[output] == uniq]
        testlim = int(sub.shape[0] * test_size / 100)

        if test is None:
            test = sub.iloc[:testlim, :]
            train = sub.iloc[testlim:, :]
        else:
            test = pd.concat([test, sub.iloc[:testlim, :]])
            train = pd.concat([train, sub.iloc[testlim:, :]])

    y_train = train[output].values
    del train[output]
    X_trains.append(train.values)
    del train

    y_test = test[output].values
    del test[output]
    X_tests.append(test.values)
    del test

    for p in threshold_set:
        sub = temp.copy()

        suby = sub[output].values
        del sub[output]
        subx = sub.values
        del sub

        sel = vart(threshold=(p * (1 - p)))
        subx = sel.fit_transform(subx)

        sub = pd.DataFrame(subx)
        sub[output] = suby

        train = None
        test = None

        for uniq in uniqs:
            subt = sub[sub[output] == uniq]
            testlim = int(subt.shape[0] * test_size / 100)

            if test is None:
                test = subt.iloc[:testlim, :]
                train = subt.iloc[testlim:, :]
            else:
                test = pd.concat([test, subt.iloc[:testlim, :]])
                train = pd.concat([train, subt.iloc[testlim:, :]])

        y_train = train[output].values
        del train[output]
        X_trains.append(train.values)
        del train

        y_test = test[output].values
        del test[output]
        X_tests.append(test.values)
        del test

    return X_trains, X_tests, y_train, y_test


def expand_df(df, output, sampling_strategy):
    from imblearn.over_sampling import SMOTE as smote
    import pandas as pd

    temp = df.copy()
    cols = list(temp.columns)
    cols.remove(output)

    y = temp[output].values
    del temp[output]
    X = temp.values
    del temp

    sm = smote(sampling_strategy=sampling_strategy)
    X, y = sm.fit_resample(X, y)

    temp = pd.DataFrame(X, columns=cols)
    temp[output] = y

    return temp