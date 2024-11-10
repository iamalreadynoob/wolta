import numpy as np
import pandas as pd


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


def extract_float(column, symbols):
    for i in range(len(column)):
        if column[i] != np.nan and column[i] is not None:
            for sym in symbols:
                column[i] = str(column[i]).replace(sym, '')

            column[i] = float(column[i])

    return column


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


def split_as_df(X, y, features, output, test_size, random_state=42, shuffle=True, stratify=None):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

    dftrain = pd.DataFrame(X_train, columns=features)
    dftrain[output] = y_train

    dftest = pd.DataFrame(X_test, columns=features)
    dftest[output] = y_test

    return dftrain, dftest


def train_test_val_split(X, y, test_size, val_size, random_state=42, shuffle=True, stratify=None, stratify_for_val=True):
    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

    req_sample = int(X.shape[0] * val_size)
    new_ratio = req_sample / X_temp.shape[0]

    if stratify_for_val is True:
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=new_ratio, random_state=random_state, shuffle=shuffle, stratify=y_temp)
        return X_train, X_test, X_val, y_train, y_test, y_val
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=new_ratio, random_state=random_state, shuffle=shuffle)
        return X_train, X_test, X_val, y_train, y_test, y_val


def synthetic_expand(df, feature_info, shape_zero):
    from collections import Counter
    from random import randint, uniform
    import numpy as np
    import pandas as pd

    arrs = {}

    for col in df.columns:
        arrs[col] = list(df[col].values)

    new_create = shape_zero - df.shape[0]

    for ftr in arrs:
        if feature_info[ftr] == 'discrete':
            counts = Counter(arrs[ftr])
            boundaries = {}

            ceil = 0

            for item in counts:
                per = counts[item] / df.shape[0] * 100
                per = round(per)

                if per == 0:
                    per += 1

                boundaries[item] = {
                    'min': ceil,
                    'max': ceil + per
                }

                ceil += per

            for i in range(new_create):
                loc = randint(0, ceil - 1)

                for item in boundaries:
                    if boundaries[item]['min'] <= loc < boundaries[item]['max']:
                        arrs[ftr].append(item)

        elif feature_info[ftr] == 'continuous':
            local_min = min(arrs[ftr])
            local_max = max(arrs[ftr])
            local_mean = np.mean(arrs[ftr])

            low_gap = abs(local_mean - local_min)
            high_gap = abs(local_max - local_mean)

            is_int = True

            for i in range(df.shape[0]):
                if arrs[ftr][i] != round(arrs[ftr][i]):
                    is_int = False
                    break

            for i in range(new_create):
                plus = randint(0, 1)

                if plus == 0:
                    if is_int is True:
                        decrease = randint(0, int(round(low_gap)))
                        result = int(round(local_mean)) - decrease
                        arrs[ftr].append(result)
                    else:
                        decrease = uniform(0, low_gap)
                        result = local_mean - decrease
                        result = round(result, 2)
                        arrs[ftr].append(result)
                else:
                    if is_int is True:
                        increase = randint(0, int(high_gap))
                        result = int(round(local_mean)) + increase
                        arrs[ftr].append(result)
                    else:
                        increase = uniform(0, high_gap)
                        result = local_mean + increase
                        result = round(result, 2)
                        arrs[ftr].append(result)

    cols = list(arrs.keys())
    temp_df = pd.DataFrame(arrs[cols[0]], columns=[cols[0]])
    for i in range(1, len(cols)):
        temp_df[cols[i]] = np.array(arrs[cols[i]])

    return temp_df


def multi_split(df, labels, test_size, times=50):
    from collections import Counter

    temp = df.copy()
    best_df = temp.copy()
    best_score = 100000000

    ideal_ratios = {}

    train_size = 1 - test_size
    sample_amount = int(temp.shape[0] * train_size)
    test_amount = temp.shape[0] - sample_amount

    for label in labels:
        classes = len(list(Counter(temp[label].values).keys()))
        ideal_ratios[label] = (test_amount / classes) / test_amount

    for i in range(times):
        temp = temp.sample(frac=1)
        score = 0

        for label in labels:
            sub_test = temp[label].values[sample_amount:]

            counts = list(Counter(sub_test).values())
            max_count = max(counts)
            min_count = min(counts)

            max_ratio = max_count / test_amount
            min_ratio = min_count / test_amount

            upper_dif = abs(max_ratio - ideal_ratios[label])
            lower_dif = abs(min_ratio - ideal_ratios[label])
            dif_mean = (upper_dif + lower_dif) / 2

            score += dif_mean

        score /= len(labels)

        if score < best_score:
            best_df = temp.copy()
            best_score = score

    traindf = best_df.iloc[:sample_amount, :]
    testdf = best_df.iloc[sample_amount:, :]

    ytrain = {}
    ytest = {}

    for label in labels:
        ytrain[label] = traindf[label].values
        del traindf[label]

        ytest[label] = testdf[label].values
        del testdf[label]

    X_train = traindf.values
    X_test = testdf.values

    return X_train, X_test, ytrain, ytest


def corr_analyse(array, columns, un_w=0.1, w_s=0.5, s_p=0.9, verbose=True, get_matrix=False):
    import numpy as np

    matrix = np.corrcoef(array.T)

    results = {
        'uncorrelated': [],
        'weak': [],
        'strong': [],
        'perfect': []
    }

    width = array.shape[0]

    for row in range(width):
        for column in range(row + 1, width):
            score = matrix[row][column]

            entry = {
                'columns': [columns[row], columns[column]],
                'score': score
            }

            if score < 0:
                score *= -1

            if s_p < score <= 1:
                results['perfect'].append(entry)
            elif w_s < score <= s_p:
                results['strong'].append(entry)
            elif un_w < score <= w_s:
                results['weak'].append(entry)
            else:
                results['uncorrelated'].append(entry)

    if verbose is True:
        for kind in ['perfect', 'strong', 'weak', 'uncorrelated']:
            print('{}\n============'.format(kind.upper()))

            if len(results[kind]) == 0:
                print('There is no relation!')
            else:
                for relation in results[kind]:
                    print('{} - {}: {}'.format(relation['columns'][0], relation['columns'][1], relation['score']))

            print('************')

    if get_matrix is True:
        return results, matrix
    else:
        return results
