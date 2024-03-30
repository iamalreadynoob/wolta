import os


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

            if amount in strategy:
                space[col] = amount

        if print_dict:
            print(space)

        return space


def make_numerics(column):
    unique_vals = list(column.unique())
    space = {}

    for i in range(len(unique_vals)):
        space[unique_vals[i]] = i

    column = column.map(space)

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


def load_by_parts(paths, strategy='default', deleted_columns=None, print_description=False):
    import pandas as pd

    if strategy == 'default':
        dfs = []
        loc = 0
        for path in paths:
            df = pd.read_csv(path)

            if deleted_columns is not None:
                for col in deleted_columns:
                    del df[col]

            dfs.append(df)

            if print_description:
                loc += 1
                print('{} out of {} paths done'.format(str(loc), str(len(paths))))

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
            df = pd.read_csv(path, dtype=dtype)

            if deleted_columns is not None:
                for col in deleted_columns:
                    del df[col]

            dfs.append(df)

            if print_description:
                loc += 1
                print('{} out of {} paths done'.format(str(loc), str(len(paths))))

        main_df = pd.concat(dfs)
        del dfs

        return main_df


def create_chunks(path, sample_amount, target_dir=None, print_description=False, chunk_name='part'):
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