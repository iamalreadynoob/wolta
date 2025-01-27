import numpy as np


def quest_selection(model_class, X_train, y_train, X_test, y_test, features, flag_one_tol, fin_tol, params=None, normal_acc=None, trials=100):
    from sklearn.metrics import accuracy_score
    import random
    import numpy as np

    if normal_acc is None:
        model = None

        if params is None:
            model = model_class()
        else:
            model = model_class(**params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        normal_acc = accuracy_score(y_test, y_pred)

    passed = []

    for feature in features:
        sub_X_train = np.delete(X_train, features.index(feature), axis=1)
        sub_X_test = np.delete(X_test, features.index(feature), axis=1)

        model = None

        if params is None:
            model = model_class()
        else:
            model = model_class(**params)

        model.fit(sub_X_train, y_train)
        y_pred = model.predict(sub_X_test)
        sub_acc = accuracy_score(y_test, y_pred)

        if sub_acc >= normal_acc or (normal_acc - sub_acc) <= flag_one_tol:
            passed.append(feature)

    if len(passed) <= 0:
        print("There is no feature to exclude")
    else:
        succeeded_combinations = []
        scores = []

        indexes = []
        for i in range(len(passed)):
            indexes.append(i)

        combinations = []
        for i in range(trials):
            sub_len = random.randint(1, len(passed) - 2)
            comb = []
            while sub_len != len(comb):
                rn = random.randint(0, len(passed) - 1)
                if not rn in comb:
                    comb.append(rn)
            combinations.append(comb)

        for comb in combinations:
            will_del_indexes = []
            for index in comb:
                will_del_indexes.append(features.index(passed[index]))
            sub_X_train = np.delete(X_train, will_del_indexes, axis=1)
            sub_X_test = np.delete(X_test, will_del_indexes, axis=1)

            model = None

            if params is None:
                model = model_class()
            else:
                model = model_class(**params)

            model.fit(sub_X_train, y_train)
            y_pred = model.predict(sub_X_test)
            sub_acc = accuracy_score(y_test, y_pred)

            if sub_acc >= normal_acc or (normal_acc - sub_acc) <= fin_tol:
                feature_comb = []
                for index in comb:
                    feature_comb.append(passed[index])

                succeeded_combinations.append(feature_comb)
                scores.append(sub_acc)

        if len(succeeded_combinations) > 0:
            for i in range(len(succeeded_combinations)):
                print('if you delete {}, then acc will {} with {} change'.format(str(succeeded_combinations[i]), str(scores[i]), str(scores[i] - normal_acc)))
        else:
            print("There is no future to suitable to exclude")

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


def rand_arr(outputs, values=None, strategy='equal', arr_size=1):
    from random import randint
    import numpy as np

    y_rand = []

    if values is None:
        strategy = 'equal'

    if strategy == 'equal':
        for _ in range(arr_size):
            y_rand.append(outputs[randint(0, len(outputs) - 1)])
    elif strategy == 'weighted':
        for _ in range(arr_size):
            destiny = randint(0, sum(values) - 1)
            ceil = 0
            for i in range(len(values)):
                ceil += values[i]
                if destiny < ceil:
                    y_rand.append(outputs[i])
                    break
    elif strategy == 'piled':
        for _ in range(arr_size):
            destiny = randint(0, values[-1] - 1)
            for i in range(len(values)):
                if destiny < values[i]:
                    y_rand.append(outputs[i])
                    break

    return np.array(y_rand)
