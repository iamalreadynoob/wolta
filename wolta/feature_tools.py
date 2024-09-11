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