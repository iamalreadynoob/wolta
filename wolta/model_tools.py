import numpy as np


def get_score(y_test, y_pred, metrics=None, average='weighted'):
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, log_loss, matthews_corrcoef, precision_score, recall_score, zero_one_loss

    if metrics is None:
        metrics = ['acc']

    output = ''
    scores = {}

    for metric in metrics:
        if metric == 'acc':
            score = accuracy_score(y_test, y_pred)
            scores[metric] = score

            if output == '':
                output = 'Accuracy Score: {}'.format(str(score))
            else:
                output += '\nAccuracy Score: {}'.format(str(score))

        elif metric == 'f1':
            score = f1_score(y_test, y_pred, average=average)
            scores[metric] = score

            if output == '':
                output = 'F1 Score (weighted): {}'.format(str(score))
            else:
                output += '\nF1 Score (weighted): {}'.format(str(score))

        elif metric == 'hamming':
            score = hamming_loss(y_test, y_pred)
            scores[metric] = score

            if output == '':
                output = 'Hamming Loss: {}'.format(str(score))
            else:
                output += '\nHamming Loss: {}'.format(str(score))

        elif metric == 'jaccard':
            score = jaccard_score(y_test, y_pred, average=average)
            scores[metric] = score

            if output == '':
                output = 'Jaccard: {}'.format(str(score))
            else:
                output += '\nJaccard: {}'.format(str(score))

        elif metric == 'log':
            score = log_loss(y_test, y_pred)
            scores[metric] = score

            if output == '':
                output = 'Log Loss: {}'.format(str(score))
            else:
                output += '\nLog Loss: {}'.format(str(score))

        elif metric == 'mcc':
            score = matthews_corrcoef(y_test, y_pred)
            scores[metric] = score

            if output == '':
                output = 'MCC: {}'.format(str(score))
            else:
                output += '\nMCC: {}'.format(str(score))

        elif metric == 'precision':
            score = precision_score(y_test, y_pred, average=average)
            scores[metric] = score

            if output == '':
                output = 'Precision Score: {}'.format(str(score))
            else:
                output += '\nPrecision Score: {}'.format(str(score))

        elif metric == 'recall':
            score = recall_score(y_test, y_pred, average=average)
            scores[metric] = score

            if output == '':
                output = 'Recall Score: {}'.format(str(score))
            else:
                output += '\nRecall Score: {}'.format(str(score))

        elif metric == 'zol':
            score = zero_one_loss(y_test, y_pred)
            scores[metric] = score

            if output == '':
                output = 'Zero One Loss: {}'.format(str(score))
            else:
                output += '\nZero One Loss: {}'.format(str(score))

    print(output)
    return scores


def get_supported_metrics():
    return ['acc', 'f1', 'hamming', 'jaccard', 'log', 'mcc', 'precision', 'recall', 'zol']


def get_avg_options():
    return ['micro', 'macro', 'binary', 'weighted', 'samples']


def do_voting(y_pred_list, combinations, strategy='avg'):
    if strategy == 'avg':
        results = []

        for comb in combinations:
            y_sum = None

            for index in comb:
                if y_sum is None:
                    y_sum = y_pred_list[index]
                else:
                    y_sum += y_pred_list[index]

            y_sum = y_sum / len(comb)
            y_sum = y_sum.astype(int)
            results.append(y_sum)

        return results

    elif strategy == 'mode':
        from math import ceil
        import numpy as np

        results = []

        for comb in combinations:
            selected = []
            for index in comb:
                selected.append(y_pred_list[index])

            stack = np.concatenate(selected, axis=1)
            del selected

            req_min = ceil(len(list(stack[0])) / 2)
            length = stack.shape[0]
            modes = []

            for i in range(length):
                result = 0
                max_times = 0
                min_space = max_times + 1
                row = list(stack[i])

                while len(row) > 0:
                    loc = 0
                    obj = row[0]
                    times = 0

                    while loc < len(row):
                        if obj == row[loc]:
                            del row[loc]
                            times += 1
                        else:
                            loc += 1

                    if times > max_times:
                        result = obj
                        max_times = times
                        min_space = max_times + 1

                    if max_times >= req_min or len(row) < min_space:
                        break

                modes.append(result)

            modes = np.array(modes)
            results.append(modes)

        return results


def do_combinations(indexes, min_item, max_item):
    import itertools
    combinations = []

    for i in range(min_item, max_item + 1):
        combs = list(itertools.combinations(indexes, i))

        for comb in combs:
            combinations.append(list(comb))

    return combinations


def examine_time(model, X_train, y_train):
    import time

    start = time.process_time()
    model.fit(X_train, y_train)
    end = time.process_time()

    consumed = end - start
    return consumed, model


class WelkinClassification:
    def __init__(self, strategy='travel', priority=None, limit=None):
        self.min = {}
        self.max = {}
        self.strategy = strategy
        self.priority = priority
        self.limit = limit

    def fit(self, X_train, y_train):
        import sys

        for i in range(y_train.shape[0]):
            if not y_train[i] in self.min:
                self.min[y_train[i]] = {}
                self.max[y_train[i]] = {}

                for j in range(X_train.shape[1]):
                    self.min[y_train[i]][j] = sys.maxsize
                    self.max[y_train[i]][j] = -sys.maxsize - 1

            row = list(X_train[i])

            for j in range(len(row)):
                if row[j] < self.min[y_train[i]][j]:
                    self.min[y_train[i]][j] = row[j]

                elif row[j] > self.max[y_train[i]][j]:
                    self.max[y_train[i]][j] = row[j]

    def predict(self, X_test):
        if self.strategy == 'travel':
            import numpy as np
            from random import randint

            pred = []

            stage_line = list(self.min.keys())
            if self.priority is not None:
                stage_line = self.priority

            for i in range(X_test.shape[0]):
                max_key = list(self.max.keys())[randint(0, len(list(self.max.keys())) - 1)]
                max_zone = 0

                row = list(X_test[i])

                for output in stage_line:
                    count = 0

                    for j in range(X_test.shape[1]):
                        if row[j] >= self.min[output][j] and row[j] <= self.max[output][j]:
                            count += 1

                    if count > max_zone:
                        max_zone = count
                        max_key = output

                        if count == X_test.shape[0]:
                            break

                pred.append(max_key)

            return np.array(pred)

        elif self.strategy == 'limit' and self.limit is not None:
            import numpy as np
            from random import randint

            pred = []

            stage_line = list(self.min.keys())
            if self.priority is not None:
                stage_line = self.priority

            for i in range(X_test.shape[0]):
                max_key = list(self.max.keys())[randint(0, len(list(self.max.keys())) - 1)]
                max_zone = 0

                row = list(X_test[i])

                for output in stage_line:
                    count = 0

                    for j in range(X_test.shape[1]):
                        if row[j] >= self.min[output][j] and row[j] <= self.max[output][j]:
                            count += 1

                    if count > max_zone:
                        max_zone = count
                        max_key = output

                        if count == X_test.shape[0] or count >= self.limit:
                            break

                pred.append(max_key)

            return np.array(pred)


class DistRegressor:
    import pandas as pd
    import numpy as np

    def __init__(self, verbose=True, clf_model=None, clf_params=None, reg_model=None, reg_params=None, efficiency='time', rus=True):
        self.type_zero_regressor = None
        self.type_one_regressor = None
        self.type_two_regressor = None
        self.rus = rus
        self.verbose = verbose
        self.efficiency = efficiency

        self.clf_model = clf_model
        self.reg_model = reg_model
        self.reg_params = reg_params

        if self.clf_model is not None:
            if clf_params is None:
                self.clf_model = clf_model()
            else:
                self.clf_model = clf_model(**clf_params)

    def fit(self, X_train, y_train):
        std = np.std(y_train)
        amin = np.amin(y_train)
        amax = np.amax(y_train)
        mean = np.mean(y_train)

        border_one = mean - std
        border_two = mean + std

        if self.verbose:
            print('Basic calculations are completed')

        clf_arr = []
        one_side = 0
        type_zero_X = []
        type_one_X = []
        type_two_X = []
        type_zero_y = []
        type_one_y = []
        type_two_y = []

        if self.efficiency == 'space':
            for i in range(y_train.shape[0]):
                if y_train[i] >= amin and y_train[i] <= border_one:
                    clf_arr.append(0)
                    one_side += 1
                elif y_train[i] > border_one and y_train[i] < border_two:
                    clf_arr.append(1)
                else:
                    clf_arr.append(2)

            del type_zero_X, type_one_X, type_two_X, type_zero_y, type_one_y, type_two_y

        elif self.efficiency == 'time':
            for i in range(y_train.shape[0]):
                if y_train[i] >= min and y_train[i] <= border_one:
                    clf_arr.append(0)
                    type_zero_X.append(list(X_train[i]))
                    type_zero_y.append(y_train[i])
                elif y_train[i] > border_one and y_train[i] < border_two:
                    clf_arr.append(1)
                    type_one_X.append(list(X_train[i]))
                    type_one_y.append(y_train[i])
                else:
                    clf_arr.append(2)
                    type_two_X.append(list(X_train[i]))
                    type_two_y.append(y_train[i])

            type_zero_X = np.array(type_zero_X)
            type_one_X = np.array(type_one_X)
            type_two_X = np.array(type_two_X)
            type_zero_y = np.array(type_zero_y)
            type_one_y = np.array(type_one_y)
            type_two_y = np.array(type_two_y)

        clf_arr = np.array(clf_arr)

        if self.verbose:
            print('Result array is ready for classification')

        small_X = None
        if self.rus:
            from imblearn.under_sampling import RandomUnderSampler

            strategy = {1: one_side}
            rand = RandomUnderSampler(sampling_strategy=strategy)

            small_X, clf_arr = rand.fit_resample(X_train, clf_arr)


        if self.clf_model is None:
            from catboost import CatBoostClassifier

            self.clf_model = CatBoostClassifier(verbose=False, iterations=20)

            if self.rus:
                self.clf_model.fit(small_X, clf_arr)
            else:
                self.clf_model.fit(X_train, clf_arr)
        else:
            if self.rus:
                self.clf_model.fit(small_X, clf_arr)
            else:
                self.clf_model.fit(X_train, clf_arr)

        del small_X, clf_arr

        if self.verbose:
            print('Classification model has been trained')

        if self.efficiency == 'space':
            sub_X = []
            sub_y = []
            for i in range(X_train.shape[0]):
                if y_train[i] >= amin and y_train[i] <= border_one:
                    sub_X.append(list(X_train[i]))
                    sub_y.append(y_train[i])

            sub_X = np.array(sub_X)
            sub_y = np.array(sub_y)

            if self.reg_model is None:
                from catboost import CatBoostRegressor

                self.type_zero_regressor = CatBoostRegressor(verbose=False, iterations=20)
                self.type_zero_regressor.fit(sub_X, sub_y)
            else:
                if self.reg_params is None:
                    self.type_zero_regressor = self.reg_model()
                else:
                    self.type_zero_regressor = self.reg_model(**self.reg_params)

                self.type_zero_regressor.fit(sub_X, sub_y)

            sub_X = []
            sub_y = []
            for i in range(X_train.shape[0]):
                if y_train[i] > border_one and y_train[i] < border_two:
                    sub_X.append(list(X_train[i]))
                    sub_y.append(y_train[i])

            sub_X = np.array(sub_X)
            sub_y = np.array(sub_y)

            if self.reg_model is None:
                from catboost import CatBoostRegressor

                self.type_one_regressor = CatBoostRegressor(verbose=False, iterations=20)
                self.type_one_regressor.fit(sub_X, sub_y)
            else:
                if self.reg_params is None:
                    self.type_one_regressor = self.reg_model()
                else:
                    self.type_one_regressor = self.reg_model(**self.reg_params)

                self.type_one_regressor.fit(sub_X, sub_y)

            sub_X = []
            sub_y = []
            for i in range(X_train.shape[0]):
                if y_train[i] >= border_two and y_train[i] <= amax:
                    sub_X.append(list(X_train[i]))
                    sub_y.append(y_train[i])

            sub_X = np.array(sub_X)
            sub_y = np.array(sub_y)

            if self.reg_model is None:
                from catboost import CatBoostRegressor

                self.type_two_regressor = CatBoostRegressor(verbose=False, iterations=20)
                self.type_two_regressor.fit(sub_X, sub_y)
            else:
                if self.reg_params is None:
                    self.type_two_regressor = self.reg_model()
                else:
                    self.type_two_regressor = self.reg_model(**self.reg_params)

                self.type_two_regressor.fit(sub_X, sub_y)

        elif self.efficiency == 'time':
            if self.reg_model is None:
                from catboost import CatBoostRegressor

                self.type_zero_regressor = CatBoostRegressor(verbose=False, iterations=20)
                self.type_one_regressor = CatBoostRegressor(verbose=False, iterations=20)
                self.type_two_regressor = CatBoostRegressor(verbose=False, iterations=20)

                self.type_zero_regressor.fit(type_zero_X, type_zero_y)
                del type_zero_X, type_zero_y

                self.type_one_regressor.fit(type_one_X, type_one_y)
                del type_one_X, type_one_y

                self.type_two_regressor.fit(type_two_X, type_two_y)
                del type_two_X, type_two_y
            else:
                if self.reg_params is None:
                    self.type_zero_regressor = self.reg_model()
                    self.type_one_regressor = self.reg_model()
                    self.type_two_regressor = self.reg_model()
                else:
                    self.type_zero_regressor = self.reg_model(**self.reg_params)
                    self.type_one_regressor = self.reg_model(**self.reg_params)
                    self.type_two_regressor = self.reg_model(**self.reg_params)

                self.type_zero_regressor.fit(type_zero_X, type_zero_y)
                del type_zero_X, type_zero_y

                self.type_one_regressor.fit(type_one_X, type_one_y)
                del type_one_X, type_one_y

                self.type_two_regressor.fit(type_two_X, type_two_y)
                del type_two_X, type_two_y

        if self.verbose:
            print('Regression models have been trained')

    def predict(self, X_test):
        y_pred = []

        for i in range(X_test.shape[0]):
            category = self.clf_model.predict([X_test[i]])

            if category == 0:
                y_pred.append(self.type_zero_regressor.predict([X_test[i]]))
            elif category == 1:
                y_pred.append(self.type_one_regressor.predict([X_test[i]]))
            else:
                y_pred.append(self.type_two_regressor.predict([X_test[i]]))

        y_pred = np.array(y_pred)
        return y_pred